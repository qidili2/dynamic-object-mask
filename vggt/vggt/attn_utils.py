"""
vggt/attn_utils.py  –  Hook utilities for VGGT attention analysis.

Token layout per frame (identical for frame-attention and global-attention):
  index 0              : camera token
  index 1 .. ps-1      : register tokens  (ps = patch_start_idx, default 5)
  index ps .. P-1      : patch tokens     Pp = P - ps = H_p * W_p

In global attention the S frames are concatenated into one long sequence
of length N = S * P:
  frame i occupies        [i*P : (i+1)*P]
  frame i camera token    [i*P]
  frame i register tokens [i*P+1 : i*P+ps]
  frame i patch tokens    [i*P+ps : (i+1)*P]

Three capture classes
─────────────────────
  GlobalAttnCapture
      Stores the FULL head-averaged attention matrix [N, N]  (N = S*P)
      for each selected global_block layer.
      Proper full softmax: each row i is softmax over ALL N keys.
      Includes all token types: camera, register, and patch.
      Processes one head at a time to keep peak GPU memory bounded.

      Memory: [S*P, S*P] float32 on CPU per layer.
        S=4 → 5496² × 4 B = 121 MB / layer
        S=8 → 10992² × 4 B = 483 MB / layer
      Choose --layers carefully for large S.

  FrameAttnCapture
      Stores the FULL head-averaged within-frame attention matrix [S, P, P]
      for each selected frame_block layer.
      Proper softmax over all P tokens in the frame.
      Includes all token types.

      Memory: S × P² × 4 B per layer.
        S=4, P=1374 → 4 × 1374² × 4 B ≈ 30 MB / layer

  BlockResidualCapture
      L2 feature-update magnitude per token.
      Uses register_forward_pre/post_hook — no attention computation.

Usage example
─────────────
    with GlobalAttnCapture(agg, layers, S, P, ps) as gcap, \\
         FrameAttnCapture(agg, layers, S, P, ps) as fcap, \\
         BlockResidualCapture(agg, layers, S, P, ps) as rcap:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                agg(images)
"""

import torch
import torch.nn.functional as F
import numpy as np
from types import MethodType
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Direction 1 & 3 : Global attention full-matrix capture
# ─────────────────────────────────────────────────────────────────────────────

class GlobalAttnCapture:
    """
    Patches selected global_blocks to capture the complete attention matrix.

    Stored per selected layer:
        data[layer_id]['attn']   np.ndarray [S*P, S*P]  float32
            Head-averaged (and batch-averaged) attention matrix.
            Full softmax: row i sums to 1 over all N = S*P keys.
            Includes ALL token types: camera (index 0 per frame),
            register (indices 1..ps-1), and patch (indices ps..P-1).
        data[layer_id]['count']  int  (number of forward passes accumulated)

    Sub-matrix naming convention
    ────────────────────────────
      block_attn(lid, i, j)   [P, P]   all tokens of frame i → all tokens of frame j
      recv_patch(lid, i, j)   [Pp]     sum of attention received by frame-j PATCH tokens
                                       from ALL tokens of frame i  (column sums, patch cols)
      cam_row(lid, i)         [N]      attention row of camera token in frame i → all tokens
    """

    def __init__(
        self,
        agg,
        selected_layers: List[int],
        S: int,
        P: int,            # total tokens per frame  (= patch_start + Pp)
        patch_start: int,  # first patch token index per frame  (default 5)
        selected_heads: Optional[List[int]] = None,
    ):
        """
        Args:
            agg:             model.aggregator
            selected_layers: indices into agg.global_blocks  (0 .. depth-1)
            S:               number of frames in the sequence
            P:               tokens per frame  (camera + register + patches)
            patch_start:     ps = 1 + num_register_tokens  (default 5)
            selected_heads:  if given, average only over these head indices;
                             otherwise average over all heads.
                             Use this to limit memory when S is large.
        """
        self.agg = agg
        self.selected_layers = list(selected_layers)
        self.S = S
        self.P = P
        self.ps = patch_start
        self.Pp = P - patch_start
        self.selected_heads = selected_heads

        # data[lid] = {'attn': np.ndarray [N, N], 'count': int}
        self.data: Dict[int, Dict] = {}
        self._saved: Dict[int, object] = {}

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        for lid in self.selected_layers:
            blk = self.agg.global_blocks[lid]
            self._saved[lid] = blk.attn.forward
            blk.attn.forward = MethodType(self._make_hook(lid), blk.attn)
        return self

    def __exit__(self, *args):
        for lid, fwd in self._saved.items():
            self.agg.global_blocks[lid].attn.forward = fwd
        self._saved.clear()

    # ── hook factory ──────────────────────────────────────────────────────────

    def _make_hook(self, lid: int):
        sel_heads = self.selected_heads
        data = self.data

        def _forward(self_attn, x, pos=None, return_cam_attn=False):
            B, N, C = x.shape   # N = S * P

            # ── compute Q, K, V ──────────────────────────────────────────────
            qkv = self_attn.qkv(x).reshape(B, N, 3, self_attn.num_heads, self_attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B, H, N, D]
            q, k, v = qkv.unbind(0)              # each [B, H, N, D]
            q, k = self_attn.q_norm(q), self_attn.k_norm(k)

            if self_attn.rope is not None:
                q = self_attn.rope(q, pos)
                k = self_attn.rope(k, pos)

            H_n = self_attn.num_heads
            scale = self_attn.scale
            h_idx = sel_heads if sel_heads is not None else list(range(H_n))

            # ── full [N, N] attention, head-by-head to limit peak GPU RAM ────
            # For S=4, P=1374: N=5496 → logits [B, N, N] ≈ 121 MB/head (float32)
            # We accumulate on CPU and average over heads + batch.
            with torch.no_grad():
                # Accumulate on GPU to avoid 16 blocking GPU→CPU copies per layer.
                # One single .cpu() at the end after averaging over heads + batch.
                attn_acc = torch.zeros(N, N, dtype=torch.float32, device=x.device)

                for hi in h_idx:
                    q_h = (q[:, hi, :, :] * scale).float()  # [B, N, D]
                    k_h = k[:, hi, :, :].float()             # [B, N, D]
                    # Full N×N logits → softmax over ALL N keys (proper global softmax)
                    logits = torch.bmm(q_h, k_h.transpose(-1, -2))  # [B, N, N]
                    attn_h = logits.softmax(dim=-1)                  # [B, N, N]
                    attn_acc += attn_h.mean(dim=0)                   # [N, N] GPU accumulation

                attn_avg = (attn_acc / len(h_idx)).cpu().numpy()     # one transfer per layer

                ld = data.setdefault(lid, {'attn': None, 'count': 0})
                if ld['attn'] is None:
                    ld['attn'] = attn_avg
                else:
                    ld['attn'] = ld['attn'] + attn_avg
                ld['count'] += 1

            # ── model output: fused attention (unchanged behaviour) ───────────
            x_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x_out = x_out.transpose(1, 2).reshape(B, N, C)
            x_out = self_attn.proj(x_out)
            x_out = self_attn.proj_drop(x_out)
            return x_out

        return _forward

    # ── accessors ─────────────────────────────────────────────────────────────

    def mean_attn(self, layer_id: int) -> Optional[np.ndarray]:
        """Return mean [N, N] attention matrix for given layer."""
        e = self.data.get(layer_id)
        if e is None or e['attn'] is None:
            return None
        return e['attn'] / e['count']

    def block_attn(self, layer_id: int, i: int, j: int) -> Optional[np.ndarray]:
        """
        [P, P] sub-block of global attention: ALL tokens of frame i → ALL tokens of frame j.
        Row dimension: frame-i tokens (camera, register, patch).
        Col dimension: frame-j tokens (camera, register, patch).
        """
        A = self.mean_attn(layer_id)
        if A is None:
            return None
        P = self.P
        return A[i*P:(i+1)*P, j*P:(j+1)*P]   # [P, P]

    def recv_patch(self, layer_id: int, i: int, j: int) -> Optional[np.ndarray]:
        """
        [Pp] received attention: sum over ALL frame-i tokens (camera+reg+patch)
        of the attention going to each frame-j PATCH token.

        This is the column sum of block_attn(i,j) restricted to patch columns,
        summed over all rows (all query token types from frame i).
        """
        blk = self.block_attn(layer_id, i, j)
        if blk is None:
            return None
        # blk shape [P, P]; patch columns: ps..P
        return blk[:, self.ps:].sum(axis=0)    # [Pp]

    def cam_row(self, layer_id: int, frame: int) -> Optional[np.ndarray]:
        """
        [N] attention row for the camera token of the given frame.
        Shows what the camera token attends to across all S*P tokens.
        """
        A = self.mean_attn(layer_id)
        if A is None:
            return None
        return A[frame * self.P, :]    # [N]

    def reg_rows(self, layer_id: int, frame: int) -> Optional[np.ndarray]:
        """
        [num_reg, N] attention rows for the register tokens of the given frame.
        num_reg = ps - 1  (indices 1..ps-1 within the frame).
        """
        A = self.mean_attn(layer_id)
        if A is None:
            return None
        row_start = frame * self.P + 1
        row_end   = frame * self.P + self.ps
        return A[row_start:row_end, :]    # [num_reg, N]

    def layer_mean_recv_patch(self, i: int, j: int) -> Optional[np.ndarray]:
        """Average recv_patch across all selected layers → [Pp]."""
        arrays = [self.recv_patch(lid, i, j) for lid in self.selected_layers]
        arrays = [a for a in arrays if a is not None]
        return None if not arrays else np.mean(np.stack(arrays, 0), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Direction 2 : Frame-wise attention full-matrix capture
# ─────────────────────────────────────────────────────────────────────────────

class FrameAttnCapture:
    """
    Patches selected frame_blocks to capture the complete within-frame attention.

    In frame attention, each frame is processed independently:
        input shape [B*S, P, C]

    Stored per selected layer:
        data[layer_id]['attn']   np.ndarray [S, P, P]  float32
            Head-averaged (and batch-averaged) within-frame attention matrix.
            Full softmax: row i sums to 1 over all P tokens in the frame.
            Includes ALL token types: camera, register, and patch.
        data[layer_id]['count']  int

    Memory: S × P² × 4 B per layer  (≈30 MB for S=4, P=1374).
    """

    def __init__(
        self,
        agg,
        selected_layers: List[int],
        S: int,
        P: int,
        patch_start: int,
        selected_heads: Optional[List[int]] = None,
    ):
        self.agg = agg
        self.selected_layers = list(selected_layers)
        self.S = S
        self.P = P
        self.ps = patch_start
        self.Pp = P - patch_start
        self.selected_heads = selected_heads

        # data[lid] = {'attn': np.ndarray [S, P, P], 'count': int}
        self.data: Dict[int, Dict] = {}
        self._saved: Dict[int, object] = {}

    def __enter__(self):
        for lid in self.selected_layers:
            blk = self.agg.frame_blocks[lid]
            self._saved[lid] = blk.attn.forward
            blk.attn.forward = MethodType(self._make_hook(lid), blk.attn)
        return self

    def __exit__(self, *args):
        for lid, fwd in self._saved.items():
            self.agg.frame_blocks[lid].attn.forward = fwd
        self._saved.clear()

    def _make_hook(self, lid: int):
        S = self.S
        sel_heads = self.selected_heads
        data = self.data

        def _forward(self_attn, x, pos=None, return_cam_attn=False):
            BS, N, C = x.shape   # BS = B*S, N = P
            B = BS // S

            qkv = self_attn.qkv(x).reshape(BS, N, 3, self_attn.num_heads, self_attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # each [BS, H, P, D]
            q, k = self_attn.q_norm(q), self_attn.k_norm(k)

            if self_attn.rope is not None:
                q = self_attn.rope(q, pos)
                k = self_attn.rope(k, pos)

            H_n = self_attn.num_heads
            scale = self_attn.scale
            h_idx = sel_heads if sel_heads is not None else list(range(H_n))

            with torch.no_grad():
                # Process head-by-head; accumulate [BS, N, N] on CPU
                # Peak GPU: [BS, N, N] per head ≈ 4*1374² ≈ 30 MB (float32)
                attn_acc = None

                for hi in h_idx:
                    q_h = (q[:, hi, :, :] * scale).float()   # [BS, N, D]
                    k_h = k[:, hi, :, :].float()              # [BS, N, D]
                    attn_h = torch.bmm(q_h, k_h.transpose(-1, -2)).softmax(dim=-1)  # [BS, N, N]
                    attn_h = attn_h.cpu()

                    if attn_acc is None:
                        attn_acc = attn_h
                    else:
                        attn_acc = attn_acc + attn_h

                # Average over heads → [BS, N, N]
                attn_avg = (attn_acc / len(h_idx)).numpy()   # [BS, P, P]

                # Reshape to [B, S, P, P], average over batch → [S, P, P]
                attn_avg = attn_avg.reshape(B, S, N, N).mean(axis=0)

                ld = data.setdefault(lid, {'attn': None, 'count': 0})
                if ld['attn'] is None:
                    ld['attn'] = attn_avg
                else:
                    ld['attn'] = ld['attn'] + attn_avg
                ld['count'] += 1

            x_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x_out = x_out.transpose(1, 2).reshape(BS, N, C)
            x_out = self_attn.proj(x_out)
            x_out = self_attn.proj_drop(x_out)
            return x_out

        return _forward

    # ── accessors ─────────────────────────────────────────────────────────────

    def mean_attn(self, layer_id: int) -> Optional[np.ndarray]:
        """Return mean [S, P, P] within-frame attention for given layer."""
        e = self.data.get(layer_id)
        if e is None or e['attn'] is None:
            return None
        return e['attn'] / e['count']

    def recv_patch(self, layer_id: int, s: int) -> Optional[np.ndarray]:
        """
        [Pp] received attention for frame s:
        sum over ALL query tokens (camera+reg+patch) of attention to each PATCH token.
        = column sums of frame_attn[s, :, ps:].
        """
        A = self.mean_attn(layer_id)
        if A is None:
            return None
        # A[s] shape [P, P]; sum all rows, restrict to patch columns
        return A[s, :, self.ps:].sum(axis=0)    # [Pp]

    def cam_row(self, layer_id: int, s: int) -> Optional[np.ndarray]:
        """[P] attention row of camera token for frame s → all tokens in same frame."""
        A = self.mean_attn(layer_id)
        if A is None:
            return None
        return A[s, 0, :]    # camera token is index 0

    def layer_mean_recv_patch(self, s: int) -> Optional[np.ndarray]:
        """Average recv_patch across all selected layers for frame s → [Pp]."""
        arrays = [self.recv_patch(lid, s) for lid in self.selected_layers]
        arrays = [a for a in arrays if a is not None]
        return None if not arrays else np.mean(np.stack(arrays, 0), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Direction 4 : Block-level residual magnitude capture
# ─────────────────────────────────────────────────────────────────────────────

class BlockResidualCapture:
    """
    Uses register_forward_pre_hook / register_forward_hook on transformer blocks
    to capture the L2 norm of the feature update per token.

    For frame_blocks:  input/output shape [B*S, P, C]
    For global_blocks: input/output shape [B, S*P, C]

    Residual = ||x_out - x_in||_2 per token (full block: attn + MLP combined).

    After context:
        frame_data[layer_id]   np.ndarray [S, Pp]  – patch residual norms
        global_data[layer_id]  np.ndarray [S, Pp]  – patch residual norms
    """

    def __init__(
        self,
        agg,
        selected_layers: List[int],
        S: int,
        P: int,
        patch_start: int,
    ):
        self.agg = agg
        self.selected_layers = list(selected_layers)
        self.S = S
        self.P = P
        self.ps = patch_start
        self.Pp = P - patch_start

        self.frame_data: Dict[int, Dict] = {}
        self.global_data: Dict[int, Dict] = {}
        self._handles: List = []

    def __enter__(self):
        S, P, ps, Pp = self.S, self.P, self.ps, self.Pp

        for lid in self.selected_layers:
            # ── frame block ───────────────────────────────────────────────────
            fb = self.agg.frame_blocks[lid]
            fb_store = {}

            def _fb_pre(m, inp, _s=fb_store):
                _s['x'] = inp[0].detach()

            def _fb_post(m, inp, out, _s=fb_store, _lid=lid):
                x_in = _s.get('x')
                if x_in is None:
                    return
                delta = (out.detach().float() - x_in.float()).norm(dim=-1)   # [BS, P]
                B = delta.shape[0] // S
                patch = delta[:, ps:].view(B, S, Pp).mean(dim=0).cpu().numpy()  # [S, Pp]
                d = self.frame_data.setdefault(_lid, {'delta': None, 'count': 0})
                d['delta'] = patch if d['delta'] is None else d['delta'] + patch
                d['count'] += 1

            self._handles.append(fb.register_forward_pre_hook(_fb_pre))
            self._handles.append(fb.register_forward_hook(_fb_post))

            # ── global block ──────────────────────────────────────────────────
            gb = self.agg.global_blocks[lid]
            gb_store = {}

            def _gb_pre(m, inp, _s=gb_store):
                _s['x'] = inp[0].detach()

            def _gb_post(m, inp, out, _s=gb_store, _lid=lid):
                x_in = _s.get('x')
                if x_in is None:
                    return
                B_g = x_in.shape[0]
                delta = (out.detach().float() - x_in.float()).norm(dim=-1)   # [B, S*P]
                patch = delta.view(B_g, S, P)[:, :, ps:].mean(dim=0).cpu().numpy()  # [S, Pp]
                d = self.global_data.setdefault(_lid, {'delta': None, 'count': 0})
                d['delta'] = patch if d['delta'] is None else d['delta'] + patch
                d['count'] += 1

            self._handles.append(gb.register_forward_pre_hook(_gb_pre))
            self._handles.append(gb.register_forward_hook(_gb_post))

        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _mean(self, store, layer_id):
        e = store.get(layer_id)
        if e is None or e['delta'] is None:
            return None
        return e['delta'] / e['count']

    def frame_residuals(self, layer_id: int) -> Optional[np.ndarray]:
        """[S, Pp] mean block residual magnitude for frame_blocks[layer_id]."""
        return self._mean(self.frame_data, layer_id)

    def global_residuals(self, layer_id: int) -> Optional[np.ndarray]:
        """[S, Pp] mean block residual magnitude for global_blocks[layer_id]."""
        return self._mean(self.global_data, layer_id)

    def layer_mean_frame(self) -> Optional[np.ndarray]:
        arrays = [self.frame_residuals(l) for l in self.selected_layers]
        arrays = [a for a in arrays if a is not None]
        return None if not arrays else np.mean(np.stack(arrays, 0), 0)

    def layer_mean_global(self) -> Optional[np.ndarray]:
        arrays = [self.global_residuals(l) for l in self.selected_layers]
        arrays = [a for a in arrays if a is not None]
        return None if not arrays else np.mean(np.stack(arrays, 0), 0)

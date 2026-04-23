# # Copyright (C) 2022-present Naver Corporation. All rights reserved.
# # Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# # --------------------------------------------------------
# # Main encoder/decoder blocks
# # --------------------------------------------------------
# # References: 
# # timm
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py


# import torch
# import torch.nn as nn 

# from itertools import repeat
# import collections.abc


# def _ntuple(n):
#     def parse(x):
#         if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
#             return x
#         return tuple(repeat(x, n))
#     return parse
# to_2tuple = _ntuple(2)

# def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0 and scale_by_keep:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor

# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

#     def extra_repr(self):
#         return f'drop_prob={round(self.drop_prob,3):0.3f}'

# class Mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         bias = to_2tuple(bias)
#         drop_probs = to_2tuple(drop)

#         self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
#         self.drop2 = nn.Dropout(drop_probs[1])

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x

# class Attention(nn.Module):

#     def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.rope = rope 

#     def forward(self, x, xpos, xmask=None, return_attn=False):
#         B, N, C = x.shape

#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
#         q, k, v = [qkv[:,:,i] for i in range(3)]
#         # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)
               
#         if self.rope is not None:
#             q = self.rope(q, xpos)
#             k = self.rope(k, xpos)
               
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn_before_softmax = attn.detach().clone()

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         # apply mask
#         if xmask is not None:
#             qmask = xmask.unsqueeze(1)  # [1, 1, 576, 1]
#             kmask = xmask.permute(0, 2, 1).unsqueeze(1)  # [1, 1, 1, 576]
#             attention_mask = ((qmask < 0.5) & (kmask > 0.5))
#             attn = torch.where(attention_mask, 0.0, attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         if return_attn:
#             return x, attn_before_softmax
#         else:
#             return x

# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, xpos):
#         x = x + self.drop_path(self.attn(self.norm1(x), xpos))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# class CrossAttention(nn.Module):
    
#     def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.projq = nn.Linear(dim, dim, bias=qkv_bias)
#         self.projk = nn.Linear(dim, dim, bias=qkv_bias)
#         self.projv = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
        
#         self.rope = rope
        
#     # def forward(self, query, key, value, qpos, kpos, qmask=None, kmask=None):
#     def forward(self, query, key, value, qpos, kpos, qmask=None, kmask=None,
#                 q_bg_tok=None, k_bg_tok=None,
#                 q_bg_ref_mean=None, k_bg_ref_mean=None,
#                 q_bg_ref_tok=None, k_bg_ref_tok=None,
#                 q_bg_ref_tok_mask=None, k_bg_ref_tok_mask=None,
#                 align_q=False, align_k=False,
#                 return_bg_means=False):
#         B, Nq, C = query.shape
#         Nk = key.shape[1]
#         Nv = value.shape[1]
        
#         q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
#         k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
#         v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
#         if self.rope is not None:
#             q = self.rope(q, qpos)
#             k = self.rope(k, kpos)
#         # ---------- NEW: compute bg means & align q/k BEFORE q@k^T ----------
#         def _bg_mean(x, bg_tok):
#             # x: [B, heads, N, dh], bg_tok: [B, N, 1] with {0,1} or [0,1]
#             if bg_tok is None:
#                 return None
#             sel = (bg_tok.squeeze(-1) > 0)  # [B, N]
#             # MVP: assume B==1 for now (your eval uses B=1)
#             if x.shape[0] != 1:
#                 return None
#             if sel[0].sum() == 0:
#                 return None
#             xb = x[0, :, sel[0], :]  # [heads, Nbg, dh]
#             return xb.mean(dim=1, keepdim=True).unsqueeze(0)  # [1, heads, 1, dh]

#         q_bg_mean = _bg_mean(q, q_bg_tok)
#         k_bg_mean = _bg_mean(k, k_bg_tok)

#         if align_q and (q_bg_mean is not None) and (q_bg_ref_mean is not None) and (q_bg_tok is not None):
#             sel = (q_bg_tok.squeeze(-1) > 0)
#             delta = (q_bg_mean - q_bg_ref_mean)  # [1, heads, 1, dh]
#             # q[0, :, sel[0], :] = q[0, :, sel[0], :] - delta[0]
#             q[0, :, :, :] = q[0, :, :, :] - delta[0] # all token

#         if align_k and (k_bg_mean is not None) and (k_bg_ref_mean is not None) and (k_bg_tok is not None):
#             sel = (k_bg_tok.squeeze(-1) > 0)
#             delta = (k_bg_mean - k_bg_ref_mean)
#             # k[0, :, sel[0], :] = k[0, :, sel[0], :] - delta[0]
#             k[0, :, :, :] = k[0, :, :, :] - delta[0] # all token
        
        
#         # ----------------- LIMITED DEBUG PRINT -----------------
#         # Print at most `max_prints` times for this CrossAttention module instance.
#         max_prints = 6
#         if not hasattr(self, "_dbg_align_cnt"):
#             self._dbg_align_cnt = 0

#         if self._dbg_align_cnt < max_prints:
#             # Only print when something actually aligned
#             did_q = align_q and (q_bg_mean is not None) and (q_bg_ref_mean is not None) and (q_bg_tok is not None)
#             did_k = align_k and (k_bg_mean is not None) and (k_bg_ref_mean is not None) and (k_bg_tok is not None)

#             if did_q or did_k:
#                 msgs = []
#                 if did_q:
#                     msgs.append(f"Q |mean(delta)|={float((q_bg_mean - q_bg_ref_mean).abs().mean()):.6f}")
#                 if did_k:
#                     msgs.append(f"K |mean(delta)|={float((k_bg_mean - k_bg_ref_mean).abs().mean()):.6f}")

#                 # Optional: show how many bg tokens are used
#                 try:
#                     nbq = int((q_bg_tok.squeeze(-1) > 0).sum().item()) if q_bg_tok is not None else -1
#                     nbk = int((k_bg_tok.squeeze(-1) > 0).sum().item()) if k_bg_tok is not None else -1
#                     msgs.append(f"bg_tokens(q,k)=({nbq},{nbk})")
#                 except Exception:
#                     pass
#                 if (q_bg_tok is not None) and (k_bg_tok is not None):
#                     print("[DBG TOK] Nq=", q_bg_tok.numel(), "Nk=", k_bg_tok.numel(),
#                         "bg_q=", int((q_bg_tok > 0).sum()), "bg_k=", int((k_bg_tok > 0).sum()))
#                 print("[DBG ALIGN]", " ".join(msgs))
#                 self._dbg_align_cnt += 1
#         # ----------------- END LIMITED DEBUG PRINT -----------------    

#         # ---------- END NEW ----------
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn_before_softmax = attn.detach().clone()

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         if qmask is not None and kmask is not None:
#             qmask = qmask.unsqueeze(1)  # [1, 1, 576, 1]
#             kmask = kmask.permute(0, 2, 1).unsqueeze(1)  # [1, 1, 1, 576]
#             attention_mask = ((qmask < 0.5) & (kmask > 0.5))
#             attn = torch.where(attention_mask, 0.0, attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         if return_bg_means:
#             return x, attn_before_softmax, q_bg_mean, k_bg_mean
#         return x, attn_before_softmax


# class DecoderBlock(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.norm3 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

#     # def forward(self, x, y, xpos, ypos, xmask, ymask, self_mask):
#     def forward(self, x, y, xpos, ypos, xmask, ymask, self_mask, q_bg_tok=None, k_bg_tok=None, q_bg_ref_mean=None, k_bg_ref_mean=None, align_q=False, align_k=False, return_bg_means=False,
#     ):
#         x_, self_attn = self.attn(self.norm1(x), xpos, xmask=self_mask, return_attn=True)
#         x = x + self.drop_path(x_)
#         y_ = self.norm_y(y)
#         # cross_attn_output, cross_attn = self.cross_attn(self.norm2(x), y_, y_, xpos, ypos, qmask=xmask, kmask=ymask)
#         # ---------- NEW ----------
#         out = self.cross_attn(
#             self.norm2(x), y_, y_, xpos, ypos,
#             qmask=xmask, kmask=ymask,
#             q_bg_tok=q_bg_tok, k_bg_tok=k_bg_tok,
#             q_bg_ref_mean=q_bg_ref_mean, k_bg_ref_mean=k_bg_ref_mean,
#             align_q=align_q, align_k=align_k,
#             return_bg_means=return_bg_means,
#         )

#         if return_bg_means:
#             cross_attn_output, cross_attn, q_bg_mean, k_bg_mean = out
#         else:
#             cross_attn_output, cross_attn = out
#             q_bg_mean, k_bg_mean = None, None
#         # ---------- END NEW ----------
#         x = x + self.drop_path(cross_attn_output)
#         x = x + self.drop_path(self.mlp(self.norm3(x)))
#         if return_bg_means:
#             return x, y, self_attn, cross_attn, q_bg_mean, k_bg_mean
#         return x, y, self_attn, cross_attn
        
        
# # patch embedding
# class PositionGetter(object):
#     """ return positions of patches """

#     def __init__(self):
#         self.cache_positions = {}
        
#     def __call__(self, b, h, w, device):
#         if not (h,w) in self.cache_positions:
#             x = torch.arange(w, device=device)
#             y = torch.arange(h, device=device)
#             self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
#         pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
#         return pos

# class PatchEmbed(nn.Module):
#     """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, init='xavier'):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
#         self.position_getter = PositionGetter()
#         self.init_type = init
        
#     def forward(self, x):
#         B, C, H, W = x.shape
#         torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
#         torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
#         x = self.proj(x)
#         pos = self.position_getter(B, x.size(2), x.size(3), x.device)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x, pos
        
#     def _init_weights(self):
#         w = self.proj.weight.data
#         if self.init_type == 'xavier':
#             torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         elif self.init_type == 'kaiming':
#             torch.nn.init.kaiming_uniform_(w.view([w.shape[0], -1]))
#         elif self.init_type == 'zero':
#             torch.nn.init.zeros_(w)
#             bias = getattr(self.proj, 'bias', None)
#             if bias is not None:
#                 torch.nn.init.zeros_(bias)
#         else:
#             raise ValueError(f"Unknown init type {self.init_type}")

# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References: 
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py


import torch
import torch.nn as nn 

from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 

    def forward(self, x, xpos, xmask=None, return_attn=False):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)
               
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
               
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_before_softmax = attn.detach().clone()

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # apply mask
        if xmask is not None:
            qmask = xmask.unsqueeze(1)  # [1, 1, 576, 1]
            kmask = xmask.permute(0, 2, 1).unsqueeze(1)  # [1, 1, 1, 576]
            attention_mask = ((qmask < 0.5) & (kmask > 0.5))
            attn = torch.where(attention_mask, 0.0, attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn_before_softmax
        else:
            return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        
    # def forward(self, query, key, value, qpos, kpos, qmask=None, kmask=None):
    def forward(
        self,
        query,
        key,
        value,
        qpos,
        kpos,
        qmask=None,
        kmask=None,
        q_bg_tok=None,
        k_bg_tok=None,
        q_bg_ref_mean=None,
        k_bg_ref_mean=None,
        q_bg_ref_tok=None,
        k_bg_ref_tok=None,
        q_bg_ref_tok_mask=None,
        k_bg_ref_tok_mask=None,
        align_q=False,
        align_k=False,
        return_bg_means=False,
    ):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        # Project to heads
        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,H,Nq,dh]
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)    # [B,H,Nk,dh]
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,H,Nv,dh]

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # ---------- BG mean helpers ----------
        def _bg_mean(x, bg_tok):
            # x: [B,H,N,dh], bg_tok: [B,N,1]
            if bg_tok is None:
                return None
            sel = (bg_tok.squeeze(-1) > 0)  # [B,N]
            if sel.sum() == 0:
                return None
            means = []
            for b in range(x.shape[0]):
                if sel[b].sum() == 0:
                    return None
                xb = x[b, :, sel[b], :]  # [H, Nbg, dh]
                means.append(xb.mean(dim=1, keepdim=True))  # [H,1,dh]
            return torch.stack(means, dim=0)  # [B,H,1,dh]

        def _align_with_tokenwise_delta(x, x_bg_tok, x_bg_ref_mean, x_bg_ref_tok, x_bg_ref_tok_mask):
            """Align x (q or k) using:
            1) mean-delta on all tokens (original behavior)
            2) token-wise override on intersection bg tokens (both sides bg)
            """
            if (x_bg_tok is None) or (x_bg_ref_mean is None):
                return x

            x_bg_mean = _bg_mean(x, x_bg_tok)
            if x_bg_mean is None:
                return x

            # 1) mean delta to all tokens
            delta_mean = (x_bg_mean - x_bg_ref_mean)  # [B,H,1,dh]
            x_aligned = x - delta_mean  # broadcast

            # 2) token-wise override on intersection bg tokens
            if (x_bg_ref_tok is None) or (x_bg_ref_tok_mask is None):
                return x_aligned

            bg_cur = (x_bg_tok.squeeze(-1) > 0)            # [B,N]
            bg_ref = (x_bg_ref_tok_mask.squeeze(-1) > 0)   # [B,N]
            inter = bg_cur & bg_ref                        # [B,N]

            for b in range(x_aligned.shape[0]):
                if inter[b].any():
                    x_aligned[b, :, inter[b], :] = x_bg_ref_tok[b, :, inter[b], :]

            return x_aligned

        # Compute means for returning/logging (before alignment)
        q_bg_mean = _bg_mean(q, q_bg_tok)
        k_bg_mean = _bg_mean(k, k_bg_tok)

        # Align Q or K if requested
        if align_q and (q_bg_ref_mean is not None):
            q = _align_with_tokenwise_delta(q, q_bg_tok, q_bg_ref_mean, q_bg_ref_tok, q_bg_ref_tok_mask)

        if align_k and (k_bg_ref_mean is not None):
            k = _align_with_tokenwise_delta(k, k_bg_tok, k_bg_ref_mean, k_bg_ref_tok, k_bg_ref_tok_mask)

        # ---------- attention ----------
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_before_softmax = attn.detach().clone()

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if qmask is not None and kmask is not None:
            qmask = qmask.unsqueeze(1)  # [B, 1, Nq, 1]
            kmask = kmask.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 1, Nk]
            attention_mask = ((qmask < 0.5) & (kmask > 0.5))
            attn = torch.where(attention_mask, 0.0, attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_bg_means:
            # Return projected q/k (after alignment) so caller can reuse as token-wise refs.
            return x, attn_before_softmax, q_bg_mean, k_bg_mean, q.detach(), k.detach()

        return x, attn_before_softmax

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    # def forward(self, x, y, xpos, ypos, xmask, ymask, self_mask):
    def forward(
        self,
        x,
        y,
        xpos,
        ypos,
        xmask,
        ymask,
        self_mask,
        q_bg_tok=None,
        k_bg_tok=None,
        q_bg_ref_mean=None,
        k_bg_ref_mean=None,
        q_bg_ref_tok=None,
        k_bg_ref_tok=None,
        q_bg_ref_tok_mask=None,
        k_bg_ref_tok_mask=None,
        align_q=False,
        align_k=False,
        return_bg_means=False,
    ):
        x_, self_attn = self.attn(self.norm1(x), xpos, xmask=self_mask, return_attn=True)
        x = x + self.drop_path(x_)
        y_ = self.norm_y(y)

        out = self.cross_attn(
            self.norm2(x), y_, y_, xpos, ypos,
            qmask=xmask, kmask=ymask,
            q_bg_tok=q_bg_tok, k_bg_tok=k_bg_tok,
            q_bg_ref_mean=q_bg_ref_mean, k_bg_ref_mean=k_bg_ref_mean,
            q_bg_ref_tok=q_bg_ref_tok, k_bg_ref_tok=k_bg_ref_tok,
            q_bg_ref_tok_mask=q_bg_ref_tok_mask, k_bg_ref_tok_mask=k_bg_ref_tok_mask,
            align_q=align_q, align_k=align_k,
            return_bg_means=return_bg_means,
        )

        if return_bg_means:
            cross_attn_output, cross_attn, q_bg_mean, k_bg_mean, q_proj, k_proj = out
        else:
            cross_attn_output, cross_attn = out
            q_bg_mean, k_bg_mean, q_proj, k_proj = None, None, None, None

        x = x + self.drop_path(cross_attn_output)
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        if return_bg_means:
            return x, y, self_attn, cross_attn, q_bg_mean, k_bg_mean, q_proj, k_proj

        return x, y, self_attn, cross_attn

class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos

class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, init='xavier'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.position_getter = PositionGetter()
        self.init_type = init
        
    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos
        
    def _init_weights(self):
        w = self.proj.weight.data
        if self.init_type == 'xavier':
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        elif self.init_type == 'kaiming':
            torch.nn.init.kaiming_uniform_(w.view([w.shape[0], -1]))
        elif self.init_type == 'zero':
            torch.nn.init.zeros_(w)
            bias = getattr(self.proj, 'bias', None)
            if bias is not None:
                torch.nn.init.zeros_(bias)
        else:
            raise ValueError(f"Unknown init type {self.init_type}")


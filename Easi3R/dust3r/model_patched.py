# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub
import torch.nn.functional as F
import torch.nn as nn
from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed
from third_party.raft import load_RAFT

from einops import rearrange

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    # def _decoder(self, f1, pos1, f2, pos2, mask1, mask2):
    #     final_output = [(f1, f2)]  # before projection
    #     original_D = f1.shape[-1]
    #     attention_maps = []

    #     # project to decoder dim
    #     f1 = self.decoder_embed(f1)
    #     f2 = self.decoder_embed(f2)

    #     final_output.append((f1, f2))
    #     for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2)):
    #         # img1 side
    #         f1, _, self_attn1, cross_attn1 = blk1(*final_output[-1][::+1], pos1, pos2, mask1, mask2, None) # mask1, mask2, mask1
    #         # img2 side
    #         f2, _, self_attn2, cross_attn2 = blk2(*final_output[-1][::-1], pos2, pos1, None, None, None) # mask2, mask1, mask2

    #         # store the result
    #         final_output.append((f1, f2))
    #         attention_maps.append((self_attn1, cross_attn1, self_attn2, cross_attn2))

    #     # normalize last output
    #     del final_output[1]  # duplicate with final_output[0]
    #     final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
    #     return zip(*final_output), zip(*attention_maps)
    def _decoder(self, feat1, pos1, feat2, pos2, mask1, mask2, bg1_tok=None, bg2_tok=None):
        # before projection
        f1, f2 = feat1, feat2
        final_output = [(f1, f2)]
        original_D = f1.shape[-1]
        attention_maps = []

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        final_output.append((f1, f2))

        # lag-1 reference for aligning K(view2) in blk1
        k1_bg_mean_prev = None
        k1_tok_prev = None  # [1, heads, N, dh] from view1 (key) in blk2 of previous layer

        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2)):
            # ------------------------------------------------------------
            # blk1: view1 <- view2
            # query = view1 tokens (f1), key = view2 tokens (f2)
            # We want to align K(view2) BEFORE q@k^T: align_k=True on blk1
            # ref for K(view2) is K(view1) from previous layer (lag-1): k1_bg_mean_prev
            # Also we want q1_bg_mean for aligning Q(view2) in blk2 (same layer)
            # ------------------------------------------------------------
            out1 = blk1(
                *final_output[-1][::+1],  # (f1, f2)
                pos1, pos2,
                mask1, mask2,
                None,
                # ---- new args ----
                q_bg_tok=bg1_tok,         # query is view1
                k_bg_tok=bg2_tok,         # key is view2  (we may align K2 here)
                q_bg_ref_mean=None,
                k_bg_ref_mean=k1_bg_mean_prev,
                k_bg_ref_tok=k1_tok_prev,
                k_bg_ref_tok_mask=bg1_tok,
                align_q=False,
                align_k=(k1_bg_mean_prev is not None),
                return_bg_means=True,
                return_qk=True,
            )

            # blk should now return: f1_new, y_dummy, self_attn1, cross_attn1, q1_bg_mean, k2_bg_mean
            f1, _, self_attn1, cross_attn1, q1_bg_mean, k2_bg_mean, q1_tok, _k2_tok = out1

            # ------------------------------------------------------------
            # blk2: view2 <- view1
            # query = view2 tokens (f2), key = view1 tokens (f1)
            # We want to align Q(view2) BEFORE q@k^T: align_q=True on blk2
            # ref for Q(view2) is Q(view1) from blk1 of SAME layer: q1_bg_mean
            # Also we collect k1_bg_mean to use as prev ref in next layer's blk1
            # ------------------------------------------------------------
            out2 = blk2(
                *final_output[-1][::-1],  # (f2, f1)  NOTE: use prev layer tokens as in your original
                pos2, pos1,
                None, None,
                None,
                # ---- new args ----
                q_bg_tok=bg2_tok,         # query is view2 (we align Q2 here)
                k_bg_tok=bg1_tok,         # key is view1
                q_bg_ref_mean=q1_bg_mean,
                k_bg_ref_mean=None,
                align_q=(q1_bg_mean is not None),
                align_k=False,
                return_bg_means=True,
            )

            # blk should now return: f2_new, y_dummy, self_attn2, cross_attn2, q2_bg_mean, k1_bg_mean
            f2, _, self_attn2, cross_attn2, q2_bg_mean, k1_bg_mean, _q2_tok, k1_tok = out2

            # store for next iteration (lag-1 for aligning K(view2))
            k1_bg_mean_prev = k1_bg_mean
            k1_tok_prev = k1_tok

            # store results
            final_output.append((f1, f2))
            attention_maps.append((self_attn1, cross_attn1, self_attn2, cross_attn2))
        
        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D、
        if not hasattr(self, "_printed_img_hw"):
            self._printed_img_hw = True
            if 'img' in view1 and 'img' in view2:
                img1 = view1['img']
                img2 = view2['img']
                # expected: [B,3,H,W]
                print("[DEBUG] view1['img']:", tuple(img1.shape), "view2['img']:", tuple(img2.shape))
            else:
                print("[DEBUG] no 'img' in view1/view2 keys:", view1.keys(), view2.keys())
        bg1 = view1.get('bg_mask', None)
        bg2 = view2.get('bg_mask', None)
        if not hasattr(self, "_dbg_bg"):
            self._dbg_bg = True
            print("[DEBUG] bg1:", None if bg1 is None else tuple(bg1.shape),
                "bg2:", None if bg2 is None else tuple(bg2.shape),
                "bg1_max:", None if bg1 is None else float(bg1.max()))

                        
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # resize the mask to the shape of the feature
        mask1 = self._resize_mask(view1, shape1) # if 'atten_mask' not in view1, return None
        mask2 = self._resize_mask(view2, shape2) # if 'atten_mask' not in view2, return None
        bg1_tok = self._resize_bg_mask(view1, shape1)
        bg2_tok = self._resize_bg_mask(view2, shape2)
        # combine all ref images into object-centric representation
        (dec1, dec2), (self_attn1, cross_attn1, self_attn2, cross_attn2) = self._decoder(
            feat1, pos1, feat2, pos2, mask1, mask2,
            bg1_tok=bg1_tok, bg2_tok=bg2_tok
        )

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame

        res1['match_feature'] = self._get_feature(feat1, shape1)
        res1['cross_atten_maps_k'] = self._get_attn_k(torch.cat(cross_attn1), shape1)
        res2['cross_atten_maps_k'] = self._get_attn_k(torch.cat(cross_attn2), shape2)
        
        return res1, res2
    
    def _resize_bg_mask(self, view, shape, patch=16, thr=0.5):
        if 'bg_mask' not in view or view['bg_mask'] is None:
            return None
        bg = view['bg_mask']  # [B,H,W] or [H,W] or [B,1,H,W]
        if bg.dim() == 2:
            bg = bg[None, None]
        elif bg.dim() == 3:
            bg = bg[:, None]
        bg = bg.float()
        if bg.max() > 1.5:
            bg = bg / 255.0

        pool = torch.nn.AvgPool2d(kernel_size=patch, stride=patch)
        bg_tok = pool(bg)  # [B,1,h,w]
        bg_tok = (bg_tok > thr).flatten(2).transpose(1, 2)  # [B,N,1] bool-ish
        return bg_tok


    def _resize_mask(self, view, shape):
        if 'atten_mask' not in view:
            return None
        H, W = shape[0].cpu().tolist()
        N_H = H // 16
        N_W = W // 16
        mask = view['atten_mask'] # dynamic_mask
        # # visualize the mask
        # import torchvision
        # torchvision.utils.save_image(mask.float(), 'dynamic_mask1.png')
        # downsample the mask to the shape of N_H x N_W
        max_pool = torch.nn.MaxPool2d(kernel_size=16, stride=16)
        mask = max_pool(mask.float().unsqueeze(1)).squeeze(1)
        # torchvision.utils.save_image(mask.float(), 'dynamic_mask_downsampled.png')
        mask = rearrange(mask, 'b nh nw -> b (nh nw) 1', nh=N_H, nw=N_W)
        return mask

    def _get_feature(self, feats, shape):
        H, W = shape[0].cpu().tolist()
        # Number of patches in height and width
        N_H = H // 16
        N_W = W // 16
        # Reshape tokens to spatial representation
        feats = rearrange(feats, 'b (nh nw) c -> b nh nw c', nh=N_H, nw=N_W)
        return feats
    
    def _get_attn_k(self, attn, shape):
        H, W = shape[0].cpu().tolist()
        N_H = H // 16
        N_W = W // 16
        reshaped_attn = rearrange(attn, 'l h (nh nw) (nh2 nw2) -> 1 nh nw nh2 nw2 l h', 
                                  nh=N_H, nw=N_W, nh2=N_H, nw2=N_W)
        reshaped_attn = reshaped_attn[...,0:,:].mean(dim=(1, 2)).view(-1, N_H, N_W, 12*12)
        return reshaped_attn

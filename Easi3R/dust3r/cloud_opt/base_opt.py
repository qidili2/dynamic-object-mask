# --------------------------------------------------------
# Base class for the global alignement procedure
# --------------------------------------------------------
from copy import deepcopy
import cv2

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm
import uuid
from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.viz import SceneViz, segment_sky, auto_cam_size
from dust3r.optim_factory import adjust_learning_rate_by_lr

from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, get_imshapes, signed_expm1, signed_log1p,
                                      cosine_schedule, linear_schedule, cycled_linear_schedule, get_conf_trf)
import dust3r.cloud_opt.init_im_poses as init_fun
from scipy.spatial.transform import Rotation
from dust3r.utils.vo_eval import save_trajectory_tum_format
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu, threshold_multiotsu
import math
import torchvision
import cv2
from sam2.build_sam import build_sam2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
import glob
from time import perf_counter as _now
import csv
import re
from collections import defaultdict
import uuid
import sys, subprocess, tempfile, yaml
import cv2
from matplotlib import cm   

inferno_cmap = cm.get_cmap("inferno") 

def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation
    
    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = to_numpy(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose


class BasePCOptimizer (nn.Module):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            other = deepcopy(args[0])
            attrs = '''edges is_symmetrized dist n_imgs pred_i pred_j imshapes 
                        min_conf_thr conf_thr conf_i conf_j im_conf
                        base_scale norm_pw_scale POSE_DIM pw_poses 
                        pw_adaptors pw_adaptors has_im_poses rand_pose imgs verbose'''.split()
            self.__dict__.update({k: other[k] for k in attrs})
        else:
            self._init_from_views(*args, **kwargs)

    def _init_from_views(self, view1, view2, pred1, pred2,
                         dist='l1',
                         conf='log',
                         min_conf_thr=2.5,
                         thr_for_init_conf=False,
                         base_scale=0.5,
                         allow_pw_adaptors=False,
                         pw_break=20,
                         rand_pose=torch.randn,
                         empty_cache=False,
                         verbose=True,
                         use_atten_mask=False,
                         use_region_pooling = False,
                         sam2_group_output_dir = None,
                         textregion_annotations_dir = None):
        super().__init__()
        if not isinstance(view1['idx'], list):
            view1['idx'] = view1['idx'].tolist()
        if not isinstance(view2['idx'], list):
            view2['idx'] = view2['idx'].tolist()
        self.edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
        self.is_symmetrized = set(self.edges) == {(j, i) for i, j in self.edges}
        self.dist = ALL_DISTS[dist]
        self.verbose = verbose
        self.empty_cache = empty_cache
        self.n_imgs = self._check_edges()
        self.sam2_group_output_dir = sam2_group_output_dir
        self.textregion_annotations_dir = textregion_annotations_dir
        
        # input data
        pred1_pts = pred1['pts3d']
        pred2_pts = pred2['pts3d_in_other_view']
        self.pred_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
        self.pred_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})
        self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)

        # work in log-scale with conf
        pred1_conf = pred1['conf']  # (Number of image_pairs, H, W)
        pred2_conf = pred2['conf']  # (Number of image_pairs, H, W)
        self.min_conf_thr = min_conf_thr
        self.thr_for_init_conf = thr_for_init_conf
        self.conf_trf = get_conf_trf(conf)

        self.conf_i = NoGradParamDict({ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)})
        self.conf_j = NoGradParamDict({ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)})
        self.im_conf = self._compute_img_conf(pred1_conf, pred2_conf)
        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        self.init_conf_maps = [c.clone() for c in self.im_conf]

        # pairwise pose parameters
        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.pw_poses = nn.Parameter(rand_pose((self.n_edges, 1+self.POSE_DIM)))  # pairwise poses
        self.pw_adaptors = nn.Parameter(torch.zeros((self.n_edges, 2)))  # slight xy/z adaptation
        self.pw_adaptors.requires_grad_(allow_pw_adaptors)
        self.has_im_poses = False
        self.rand_pose = rand_pose

        # possibly store images, camera_pose, instance for show_pointcloud
        self.imgs = None
        if 'img' in view1 and 'img' in view2:
            imgs = [torch.zeros((3,)+hw) for hw in self.imshapes]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                imgs[idx] = view1['img'][v]
                idx = view2['idx'][v]
                imgs[idx] = view2['img'][v]
            self.imgs = rgb(imgs)

        self.dynamic_masks = None
        if 'dynamic_mask' in view1 and 'dynamic_mask' in view2:
            dynamic_masks = [torch.zeros(hw) for hw in self.imshapes]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                dynamic_masks[idx] = view1['dynamic_mask'][v]
                idx = view2['idx'][v]
                dynamic_masks[idx] = view2['dynamic_mask'][v]
            self.dynamic_masks = dynamic_masks

        self.camera_poses = None
        if 'camera_pose' in view1 and 'camera_pose' in view2:
            camera_poses = [torch.zeros((4, 4)) for _ in range(self.n_imgs)]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                camera_poses[idx] = view1['camera_pose'][v]
                idx = view2['idx'][v]
                camera_poses[idx] = view2['camera_pose'][v]
            self.camera_poses = camera_poses

        self.img_pathes = None
        if 'instance' in view1 and 'instance' in view2:
            img_pathes = ['' for _ in range(self.n_imgs)]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                img_pathes[idx] = view1['instance'][v]
                idx = view2['idx'][v]
                img_pathes[idx] = view2['instance'][v]
            self.img_pathes = img_pathes

        if use_atten_mask:
            # attention map
            cross_att_k_i_mean, cross_att_k_i_var, cross_att_k_j_mean, cross_att_k_j_var = self.aggregate_attention_maps(pred1, pred2)
            
            if use_region_pooling:
                # if not hasattr(self, "region_groups") or self.region_groups is None or len(self.region_groups) != self.n_imgs:
                #     self.generate_sam2_region_groups(min_size=100, vis_dir=sam2_group_output_dir)  
                #     # self.generate_sam2_region_groups(min_size=100)  
                if (not hasattr(self, "region_groups")) or (self.region_groups is None) or (len(self.region_groups) != self.n_imgs):
                    # 你也可以把 vis_dir=None；保留的话会存每帧的 group 可视化和 .npy
                    self.generate_region_groups_with_tracking(
                        proposal_backend="sam1",  
                        min_size=100,             
                        vis_dir=sam2_group_output_dir            
                    )
                    print(f"[INIT] Successfully generated {len(self.region_groups)} region groups")


                H_img, W_img = self.imshapes[0]
                group_img = torch.stack([g.to(self.device) for g in self.region_groups], 0)        # [B,H,W] int
                group_tok = self._downsample_groups_to_tokens(group_img, H_img, W_img, patch=16)    # [B,Ht,Wt] int

                cross_att_k_i_mean = self._region_group_mean_pool_map(cross_att_k_i_mean, group_tok, include_background=False)
                cross_att_k_i_var  = self._region_group_mean_pool_map(cross_att_k_i_var,  group_tok, include_background=False)
                cross_att_k_j_mean = self._region_group_mean_pool_map(cross_att_k_j_mean, group_tok, include_background=False)
                cross_att_k_j_var  = self._region_group_mean_pool_map(cross_att_k_j_var,  group_tok, include_background=False)
                
            def fuse_attention_channels(att_maps):
                # att_maps: B, H, W, C
                # normalize
                att_maps_min = att_maps.min()
                att_maps_max = att_maps.max()
                att_maps_normalized = (att_maps - att_maps_min) / (att_maps_max - att_maps_min + 1e-6)
                # average channel
                att_maps_fused = att_maps_normalized.mean(dim=-1) # B, H, W
                # normalize
                att_maps_fused_min = att_maps_fused.min()
                att_maps_fused_max = att_maps_fused.max()
                att_maps_fused = (att_maps_fused - att_maps_fused_min) / (att_maps_fused_max - att_maps_fused_min + 1e-6)
                return att_maps_normalized, att_maps_fused
            
            self.cross_att_k_i_mean, self.cross_att_k_i_mean_fused = fuse_attention_channels(cross_att_k_i_mean)
            self.cross_att_k_i_var, self.cross_att_k_i_var_fused = fuse_attention_channels(cross_att_k_i_var)
            self.cross_att_k_j_mean, self.cross_att_k_j_mean_fused = fuse_attention_channels(cross_att_k_j_mean)
            self.cross_att_k_j_var, self.cross_att_k_j_var_fused = fuse_attention_channels(cross_att_k_j_var)
            
            # create dynamic mask
            dynamic_map = (1-self.cross_att_k_i_mean_fused) * self.cross_att_k_i_var_fused * self.cross_att_k_j_mean_fused * (1-self.cross_att_k_j_var_fused)
            dynamic_map_min = dynamic_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] # B, 1, 1
            dynamic_map_max = dynamic_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] # B, 1, 1
            
            self.dynamic_map = (dynamic_map - dynamic_map_min) / (dynamic_map_max - dynamic_map_min + 1e-6)

            
            if os.path.exists(textregion_annotations_dir):
                self.validate_and_adjust_dynamic_map_with_gt(textregion_annotations_dir)
            else:
                print(f"[TR Validation] GT directory not found: {textregion_annotations_dir}")
                
            try:
                print("Starting variance analysis...")
                variances, attention_values = self.compute_region_attention_variance_and_visualize(
                    save_folder=sam2_group_output_dir if sam2_group_output_dir else 'demo_tmp/region_variance_vis'
                )
                print(f"Variance analysis completed successfully! Results in: {sam2_group_output_dir if sam2_group_output_dir else 'demo_tmp/region_variance_vis'}")
            except Exception as e:
                print(f"Warning: Could not compute variance analysis during init: {e}")
                import traceback
                traceback.print_exc()
            # feature
            pred1_feat = pred1['match_feature']
            feat_i = NoGradParamDict({ij: nn.Parameter(pred1_feat[n], requires_grad=False) for n, ij in enumerate(self.str_edges)})
            stacked_feat_i = [feat_i[k] for k in self.str_edges]
            stacked_feat = [None] * len(self.imshapes)
            for i, ei in enumerate(torch.tensor([i for i, j in self.edges])):
                stacked_feat[ei]=stacked_feat_i[i]
            self.stacked_feat = torch.stack(stacked_feat).float().detach()

            self.refined_dynamic_map, self.dynamic_map_labels = cluster_attention_maps(self.stacked_feat, self.dynamic_map, n_clusters=64)
            
    # @torch.no_grad()
    # def validate_and_adjust_dynamic_map_with_gt(self,
    #                                         textregion_root: str,
    #                                         prefer_npy: bool = True,
    #                                         bin_suffix: str = "_bin",
    #                                         use_first_frame: bool = True):
    #     """
    #     使用GT annotations验证dynamic map的方向性，如果GT objects区域的attention普遍较低，
    #     则反转dynamic map
        
    #     Args:
    #         gt_annotations_dir: GT标注目录路径，如 '/mnt/data0/andy/Easi3R/DAVIS/Annotations/480p'
        
    #     Returns:
    #         bool: 是否进行了反转
    #     """

    #     if not hasattr(self, 'dynamic_map') or self.dynamic_map is None:
    #         print("[TR Validation] dynamic_map not found, skipping validation")
    #         return False
    #     if not hasattr(self, 'img_pathes') or self.img_pathes is None or len(self.img_pathes) == 0:
    #         print("[TR Validation] img_pathes not found, cannot match TextRegion masks")
    #         return False

    #     def _seq_and_stem(img_path: str):
    #         # 解析 sequence 名与帧名（不含扩展名）
    #         parts = img_path.split('/')
    #         if 'JPEGImages' in parts:
    #             i = parts.index('JPEGImages')
    #             if len(parts) <= i + 2:
    #                 return None, None
    #             seq = parts[i + 2]                           # e.g. "kite-surf"
    #             stem = os.path.splitext(parts[-1])[0]        # e.g. "00000"
    #             return seq, stem
    #         # 兜底（路径不含 JPEGImages）
    #         seq = os.path.basename(os.path.dirname(img_path))
    #         stem = os.path.splitext(os.path.basename(img_path))[0]
    #         return seq, stem

    #     # 收集前景/背景 attention
    #     fg_scores, bg_scores = [], []

    #     frames = range(1) if use_first_frame else range(self.n_imgs)

    #     for frame_idx in frames:
    #         img_path = self.img_pathes[frame_idx]
    #         seq, stem = _seq_and_stem(img_path)
    #         if seq is None or stem is None:
    #             print(f"[TR Validation] Warning: cannot parse seq/stem from {img_path}")
    #             continue

    #         # TextRegion 二值文件路径
    #         seq_dir = os.path.join(textregion_root, seq)
    #         npy_path = os.path.join(seq_dir, f"{stem}{bin_suffix}.npy")
    #         png_path = os.path.join(seq_dir, f"{stem}{bin_suffix}.png")

    #         # 读取二值 mask
    #         bin_mask = None
    #         if prefer_npy and os.path.exists(npy_path):
    #             try:
    #                 bin_mask = np.load(npy_path).astype(np.uint8)
    #                 print(f"[TR Validation] loaded {npy_path}")
    #             except Exception as e:
    #                 print(f"[TR Validation] failed to load {npy_path}: {e}")
    #         if bin_mask is None and os.path.exists(png_path):
    #             m = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    #             if m is None:
    #                 print(f"[TR Validation] cannot read {png_path}")
    #             else:
    #                 bin_mask = (m > 127).astype(np.uint8)
    #                 print(f"[TR Validation] loaded {png_path}")
    #         if bin_mask is None and not prefer_npy and os.path.exists(npy_path):
    #             try:
    #                 bin_mask = np.load(npy_path).astype(np.uint8)
    #                 print(f"[TR Validation] loaded {npy_path}")
    #             except Exception as e:
    #                 print(f"[TR Validation] failed to load {npy_path}: {e}")

    #         if bin_mask is None:
    #             print(f"[TR Validation] mask not found for {seq}/{stem} under {textregion_root}")
    #             continue

    #         # 尺寸对齐到 dynamic_map
    #         H_att, W_att = self.dynamic_map.shape[1], self.dynamic_map.shape[2]
    #         if bin_mask.shape != (H_att, W_att):
    #             bin_mask = cv2.resize(bin_mask.astype(np.uint8), (W_att, H_att), interpolation=cv2.INTER_NEAREST)

    #         dm = self.dynamic_map[frame_idx]
    #         obj = (bin_mask > 0)
    #         bg  = ~obj

    #         if obj.sum() == 0 or bg.sum() == 0:
    #             print(f"[TR Validation] frame {frame_idx}: foreground/background pixels missing, skip.")
    #             continue

    #         fg_scores.extend(dm[obj].detach().cpu().numpy().tolist())
    #         bg_scores.extend(dm[bg].detach().cpu().numpy().tolist())

    #         if use_first_frame:
    #             break  # 只用首帧时立刻退出

    #     if len(fg_scores) == 0:
    #         print("[TR Validation] No valid foreground scores collected (missing masks?)")
    #         return False

    #     mean_fg = float(np.mean(fg_scores))
    #     mean_bg = float(np.mean(bg_scores)) if len(bg_scores) > 0 else 0.5

    #     print("[TR Validation] Statistics from TextRegion masks:")
    #     print(f"  Foreground mean attention: {mean_fg:.4f}")
    #     print(f"  Background mean attention: {mean_bg:.4f}")
    #     print(f"  Total FG pixels: {len(fg_scores)}")
    #     print(f"  Total BG pixels: {len(bg_scores)}")
        
    #     # 判定是否翻转
    #     should_invert = mean_fg < mean_bg
    #     if should_invert:
    #         print("[TR Validation] Foreground attention LOWER than background -> invert dynamic_map")
    #         self.dynamic_map = 1.0 - self.dynamic_map
    #         if hasattr(self, 'refined_dynamic_map') and self.refined_dynamic_map is not None:
    #             self.refined_dynamic_map = 1.0 - self.refined_dynamic_map
    #             print("[TR Validation] Also inverted refined_dynamic_map")

    #         # 仅打印翻转后的统计（基于已有分数 1-x）
    #         inv_fg = [1.0 - s for s in fg_scores]
    #         inv_bg = [1.0 - s for s in bg_scores]
    #         print("[TR Validation] After inversion:")
    #         print(f"  Foreground mean attention: {np.mean(inv_fg):.4f}")
    #         print(f"  Background mean attention: {np.mean(inv_bg):.4f}")
    #     else:
    #         print("[TR Validation] Direction looks correct (no inversion).")

    #     # **新增**: 保存flip状态，供后续filter使用
    #     self.dynamic_map_was_inverted = should_invert
    #     print(f"[TR Validation] *** Set dynamic_map_was_inverted = {should_invert} ***")
        
    #     return should_invert
    @torch.no_grad()
    def validate_and_adjust_dynamic_map_with_gt(self,
                                            textregion_root: str,
                                            prefer_npy: bool = True,
                                            bin_suffix: str = "_bin",
                                            min_object_area_ratio: float = 0.01,
                                            foreground_overlap_threshold: float = 0.6,
                                            min_frames_ratio: float = 0.6):
        """
        使用GT annotations验证dynamic map的方向性，如果GT objects区域的attention普遍较低，
        则反转dynamic map
        
        改进的flip判断逻辑：
        - 使用跨帧的object attention（而不是只看第一帧）
        - 使用跨帧的平均面积判断object大小
        - 放宽前景判断：只要object有30%以上在前景区域就算前景object
        - 过滤太小的objects（面积比例 < min_object_area_ratio）
        - 过滤出现帧数太少的objects（帧数比例 < min_frames_ratio）
        - 找到前景中最高attention的object
        - 用这个object的mean attention vs 背景的mean attention来判断是否flip
        
        Args:
            textregion_root: TextRegion标注目录路径
            prefer_npy: 是否优先读取.npy文件
            bin_suffix: 二值文件后缀
            use_first_frame: 是否只使用第一帧来判断前景/背景归属
            min_object_area_ratio: 最小object面积比例（相对于前景总面积），防止边缘小object
            foreground_overlap_threshold: object至少要有多少比例在前景区域才算前景object（默认30%）
            min_frames_ratio: object至少要在多少比例的帧中出现才有效（默认30%）
        
        Returns:
            bool: 是否进行了反转
        """

        if not hasattr(self, 'dynamic_map') or self.dynamic_map is None:
            print("[TR Validation] dynamic_map not found, skipping validation")
            return False
        if not hasattr(self, 'img_pathes') or self.img_pathes is None or len(self.img_pathes) == 0:
            print("[TR Validation] img_pathes not found, cannot match TextRegion masks")
            return False
        if not hasattr(self, 'region_groups') or self.region_groups is None:
            print("[TR Validation] region_groups not found, cannot analyze objects across frames")
            return False

        def _seq_and_stem(img_path: str):
            # 解析 sequence 名与帧名（不含扩展名）
            parts = img_path.split('/')
            if 'JPEGImages' in parts:
                i = parts.index('JPEGImages')
                if len(parts) <= i + 2:
                    return None, None
                seq = parts[i + 2]                           # e.g. "kite-surf"
                stem = os.path.splitext(parts[-1])[0]        # e.g. "00000"
                return seq, stem
            # 兜底（路径不含 JPEGImages）
            seq = os.path.basename(os.path.dirname(img_path))
            stem = os.path.splitext(os.path.basename(img_path))[0]
            return seq, stem

        # Step 1: 读取第一帧的textregion mask，用于判断哪些objects属于前景
        frame_idx = 0
        img_path = self.img_pathes[frame_idx]
        seq, stem = _seq_and_stem(img_path)
        if seq is None or stem is None:
            print(f"[TR Validation] Warning: cannot parse seq/stem from {img_path}")
            return False

        # TextRegion 二值文件路径
        seq_dir = os.path.join(textregion_root, seq)
        npy_path = os.path.join(seq_dir, f"{stem}{bin_suffix}.npy")
        png_path = os.path.join(seq_dir, f"{stem}{bin_suffix}.png")

        # 读取二值 mask
        bin_mask = None
        if prefer_npy and os.path.exists(npy_path):
            try:
                bin_mask = np.load(npy_path).astype(np.uint8)
                print(f"[TR Validation] loaded {npy_path}")
            except Exception as e:
                print(f"[TR Validation] failed to load {npy_path}: {e}")
        if bin_mask is None and os.path.exists(png_path):
            m = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                print(f"[TR Validation] cannot read {png_path}")
            else:
                bin_mask = (m > 127).astype(np.uint8)
                print(f"[TR Validation] loaded {png_path}")
        if bin_mask is None and not prefer_npy and os.path.exists(npy_path):
            try:
                bin_mask = np.load(npy_path).astype(np.uint8)
                print(f"[TR Validation] loaded {npy_path}")
            except Exception as e:
                print(f"[TR Validation] failed to load {npy_path}: {e}")

        if bin_mask is None:
            print(f"[TR Validation] mask not found for {seq}/{stem} under {textregion_root}")
            return False

        # Step 2: 分析第一帧的region_groups，判断每个object是否为前景object
        from scipy import ndimage
        
        # 获取第一帧的region_groups并转换为tensor
        first_frame_groups = self.region_groups[frame_idx]
        if not isinstance(first_frame_groups, torch.Tensor):
            first_frame_groups = torch.tensor(first_frame_groups, device=self.dynamic_map.device)
        else:
            first_frame_groups = first_frame_groups.to(self.dynamic_map.device)
        
        # **关键修改**: textregion_mask需要resize到与region_groups相同的尺寸（高分辨率）
        # 而不是dynamic_map的尺寸（token级别）
        H_rg, W_rg = first_frame_groups.shape
        print(f"[TR Validation] Region groups shape: {H_rg} x {W_rg}")
        print(f"[TR Validation] Dynamic map shape: {self.dynamic_map.shape[1]} x {self.dynamic_map.shape[2]}")
        
        if bin_mask.shape != (H_rg, W_rg):
            bin_mask = cv2.resize(bin_mask.astype(np.uint8), (W_rg, H_rg), interpolation=cv2.INTER_NEAREST)
            print(f"[TR Validation] Resized textregion mask to {H_rg} x {W_rg}")
        
        textregion_mask = torch.from_numpy(bin_mask).bool().to(first_frame_groups.device)
        
        # 获取所有unique object IDs
        all_object_ids = torch.unique(first_frame_groups)
        all_object_ids = all_object_ids[all_object_ids != 0].tolist()  # 排除background(0)
        
        print(f"\n[TR Validation] Analyzing {len(all_object_ids)} objects across all frames...")
        
        # 判断每个object是否为前景object（基于第一帧）
        foreground_objects = set()
        background_objects = set()
        
        for obj_id in all_object_ids:
            obj_mask = (first_frame_groups == obj_id)
            obj_pixels = obj_mask.sum().item()
            
            if obj_pixels == 0:
                continue
            
            # 计算该object在textregion前景中的比例
            obj_in_foreground = (obj_mask & textregion_mask).sum().item()
            foreground_ratio = obj_in_foreground / obj_pixels
            
            # **放宽判断**：只要有30%以上在前景区域，就认为是前景object
            if foreground_ratio >= foreground_overlap_threshold:
                foreground_objects.add(obj_id)
                status = "FOREGROUND"
            else:
                background_objects.add(obj_id)
                status = "BACKGROUND"
            
            print(f"  Object {obj_id}: {obj_in_foreground}/{obj_pixels} ({foreground_ratio*100:.1f}%) in textregion -> {status}")
        
        print(f"\n[TR Validation] Classification: {len(foreground_objects)} foreground objects, {len(background_objects)} background objects")
        
        if len(foreground_objects) == 0:
            print("[TR Validation] WARNING: No foreground objects found!")
            return False
        
        # **修改**: 计算前景objects的跨帧平均面积，用于过滤太小的objects
        print(f"\n[TR Validation] Computing cross-frame average size for foreground objects...")
        
        foreground_object_avg_areas = {}
        total_foreground_pixels_avg = 0
        
        # Step 1: 计算每个前景object的跨帧平均大小
        for obj_id in foreground_objects:
            obj_sizes = []  # 记录该object在每一帧的大小
            
            for frame_idx in range(self.n_imgs):
                frame_groups = self.region_groups[frame_idx]
                if not isinstance(frame_groups, torch.Tensor):
                    frame_groups = torch.tensor(frame_groups, device=self.dynamic_map.device)
                else:
                    frame_groups = frame_groups.to(self.dynamic_map.device)
                
                obj_mask = (frame_groups == obj_id)
                obj_pixels = obj_mask.sum().item()
                
                if obj_pixels > 0:
                    obj_sizes.append(obj_pixels)
            
            # 计算该object的平均大小
            if len(obj_sizes) > 0:
                avg_pixels = np.mean(obj_sizes)
                foreground_object_avg_areas[obj_id] = {
                    'avg_pixels': avg_pixels,
                    'frames_present': len(obj_sizes),
                    'min_pixels': min(obj_sizes),
                    'max_pixels': max(obj_sizes)
                }
                total_foreground_pixels_avg += avg_pixels
        
        # Step 2: 计算面积比例（相对于所有前景objects的总平均面积）
        for obj_id in foreground_object_avg_areas:
            avg_pixels = foreground_object_avg_areas[obj_id]['avg_pixels']
            ratio = avg_pixels / total_foreground_pixels_avg if total_foreground_pixels_avg > 0 else 0
            foreground_object_avg_areas[obj_id]['ratio'] = ratio
            
            print(f"  Object {obj_id}: avg_size={avg_pixels:.1f} pixels "
                  f"(ratio={ratio*100:.2f}%, frames={foreground_object_avg_areas[obj_id]['frames_present']})")
        
        # Step 3: 过滤掉太小的前景objects（基于跨帧平均面积）
        # 以及过滤出现帧数太少的objects
        valid_foreground_objects = set()
        filtered_small_objects = []
        filtered_short_duration_objects = []
        
        for obj_id in foreground_objects:
            if obj_id not in foreground_object_avg_areas:
                # 该object在所有帧都不存在（理论上不应该发生）
                filtered_short_duration_objects.append(obj_id)
                continue
            
            info = foreground_object_avg_areas[obj_id]
            frames_ratio = info['frames_present'] / self.n_imgs
            
            # 检查是否满足最小帧数要求
            if frames_ratio < min_frames_ratio:
                filtered_short_duration_objects.append(obj_id)
                continue
            
            # 检查是否满足最小面积要求
            if info['ratio'] < min_object_area_ratio:
                filtered_small_objects.append(obj_id)
                continue
            
            # 通过所有检查
            valid_foreground_objects.add(obj_id)
        
        # 打印过滤信息
        if len(filtered_small_objects) > 0:
            print(f"\n[TR Validation] Filtered {len(filtered_small_objects)} small foreground objects (< {min_object_area_ratio*100:.1f}% of avg foreground):")
            for obj_id in filtered_small_objects[:5]:  # 只打印前5个
                if obj_id in foreground_object_avg_areas:
                    info = foreground_object_avg_areas[obj_id]
                    print(f"  Object {obj_id}: avg_size={info['avg_pixels']:.1f} pixels ({info['ratio']*100:.2f}%)")
            if len(filtered_small_objects) > 5:
                print(f"  ... and {len(filtered_small_objects)-5} more")
        else:
            print(f"\n[TR Validation] No small foreground object Exist")
        
        if len(filtered_short_duration_objects) > 0:
            print(f"\n[TR Validation] Filtered {len(filtered_short_duration_objects)} short-duration foreground objects (< {min_frames_ratio*100:.1f}% of frames):")
            for obj_id in filtered_short_duration_objects[:5]:  # 只打印前5个
                if obj_id in foreground_object_avg_areas:
                    info = foreground_object_avg_areas[obj_id]
                    frames_ratio = info['frames_present'] / self.n_imgs
                    print(f"  Object {obj_id}: present in {info['frames_present']}/{self.n_imgs} frames ({frames_ratio*100:.1f}%)")
            if len(filtered_short_duration_objects) > 5:
                print(f"  ... and {len(filtered_short_duration_objects)-5} more")
        else:
            print(f"\n[TR Validation] No short-duration foreground object Exist")
        
        if len(valid_foreground_objects) == 0:
            print("[TR Validation] WARNING: No valid foreground objects after filtering small ones!")
            print(f"[TR Validation] Try lowering min_object_area_ratio (current: {min_object_area_ratio})")
            return False
        
        print(f"[TR Validation] Valid foreground objects: {len(valid_foreground_objects)} (after filtering small ones)")
        
        # Step 3: 计算每个object在所有帧上的平均attention
        print(f"\n[TR Validation] Computing cross-frame attention for each object...")
        
        # 获取dynamic_map的尺寸（token级别）
        H_att, W_att = self.dynamic_map.shape[1], self.dynamic_map.shape[2]
        print(f"[TR Validation] Attention map is at token level: {H_att} x {W_att}")
        
        object_attentions = {}  # {obj_id: mean_attention_across_frames}
        
        for obj_id in all_object_ids:
            # 收集该object在所有帧上的attention
            obj_att_values = []
            
            for frame_idx in range(self.n_imgs):
                # 获取该帧的region_groups（高分辨率）
                frame_groups = self.region_groups[frame_idx]
                if not isinstance(frame_groups, torch.Tensor):
                    frame_groups = torch.tensor(frame_groups, device=self.dynamic_map.device)
                else:
                    frame_groups = frame_groups.to(self.dynamic_map.device)
                
                # 创建该object的mask（高分辨率）
                obj_mask_hr = (frame_groups == obj_id)  # [H_rg, W_rg]
                
                if obj_mask_hr.sum() == 0:
                    continue  # 该object在这一帧不存在
                
                # **关键修改**: 将高分辨率mask下采样到token级别
                # 使用max pooling: 如果patch中有任何像素属于该object，则该token属于该object
                H_rg, W_rg = frame_groups.shape
                patch_size = H_rg // H_att  # 假设是整数倍，通常是16
                
                # 下采样到token级别
                obj_mask_token = torch.nn.functional.max_pool2d(
                    obj_mask_hr.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=patch_size,
                    stride=patch_size
                ).squeeze().bool()  # [H_att, W_att]
                
                if obj_mask_token.sum() == 0:
                    continue
                
                # 获取该object在这一帧的attention
                frame_att = self.dynamic_map[frame_idx]  # [H_att, W_att]
                obj_att = frame_att[obj_mask_token].detach().cpu().numpy()
                obj_att_values.extend(obj_att.tolist())
            
            if len(obj_att_values) > 0:
                mean_att = float(np.mean(obj_att_values))
                object_attentions[obj_id] = mean_att
                
                # 判断该object的类型，并标记
                if obj_id in valid_foreground_objects:
                    obj_type = "FG"
                elif obj_id in filtered_small_objects:
                    obj_type = "FG-small"  # 前景但太小被过滤
                elif obj_id in filtered_short_duration_objects:
                    obj_type = "FG-short"  # 前景但出现时间太短被过滤
                elif obj_id in foreground_objects:
                    obj_type = "FG-filtered"  # 前景但被其他原因过滤
                else:
                    obj_type = "BG"
                
                print(f"  Object {obj_id} ({obj_type}): mean_att={mean_att:.4f} (from {len(obj_att_values)} pixels across frames)")
        self.tr_fg_objects_all = set(foreground_objects)
        self.tr_bg_objects_all = set(background_objects)
        self.tr_fg_valid_objects = set(valid_foreground_objects)
        self.tr_fg_small_filtered = set(filtered_small_objects)
        self.tr_fg_short_filtered = set(filtered_short_duration_objects)
        print(f"[TR Validation] Cached foreground IDs for Step 7: {len(self.tr_fg_valid_objects)} valid / {len(self.tr_fg_objects_all)} total FG")

        if len(valid_foreground_objects) == 0:
            print("[TR Validation] WARNING: No valid foreground objects after filtering small/short ones!")
            print(f"[TR Validation] Try lowering min_object_area_ratio (current: {min_object_area_ratio}) or min_frames_ratio (current: {min_frames_ratio})")
            # 仍然缓存空的有效前景集合，避免外部访问属性时报错
            self.dynamic_map_was_inverted = False
            return False
        
        # Step 4: 找到前景objects中attention最高的
        # **修改**: 只考虑valid_foreground_objects（已过滤小objects）
        foreground_attentions = {obj_id: att for obj_id, att in object_attentions.items() 
                                if obj_id in valid_foreground_objects}
        background_attentions = {obj_id: att for obj_id, att in object_attentions.items() 
                                if obj_id in background_objects}
        
        if len(foreground_attentions) == 0:
            print("[TR Validation] WARNING: No valid foreground objects with attention!")
            return False
        
        # 找到前景中最高attention的object
        max_fg_obj_id = max(foreground_attentions, key=foreground_attentions.get)
        max_fg_attention = foreground_attentions[max_fg_obj_id]
        
        # 计算背景的平均attention
        if len(background_attentions) > 0:
            mean_bg_attention = float(np.mean(list(background_attentions.values())))
        else:
            # 如果没有背景objects，直接计算背景像素的attention
            # 需要将textregion_mask下采样到token级别
            H_rg, W_rg = first_frame_groups.shape
            H_att, W_att = self.dynamic_map.shape[1], self.dynamic_map.shape[2]
            patch_size = H_rg // H_att
            
            # 下采样textregion_mask到token级别（使用min pooling，只要patch中有任何背景像素就算背景）
            bg_mask_hr = ~textregion_mask  # 高分辨率背景mask
            bg_mask_token = torch.nn.functional.max_pool2d(
                bg_mask_hr.float().unsqueeze(0).unsqueeze(0),
                kernel_size=patch_size,
                stride=patch_size
            ).squeeze().bool()  # [H_att, W_att]
            
            bg_att_values = []
            for frame_idx in range(self.n_imgs):
                frame_att = self.dynamic_map[frame_idx]
                bg_att = frame_att[bg_mask_token].detach().cpu().numpy()
                bg_att_values.extend(bg_att.tolist())
            mean_bg_attention = float(np.mean(bg_att_values)) if len(bg_att_values) > 0 else 0.5
        
        # Step 5: 判断是否需要flip
        print(f"\n[TR Validation] ========== FLIP DECISION ==========")
        print(f"  Foreground objects: {len(foreground_attentions)}")
        print(f"    -> Highest attention: Object {max_fg_obj_id} = {max_fg_attention:.4f}")
        if len(foreground_attentions) > 1:
            sorted_fg = sorted(foreground_attentions.items(), key=lambda x: x[1], reverse=True)
            print(f"    -> Top 3 foreground objects:")
            for obj_id, att in sorted_fg[:min(3, len(sorted_fg))]:
                print(f"       Object {obj_id}: {att:.4f}")
        
        print(f"  Background mean attention: {mean_bg_attention:.4f}")
        if len(background_attentions) > 0:
            print(f"    -> From {len(background_attentions)} background objects")
        else:
            print(f"    -> From background pixels (no background objects)")
            
        all_att_values = list(foreground_attentions.values()) + list(background_attentions.values())
        if len(all_att_values) > 0:
            threshold = adaptive_multiotsu_variance(np.array(all_att_values))
        else:
            threshold = 0.5  # fallback

        print(f"[TR Validation] Global Otsu threshold for attention = {threshold:.4f}")
        
        # should_invert = max_fg_attention < mean_bg_attention or threshold < mean_bg_attention
        should_invert = max_fg_attention < mean_bg_attention
        
        if should_invert:
            print(f"\n[TR Validation] DECISION: FLIP (YES)")
            print(f"  Reason: Highest FG object attention ({max_fg_attention:.4f}) < Background ({mean_bg_attention:.4f})")
            print(f"  -> This suggests HIGH attention = STATIC, so we INVERT")
            
            self.dynamic_map = 1.0 - self.dynamic_map
            if hasattr(self, 'refined_dynamic_map') and self.refined_dynamic_map is not None:
                self.refined_dynamic_map = 1.0 - self.refined_dynamic_map
                print("[TR Validation] Also inverted refined_dynamic_map")

            # 打印翻转后的统计
            inv_max_fg = 1.0 - max_fg_attention
            inv_mean_bg = 1.0 - mean_bg_attention
            print(f"[TR Validation] After inversion:")
            print(f"  Highest FG object attention: {inv_max_fg:.4f}")
            print(f"  Background mean attention: {inv_mean_bg:.4f}")
        else:
            print(f"\n[TR Validation] DECISION: NO FLIP")
            print(f"  Reason: Highest FG object attention ({max_fg_attention:.4f}) >= Background ({mean_bg_attention:.4f})")
            print(f"  -> Direction looks correct, no inversion needed")

        # 保存flip状态，供后续filter使用
        self.dynamic_map_was_inverted = should_invert
        print(f"\n[TR Validation] *** Set dynamic_map_was_inverted = {should_invert} ***")
        print(f"[TR Validation] =======================================\n")
        
        return should_invert
    # # 引用region级别token的help function
    # def _downsample_mask_to_tokens(self, mask_bhw, H_img, W_img, patch=16):
    #     """
    #     mask_bhw: torch.bool [B,H_img,W_img]
    #     输出: torch.bool [B, H_img//patch, W_img//patch]
    #     """
    #     B, H, W = mask_bhw.shape
    #     assert H == H_img and W == W_img
    #     x = mask_bhw.float().unsqueeze(1)  # [B,1,H,W]
    #     ds = torch.nn.functional.avg_pool2d(x, kernel_size=patch, stride=patch)
    #     return (ds >= 0.5).squeeze(1)
    @torch.no_grad()
    def make_hr_masks_from_regions(self, attn_map_bhw: torch.Tensor, use_refined=True, include_background=False, patch=16):
        """
        基于低分辨率 attention + 高分辨率 region_groups 生成高分辨率二值 mask
        attn_map_bhw: [B,Ht,Wt] 低分辨率 attention（比如 self.refined_dynamic_map 或 self.dynamic_map）
        返回：list[H_img, W_img] 的 bool mask（与 self.region_groups 同分辨率）
        """
        assert hasattr(self, "region_groups") and len(self.region_groups) == attn_map_bhw.shape[0], \
            "region_groups 未准备好或 batch 大小不一致"

        B, Ht, Wt = attn_map_bhw.shape
        H_img, W_img = self.imshape  # 你的 vis 里也是用这个 target_size【turn8file1†L7-L15】

        # 1) 下采样高分辨率 region_id → token 网格
        groups_hr = torch.stack(self.region_groups, dim=0).long().to(attn_map_bhw.device)   # [B,H_img,W_img]
        groups_token = self._downsample_groups_to_tokens(groups_hr, H_img, W_img, patch=patch)  # [B,Ht,Wt]【turn7file10†L1-L3】

        # 2) 在 token 网格上对每个 region 求 mean（可选是否包含背景0）
        region_mean_on_tokens = self._region_group_mean_pool_map(attn_map_bhw, groups_token, include_background=include_background)  # [B,Ht,Wt]【turn7file11†L4-L13】【turn7file11†L15-L29】

        # 3) 自适应阈值（复用 adaptive_multiotsu_variance）
        #    注意：你的 vis 和 optimizer 里就是拿 upsampled_attns 全局调一次阈值【turn8file1†L38-L40】【turn8file8†L6-L12】
        thr = adaptive_multiotsu_variance(region_mean_on_tokens.detach().cpu().numpy())     #【turn8file2†L20-L29】【turn8file2†L36-L47】

        # 4) 先在 token 网格上二值化，再把 region 选择回到高分辨率
        sel_token = (region_mean_on_tokens > thr)   # [B,Ht,Wt]，更稳的是“按 region 的均值”而不是像素阈值

        # 把“被选中的 token 内的 region id”提取出来，并在 HR 上回填
        hr_masks = []
        for b in range(B):
            gids_token = groups_token[b]                 # [Ht,Wt]
            gids_hr    = groups_hr[b]                    # [H_img,W_img]
            sel_ids = torch.unique(gids_token[sel_token[b]]).tolist()
            if 0 in sel_ids and not include_background:
                sel_ids = [g for g in sel_ids if g != 0]
            if len(sel_ids) == 0:
                hr_masks.append(torch.zeros_like(gids_hr, dtype=torch.bool))
                continue
            mask_hr = torch.zeros_like(gids_hr, dtype=torch.bool)
            for g in sel_ids:
                mask_hr |= (gids_hr == g)
            hr_masks.append(mask_hr)
        return hr_masks
    
    def _downsample_groups_to_tokens(self, groups_bhw: torch.Tensor, H_img: int, W_img: int, patch: int = 16):
        """
        groups_bhw: [B,H_img,W_img] int64（0=背景,1..K=区域）
        直接用最近邻缩放到 token 网格： [B, H_img//patch, W_img//patch] int64
        """
        x = groups_bhw.unsqueeze(1).float()  # [B,1,H,W]
        Ht, Wt = H_img // patch, W_img // patch
        ds = torch.nn.functional.interpolate(x, size=(Ht, Wt), mode='nearest').squeeze(1).long()
        return ds  # [B,Ht,Wt]
    
    def _region_group_mean_pool_map(self, x_bhw_or_bhwc: torch.Tensor, groups_bhw: torch.Tensor, include_background: bool = False):
        """
        x: [B,Ht,Wt] 或 [B,Ht,Wt,C]
        groups_bhw: [B,Ht,Wt] int （0=背景,1..K）
        对每个batch、每个region id 计算均值，并把该region内的值替换为均值。
        """
        if x_bhw_or_bhwc.dim() == 3:
            x4 = x_bhw_or_bhwc[..., None]; squeeze = True
        else:
            x4 = x_bhw_or_bhwc; squeeze = False  # [B,H,W,C]

        B, H, W, C = x4.shape
        out = x4.clone()
        for b in range(B):
            gids = groups_bhw[b]                             # [H,W]
            uniq = torch.unique(gids)
            if not include_background:
                uniq = uniq[uniq != 0]
            for g in uniq.tolist():
                mask = (gids == g).unsqueeze(-1)            # [H,W,1]
                denom = mask.float().sum()                  
                if denom.item() < 1: 
                    continue
                mean_val = (x4[b] * mask).sum(dim=(0,1), keepdim=True) / denom
                out[b] = torch.where(mask.expand_as(out[b]), mean_val.expand_as(out[b]), out[b])
        return out.squeeze(-1) if squeeze else out

    # # 引用region级别token的help function
    # def _region_mean_pool_map(self, x_bhw_or_bhwc, reg_bhw_bool, mode='inside'):
    #     """
    #     对张量在 region 内做均值并回填到区域内。支持 [B,H,W] 或 [B,H,W,C]
    #     """
    #     if x_bhw_or_bhwc.dim() == 3:
    #         x = x_bhw_or_bhwc[..., None]   # [B,H,W,1]
    #         squeeze = True
    #     else:
    #         x = x_bhw_or_bhwc              # [B,H,W,C]
    #         squeeze = False
           
    #     B,H,W,C = x.shape
    #     reg = reg_bhw_bool.unsqueeze(-1).expand_as(x)  # [B,H,W,C]
    #     if mode == 'outside':
    #         reg = ~reg  
    #     denom = reg.float().sum(dim=(1,2), keepdim=True).clamp_min(1.0)
    #     pooled = (x * reg).sum(dim=(1,2), keepdim=True) / denom        # [B,1,1,C]
    #     out = torch.where(reg, pooled.expand_as(x), x)
    #     return out.squeeze(-1) if squeeze else out
    


    def aggregate_attention_maps(self, pred1, pred2):
        
        def aggregate_attention(attention_maps, aggregate_j=True):
            attention_maps = NoGradParamDict({ij: nn.Parameter(attention_maps[n], requires_grad=False) 
                                            for n, ij in enumerate(self.str_edges)})
            aggregated_maps = {}
            for edge, attention_map in attention_maps.items():
                idx = edge.split('_')[1 if aggregate_j else 0]
                att = attention_map.clone()
                if idx not in aggregated_maps: 
                    aggregated_maps[idx] = [att]
                else:
                    aggregated_maps[idx].append(att)
            stacked_att_mean = [None] * len(self.imshapes)
            stacked_att_var = [None] * len(self.imshapes)
            for i, aggregated_map in aggregated_maps.items():
                att = torch.stack(aggregated_map, dim=-1)
                att[0,0] = (att[0,1] + att[1,0])/2
                stacked_att_mean[int(i)] = att.mean(dim=-1)
                stacked_att_var[int(i)] = att.std(dim=-1)
            return torch.stack(stacked_att_mean).float().detach(), torch.stack(stacked_att_var).float().detach()
        
        cross_att_k_i_mean, cross_att_k_i_var = aggregate_attention(pred1['cross_atten_maps_k'], aggregate_j=True)
        cross_att_k_j_mean, cross_att_k_j_var = aggregate_attention(pred2['cross_atten_maps_k'], aggregate_j=False)
        return cross_att_k_i_mean, cross_att_k_i_var, cross_att_k_j_mean, cross_att_k_j_var

    def save_attention_maps(self, save_folder='demo_tmp/attention_vis'):
        self.vis_attention_masks(1-self.cross_att_k_i_mean_fused, save_folder=save_folder, save_name='cross_att_k_i_mean')
        self.vis_attention_masks(self.cross_att_k_i_var_fused, save_folder=save_folder, save_name='cross_att_k_i_var')
        self.vis_attention_masks(1-self.cross_att_k_j_mean_fused, save_folder=save_folder, save_name='cross_att_k_j_mean')
        self.vis_attention_masks(self.cross_att_k_j_var_fused, save_folder=save_folder, save_name='cross_att_k_j_var')
        self.vis_attention_masks(self.dynamic_map, save_folder=save_folder, save_name='dynamic_map')
        # self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map')
        # self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map_labels', \
        #                     cluster_labels=self.dynamic_map_labels)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]

    @property
    def imsizes(self):
        return [(w, h) for h, w in self.imshapes]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def state_dict(self, trainable=True):
        all_params = super().state_dict()
        return {k: v for k, v in all_params.items() if k.startswith(('_', 'pred_i.', 'pred_j.', 'conf_i.', 'conf_j.')) != trainable}

    def load_state_dict(self, data):
        return super().load_state_dict(self.state_dict(trainable=False) | data)

    def _check_edges(self):
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), 'bad pair indices: missing values '
        return len(indices)

    @torch.no_grad()
    def _compute_img_conf(self, pred1_conf, pred2_conf):
        im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
        for e, (i, j) in enumerate(self.edges):
            im_conf[i] = torch.maximum(im_conf[i], pred1_conf[e])
            im_conf[j] = torch.maximum(im_conf[j], pred2_conf[e])
        return im_conf

    def get_adaptors(self):
        adapt = self.pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def _get_poses(self, poses):
        # normalize rotation
        Q = poses[:, :4]
        T = signed_expm1(poses[:, 4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT

    def _set_pose(self, poses, idx, R, T=None, scale=None, force=False):
        # all poses == cam-to-world
        pose = poses[idx]
        if not (pose.requires_grad or force):
            return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    def get_pw_norm_scale_factor(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    def get_pw_scale(self):
        scale = self.pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale

    def get_pw_poses(self):  # cam to world
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    def get_masks(self):
        if self.thr_for_init_conf:
            return [(conf > self.min_conf_thr) for conf in self.init_conf_maps]
        else:
            return [(conf > self.min_conf_thr) for conf in self.im_conf]

    def get_feats(self):
        return self.stacked_feat

    def get_atts(self):
        return self.refined_dynamic_map

    def depth_to_pts3d(self):
        raise NotImplementedError()

    def get_pts3d(self, raw=False, **kwargs):
        res = self.depth_to_pts3d(**kwargs)
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def _set_focal(self, idx, focal, force=False):
        raise NotImplementedError()

    def get_focals(self):
        raise NotImplementedError()

    def get_known_focal_mask(self):
        raise NotImplementedError()

    def get_principal_points(self):
        raise NotImplementedError()

    def get_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.im_conf]
    
    def get_init_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.init_conf_maps]

    def get_im_poses(self):
        raise NotImplementedError()

    def _set_depthmap(self, idx, depth, force=False):
        raise NotImplementedError()

    def get_depthmaps(self, raw=False):
        raise NotImplementedError()

    def clean_pointcloud(self, **kw):
        cams = inv(self.get_im_poses())
        K = self.get_intrinsics()
        depthmaps = self.get_depthmaps()
        all_pts3d = self.get_pts3d()

        new_im_confs = clean_pointcloud(self.im_conf, K, cams, depthmaps, all_pts3d, **kw)

        for i, new_conf in enumerate(new_im_confs):
            self.im_conf[i].data[:] = new_conf
        return self

    def get_tum_poses(self):
        poses = self.get_im_poses()
        tt = np.arange(len(poses)).astype(float)
        tum_poses = [c2w_to_tumpose(p) for p in poses]
        tum_poses = np.stack(tum_poses, 0)
        return [tum_poses, tt]

    def save_tum_poses(self, path):
        traj = self.get_tum_poses()
        save_trajectory_tum_format(traj, path)
        return traj[0] # return the poses
    
    def save_focals(self, path):
        # convert focal to txt
        focals = self.get_focals()
        np.savetxt(path, focals.detach().cpu().numpy(), fmt='%.6f')
        return focals

    def save_intrinsics(self, path):
        K_raw = self.get_intrinsics()
        K = K_raw.reshape(-1, 9)
        np.savetxt(path, K.detach().cpu().numpy(), fmt='%.6f')
        return K_raw

    def save_conf_maps(self, path):
        conf = self.get_conf()
        for i, c in enumerate(conf):
            np.save(f'{path}/conf_{i}.npy', c.detach().cpu().numpy())
        return conf
    
    def save_init_conf_maps(self, path):
        conf = self.get_init_conf()
        for i, c in enumerate(conf):
            np.save(f'{path}/init_conf_{i}.npy', c.detach().cpu().numpy())
        return conf

    def save_rgb_imgs(self, path):
        imgs = self.imgs
        for i, img in enumerate(imgs):
            # convert from rgb to bgr
            img = img[..., ::-1]
            cv2.imwrite(f'{path}/frame_{i:04d}.png', img*255)
        return imgs
    
    def save_dynamic_masks(self, path):
        """
        保留：
        - dynamic_mask_{i}.png（二值 0/255）
        - 0_dynamic_masks.mp4（二值视频）
        - return: dynamic_masks（或 sam2_dynamic_masks）

        新增：
        - {i:05d}_dynamic_color.png（同一帧所有“动态 object”的彩色合成）
        - 1_dynamic_objects_color.mp4（彩色视频）

        注意：统一确保写帧是 uint8 3通道，避免 OpenCV depth 错误。
        """

        os.makedirs(path, exist_ok=True)

        # —— 一定先定义，避免异常后 NameError ——
        dynamic_masks = getattr(self, 'sam2_dynamic_masks', None)
        if dynamic_masks is None:
            dynamic_masks = getattr(self, 'dynamic_masks', None)
        if dynamic_masks is None or len(dynamic_masks) == 0:
            raise RuntimeError("save_dynamic_masks: dynamic masks not ready. "
                            "Call get_motion_mask_from_attns()/get_motion_mask_from_pairs() first.")

        # 辅助：把任意张量/数组 -> numpy
        def to_numpy_uint8(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            else:
                x = np.array(x)
            return x

        # --- 帧目录 ---
        frames_dir_bin = os.path.join(path, 'frames_dynamic_masks')
        frames_dir_color = os.path.join(path, 'frames_dynamic_objects_color')
        os.makedirs(frames_dir_bin, exist_ok=True)
        os.makedirs(frames_dir_color, exist_ok=True)

        # --- 彩色调色板（id->BGR） ---
        def id_to_color(oid: int) -> tuple:
            if oid <= 0:
                return (0, 0, 0)
            H = (oid * 37) % 180  # HSV 色相
            S, V = 200, 255
            hsv = np.uint8([[[H, S, V]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        have_fg = hasattr(self, 'region_groups') and hasattr(self, 'dynamic_object_ids')
        dyn_ids = set(int(x) for x in getattr(self, 'dynamic_object_ids', []) if int(x) > 0)

        # ===== 逐帧输出 =====
        for i, dm in enumerate(dynamic_masks):
            dm_np = to_numpy_uint8(dm)

            # —— 二值图：strict uint8 单通道，然后堆成 3 通道写帧 ——
            bin_mask = ((dm_np > 0).astype(np.uint8)) * 255  # (H,W) uint8 in {0,255}
            cv2.imwrite(os.path.join(path, f'dynamic_mask_{i}.png'), bin_mask)
            bin_bgr = np.dstack([bin_mask, bin_mask, bin_mask])  # 直接堆，不用 cvtColor，避免类型坑
            cv2.imwrite(os.path.join(frames_dir_bin, f'frame_{i:04d}.png'), bin_bgr)

            # —— 彩色 object 合成图 —— 
            if have_fg:
                # region_groups: 所有 SAM2 object 的 id（0=背景）
                rg = to_numpy_uint8(self.region_groups[i]).astype(np.uint8)

                # 1) 从 dynamic_object_ids 取出本序列的“动态 object id”
                dyn_ids = getattr(self, "dynamic_object_ids", [])
                dyn_ids = [int(x) for x in dyn_ids if int(x) > 0 and int(x) < 256]

                if len(dyn_ids) == 0:
                    # 没有任何动态 object，就全背景
                    out_id = np.zeros_like(rg, dtype=np.uint8)
                else:
                    # 2) 给这些动态 id 重新编号成 1..K（避免 id 特别大，便于 eval）
                    dyn_ids_sorted = sorted(set(dyn_ids))
                    id_remap = {oid: (k + 1) for k, oid in enumerate(dyn_ids_sorted)}
                    # out_id 只包含动态 objects，其它全是 0
                    out_id = np.zeros_like(rg, dtype=np.uint8)
                    for oid, new_id in id_remap.items():
                        out_id[rg == oid] = new_id

                # === 这里写出的灰度 PNG 就是 “只含动态 object 的 label mask” ===
                cv2.imwrite(os.path.join(path, f'{i:05d}.png'), out_id)

                # 3) 可视化用的彩色图，同样只画动态 objects
                H, W = out_id.shape
                color_img = np.zeros((H, W, 3), dtype=np.uint8)
                unique_ids = np.unique(out_id)
                for uid in unique_ids:
                    uid = int(uid)
                    if uid == 0:
                        continue
                    color_img[out_id == uid] = id_to_color(uid)

            else:
                # 没有 region_groups/dynamic_object_ids 时的降级可视化
                H, W = bin_mask.shape
                color_img = np.zeros((H, W, 3), dtype=np.uint8)
                color_img[bin_mask > 0] = (0, 255, 255)

            # 确保 uint8 3通道
            color_img = color_img.astype(np.uint8)
            cv2.imwrite(os.path.join(path, f'dynamic_mask_color_{i}.png'), color_img)
            cv2.imwrite(os.path.join(frames_dir_color, f'frame_{i:04d}.png'), color_img)

        # ===== 生成视频（ffmpeg） =====
        if len(dynamic_masks) > 0:
            video_output_path_bin = os.path.join(path, '0_dynamic_masks.mp4')
            os.system(
                f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir_bin}/frame_%04d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_output_path_bin}"'
            )

            video_output_path_color = os.path.join(path, '1_dynamic_objects_color.mp4')
            os.system(
                f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir_color}/frame_%04d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_output_path_color}"'
            )

        # 返回原 dynamic masks
        return dynamic_masks
    
    def save_init_fused_dynamic_masks(self, path):
        # save init_dynamic_masks video
        if len(self.init_dynamic_masks) > 0:
            h, w = self.init_dynamic_masks[0].shape
            # save all frames first
            frames_dir = os.path.join(path, 'frames_init_dynamic_masks')
            os.makedirs(frames_dir, exist_ok=True)
            for i, mask in enumerate(self.init_dynamic_masks):
                frame = (mask * 255).detach().cpu().numpy().astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_output_path = os.path.join(path, '0_init_dynamic_masks.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir}/frame_%04d.png" '
                     f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                     f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                     f'-movflags +faststart -b:v 5000k "{video_output_path}"')

        # save dynamic_masks video
        if len(self.dynamic_masks) > 0:
            h, w = self.dynamic_masks[0].shape
            # save all frames first
            frames_dir = os.path.join(path, 'frames_fused_dynamic_masks')
            os.makedirs(frames_dir, exist_ok=True)
            for i, mask in enumerate(self.dynamic_masks):
                frame = (mask * 255).detach().cpu().numpy().astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_output_path = os.path.join(path, '0_fused_dynamic_masks.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir}/frame_%04d.png" '
                     f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                     f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                     f'-movflags +faststart -b:v 5000k "{video_output_path}"')

    def save_depth_maps(self, path):
        depth_maps = self.get_depthmaps()
        images = []
        
        for i, depth_map in enumerate(depth_maps):
            # Apply color map to depth map
            depth_map_colored = cv2.applyColorMap((depth_map * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
            img_path = f'{path}/frame_{(i):04d}.png'
            cv2.imwrite(img_path, depth_map_colored)
            images.append(Image.open(img_path))
            np.save(f'{path}/frame_{(i):04d}.npy', depth_map.detach().cpu().numpy())
        
        images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
        
        return depth_maps

    def forward(self, ret_details=False):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs))

        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # distance in image i and j
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj
        loss /= self.n_edges  # average over all pairs

        if ret_details:
            return loss, details
        return loss

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, save_score_path=None, save_score_only=False, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == 'msp' or init == 'mst':
            init_fun.init_minimum_spanning_tree(self, save_score_path=save_score_path, save_score_only=save_score_only, niter_PnP=niter_PnP)
            if save_score_only: # if only want the score map
                return None
        elif init == 'known_poses':
            self.preset_pose(known_poses=self.camera_poses, requires_grad=True)
            init_fun.init_from_known_poses(self, min_conf_thr=self.min_conf_thr,
                                           niter_PnP=niter_PnP)
        else:
            raise ValueError(f'bad value for {init=}')

        return global_alignment_loop(self, **kw)

    @torch.no_grad()
    def mask_sky(self):
        res = deepcopy(self)
        for i in range(self.n_imgs):
            sky = segment_sky(self.imgs[i])
            res.im_conf[i][sky] = 0
        return res

    def show(self, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
        viz = SceneViz()
        if self.imgs is None:
            colors = np.random.randint(0, 256, size=(self.n_imgs, 3))
            colors = list(map(tuple, colors.tolist()))
            for n in range(self.n_imgs):
                viz.add_pointcloud(self.get_pts3d()[n], colors[n], self.get_masks()[n])
        else:
            viz.add_pointcloud(self.get_pts3d(), self.imgs, self.get_masks())
            colors = np.random.randint(256, size=(self.n_imgs, 3))

        # camera poses
        im_poses = to_numpy(self.get_im_poses())
        if cam_size is None:
            cam_size = auto_cam_size(im_poses)
        viz.add_cameras(im_poses, self.get_focals(), colors=colors,
                        images=self.imgs, imsizes=self.imsizes, cam_size=cam_size)
        if show_pw_cams:
            pw_poses = self.get_pw_poses()
            viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

            if show_pw_pts3d:
                pts = [geotrf(pw_poses[e], self.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(self.edges)]
                viz.add_pointcloud(pts, (128, 0, 128))

        viz.show(**kw)
        return viz

    @torch.no_grad()
    def generate_sam2_region_groups(self, min_size: int = 100, vis_dir: str | None = None):
        """
        用 SAM2AutomaticMaskGenerator 逐帧生成 proposals，
        依据 area 从大到小依次“占坑”，得到互斥的 group_ids（0=背景，1..K=区域）。
        保存到 self.region_groups: list[LongTensor HxW]

        额外：详细计时输出，定位耗时瓶颈。
        """
        def _sync_if_cuda():
            try:
                if torch.cuda.is_available() and torch.cuda.current_device() is not None:
                    torch.cuda.synchronize()
            except Exception:
                pass

        def colorize_groups(group_ids: np.ndarray) -> np.ndarray:
            assert group_ids.ndim == 2
            palette = np.array([
                [  0,   0,   0],
                [ 31, 119, 180], [255, 127,  14], [ 44, 160,  44], [214,  39,  40],
                [148, 103, 189], [140,  86,  75], [227, 119, 194], [127, 127, 127],
                [188, 189,  34], [ 23, 190, 207], [174, 199, 232], [255, 187, 120],
                [152, 223, 138], [255, 152, 150], [197, 176, 213], [196, 156, 148],
                [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
            ], dtype=np.uint8)
            H, W = group_ids.shape
            colored = np.zeros((H, W, 3), dtype=np.uint8)
            ids = group_ids.astype(np.int64)
            uniq = np.unique(ids)
            for g in uniq:
                if g == 0:
                    colored[ids == 0] = palette[0]
                else:
                    colored[ids == g] = palette[(g % (len(palette)-1)) + 1]
            return colored

        def _draw_frame_id(img_bgr: np.ndarray, i: int, n: int) -> np.ndarray:
            out = img_bgr.copy()
            txt = f"Frame {i+1}/{n}"
            cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            return out

        def _save_progress_bar(progress_png: str, cur: int, total: int, w: int = 480, h: int = 48):
            p = max(0.0, min(1.0, (cur+1)/max(1,total)))
            bar_w = int((w-20) * p)
            img = np.full((h, w, 3), 32, np.uint8)
            cv2.rectangle(img, (10, h//2-10), (10 + bar_w, h//2+10), (80,200,80), -1)
            cv2.rectangle(img, (10, h//2-10), (w-10, h//2+10), (200,200,200), 2)
            txt = f"{cur+1}/{total}"
            cv2.putText(img, txt, (w-120, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(progress_png, img)

        device = self.device if torch.cuda.is_available() else 'cpu'

        # === 模型构建 & 设备管理 ===
        t0 = _now()

        USE_SAM = True  # True=用SAM(v1)，False=用SAM2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def _get_any_param_device(m):
            try:
                return next(m.parameters()).device
            except StopIteration:
                return torch.device('cpu')

        def _force_to_cuda(model, name="model"):
            dev = _get_any_param_device(model)
            print(f"[PROFILE] {name} param device = {dev}")
            if torch.cuda.is_available() and dev.type != 'cuda':
                model = model.to('cuda')
                torch.cuda.synchronize()
                print(f"[PROFILE] {name} moved to CUDA explicitly. Now on {_get_any_param_device(model)}")
            model.eval()
            return model

        # 1) 打印 PyTorch/CUDA 情况
        print(f"[PROFILE] torch.cuda.is_available = {torch.cuda.is_available()}")
        print(f"[PROFILE] torch.version.cuda = {getattr(torch.version, 'cuda', None)}")
        print(f"[PROFILE] cudnn.version = {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}")
        if torch.cuda.is_available():
            print(f"[PROFILE] CUDA device count = {torch.cuda.device_count()}")
            print(f"[PROFILE] current device = {torch.cuda.current_device()}, name = {torch.cuda.get_device_name(torch.cuda.current_device())}")

        # 2) 构建模型
        if USE_SAM:
            # --- SAM(v1) ---
            sam_ckpt = "third_party/segment-anything/checkpoints/sam_vit_l_0b3195.pth"  
            sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt)  
            sam = _force_to_cuda(sam, name="SAM")
            _sync_if_cuda()
            t1 = _now()
            amg = SamAutomaticMaskGenerator(
                sam,
                crop_n_layers=0,
                pred_iou_thresh=0.75,
                output_mode="binary_mask",
            )
        else:
            # --- SAM2 ---
            sam2 = build_sam2(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
                device=device 
            )
            sam2 = _force_to_cuda(sam2, name="SAM2")
            _sync_if_cuda()
            t1 = _now()
            amg = SAM2AutomaticMaskGenerator(
                sam2,
                crop_n_layers=0,
                pred_iou_thresh=0.75,
                output_mode="binary_mask",
            )

        print(f"[PROFILE] build+move: {t1 - t0:.3f}s")

        _sync_if_cuda()
        t2 = _now()

        # 控制台输出模型构建耗时
        try:
            tqdm.tqdm.write(f"[PROFILE] build_sam2: {t1 - t0:.3f}s, AMG init: {t2 - t1:.3f}s")
        except Exception:
            print(f"[PROFILE] build_sam2: {t1 - t0:.3f}s, AMG init: {t2 - t1:.3f}s")

        progress_png = None
        timings_csv = None
        if vis_dir:
            os.makedirs(os.path.join(vis_dir, "groups"), exist_ok=True)
            os.makedirs(os.path.join(vis_dir, "overlays"), exist_ok=True)
            progress_png = os.path.join(vis_dir, "progress.png")
            timings_csv = os.path.join(vis_dir, "sam2_region_group_timings.csv")

        # 计时表头
        per_frame_timings = []
        csv_header = [
            "frame","H","W","n_props","n_groups",
            "t_imgprep","t_generate","t_sort","t_assign","t_tensor","t_vis","t_io","t_total"
        ]

        region_groups = []

        iterator = range(self.n_imgs)
        try:
            iterator = tqdm.tqdm(iterator, desc="SAM2 region grouping", total=self.n_imgs)
        except Exception:
            pass

        for i in iterator:
            frame_start = _now()

            # --- 图像准备 ---
            tA = _now()
            img_rgb = (self.imgs[i] * 255).astype(np.uint8)   # HxWx3 RGB uint8
            H, W = img_rgb.shape[:2]
            _sync_if_cuda()
            tB = _now()

            # --- 生成 proposals ---
            props = amg.generate(img_rgb)                     # list of dicts
            _sync_if_cuda()
            tC = _now()

            # --- 排序 ---
            props = sorted(props, key=lambda p: p['area'], reverse=True)
            _sync_if_cuda()
            tD = _now()

            # --- 占坑分配 ---
            group_ids = np.full((H, W), fill_value=-1, dtype=np.int32)  # -1=未分配
            gid = 0
            for p in props:
                m = p['segmentation'].astype(bool)
                to_assign = (group_ids == -1) & m
                if to_assign.sum() < min_size:
                    continue
                gid += 1
                group_ids[to_assign] = gid
            group_ids[group_ids == -1] = 0                                # 0=背景
            _sync_if_cuda()
            tE = _now()

            # --- Tensor 转移到目标 device ---
            g_tensor = torch.from_numpy(group_ids)
            if str(device).startswith("cuda"):
                g_tensor = g_tensor.to(device, non_blocking=True)
            region_groups.append(g_tensor.long())
            _sync_if_cuda()
            tF = _now()

            # --- 可视化 & 写盘（可选） ---
            vis_cost = 0.0
            io_cost  = 0.0
            if vis_dir:
                # overlay
                v0 = _now()
                lut = np.array([[0,0,0],[0,114,189],[217,83,25],[237,177,32],[126,47,142],
                                [119,172,48],[77,190,238],[162,20,47],[0,128,128],[128,0,128]], dtype=np.uint8)
                color = lut[(group_ids % len(lut))]
                overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, color, 0.4, 0)
                overlay = _draw_frame_id(overlay, i, self.n_imgs)
                v1 = _now()
                cv2.imwrite(os.path.join(vis_dir, "overlays", f"frame_{i:04d}.png"), overlay)
                v2 = _now()

                # groups 伪彩
                color_rgb = colorize_groups(group_ids)
                groups_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                groups_bgr = _draw_frame_id(groups_bgr, i, self.n_imgs)
                v3 = _now()
                cv2.imwrite(os.path.join(vis_dir, "groups", f"frame_{i:04d}.png"), groups_bgr)
                v4 = _now()

                # 进度条图片
                if progress_png:
                    _save_progress_bar(progress_png, i, self.n_imgs)
                v5 = _now()

                vis_cost = (v1 - v0) + (v3 - v2)  # 图像合成/伪彩/标注
                io_cost  = (v2 - v1) + (v4 - v3) + (v5 - v4)  # 写盘 + 进度图
                npy_dir = os.path.join(vis_dir, "groups_npy")
                os.makedirs(npy_dir, exist_ok=True)
                np.save(os.path.join(npy_dir, f"group_{i:04d}.npy"),
                        group_ids.astype(np.int32))

            frame_end = _now()

            # 分段耗时
            t_imgprep = tB - tA
            t_generate = tC - tB
            t_sort = tD - tC
            t_assign = tE - tD
            t_tensor = tF - tE
            t_vis = vis_cost
            t_io = io_cost
            t_total = frame_end - frame_start

            row = [
                i, H, W, len(props), gid,
                round(t_imgprep, 6), round(t_generate, 6), round(t_sort, 6),
                round(t_assign, 6), round(t_tensor, 6), round(t_vis, 6),
                round(t_io, 6), round(t_total, 6)
            ]
            per_frame_timings.append(row)

            # 逐帧简要打印（方便你观察是否 amg.generate 或 assign 过慢）
            try:
                tqdm.tqdm.write(
                    f"[PROFILE f{i:04d}] gen={t_generate:.3f}s  assign={t_assign:.3f}s  "
                    f"vis={t_vis:.3f}s  io={t_io:.3f}s  total={t_total:.3f}s  "
                    f"props={len(props)} groups={gid} ({H}x{W})"
                )
            except Exception:
                print(
                    f"[PROFILE f{i:04d}] gen={t_generate:.3f}s  assign={t_assign:.3f}s  "
                    f"vis={t_vis:.3f}s  io={t_io:.3f}s  total={t_total:.3f}s  "
                    f"props={len(props)} groups={gid} ({H}x{W})"
                )

        # 保存 CSV
        if timings_csv:
            try:
                with open(timings_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_header)
                    writer.writerows(per_frame_timings)
                try:
                    tqdm.tqdm.write(f"[PROFILE] Per-frame timings saved to: {timings_csv}")
                except Exception:
                    print(f"[PROFILE] Per-frame timings saved to: {timings_csv}")
            except Exception as e:
                try:
                    tqdm.tqdm.write(f"[PROFILE] Failed to write timings CSV: {e}")
                except Exception:
                    print(f"[PROFILE] Failed to write timings CSV: {e}")

        # 汇总统计（平均值）
        if per_frame_timings:
            arr = np.array([[float(x) for x in row[5:]] for row in per_frame_timings])  # 只取耗时列
            avg = arr.mean(axis=0)
            labels = ["imgprep","generate","sort","assign","tensor","vis","io","total"]
            summary = ", ".join([f"{k}={v:.3f}s" for k, v in zip(labels, avg)])
            try:
                tqdm.tqdm.write(f"[PROFILE AVG] {summary}")
            except Exception:
                print(f"[PROFILE AVG] {summary}")

        self.region_groups = region_groups  # list[H,W] int64，0=背景

    @torch.no_grad()
    def vis_attention_masks(self, attns_fused, save_folder='demo_tmp/attention_vis', save_name='attention_channels_all_frames', cluster_labels=None):
        B, H, W = attns_fused.shape

        # ensure self.imshape exists, otherwise use the original size
        target_size = getattr(self, 'imshape', (H, W))
        
        # upsample the attention maps
        upsampled_attns = torch.nn.functional.interpolate(
            attns_fused.unsqueeze(1),  # [B, 1, H, W]
            size=target_size, 
            mode='nearest'
        ).squeeze(1)  # [B, H', W']
        
        # if there is cluster_labels, also upsample it
        if cluster_labels is not None:
            upsampled_labels = torch.nn.functional.interpolate(
                cluster_labels.float().unsqueeze(1),  # [B, 1, H, W]
                size=target_size,
                mode='nearest'
            ).squeeze(1).long()  # [B, H', W']
        
        # use matplotlib's Spectral_r color map
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('Spectral_r')
        
        # apply color map to each attention map
        H_up, W_up = upsampled_attns.shape[1:]
        stacked_att_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_attns.device)
        for i in range(B):
            att_np = upsampled_attns[i].cpu().numpy()
            colored_att = cmap(att_np)[:, :, :3]  # remove alpha channel
            colored_att_torch = torch.from_numpy(colored_att).float().permute(2, 0, 1).to(upsampled_attns.device)
            stacked_att_img[i] = colored_att_torch

        # calculate mask
        stacked_mask = (upsampled_attns > adaptive_multiotsu_variance(upsampled_attns.cpu().numpy()))

        if cluster_labels is not None:
            import matplotlib.pyplot as plt
            num_clusters = upsampled_labels.max().item() + 1
            colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))[:, :3]
            colors = torch.from_numpy(colors).float().to(upsampled_labels.device)
            
            stacked_mask_img = torch.zeros((B, 3, H_up, W_up), device=upsampled_labels.device)
            for i in range(num_clusters):
                mask = (upsampled_labels == i) & stacked_mask 
                mask = mask.unsqueeze(1)  # [B, 1, H', W']
                stacked_mask_img += mask * colors[i].view(1, 3, 1, 1)
        else:
            stacked_mask_img = stacked_mask.unsqueeze(1).expand(-1, 3, -1, -1)  # [B, 3, H', W']

        # create grid layout  
        grid_size = int(math.ceil(math.sqrt(B)))
        # for stacked_att and cluster_map create grid
        grid_att = torchvision.utils.make_grid(stacked_att_img, nrow=grid_size, padding=2, normalize=False)
        grid_cluster = torchvision.utils.make_grid(stacked_mask_img, nrow=grid_size, padding=2, normalize=False)
        # concatenate two grids in vertical direction
        final_grid = torch.cat([grid_att, grid_cluster], dim=1)
        torchvision.utils.save_image(final_grid, os.path.join(save_folder, f'0_{save_name}_fused.png'))

        # vis
        fused_save_folder = os.path.join(save_folder, f'0_{save_name}_fused')
        os.makedirs(fused_save_folder, exist_ok=True)

        # save video
        if B > 0:
            # create frames directory for stacked_att_img
            frames_att_dir = os.path.join(fused_save_folder, 'frames_att')
            os.makedirs(frames_att_dir, exist_ok=True)
            
            for i in range(B):
                att_frame = stacked_att_img[i].cpu().numpy().transpose(1, 2, 0)  # convert to HWC format
                frame = (att_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR format
                cv2.imwrite(f'{frames_att_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_att_path = os.path.join(fused_save_folder, f'0_{save_name}_att_video.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_att_dir}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{video_att_path}"')

            # create frames directory for stacked_mask_img
            frames_mask_dir = os.path.join(fused_save_folder, 'frames_mask')
            os.makedirs(frames_mask_dir, exist_ok=True)
            
            for i in range(B):
                mask_frame = stacked_mask_img[i].cpu().numpy().transpose(1, 2, 0)  # convert to HWC format
                frame = (mask_frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to BGR format
                cv2.imwrite(f'{frames_mask_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_mask_path = os.path.join(fused_save_folder, f'0_{save_name}_mask_video.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_mask_dir}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{video_mask_path}"')
            
    def generate_whole_masks_from_grid(
        self, 
        img_rgb, 
        min_size=500, 
        grid=16,
        fill_background=True,      # 新增：是否用小mask填补背景
        small_min_size=100,        # 新增：小mask的最小面积
        small_grid=32              # 新增：用于生成小mask的更密集grid
    ):
        """
        基于 LangSplat 完整逻辑：
        1. 用网格点生成 whole (large) masks，占领主要区域
        2. 如果 fill_background=True，用更密集的grid生成small masks填补背景
        3. 返回分层的、非重叠的 segmentation masks
        
        返回：按面积从大到小的 mask 列表
        """
        import numpy as np
        import torch
        H, W, _ = img_rgb.shape

        # ===== 步骤1: 构建 SAM2 predictor =====
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam2 = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
            device=device
        )
        predictor = SAM2ImagePredictor(sam2)
        predictor.set_image(img_rgb)
        
        # ===== 步骤2: 生成 LARGE (whole) masks =====
        print(f"[LangSplat] Generating LARGE masks from {grid}x{grid} grid...")
        
        def generate_masks_from_grid(predictor, H, W, grid_size, select_mode='largest'):
            """
            从grid生成masks
            select_mode: 'largest' 选最大面积, 'all' 返回所有masks
            """
            masks_raw = []
            ys = np.linspace(0, H-1, grid_size, dtype=np.int32)
            xs = np.linspace(0, W-1, grid_size, dtype=np.int32)
            
            for y in ys:
                for x in xs:
                    pts = np.array([[x, y]], dtype=np.float32)
                    lbls = np.array([1], dtype=np.int32)
                    
                    out = predictor.predict(
                        point_coords=pts, 
                        point_labels=lbls, 
                        multimask_output=True
                    )
                    
                    if isinstance(out, dict):
                        masks = out["masks"]
                    elif isinstance(out, tuple):
                        masks = out[0]
                    else:
                        masks = out
                    
                    masks = masks.astype(bool)
                    
                    if select_mode == 'largest':
                        # 选择最大的mask (whole)
                        areas = masks.reshape(masks.shape[0], -1).sum(1)
                        k = int(np.argmax(areas))
                        masks_raw.append({
                            'segmentation': masks[k],
                            'area': int(areas[k])
                        })
                    elif select_mode == 'smallest':
                        # 选择最小的mask (small/part)
                        areas = masks.reshape(masks.shape[0], -1).sum(1)
                        k = int(np.argmin(areas))
                        masks_raw.append({
                            'segmentation': masks[k],
                            'area': int(areas[k])
                        })
                    elif select_mode == 'all':
                        # 返回所有masks
                        for idx, mask in enumerate(masks):
                            area = int(mask.sum())
                            masks_raw.append({
                                'segmentation': mask,
                                'area': area
                            })
            
            return masks_raw
        
        # 生成large masks
        large_masks = generate_masks_from_grid(predictor, H, W, grid, select_mode='largest')
        large_masks = [m for m in large_masks if m['area'] >= min_size]
        print(f"[LangSplat] Generated {len(large_masks)} large masks")
        
        # 去重
        def deduplicate_masks(masks, iou_threshold=0.8):
            masks.sort(key=lambda d: d['area'], reverse=True)
            
            def compute_iou(mask1, mask2):
                intersection = (mask1 & mask2).sum()
                union = (mask1 | mask2).sum()
                return intersection / (union + 1e-8)
            
            deduplicated = []
            for d in masks:
                m = d['segmentation']
                is_duplicate = False
                for kept in deduplicated:
                    iou = compute_iou(m, kept['segmentation'])
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    deduplicated.append(d)
            
            return deduplicated
        
        large_masks = deduplicate_masks(large_masks)
        print(f"[LangSplat] After deduplication: {len(large_masks)} unique large masks")
        
        # ===== 步骤3: LangSplat 分层分配逻辑 =====
        print(f"[LangSplat] Assigning regions in hierarchical order...")
        
        # -1=未分配, 0=background(最后填充), 1+=objects
        group_ids = np.full((H, W), fill_value=-1, dtype=np.int32)
        group_counter = 1
        final_masks = []
        
        # 第一轮：分配large masks
        print(f"[LangSplat] Round 1: Assigning {len(large_masks)} LARGE masks...")
        for mask_dict in large_masks:
            mask_original = mask_dict['segmentation']
            
            non_assigned_area = (group_ids == -1)
            to_assign_area = mask_original & non_assigned_area
            
            assigned_area = to_assign_area.sum()
            if assigned_area < min_size:
                continue
            
            group_ids[to_assign_area] = group_counter
            
            final_masks.append({
                'segmentation': to_assign_area,
                'area': int(assigned_area),
                'obj_id': group_counter,
                'level': 'large'
            })
            
            group_counter += 1
        
        large_assigned = group_counter - 1
        print(f"[LangSplat] Round 1 complete: {large_assigned} large regions assigned")
        
        # 第二轮：如果需要，用small masks填补背景
        if fill_background:
            # 检查剩余未分配区域
            remaining_area = (group_ids == -1).sum()
            total_area = H * W
            background_ratio = remaining_area / total_area
            
            print(f"[LangSplat] Remaining unassigned area: {background_ratio:.1%}")
            
            if background_ratio > 0.05:  # 如果背景超过5%，进行填充
                print(f"[LangSplat] Round 2: Generating SMALL masks from {small_grid}x{small_grid} grid...")
                
                # 生成small masks - 选择最小的mask或中等的mask
                small_masks = generate_masks_from_grid(
                    predictor, H, W, small_grid, 
                    select_mode='smallest'  # 或 'all' 获取所有尺度
                )
                small_masks = [m for m in small_masks if m['area'] >= small_min_size]
                small_masks = deduplicate_masks(small_masks, iou_threshold=0.7)
                
                print(f"[LangSplat] Generated {len(small_masks)} small masks")
                
                # 按面积排序，优先分配较大的small masks
                small_masks.sort(key=lambda d: d['area'], reverse=True)
                
                print(f"[LangSplat] Round 2: Filling background with small masks...")
                for mask_dict in small_masks:
                    mask_original = mask_dict['segmentation']
                    
                    non_assigned_area = (group_ids == -1)
                    to_assign_area = mask_original & non_assigned_area
                    
                    assigned_area = to_assign_area.sum()
                    if assigned_area < small_min_size:
                        continue
                    
                    group_ids[to_assign_area] = group_counter
                    
                    final_masks.append({
                        'segmentation': to_assign_area,
                        'area': int(assigned_area),
                        'obj_id': group_counter,
                        'level': 'small'
                    })
                    
                    group_counter += 1
                
                small_assigned = group_counter - 1 - large_assigned
                print(f"[LangSplat] Round 2 complete: {small_assigned} small regions assigned")
                
                # 最终剩余
                final_remaining = (group_ids == -1).sum()
                final_background_ratio = final_remaining / total_area
                print(f"[LangSplat] Final unassigned area: {final_background_ratio:.1%}")
        
        # 将剩余未分配区域标记为background (0)
        group_ids[group_ids == -1] = 0
        
        print(f"[LangSplat] Total regions: {len(final_masks)} " +
            f"(large: {sum(1 for m in final_masks if m['level']=='large')}, " +
            f"small: {sum(1 for m in final_masks if m['level']=='small')})")
        
        # 按面积排序：large在前，small在后
        final_masks.sort(key=lambda d: (d['level'] != 'large', -d['area']))
        
        return final_masks

    @torch.no_grad()
    def generate_region_groups_with_tracking(
        self,
        proposal_backend="sam2",  
        min_size=500,             
        max_objects=30,          
        vis_dir=None              
    ):
        """
        直接使用SAM生成的masks作为SAM2的初始regions进行追踪
        
        策略：
        1. 在第0帧用SAM生成高质量的segmentation masks
        2. 将这些masks直接作为SAM2的初始regions
        3. 使用SAM2进行整个视频的追踪
        """
        import numpy as np
        import torch
        import cv2
        import os
        from time import perf_counter as _now
        import shutil
        original_cwd = os.getcwd()
        print(f"[DEBUG] Original CWD: {original_cwd}")
        
        def convert_to_float32_comprehensive(model):
            """彻底转换所有参数和缓冲区为float32"""
            def _convert_module(module):
                for name, param in module.named_parameters(recurse=False):
                    if param.dtype in [torch.bfloat16, torch.float16]:
                        param.data = param.data.float()
                        if param.grad is not None:
                            param.grad.data = param.grad.data.float()
                
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer.dtype in [torch.bfloat16, torch.float16]:
                        module.register_buffer(name, buffer.float())
                
                for child in module.children():
                    _convert_module(child)
            
            _convert_module(model)
            return model

        def _convert_inference_state_dtype(inference_state):
            """转换inference state中的tensor数据类型"""
            if not isinstance(inference_state, dict):
                return
                
            for key, value in inference_state.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype in [torch.bfloat16, torch.float16]:
                        inference_state[key] = value.float()
                elif isinstance(value, dict):
                    _convert_inference_state_dtype(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, torch.Tensor) and item.dtype in [torch.bfloat16, torch.float16]:
                            value[i] = item.float()
                        elif isinstance(item, dict):
                            _convert_inference_state_dtype(item)

        def create_consistent_color_palette():
            """创建一致的颜色调色板"""
            colors = [
                [0, 0, 0],        # 0: 背景 - 黑色
                [31, 119, 180],   # 1: 蓝色
                [255, 127, 14],   # 2: 橙色  
                [44, 160, 44],    # 3: 绿色
                [214, 39, 40],    # 4: 红色
                [148, 103, 189],  # 5: 紫色
                [140, 86, 75],    # 6: 棕色
                [227, 119, 194],  # 7: 粉色
                [127, 127, 127],  # 8: 灰色
                [188, 189, 34],   # 9: 黄绿色
                [23, 190, 207],   # 10: 青色
                [174, 199, 232],  # 11: 浅蓝色
                [255, 187, 120],  # 12: 浅橙色
                [152, 223, 138],  # 13: 浅绿色
                [255, 152, 150],  # 14: 浅红色
                [197, 176, 213],  # 15: 浅紫色
                [196, 156, 148],  # 16: 浅棕色
                [247, 182, 210],  # 17: 浅粉色
                [199, 199, 199],  # 18: 浅灰色
                [219, 219, 141],  # 19: 浅黄色
                [158, 218, 229],  # 20: 浅青色
            ]
            return np.array(colors, dtype=np.uint8)

        def colorize_groups_consistent(group_ids: np.ndarray, color_palette: np.ndarray) -> np.ndarray:
            """使用一致的颜色调色板着色"""
            assert group_ids.ndim == 2
            H, W = group_ids.shape
            colored = np.zeros((H, W, 3), dtype=np.uint8)
            ids = group_ids.astype(np.int64)
            uniq = np.unique(ids)
            
            for g in uniq:
                if g == 0:
                    colored[ids == 0] = color_palette[0]  # 背景
                else:
                    color_idx = (g % (len(color_palette) - 1)) + 1
                    colored[ids == g] = color_palette[color_idx]
            return colored

        def mask_to_points_and_box(mask):
            """从mask提取点击点和边界框"""
            y_coords, x_coords = np.where(mask)
            if len(x_coords) == 0:
                return None, None, None
                
            # 计算边界框
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            
            # 计算质心作为正点击
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            points = np.array([[center_x, center_y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            
            return points, labels, box

        # 强制使用CUDA设备，禁用混合精度
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            device = self.device

        def _force_to_cuda(model, name="model"):
            if torch.cuda.is_available():
                model = model.to('cuda')
                torch.cuda.synchronize()
            model.eval()
            return model
        
        # 构建SAM用于初始mask生成
        USE_SAM = (proposal_backend == "sam1")
        
        if USE_SAM:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            sam_ckpt = "third_party/segment-anything/checkpoints/sam_vit_l_0b3195.pth"
            sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt)
            sam = _force_to_cuda(sam, name="SAM")
            amg = SamAutomaticMaskGenerator(
                sam,
                crop_n_layers=0,
                pred_iou_thresh=0.75,
                output_mode="binary_mask",
            )
        # else:
        #     from sam2.build_sam import build_sam2
        #     from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        #     with torch.cuda.amp.autocast(enabled=False):
        #         sam2_amg = build_sam2(
        #             "configs/sam2.1/sam2.1_hiera_l.yaml", 
        #             "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
        #             device=device
        #         )
        #         sam2_amg = convert_to_float32_comprehensive(sam2_amg)
        #         sam2_amg = _force_to_cuda(sam2_amg, name="SAM2_AMG")
        #         amg = SAM2AutomaticMaskGenerator(
        #             sam2_amg,
        #             crop_n_layers=0,
        #             pred_iou_thresh=0.75,
        #             output_mode="binary_mask",
        #         )
            
        
        # 构建SAM2 video predictor用于追踪
        print(f"[SAM→SAM2] Building SAM2 video predictor...")
        with torch.cuda.amp.autocast(enabled=False):
            from sam2.build_sam import build_sam2_video_predictor
            sam2_predictor = build_sam2_video_predictor(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "third_party/sam2/checkpoints/sam2.1_hiera_large.pt", 
                device=device
            )
            sam2_predictor = convert_to_float32_comprehensive(sam2_predictor)
            sam2_predictor = _force_to_cuda(sam2_predictor, name="SAM2_VideoPredictor")
        
        # 准备视频路径
        unique_id = str(uuid.uuid4())[:8]
        temp_video_dir = f"/tmp/temp_frames_sam2sam2_{unique_id}"
        if os.path.exists("/dev/shm"):
            temp_video_dir = f"/dev/shm/temp_frames_sam2sam2_{unique_id}"
        os.makedirs(temp_video_dir, exist_ok=True)
        
        # 保存所有帧到临时目录
        print(f"[SAM→SAM2] Saving {self.n_imgs} frames...")
        for i in range(self.n_imgs):
            img_rgb = (self.imgs[i] * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(temp_video_dir, f"{i:06d}.jpg"), img_bgr)
        
        # 存储结果
        region_groups = []
        color_palette = create_consistent_color_palette()
        
        # 可视化相关
        if vis_dir:
            os.makedirs(os.path.join(vis_dir, "groups"), exist_ok=True)
            os.makedirs(os.path.join(vis_dir, "overlays"), exist_ok=True)
            os.makedirs(os.path.join(vis_dir, "groups_npy"), exist_ok=True)
            
            # 保存初始SAM结果
            os.makedirs(os.path.join(vis_dir, "sam_initial"), exist_ok=True)

        try:
            # 步骤1: 在第0帧用SAM生成高质量masks
            print(f"[SAM→SAM2] Generating initial SAM masks on frame 0...")
            img_rgb = (self.imgs[0] * 255).astype(np.uint8)
            
            sam_results = self.generate_whole_masks_from_grid(
                img_rgb, min_size=min_size
            )
            sam_results = sam_results[:max_objects]

            print(f"[SAM→SAM2] Selected {len(sam_results)} initial masks " f"({'SAM1-AMG' if USE_SAM else 'SAM2-grid-whole'})")
            # with torch.cuda.amp.autocast(enabled=False):
            #     sam_results = amg.generate(img_rgb)
            
            # # 筛选和排序SAM结果
            # sam_results = sorted(sam_results, key=lambda x: x['area'], reverse=True)
            # sam_results = [r for r in sam_results if r['area'] >= min_size][:max_objects]
            
            # print(f"[SAM→SAM2] Selected {len(sam_results)} high-quality masks from SAM")
            # 可视化初始SAM结果
            if vis_dir:
                H, W = self.imshapes[0]
                sam_group_ids = np.zeros((H, W), dtype=np.int32)
                for i, result in enumerate(sam_results):
                    mask = result['segmentation'].astype(bool)
                    sam_group_ids[mask] = i + 1
                
                sam_colored = colorize_groups_consistent(sam_group_ids, color_palette)
                sam_bgr = cv2.cvtColor(sam_colored, cv2.COLOR_RGB2BGR)
                cv2.putText(sam_bgr, f"SAM Initial Masks: {len(sam_results)} objects", 
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imwrite(os.path.join(vis_dir, "sam_initial", "sam_masks_frame_0.png"), sam_bgr)
                
                # 保存overlay
                sam_overlay = cv2.addWeighted(
                    cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6,
                    sam_colored, 0.4, 0
                )
                cv2.putText(sam_overlay, f"SAM Initial Masks: {len(sam_results)} objects", 
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imwrite(os.path.join(vis_dir, "sam_initial", "sam_overlay_frame_0.png"), sam_overlay)
            
            # 步骤2: 初始化SAM2 video predictor
            print(f"[SAM→SAM2] Initializing SAM2 video tracking...")
            with torch.cuda.amp.autocast(enabled=False):
                inference_state = sam2_predictor.init_state(video_path=temp_video_dir)
                sam2_predictor.reset_state(inference_state)
                _convert_inference_state_dtype(inference_state)
            
            # 步骤3: 将SAM masks添加到SAM2追踪器
            print(f"[SAM→SAM2] Adding {len(sam_results)} SAM masks to SAM2 tracker...")
            added_objects = []
            
            for obj_id, sam_result in enumerate(sam_results, start=1):
                mask = sam_result['segmentation'].astype(bool)
                
                # 从SAM mask提取prompt信息
                points, labels, box = mask_to_points_and_box(mask)
                
                if points is not None:
                    try:
                        with torch.cuda.amp.autocast(enabled=False):
                            # 使用box + points的组合prompt，更稳定
                            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=obj_id,
                                points=points,
                                labels=labels,
                                box=box,  # 同时提供box和points
                            )
                        
                        added_objects.append(obj_id)
                        print(f"[SAM→SAM2] Added object {obj_id} (area: {sam_result['area']})")
                        
                    except Exception as e:
                        print(f"[SAM→SAM2] Failed to add object {obj_id}: {e}")
                        if "BFloat16" in str(e) or "Float" in str(e):
                            print(f"[SAM→SAM2] Data type error, stopping object addition")
                            break
            
            print(f"[SAM→SAM2] Successfully added {len(added_objects)} objects to SAM2 tracker")
            
            # 步骤4: 运行SAM2追踪整个视频
            print(f"[SAM→SAM2] Running SAM2 tracking for all {self.n_imgs} frames...")
            
            # 一次性获取所有帧的追踪结果
            video_segments = {}
            
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
                        if out_frame_idx < self.n_imgs:
                            frame_masks = {}
                            
                            for i, obj_id in enumerate(out_obj_ids):
                                if obj_id in added_objects:
                                    mask_logits = out_mask_logits[i]
                                    
                                    # 数据类型处理
                                    if mask_logits.dtype in [torch.bfloat16, torch.float16]:
                                        mask_logits = mask_logits.float()
                                    
                                    while mask_logits.dim() > 2:
                                        mask_logits = mask_logits.squeeze(0)
                                    
                                    mask = (mask_logits > 0.0).cpu().numpy()
                                    H, W = self.imshapes[out_frame_idx]
                                    
                                    if mask.ndim == 2 and mask.shape == (H, W):
                                        frame_masks[obj_id] = mask
                            
                            video_segments[out_frame_idx] = frame_masks
                            
                            if out_frame_idx % 10 == 0:
                                print(f"[SAM→SAM2] Processed frame {out_frame_idx}, {len(frame_masks)} objects")
                        
                        if out_frame_idx >= self.n_imgs - 1:
                            break
                            
            except Exception as e:
                print(f"[SAM→SAM2] Tracking failed: {e}")
                # 创建空结果
                for frame_idx in range(self.n_imgs):
                    video_segments[frame_idx] = {}
            
            # 步骤5: 转换结果为region_groups
            print(f"[SAM→SAM2] Converting tracking results to region groups...")
            
            for frame_idx in range(self.n_imgs):
                H, W = self.imshapes[frame_idx]
                group_ids = np.zeros((H, W), dtype=np.int32)
                
                if frame_idx in video_segments:
                    for obj_id, mask in video_segments[frame_idx].items():
                        if mask.shape == (H, W):
                            group_ids[mask] = obj_id
                
                # 保存tensor结果
                g_tensor = torch.from_numpy(group_ids.astype(np.int64))
                if str(device).startswith("cuda"):
                    g_tensor = g_tensor.to(device, non_blocking=True)
                region_groups.append(g_tensor)
                
                # 可视化保存
                if vis_dir:
                    # 保存group可视化
                    color_rgb = colorize_groups_consistent(group_ids, color_palette)
                    groups_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                    
                    # 添加信息文字
                    n_objects = len(video_segments.get(frame_idx, {}))
                    txt = f"Frame {frame_idx+1}/{self.n_imgs} | Objects: {n_objects}"
                    cv2.putText(groups_bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
                    cv2.putText(groups_bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                    
                    cv2.imwrite(os.path.join(vis_dir, "groups", f"frame_{frame_idx:04d}.png"), groups_bgr)
                    
                    # 保存overlay
                    img_rgb = (self.imgs[frame_idx] * 255).astype(np.uint8)
                    overlay = cv2.addWeighted(
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6,
                        color_rgb, 0.4, 0
                    )
                    cv2.putText(overlay, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
                    cv2.putText(overlay, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(vis_dir, "overlays", f"frame_{frame_idx:04d}.png"), overlay)
                    
                    # 保存numpy数组
                    np.save(os.path.join(vis_dir, "groups_npy", f"group_{frame_idx:04d}.npy"), 
                        group_ids.astype(np.int32))
        
        finally:
            # 清理所有可能的状态污染
            if hasattr(self, '_temp_video_path'):
                delattr(self, '_temp_video_path')
            current_cwd = os.getcwd()
            print(f"[DEBUG] CWD after tracking: {current_cwd}")
            if current_cwd != original_cwd:
                print(f"[WARNING] Working directory changed!")
                os.chdir(original_cwd)
            # 重置工作目录
            import os
            original_cwd = getattr(self, '_original_cwd', None)
            if original_cwd and os.path.exists(original_cwd):
                os.chdir(original_cwd)

        # 生成可视化视频
        if vis_dir and len(region_groups) > 0:
            print(f"[SAM→SAM2] Generating visualization videos...")
            
            # 生成groups视频
            groups_video_path = os.path.join(vis_dir, "0_sam_to_sam2_groups.mp4")
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{os.path.join(vis_dir, "groups")}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{groups_video_path}"')
            
            # 生成overlay视频  
            overlay_video_path = os.path.join(vis_dir, "0_sam_to_sam2_overlays.mp4")
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{os.path.join(vis_dir, "overlays")}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{overlay_video_path}"')
            
            print(f"[SAM→SAM2] Videos saved: {groups_video_path}, {overlay_video_path}")

        self.region_groups = region_groups  
        self.video_segments = video_segments
        print(f"[SAM→SAM2] Completed SAM→SAM2 tracking for {len(region_groups)} frames")
        print(f"[SAM→SAM2] Total objects tracked: {len(added_objects)}")
        print(f"[SAM→SAM2] Using direct SAM masks as initial regions (no point conversion)")
        
        return region_groups

    @torch.no_grad()
    def compute_region_attention_variance_and_visualize(self, save_folder='demo_tmp/region_variance_vis', window_size=5):
        """
        计算每个region object在所有帧上mean pool后的attention方差，
        并生成基于方差的可视化（方差越大越红）
        新增：滑动窗口方差计算
        
        Args:
            save_folder: 保存结果的文件夹
            window_size: 滑动窗口大小
        """
        if not hasattr(self, 'video_segments') or not hasattr(self, 'region_groups'):
            print("Error: video_segments or region_groups not found. Please run region tracking first.")
            return None, None
        
        # Debug: Check what attention-related attributes exist
        attention_attrs = []
        for attr in ['dynamic_map', 'refined_dynamic_map', 'cross_att_k_i_mean_fused', 'cross_att_k_i_var_fused']:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    attention_attrs.append(f"{attr}: {type(val)} shape={getattr(val, 'shape', 'N/A')}")
                else:
                    attention_attrs.append(f"{attr}: None")
            else:
                attention_attrs.append(f"{attr}: not found")
        
        print("Available attention attributes:")
        for attr_info in attention_attrs:
            print(f"  - {attr_info}")
        
        # Use dynamic_map directly for variance analysis
        if hasattr(self, 'dynamic_map') and self.dynamic_map is not None:
            attention_source = self.dynamic_map
            print(f"Using dynamic_map with shape: {self.dynamic_map.shape}")
        else:
            print("Error: dynamic_map not found. Please ensure use_atten_mask=True when initializing.")
            return None, None
        
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'frames_variance'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'frames_overlay'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'frames_mean'), exist_ok=True)  # 新增
        os.makedirs(os.path.join(save_folder, 'frames_mean_overlay'), exist_ok=True)  # 新增
        os.makedirs(os.path.join(save_folder, 'frames_window_variance'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'frames_window_overlay'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'window_variance_npy'), exist_ok=True)
        
        # 收集所有object的所有帧的mean pooled attention值
        object_attention_values = defaultdict(list)
        frame_object_means = []
        
        print(f"Computing mean pooled attention for {len(self.region_groups)} frames...")
        
        # 获取attention map和region groups的尺寸
        attention_H, attention_W = attention_source.shape[1], attention_source.shape[2]
        print(f"Attention map size: {attention_H} x {attention_W}")
        if len(self.region_groups) > 0:
            region_H, region_W = self.region_groups[0].shape
            print(f"Region groups size: {region_H} x {region_W}")
        
        for frame_idx in range(len(self.region_groups)):
            frame_means = {}
            group_tensor = self.region_groups[frame_idx]
            attention_map = attention_source[frame_idx]
            
            if group_tensor.device != attention_map.device:
                attention_map = attention_map.to(group_tensor.device)
            
            if group_tensor.shape != attention_map.shape:
                group_tensor_resized = torch.nn.functional.interpolate(
                    group_tensor.float().unsqueeze(0).unsqueeze(0),
                    size=(attention_H, attention_W),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            else:
                group_tensor_resized = group_tensor
            
            unique_objects = torch.unique(group_tensor_resized)
            unique_objects = unique_objects[unique_objects != 0]
            
            for obj_id in unique_objects:
                obj_mask = (group_tensor_resized == obj_id)
                if obj_mask.sum() > 0:
                    mean_attention = attention_map[obj_mask].mean().item()
                    object_attention_values[obj_id.item()].append(mean_attention)
                    frame_means[obj_id.item()] = mean_attention
            
            frame_object_means.append(frame_means)
        
        # ========== 计算全局方差和均值 ==========
        object_variances = {}
        object_means = {}  # 新增：保存每个object的mean attention
        for obj_id, attention_values in object_attention_values.items():
            if len(attention_values) >= 2:
                variance = np.var(attention_values)
                object_variances[obj_id] = variance
                mean_val = np.mean(attention_values)
                object_means[obj_id] = mean_val
            else:
                object_variances[obj_id] = 0.0
                object_means[obj_id] = attention_values[0] if len(attention_values) == 1 else 0.0
        
        print(f"Computed global variance for {len(object_variances)} objects")
        print(f"Computed global mean attention for {len(object_means)} objects")
        
        # ========== 计算滑动窗口方差 ==========
        print(f"\nComputing sliding window variance (window_size={window_size})...")
        
        window_variances = defaultdict(list)
        
        for obj_id, attention_values in object_attention_values.items():
            n_frames = len(attention_values)
            
            for start_idx in range(n_frames - window_size + 1):
                end_idx = start_idx + window_size
                window_values = attention_values[start_idx:end_idx]
                
                if len(window_values) >= 2:
                    window_var = np.var(window_values)
                else:
                    window_var = 0.0
                
                window_variances[obj_id].append({
                    'start_frame': start_idx,
                    'end_frame': end_idx - 1,
                    'variance': window_var,
                    'mean': np.mean(window_values)
                })
        
        print(f"Computed sliding window variance for {len(window_variances)} objects")
        
        # 保存滑动窗口方差数据为npy
        window_var_data = {}
        for obj_id, windows in window_variances.items():
            window_var_data[obj_id] = {
                'windows': windows,
                'attention_values': object_attention_values[obj_id]
            }
        
        np.save(os.path.join(save_folder, 'window_variance_npy', 'all_objects_window_variance.npy'), 
                window_var_data, allow_pickle=True)
        print(f"Saved window variance data to: window_variance_npy/all_objects_window_variance.npy")
        
        # 新增：保存全局variance和mean数据为npy
        global_stats_data = {}
        for obj_id in object_attention_values.keys():
            global_stats_data[obj_id] = {
                'global_variance': object_variances.get(obj_id, 0.0),
                'global_mean': object_means.get(obj_id, 0.0),
                'attention_values': object_attention_values[obj_id]
            }
        
        np.save(os.path.join(save_folder, 'window_variance_npy', 'all_objects_global_stats.npy'), 
                global_stats_data, allow_pickle=True)
        print(f"Saved global variance and mean data to: window_variance_npy/all_objects_global_stats.npy")
        
        # ========== 归一化方差 ==========
        if object_variances:
            alpha = 0.3
            variances_array = np.array(list(object_variances.values()))
            percentile_value = np.percentile(variances_array, (1 - alpha) * 100)
            
            if alpha > 0:
                max_value = percentile_value * (1 - alpha) / alpha
            else:
                max_value = variances_array.max()
            
            print(f"\nVariance normalization: alpha = {alpha}")
            print(f"  {(1-alpha)*100:.0f}th percentile = {percentile_value:.6f}")
            print(f"  calculated max_value = {max_value:.6f}")
            print(f"  actual max variance = {variances_array.max():.6f}")
            
            normalized_variances = {}
            for obj_id, var in object_variances.items():
                if max_value > 1e-8:
                    normalized_variances[obj_id] = min(var / max_value, 1.0)
                else:
                    normalized_variances[obj_id] = 0.0
            
            # 归一化窗口方差
            normalized_window_variances = defaultdict(list)
            for obj_id, windows in window_variances.items():
                for window_info in windows:
                    normalized_var = min(window_info['variance'] / max_value, 1.0) if max_value > 1e-8 else 0.0
                    normalized_window_variances[obj_id].append({
                        **window_info,
                        'normalized_variance': normalized_var
                    })
        else:
            normalized_variances = {}
            normalized_window_variances = defaultdict(list)
        
        # 新增：归一化mean attention
        normalized_means = {}
        if object_means:
            means_array = np.array(list(object_means.values()))
            min_mean = means_array.min()
            max_mean = means_array.max()
            print(f"\nMean attention normalization:")
            print(f"  min mean = {min_mean:.6f}")
            print(f"  max mean = {max_mean:.6f}")
            
            for obj_id, mean_val in object_means.items():
                if max_mean > min_mean:
                    normalized_means[obj_id] = (mean_val - min_mean) / (max_mean - min_mean)
                else:
                    normalized_means[obj_id] = 0.5
        
        def variance_to_color(variance_norm):
            """将归一化方差映射到 inferno colormap 颜色"""
            # 保证在 [0,1] 范围内
            x = float(np.clip(variance_norm, 0.0, 1.0))
            r, g, b, _ = inferno_cmap(x)   # 0-1 浮点
            return (int(r * 255), int(g * 255), int(b * 255))

        def mean_to_color(mean_norm):
            """将归一化 mean 映射到 inferno colormap 颜色"""
            x = float(np.clip(mean_norm, 0.0, 1.0))
            r, g, b, _ = inferno_cmap(x)
            return (int(r * 255), int(g * 255), int(b * 255))
        
        # ========== 生成全局方差可视化 ==========
        variance_frames = []
        overlay_frames = []
        
        for frame_idx in range(len(self.region_groups)):
            group_tensor_original = self.region_groups[frame_idx]
            H, W = group_tensor_original.shape
            variance_vis = np.zeros((H, W, 3), dtype=np.uint8)
            
            unique_objects = torch.unique(group_tensor_original)
            unique_objects = unique_objects[unique_objects != 0]
            
            for obj_id in unique_objects:
                obj_id_val = obj_id.item()
                if obj_id_val in normalized_variances:
                    obj_mask = (group_tensor_original == obj_id).cpu().numpy()
                    var_norm = normalized_variances[obj_id_val]
                    color = variance_to_color(var_norm)
                    variance_vis[obj_mask] = color
            
            variance_vis_bgr = cv2.cvtColor(variance_vis, cv2.COLOR_RGB2BGR)
            text_info = f"Global Variance | Frame {frame_idx+1}/{len(self.region_groups)}"
            cv2.putText(variance_vis_bgr, text_info, (12, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imwrite(os.path.join(save_folder, 'frames_variance', f'frame_{frame_idx:04d}.png'), 
                        variance_vis_bgr)
            variance_frames.append(variance_vis_bgr)
            
            if hasattr(self, 'imgs') and self.imgs is not None:
                img_rgb = (self.imgs[frame_idx] * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(img_bgr, 0.6, variance_vis_bgr, 0.4, 0)
                cv2.putText(overlay, text_info, (12, 28), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(save_folder, 'frames_overlay', f'frame_{frame_idx:04d}.png'), 
                            overlay)
                overlay_frames.append(overlay)
        
        # ========== 生成全局mean可视化 ==========
        print(f"\nGenerating global mean attention visualizations...")
        mean_frames = []
        mean_overlay_frames = []
        
        for frame_idx in range(len(self.region_groups)):
            group_tensor_original = self.region_groups[frame_idx]
            H, W = group_tensor_original.shape
            mean_vis = np.zeros((H, W, 3), dtype=np.uint8)
            
            unique_objects = torch.unique(group_tensor_original)
            unique_objects = unique_objects[unique_objects != 0]
            
            for obj_id in unique_objects:
                obj_id_val = obj_id.item()
                if obj_id_val in normalized_means:
                    obj_mask = (group_tensor_original == obj_id).cpu().numpy()
                    mean_norm = normalized_means[obj_id_val]
                    color = mean_to_color(mean_norm)
                    mean_vis[obj_mask] = color
            
            mean_vis_bgr = cv2.cvtColor(mean_vis, cv2.COLOR_RGB2BGR)
            text_info = f"Global Mean Attention | Frame {frame_idx+1}/{len(self.region_groups)}"
            cv2.putText(mean_vis_bgr, text_info, (12, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imwrite(os.path.join(save_folder, 'frames_mean', f'frame_{frame_idx:04d}.png'), 
                        mean_vis_bgr)
            mean_frames.append(mean_vis_bgr)
            
            if hasattr(self, 'imgs') and self.imgs is not None:
                img_rgb = (self.imgs[frame_idx] * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                mean_overlay = cv2.addWeighted(img_bgr, 0.6, mean_vis_bgr, 0.4, 0)
                cv2.putText(mean_overlay, text_info, (12, 28), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(save_folder, 'frames_mean_overlay', f'frame_{frame_idx:04d}.png'), 
                            mean_overlay)
                mean_overlay_frames.append(mean_overlay)
        
        # ========== 生成滑动窗口方差可视化 ==========
        print(f"\nGenerating sliding window variance visualizations...")
        
        window_variance_frames = []
        window_overlay_frames = []
        
        for frame_idx in range(len(self.region_groups)):
            group_tensor_original = self.region_groups[frame_idx]
            H, W = group_tensor_original.shape
            window_var_vis = np.zeros((H, W, 3), dtype=np.uint8)
            
            unique_objects = torch.unique(group_tensor_original)
            unique_objects = unique_objects[unique_objects != 0]
            
            for obj_id in unique_objects:
                obj_id_val = obj_id.item()
                if obj_id_val in normalized_window_variances:
                    relevant_windows = [
                        w for w in normalized_window_variances[obj_id_val]
                        if w['start_frame'] <= frame_idx <= w['end_frame']
                    ]
                    
                    if relevant_windows:
                        avg_normalized_var = np.mean([w['normalized_variance'] for w in relevant_windows])
                        obj_mask = (group_tensor_original == obj_id).cpu().numpy()
                        color = variance_to_color(avg_normalized_var)
                        window_var_vis[obj_mask] = color
            
            window_var_vis_bgr = cv2.cvtColor(window_var_vis, cv2.COLOR_RGB2BGR)
            text_info = f"Window Variance (size={window_size}) | Frame {frame_idx+1}/{len(self.region_groups)}"
            cv2.putText(window_var_vis_bgr, text_info, (12, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imwrite(os.path.join(save_folder, 'frames_window_variance', f'frame_{frame_idx:04d}.png'), 
                        window_var_vis_bgr)
            window_variance_frames.append(window_var_vis_bgr)
            
            if hasattr(self, 'imgs') and self.imgs is not None:
                img_rgb = (self.imgs[frame_idx] * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                window_overlay = cv2.addWeighted(img_bgr, 0.6, window_var_vis_bgr, 0.4, 0)
                cv2.putText(window_overlay, text_info, (12, 28), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(save_folder, 'frames_window_overlay', f'frame_{frame_idx:04d}.png'), 
                            window_overlay)
                window_overlay_frames.append(window_overlay)
        
        # ========== 生成视频 ==========
        if variance_frames:
            variance_video_path = os.path.join(save_folder, '0_global_variance_heatmap.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{save_folder}/frames_variance/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{variance_video_path}"')
            
            if overlay_frames:
                overlay_video_path = os.path.join(save_folder, '0_global_variance_overlay.mp4')
                os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{save_folder}/frames_overlay/frame_%04d.png" '
                        f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                        f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                        f'-movflags +faststart -b:v 5000k "{overlay_video_path}"')
        
        # 新增：生成mean视频
        if mean_frames:
            mean_video_path = os.path.join(save_folder, '0_global_mean_heatmap.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{save_folder}/frames_mean/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{mean_video_path}"')
            
            if mean_overlay_frames:
                mean_overlay_video_path = os.path.join(save_folder, '0_global_mean_overlay.mp4')
                os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{save_folder}/frames_mean_overlay/frame_%04d.png" '
                        f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                        f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                        f'-movflags +faststart -b:v 5000k "{mean_overlay_video_path}"')
        
        if window_variance_frames:
            window_var_video_path = os.path.join(save_folder, '0_window_variance_heatmap.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{save_folder}/frames_window_variance/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{window_var_video_path}"')
            
            if window_overlay_frames:
                window_overlay_video_path = os.path.join(save_folder, '0_window_variance_overlay.mp4')
                os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{save_folder}/frames_window_overlay/frame_%04d.png" '
                        f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                        f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                        f'-movflags +faststart -b:v 5000k "{window_overlay_video_path}"')
        
        # ========== 生成统计图表 (使用constrained_layout代替tight_layout) ==========
        if object_variances:
            # Use constrained_layout from the start - 扩展为4x2布局以包含mean图表
            fig = plt.figure(figsize=(16, 16), constrained_layout=True)
            gs = fig.add_gridspec(4, 2)
            
            # 1. 全局方差柱状图
            ax1 = fig.add_subplot(gs[0, 0])
            obj_ids = list(object_variances.keys())
            variances = list(object_variances.values())
            colors = [variance_to_color(normalized_variances[obj_id]) for obj_id in obj_ids]
            colors_rgb = [(c[0]/255, c[1]/255, c[2]/255) for c in colors]
            ax1.bar(range(len(obj_ids)), variances, color=colors_rgb)
            ax1.set_xlabel('Object ID')
            ax1.set_ylabel('Global Attention Variance')
            ax1.set_title('Global Variance per Object')
            ax1.set_xticks(range(len(obj_ids)))
            ax1.set_xticklabels([f'{i}' for i in obj_ids], rotation=45)
            
            # 2. 全局mean柱状图（新增）
            ax2 = fig.add_subplot(gs[0, 1])
            means = [object_means[obj_id] for obj_id in obj_ids]
            mean_colors = [mean_to_color(normalized_means[obj_id]) for obj_id in obj_ids]
            mean_colors_rgb = [(c[0]/255, c[1]/255, c[2]/255) for c in mean_colors]
            ax2.bar(range(len(obj_ids)), means, color=mean_colors_rgb)
            ax2.set_xlabel('Object ID')
            ax2.set_ylabel('Global Mean Attention')
            ax2.set_title('Global Mean Attention per Object')
            ax2.set_xticks(range(len(obj_ids)))
            ax2.set_xticklabels([f'{i}' for i in obj_ids], rotation=45)
            
            # 3. 全局方差分布
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.hist(variances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('Global Variance Value')
            ax3.set_ylabel('Number of Objects')
            ax3.set_title('Distribution of Global Variances')
            
            # 4. 全局mean分布（新增）
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(means, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.set_xlabel('Global Mean Attention Value')
            ax4.set_ylabel('Number of Objects')
            ax4.set_title('Distribution of Global Mean Attention')
            
            # 5. 时序变化图（全局）
            ax5 = fig.add_subplot(gs[2, 0])
            top_variance_objs = sorted(object_variances.items(), key=lambda x: x[1], reverse=True)[:5]
            for obj_id, var in top_variance_objs:
                if obj_id in object_attention_values:
                    values = object_attention_values[obj_id]
                    frames = list(range(len(values)))
                    color_rgb = [c/255 for c in variance_to_color(normalized_variances[obj_id])]
                    ax5.plot(frames, values, label=f'Obj{obj_id} (var={var:.4f})', 
                            color=color_rgb, linewidth=2, marker='o', markersize=3)
            ax5.set_xlabel('Frame Index')
            ax5.set_ylabel('Mean Attention Value')
            ax5.set_title(f'Attention Temporal Changes (Top 5 by Global Variance)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Mean vs Variance散点图（新增）
            ax6 = fig.add_subplot(gs[2, 1])
            scatter_means = [object_means[obj_id] for obj_id in obj_ids]
            scatter_vars = [object_variances[obj_id] for obj_id in obj_ids]
            ax6.scatter(scatter_means, scatter_vars, alpha=0.6, s=50, c='purple')
            ax6.set_xlabel('Global Mean Attention')
            ax6.set_ylabel('Global Variance')
            ax6.set_title('Mean vs Variance Scatter Plot')
            ax6.grid(True, alpha=0.3)
            
            # 7. 窗口方差热图（显示top object）
            ax7 = fig.add_subplot(gs[3, 0])
            top_objs_for_heatmap = [obj_id for obj_id, _ in top_variance_objs[:10]]
            
            if top_objs_for_heatmap and window_variances:
                max_windows = max(len(window_variances.get(obj_id, [])) for obj_id in top_objs_for_heatmap)
                heatmap_data = []
                valid_obj_ids = []
                
                for obj_id in top_objs_for_heatmap:
                    if obj_id in window_variances and len(window_variances[obj_id]) > 0:
                        window_vars = [w['variance'] for w in window_variances[obj_id]]
                        padded_vars = window_vars + [np.nan] * (max_windows - len(window_vars))
                        heatmap_data.append(padded_vars)
                        valid_obj_ids.append(obj_id)
                
                if heatmap_data:
                    heatmap_array = np.array(heatmap_data)
                    im = ax7.imshow(heatmap_array, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
                    ax7.set_xlabel('Window Index')
                    ax7.set_ylabel('Object ID')
                    ax7.set_title(f'Window Variance Heatmap (window_size={window_size})')
                    ax7.set_yticks(range(len(valid_obj_ids)))
                    ax7.set_yticklabels([f'Obj{i}' for i in valid_obj_ids])
                    # Use figure.colorbar instead of plt.colorbar
                    fig.colorbar(im, ax=ax7, label='Variance')
                else:
                    ax7.text(0.5, 0.5, 'No window variance data available', 
                            ha='center', va='center', transform=ax7.transAxes)
            else:
                ax7.text(0.5, 0.5, 'No objects for heatmap', 
                        ha='center', va='center', transform=ax7.transAxes)
            
            # 8. 示例object的窗口方差曲线
            ax8 = fig.add_subplot(gs[3, 1])
            plotted_any = False
            for obj_id, _ in top_variance_objs[:3]:
                if obj_id in window_variances and len(window_variances[obj_id]) > 0:
                    window_indices = [w['start_frame'] for w in window_variances[obj_id]]
                    window_vars = [w['variance'] for w in window_variances[obj_id]]
                    ax8.plot(window_indices, window_vars, 
                            label=f'Obj{obj_id}', linewidth=2, marker='o', markersize=4)
                    plotted_any = True
            
            if plotted_any:
                ax8.set_xlabel('Window Start Frame')
                ax8.set_ylabel('Window Variance')
                ax8.set_title(f'Window Variance Over Time (Top 3 Objects)')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
            else:
                ax8.text(0.5, 0.5, 'No window data available', 
                        ha='center', va='center', transform=ax8.transAxes)
            
            plt.savefig(os.path.join(save_folder, '0_variance_analysis_comprehensive.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # ========== 保存CSV统计 ==========
        object_stats = []
        for obj_id in sorted(object_attention_values.keys()):
            attention_values = object_attention_values[obj_id]
            mean_attention = np.mean(attention_values)
            variance_attention = object_variances.get(obj_id, 0.0)
            
            # Calculate average pixel area
            pixel_areas = []
            for frame_idx in range(len(self.region_groups)):
                group_tensor = self.region_groups[frame_idx]
                obj_mask = (group_tensor == obj_id)
                pixel_count = obj_mask.sum().item()
                if pixel_count > 0:
                    pixel_areas.append(pixel_count)
            
            avg_pixel_area = np.mean(pixel_areas) if pixel_areas else 0.0
            num_frames_present = len(pixel_areas)
            
            # Calculate mean window variance
            mean_window_var = 0.0
            if obj_id in window_variances and len(window_variances[obj_id]) > 0:
                mean_window_var = np.mean([w['variance'] for w in window_variances[obj_id]])
            
            object_stats.append({
                'object_id': obj_id,
                'attention_mean': mean_attention,
                'global_variance': variance_attention,
                'mean_window_variance': mean_window_var,
                'avg_pixel_area': avg_pixel_area,
                'num_frames_present': num_frames_present,
                'total_frames': len(self.region_groups)
            })
        
        csv_path = os.path.join(save_folder, 'object_statistics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['object_id', 'attention_mean', 'global_variance', 'mean_window_variance',
                        'avg_pixel_area', 'num_frames_present', 'total_frames']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(object_stats)
        
        print(f"\nObject statistics saved to: {csv_path}")
        
        # ========== 保存文本结果 ==========
        results_file = os.path.join(save_folder, 'variance_results.txt')
        with open(results_file, 'w') as f:
            f.write("Region Attention Variance Analysis\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total objects tracked: {len(object_variances)}\n")
            f.write(f"Total frames: {len(self.region_groups)}\n")
            f.write(f"Window size: {window_size}\n\n")
            
            f.write("Global Variance Rankings (High to Low):\n")
            sorted_vars = sorted(object_variances.items(), key=lambda x: x[1], reverse=True)
            for rank, (obj_id, var) in enumerate(sorted_vars, 1):
                mean_win_var = 0.0
                if obj_id in window_variances and len(window_variances[obj_id]) > 0:
                    mean_win_var = np.mean([w['variance'] for w in window_variances[obj_id]])
                f.write(f"{rank:2d}. Object {obj_id:2d}: global_var={var:.6f}, mean_window_var={mean_win_var:.6f}\n")
            
            f.write(f"\nGlobal Variance Statistics:\n")
            f.write(f"Mean variance: {np.mean(list(object_variances.values())):.6f}\n")
            f.write(f"Std variance:  {np.std(list(object_variances.values())):.6f}\n")
            f.write(f"Min variance:  {np.min(list(object_variances.values())):.6f}\n")
            f.write(f"Max variance:  {np.max(list(object_variances.values())):.6f}\n")
            
            f.write(f"\nGlobal Mean Attention Statistics:\n")
            f.write(f"Mean of means: {np.mean(list(object_means.values())):.6f}\n")
            f.write(f"Std of means:  {np.std(list(object_means.values())):.6f}\n")
            f.write(f"Min mean:      {np.min(list(object_means.values())):.6f}\n")
            f.write(f"Max mean:      {np.max(list(object_means.values())):.6f}\n")
        
        print(f"\nVariance and Mean analysis complete!")
        print(f"Results saved to: {save_folder}")
        print(f"  - Global variance videos and overlays")
        print(f"  - Global mean attention videos and overlays")
        print(f"  - Window variance videos and overlays") 
        print(f"  - Window variance data (NPY)")
        print(f"  - Global variance and mean data (NPY)")
        print(f"  - Comprehensive statistics plot (with mean analysis)")
        print(f"  - CSV and text summaries")
        
        return object_variances, object_attention_values
def global_alignment_loop(net, lr=0.01, niter=300, schedule='cosine', lr_min=1e-3, temporal_smoothing_weight=0, depth_map_save_dir=None):
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print('Global alignement - optimizing for:')
        print([name for name, value in net.named_parameters() if value.requires_grad])

    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    loss = float('inf')
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                if bar.n % 500 == 0 and depth_map_save_dir is not None:
                    if not os.path.exists(depth_map_save_dir):
                        os.makedirs(depth_map_save_dir)
                    # visualize the depthmaps
                    depth_maps = net.get_depthmaps()
                    for i, depth_map in enumerate(depth_maps):
                        depth_map_save_path = os.path.join(depth_map_save_dir, f'depthmaps_{i}_iter_{bar.n}.png')
                        plt.imsave(depth_map_save_path, depth_map.detach().cpu().numpy(), cmap='jet')
                    print(f"Saved depthmaps at iteration {bar.n} to {depth_map_save_dir}")
                loss, lr = global_alignment_iter(net, bar.n, niter, lr_base, lr_min, optimizer, schedule, 
                                                 temporal_smoothing_weight=temporal_smoothing_weight)
                bar.set_postfix_str(f'{lr=:g} loss={loss:g}')
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule, 
                                            temporal_smoothing_weight=temporal_smoothing_weight)
    return loss


def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule, temporal_smoothing_weight=0):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    elif schedule.startswith('cycle'):
        try:
            num_cycles = int(schedule[5:])
        except ValueError:
            num_cycles = 2
        lr = cycled_linear_schedule(t, lr_base, lr_min, num_cycles=num_cycles)
    else:
        raise ValueError(f'bad lr {schedule=}')
    
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()

    if net.empty_cache:
        torch.cuda.empty_cache()
    
    loss = net(epoch=cur_iter)
    
    if net.empty_cache:
        torch.cuda.empty_cache()
    
    loss.backward()
    
    if net.empty_cache:
        torch.cuda.empty_cache()
    
    optimizer.step()
    
    return float(loss), lr



@torch.no_grad()
def clean_pointcloud( im_confs, K, cams, depthmaps, all_pts3d, 
                      tol=0.001, bad_conf=0, dbg=()):
    """ Method: 
    1) express all 3d points in each camera coordinate frame
    2) if they're in front of a depthmap --> then lower their confidence
    """
    assert len(im_confs) == len(cams) == len(K) == len(depthmaps) == len(all_pts3d)
    assert 0 <= tol < 1
    res = [c.clone() for c in im_confs]

    # reshape appropriately
    all_pts3d = [p.view(*c.shape,3) for p,c in zip(all_pts3d, im_confs)]
    depthmaps = [d.view(*c.shape) for d,c in zip(depthmaps, im_confs)]
    
    for i, pts3d in enumerate(all_pts3d):
        for j in range(len(all_pts3d)):
            if i == j: continue

            # project 3dpts in other view
            proj = geotrf(cams[j], pts3d)
            proj_depth = proj[:,:,2]
            u,v = geotrf(K[j], proj, norm=1, ncol=2).round().long().unbind(-1)

            # check which points are actually in the visible cone
            H, W = im_confs[j].shape
            msk_i = (proj_depth > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
            msk_j = v[msk_i], u[msk_i]

            # find bad points = those in front but less confident
            bad_points = (proj_depth[msk_i] < (1-tol) * depthmaps[j][msk_j]) & (res[i][msk_i] < res[j][msk_j])

            bad_msk_i = msk_i.clone()
            bad_msk_i[msk_i] = bad_points
            res[i][bad_msk_i] = res[i][bad_msk_i].clip_(max=bad_conf)

    return res


@torch.no_grad()
def cluster_attention_maps(feature, dynamic_map, n_clusters=64):
    """use KMeans to cluster the attention maps using feature
    
    Args:
        feature: encoder feature [B,H,W,C]
        dynamic_map: dynamic_map feature [B,H,W]
        n_clusters: number of clusters
        
    Returns:
        normalized_map: normalized cluster map [B,H,W]
        cluster_labels: reshaped cluster labels [B,H,W]
    """
    # data preprocessing
    B, H, W, C = feature.shape
    feature_np = feature.cpu().numpy()
    flattened_feature = feature_np.reshape(-1, C)
    
    # KMeans clustering
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(flattened_feature)
    
    # calculate the average dynamic score for each cluster
    dynamic_map_np = dynamic_map.cpu().numpy()
    flattened_dynamic = dynamic_map_np.reshape(-1)
    cluster_dynamic_scores = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        cluster_dynamic_scores[i] = np.mean(flattened_dynamic[cluster_mask])
    
    # map the cluster labels to the dynamic score
    cluster_map = cluster_dynamic_scores[cluster_labels]
    normalized_map = cluster_map.reshape(B, H, W)

    # reshape cluster_labels
    reshaped_labels = cluster_labels.reshape(B, H, W)
    
    # convert to torch tensor
    normalized_map = torch.from_numpy(normalized_map).float()
    cluster_labels = torch.from_numpy(reshaped_labels).long()
    
    normalized_map_min = normalized_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    normalized_map_max = normalized_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    normalized_map = (normalized_map - normalized_map_min) / (normalized_map_max - normalized_map_min + 1e-6)

    return normalized_map, cluster_labels

def adaptive_multiotsu_variance(img, verbose=False, max_classes=3):
    """
    自适应 multi-Otsu（按类间方差/√K 最大化）：
      - 如果输入长度 < 4，则将 max_classes 设为 len(img)
      - 清理 NaN/Inf
      - 唯一值过少/常量数组/阈值失败提供稳健兜底

    Returns
    -------
    float : 选定的阈值（使用最后一个切分阈值）
    """
    # ---- 预处理 ----
    arr = np.asarray(img, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        if verbose: print("[AMOV] Empty/invalid input -> return 0.5")
        return 0.5

    n = arr.size
    # 按要求：长度 < 4 时，max_classes = len(img)
    if n < max_classes:
        max_classes = max(2, int(n))  # 至少 2 类，避免无效

    # 唯一值检查
    uniques = np.unique(arr)
    u = uniques.size
    if u == 1:
        if verbose: print(f"[AMOV] Constant array (value={uniques[0]:.6f}) -> return that value")
        return float(uniques[0])

    # classes 上界不能超过唯一值数量
    max_classes_eff = int(max(2, min(max_classes, u)))
    best_score = -np.inf
    best_threshold = None
    best_n_classes = None
    scores = {}

    # ---- 逐 K 计算 ----
    for n_classes in range(2, max_classes_eff + 1):
        try:
            thresholds = threshold_multiotsu(arr, classes=n_classes)
        except Exception as e:
            if verbose: print(f"[AMOV] threshold_multiotsu failed for K={n_classes}: {e}")
            continue

        if thresholds is None or len(thresholds) != (n_classes - 1):
            if verbose: print(f"[AMOV] Invalid thresholds for K={n_classes}: {thresholds}")
            continue

        # 数值稳定：保证严格递增
        if not np.all(np.diff(thresholds) > 0):
            if verbose: print(f"[AMOV] Non-increasing thresholds for K={n_classes}: {thresholds}")
            continue

        # 分桶
        regions = np.digitize(arr, bins=thresholds, right=False)  # 0..(K-1)

        # 若某个 class 为空，跳过（skimage 理论上可避免，但保险）
        means = []
        empty_class = False
        for i in range(n_classes):
            cls_vals = arr[regions == i]
            if cls_vals.size == 0:
                empty_class = True
                break
            means.append(cls_vals.mean())

        if empty_class:
            if verbose: print(f"[AMOV] Empty class for K={n_classes}, skip")
            continue

        means = np.array(means, dtype=np.float64)

        # 评分：类间均值的方差 / sqrt(K)
        # （保持你原公式，但可换更标准的“类间方差（加权）”；此处按你原式）
        var_between = np.var(means)
        score = var_between / np.sqrt(n_classes)

        scores[n_classes] = score
        if score > best_score:
            best_score = score
            best_threshold = float(thresholds[-1])  # 取最后一个阈值
            best_n_classes = n_classes

    # ---- 兜底 ----
    if best_threshold is None:
        # multi-otsu 全部失败时：用中位数兜底（比均值更稳）
        fallback = float(np.median(arr))
        if verbose: print(f"[AMOV] All K failed -> fallback median={fallback:.6f}")
        return fallback

    if verbose:
        print("[AMOV] number of classes score:")
        for k in sorted(scores.keys()):
            tag = " (best)" if k == best_n_classes else ""
            print(f"  K={k}: score={scores[k]:.6f}{tag}")
        print(f"[AMOV] selected K={best_n_classes}, threshold={best_threshold:.6f}")

    return best_threshold

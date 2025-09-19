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
                         sam2_group_output_dir = None):
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
                        reinit_every=10,         
                        min_size=100,             
                        iou_new_thr=0.20,         
                        vis_dir=sam2_group_output_dir            
                    )

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

            # feature
            pred1_feat = pred1['match_feature']
            feat_i = NoGradParamDict({ij: nn.Parameter(pred1_feat[n], requires_grad=False) for n, ij in enumerate(self.str_edges)})
            stacked_feat_i = [feat_i[k] for k in self.str_edges]
            stacked_feat = [None] * len(self.imshapes)
            for i, ei in enumerate(torch.tensor([i for i, j in self.edges])):
                stacked_feat[ei]=stacked_feat_i[i]
            self.stacked_feat = torch.stack(stacked_feat).float().detach()

            self.refined_dynamic_map, self.dynamic_map_labels = cluster_attention_maps(self.stacked_feat, self.dynamic_map, n_clusters=64)
            

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
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map')
        self.vis_attention_masks(self.refined_dynamic_map, save_folder=save_folder, save_name='refined_dynamic_map_labels', \
                            cluster_labels=self.dynamic_map_labels)

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
        dynamic_masks = self.dynamic_masks if getattr(self, 'sam2_dynamic_masks', None) is None else self.sam2_dynamic_masks
        for i, dynamic_mask in enumerate(dynamic_masks):
            cv2.imwrite(f'{path}/dynamic_mask_{i}.png', (dynamic_mask * 255).detach().cpu().numpy().astype(np.uint8))

        # save video - use ffmpeg instead of OpenCV
        if len(dynamic_masks) > 0:
            h, w = dynamic_masks[0].shape
            # save all frames first
            frames_dir = os.path.join(path, 'frames_dynamic_masks')
            os.makedirs(frames_dir, exist_ok=True)
            for i, mask in enumerate(dynamic_masks):
                frame = (mask * 255).detach().cpu().numpy().astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame)
            
            # use ffmpeg to generate video, frame rate set to 24
            video_output_path = os.path.join(path, '0_dynamic_masks.mp4')
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{frames_dir}/frame_%04d.png" '
                     f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                     f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                     f'-movflags +faststart -b:v 5000k "{video_output_path}"')

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
            sam_ckpt = "third_party/segment-anything/checkpoints/sam_vit_l_0b3195.pth"  # 改成你的路径
            sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt)  # 先不 .to
            sam = _force_to_cuda(sam, name="SAM")
            _sync_if_cuda()
            t1 = _now()
            amg = SamAutomaticMaskGenerator(
                sam,
                # 你的超参
                crop_n_layers=0,
                pred_iou_thresh=0.75,
                # 其它保持默认或自行添加
                output_mode="binary_mask",
            )
        else:
            # --- SAM2 ---
            sam2 = build_sam2(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
                device=device  # 有些实现不会完全按这个生效，所以仍强制检查
            )
            sam2 = _force_to_cuda(sam2, name="SAM2")
            _sync_if_cuda()
            t1 = _now()
            amg = SAM2AutomaticMaskGenerator(
                sam2,
                # 你的超参
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
            
    @torch.no_grad()
    def generate_region_groups_with_tracking(
        self,
        proposal_backend="sam1",  
        reinit_every=10,          
        min_size=100,             
        iou_new_thr=0.20,         
        vis_dir=None              
    ):
        """
        修复版本：确保对象ID在整个视频中保持一致
        
        关键改进：
        1. 使用全局ID映射表维护对象一致性
        2. 重置时通过IoU匹配重新关联对象
        3. 统一的颜色调色板确保同一对象同一颜色
        """
        import numpy as np
        import torch
        import cv2
        import os
        from time import perf_counter as _now
        import csv
        from collections import defaultdict
        import shutil
        
        def compute_iou(mask1, mask2):
            """计算两个mask的IoU"""
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            return intersection / (union + 1e-8)

        def create_consistent_color_palette():
            """创建一致的颜色调色板"""
            # 使用固定的颜色序列，确保同一ID总是同一颜色
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
                    # 使用模运算确保颜色循环，但同一ID总是同一颜色
                    color_idx = (g % (len(color_palette) - 1)) + 1
                    colored[ids == g] = color_palette[color_idx]
            return colored

        def match_objects_by_iou(prev_masks, curr_masks, iou_threshold=0.3):
            """通过IoU匹配前后两帧的对象"""
            matching = {}  # curr_id -> prev_id
            used_prev_ids = set()
            
            # 计算所有可能的IoU匹配
            iou_matrix = {}
            for curr_id, curr_mask in curr_masks.items():
                for prev_id, prev_mask in prev_masks.items():
                    if prev_id not in used_prev_ids:
                        iou = compute_iou(curr_mask, prev_mask)
                        if iou > iou_threshold:
                            iou_matrix[(curr_id, prev_id)] = iou
            
            # 贪心匹配：按IoU从大到小
            sorted_matches = sorted(iou_matrix.items(), key=lambda x: x[1], reverse=True)
            
            for (curr_id, prev_id), iou in sorted_matches:
                if curr_id not in matching and prev_id not in used_prev_ids:
                    matching[curr_id] = prev_id
                    used_prev_ids.add(prev_id)
            
            return matching

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

        def create_fresh_sam2_predictor(device):
            """创建一个新的SAM2 predictor实例"""
            with torch.cuda.amp.autocast(enabled=False):
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                
                from sam2.build_sam import build_sam2_video_predictor
                predictor = build_sam2_video_predictor(
                    "configs/sam2.1/sam2.1_hiera_l.yaml",
                    "third_party/sam2/checkpoints/sam2.1_hiera_large.pt", 
                    device=device
                )
                
                predictor = convert_to_float32_comprehensive(predictor)
                predictor.eval()
                
                if torch.cuda.is_available() and device.type == 'cuda':
                    predictor = predictor.to(device)
                
                return predictor

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
        
        # 构建SAM用于proposal generation
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
        else:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            with torch.cuda.amp.autocast(enabled=False):
                sam2 = build_sam2(
                    "configs/sam2.1/sam2.1_hiera_l.yaml", 
                    "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
                    device=device
                )
                sam2 = convert_to_float32_comprehensive(sam2)
                sam2 = _force_to_cuda(sam2, name="SAM2")
                amg = SAM2AutomaticMaskGenerator(
                    sam2,
                    crop_n_layers=0,
                    pred_iou_thresh=0.75,
                    output_mode="binary_mask",
                )
        
        # 准备视频路径
        temp_video_dir = "/tmp/temp_frames_consistent"
        if os.path.exists("/dev/shm"):
            temp_video_dir = "/dev/shm/temp_frames_consistent"
        os.makedirs(temp_video_dir, exist_ok=True)
        
        # 保存所有帧到临时目录
        print(f"[TRACKING] Saving {self.n_imgs} frames for consistent tracking...")
        for i in range(self.n_imgs):
            img_rgb = (self.imgs[i] * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(temp_video_dir, f"{i:06d}.jpg"), img_bgr)
        
        # 核心数据结构
        region_groups = []
        global_object_registry = {}  # global_obj_id -> latest_mask
        next_global_obj_id = 1
        prev_frame_masks = {}  # 上一帧的masks，用于IoU匹配
        color_palette = create_consistent_color_palette()
        
        # 可视化相关
        if vis_dir:
            os.makedirs(os.path.join(vis_dir, "groups"), exist_ok=True)
            os.makedirs(os.path.join(vis_dir, "overlays"), exist_ok=True)
            os.makedirs(os.path.join(vis_dir, "groups_npy"), exist_ok=True)

        try:
            # 初始化SAM2 predictor
            sam2_predictor = create_fresh_sam2_predictor(device)
            
            # 初始化inference state
            print(f"[TRACKING] Initializing consistent tracking...")
            with torch.cuda.amp.autocast(enabled=False):
                inference_state = sam2_predictor.init_state(video_path=temp_video_dir)
                sam2_predictor.reset_state(inference_state)
                _convert_inference_state_dtype(inference_state)
            
            # 第0帧：生成初始proposals
            print(f"[TRACKING] Generating initial proposals...")
            img_rgb = (self.imgs[0] * 255).astype(np.uint8)
            
            with torch.cuda.amp.autocast(enabled=False):
                props = amg.generate(img_rgb)
            
            props = sorted(props, key=lambda p: p['area'], reverse=True)
            props = [p for p in props if p['area'] >= min_size][:12]
            
            # 添加初始对象到SAM2，建立全局ID映射
            sam2_to_global_mapping = {}  # SAM2内部ID -> 全局ID
            
            for prop in props:
                mask = prop['segmentation'].astype(bool)
                y_coords, x_coords = np.where(mask)
                if len(x_coords) > 0:
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    
                    points = np.array([[center_x, center_y]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    
                    try:
                        with torch.cuda.amp.autocast(enabled=False):
                            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=next_global_obj_id,  # 使用全局ID作为SAM2的ID
                                points=points,
                                labels=labels,
                            )
                        
                        sam2_to_global_mapping[next_global_obj_id] = next_global_obj_id
                        global_object_registry[next_global_obj_id] = mask
                        next_global_obj_id += 1
                        
                    except Exception as e:
                        print(f"[TRACKING] Failed to add initial object: {e}")
                        if "BFloat16" in str(e) or "Float" in str(e):
                            break
            
            print(f"[TRACKING] Added {len(sam2_to_global_mapping)} initial objects")
            
            # 连续追踪所有帧
            frames_since_reset = 0
            
            for frame_idx in range(self.n_imgs):
                frame_start = _now()
                
                # 检查是否需要重置inference state
                if frames_since_reset >= reinit_every and frame_idx > 0:
                    print(f"[TRACKING] Resetting inference state at frame {frame_idx} for stability...")
                    
                    try:
                        # 保存当前帧的对象状态（重置前获取）
                        current_objects_backup = global_object_registry.copy()
                        
                        # 重置inference state
                        with torch.cuda.amp.autocast(enabled=False):
                            sam2_predictor.reset_state(inference_state)
                            _convert_inference_state_dtype(inference_state)
                        
                        # 清空映射，准备重新建立
                        sam2_to_global_mapping.clear()
                        
                        # 重新添加所有现有对象到当前帧
                        for global_id, last_mask in current_objects_backup.items():
                            y_coords, x_coords = np.where(last_mask)
                            if len(x_coords) > 0:
                                center_x = int(np.mean(x_coords))
                                center_y = int(np.mean(y_coords))
                                
                                points = np.array([[center_x, center_y]], dtype=np.float32)
                                labels = np.array([1], dtype=np.int32)
                                
                                try:
                                    with torch.cuda.amp.autocast(enabled=False):
                                        _, _, _ = sam2_predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=frame_idx,
                                            obj_id=global_id,  # 使用原来的全局ID
                                            points=points,
                                            labels=labels,
                                        )
                                    sam2_to_global_mapping[global_id] = global_id
                                except Exception as e:
                                    print(f"[TRACKING] Failed to re-add object {global_id}: {e}")
                        
                        frames_since_reset = 0
                        print(f"[TRACKING] Successfully reset with {len(sam2_to_global_mapping)} objects")
                        
                    except Exception as e:
                        print(f"[TRACKING] Reset failed: {e}")
                
                # 获取当前帧的追踪结果
                frame_masks = {}
                H, W = self.imshapes[frame_idx]
                
                try:
                    # 运行propagation到当前帧
                    with torch.cuda.amp.autocast(enabled=False):
                        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
                            if out_frame_idx == frame_idx:
                                for i, sam2_obj_id in enumerate(out_obj_ids):
                                    if sam2_obj_id in sam2_to_global_mapping:
                                        global_id = sam2_to_global_mapping[sam2_obj_id]
                                        
                                        mask_logits = out_mask_logits[i]
                                        if mask_logits.dtype in [torch.bfloat16, torch.float16]:
                                            mask_logits = mask_logits.float()
                                        
                                        while mask_logits.dim() > 2:
                                            mask_logits = mask_logits.squeeze(0)
                                        
                                        mask = (mask_logits > 0.0).cpu().numpy()
                                        if mask.ndim == 2 and mask.shape == (H, W):
                                            frame_masks[global_id] = mask
                                            global_object_registry[global_id] = mask
                                break
                            elif out_frame_idx > frame_idx:
                                break
                                
                except Exception as e:
                    print(f"[TRACKING] Propagation failed for frame {frame_idx}: {e}")
                    if "BFloat16" in str(e) or "Float" in str(e):
                        print(f"[TRACKING] Data type error, using previous frame masks")
                        # 如果有前一帧的masks，用IoU匹配来估计当前帧
                        if prev_frame_masks:
                            frame_masks = prev_frame_masks.copy()
                
                # 可选：检测新对象（较少频率）
                if frame_idx % (reinit_every * 3) == 0 and frame_idx > 0:
                    print(f"[TRACKING] Checking for new objects at frame {frame_idx}")
                    try:
                        img_rgb = (self.imgs[frame_idx] * 255).astype(np.uint8)
                        with torch.cuda.amp.autocast(enabled=False):
                            new_props = amg.generate(img_rgb)
                        
                        new_props = sorted(new_props, key=lambda p: p['area'], reverse=True)
                        new_props = [p for p in new_props if p['area'] >= min_size]
                        
                        # 检查真正新的proposals
                        for prop in new_props[:3]:  # 限制新对象数量
                            prop_mask = prop['segmentation'].astype(bool)
                            is_new = True
                            
                            for existing_mask in frame_masks.values():
                                if compute_iou(prop_mask, existing_mask) > iou_new_thr:
                                    is_new = False
                                    break
                            
                            if is_new:
                                y_coords, x_coords = np.where(prop_mask)
                                if len(x_coords) > 0:
                                    center_x = int(np.mean(x_coords))
                                    center_y = int(np.mean(y_coords))
                                    
                                    points = np.array([[center_x, center_y]], dtype=np.float32)
                                    labels = np.array([1], dtype=np.int32)
                                    
                                    try:
                                        with torch.cuda.amp.autocast(enabled=False):
                                            _, _, _ = sam2_predictor.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=frame_idx,
                                                obj_id=next_global_obj_id,
                                                points=points,
                                                labels=labels,
                                            )
                                        
                                        sam2_to_global_mapping[next_global_obj_id] = next_global_obj_id
                                        global_object_registry[next_global_obj_id] = prop_mask
                                        frame_masks[next_global_obj_id] = prop_mask
                                        next_global_obj_id += 1
                                        print(f"[TRACKING] Added new object {next_global_obj_id-1}")
                                        
                                    except Exception as e:
                                        print(f"[TRACKING] Failed to add new object: {e}")
                                        break
                                        
                    except Exception as e:
                        print(f"[TRACKING] New object detection failed: {e}")
                
                # 生成group_ids（使用一致的全局ID）
                group_ids = np.zeros((H, W), dtype=np.int32)
                for global_id, mask in frame_masks.items():
                    if mask.shape == (H, W):
                        group_ids[mask] = global_id
                
                # 保存结果
                g_tensor = torch.from_numpy(group_ids.astype(np.int64))
                if str(device).startswith("cuda"):
                    g_tensor = g_tensor.to(device, non_blocking=True)
                region_groups.append(g_tensor)
                
                # 可视化保存（使用一致的颜色）
                if vis_dir:
                    # 保存group可视化（使用一致的颜色调色板）
                    color_rgb = colorize_groups_consistent(group_ids, color_palette)
                    groups_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR) 
                    
                    # 添加帧信息
                    txt = f"Frame {frame_idx+1}/{self.n_imgs} | Objects: {len(frame_masks)}"
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
                
                # 更新前一帧masks
                prev_frame_masks = frame_masks.copy()
                frames_since_reset += 1
                
                frame_end = _now()
                if frame_idx % 10 == 0:
                    print(f"[TRACKING] Frame {frame_idx}: {len(frame_masks)} objects, time: {frame_end - frame_start:.3f}s")
        
        finally:
            # 清理
            try:
                shutil.rmtree(temp_video_dir, ignore_errors=True)
            except:
                pass
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 生成可视化视频（使用ffmpeg）
        if vis_dir and len(region_groups) > 0:
            print(f"[TRACKING] Generating visualization videos...")
            
            # 生成groups视频
            groups_video_path = os.path.join(vis_dir, "0_groups_consistent.mp4")
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{os.path.join(vis_dir, "groups")}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{groups_video_path}"')
            
            # 生成overlay视频  
            overlay_video_path = os.path.join(vis_dir, "0_overlays_consistent.mp4")
            os.system(f'/usr/bin/ffmpeg -y -framerate 24 -i "{os.path.join(vis_dir, "overlays")}/frame_%04d.png" '
                    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                    f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                    f'-movflags +faststart -b:v 5000k "{overlay_video_path}"')
            
            print(f"[TRACKING] Videos saved: {groups_video_path}, {overlay_video_path}")

        self.region_groups = region_groups
        print(f"[TRACKING] Completed consistent SAM2 tracking for {len(region_groups)} frames")
        print(f"[TRACKING] Final global object count: {next_global_obj_id - 1}")
        print(f"[TRACKING] Color consistency: Each object ID will have the same color throughout the video")
        
        return region_groups


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

def adaptive_multiotsu_variance(img, verbose=False):
    """adaptive multi-threshold Otsu algorithm based on inter-class variance maximization
    
    Args:
        img: input image array
        verbose: whether to print detailed information
        
    Returns:
        tuple: (best threshold, best number of classes)
    """
    max_classes = 4
    best_score = -float('inf')
    best_threshold = None
    best_n_classes = None
    scores = {}
    
    for n_classes in range(2, max_classes + 1):
        thresholds = threshold_multiotsu(img, classes=n_classes)
        
        regions = np.digitize(img, bins=thresholds)
        var_between = np.var([img[regions == i].mean() for i in range(n_classes)])
        
        score = var_between / np.sqrt(n_classes)
        scores[n_classes] = score
        
        if score > best_score:
            best_score = score
            best_threshold = thresholds[-1]
            best_n_classes = n_classes
    
    if verbose:
        print("number of classes score:")
        for n_classes, score in scores.items():
            print(f"number of classes {n_classes}: score {score:.4f}" + 
                  (" (best)" if n_classes == best_n_classes else ""))
        print(f"final selected number of classes: {best_n_classes}")
    
    return best_threshold
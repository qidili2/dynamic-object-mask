import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib

from dust3r.cloud_opt.base_opt import BasePCOptimizer, edge_str, adaptive_multiotsu_variance
from dust3r.cloud_opt.pair_viewer import PairViewer
from dust3r.utils.geometry import xy_grid, geotrf, depthmap_to_pts3d
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.utils.goem_opt import DepthBasedWarping, OccMask, WarpImage_with_valid_mask, depth_regularization_si_weighted, tum_to_pose_matrix
from third_party.raft import load_RAFT
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
from skimage.filters import threshold_otsu, threshold_multiotsu
import os
import cv2

def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0, per_pixel_thre=50.):
    loss_raw_shape = F.smooth_l1_loss(estimate*mask, gt*mask, beta=beta, reduction='none')
    if per_pixel_thre > 0:
        per_pixel_mask = (loss_raw_shape < per_pixel_thre) * mask
    else:
        per_pixel_mask = mask
    return torch.sum(loss_raw_shape * per_pixel_mask) / (torch.sum(per_pixel_mask) + 1e-6)

def mse_loss_fn(estimate, gt, mask):
    v = torch.sum((estimate*mask-gt*mask)**2) / (torch.sum(mask) + 1e-6)
    return v  # , v.item()

class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, shared_focal=False, flow_loss_fn='smooth_l1', flow_loss_weight=0.0, 
                 depth_regularize_weight=0.0, num_total_iter=300, temporal_smoothing_weight=0, translation_weight=0.1, flow_loss_start_epoch=0.15, flow_loss_thre=50,
                 sintel_ckpt=False, use_self_mask=False, pxl_thre=50, sam2_mask_refine=True, motion_mask_thre=0.35, batchify=True, use_atten_mask=False, **kwargs):
        self.textregion_bg_track_dir = kwargs.pop("textregion_bg_track_dir", None)
        if self.textregion_bg_track_dir is not None:
            print(f"[Init] textregion_bg_track_dir = {self.textregion_bg_track_dir}")
            
        super().__init__(*args, use_atten_mask=use_atten_mask, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break
        self.num_total_iter = num_total_iter
        self.temporal_smoothing_weight = temporal_smoothing_weight
        self.translation_weight = translation_weight
        self.flow_loss_flag = False
        self.flow_loss_start_epoch = flow_loss_start_epoch
        self.flow_loss_thre = flow_loss_thre
        self.optimize_pp = optimize_pp
        self.pxl_thre = pxl_thre
        self.motion_mask_thre = motion_mask_thre
        self.batchify = batchify
        self.use_atten_mask = use_atten_mask
        

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.shared_focal = shared_focal
        if self.shared_focal:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes[:1])  # camera intrinsics
        else:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        if self.batchify:
            self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area) #(num_imgs, H*W)
            self.im_poses = ParameterStack(self.im_poses, is_param=True)
            self.im_focals = ParameterStack(self.im_focals, is_param=True)
            self.im_pp = ParameterStack(self.im_pp, is_param=True)
            self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
            self.register_buffer('_grid', ParameterStack(
                [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))
            # pre-compute pixel weights
            self.register_buffer('_weight_i', ParameterStack(
                [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
            self.register_buffer('_weight_j', ParameterStack(
                [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))
            # precompute aa
            self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
            self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
            
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

        self.depth_wrapper = DepthBasedWarping()
        self.backward_warper = WarpImage_with_valid_mask()
        self.depth_regularizer = depth_regularization_si_weighted
        if flow_loss_fn == 'smooth_l1':
            self.flow_loss_fn = smooth_L1_loss_fn
        elif flow_loss_fn == 'mse':
            self.low_loss_fn = mse_loss_fn

        self.flow_loss_weight = flow_loss_weight
        self.depth_regularize_weight = depth_regularize_weight
        if self.flow_loss_weight > 0:
            self.flow_ij, self.flow_ji, self.flow_valid_mask_i, self.flow_valid_mask_j = self.get_flow(sintel_ckpt) # (num_pairs, 2, H, W)
            if use_self_mask and not use_atten_mask: self.get_motion_mask_from_pairs(*args)
            # turn off the gradient for the flow
            self.flow_ij.requires_grad_(False)
            self.flow_ji.requires_grad_(False)
            self.flow_valid_mask_i.requires_grad_(False)
            self.flow_valid_mask_j.requires_grad_(False)

        if use_self_mask and use_atten_mask: 
            self.get_motion_mask_from_attns()

        if sam2_mask_refine: 
            with torch.no_grad():
                self.refine_motion_mask_w_sam2()
        else:
            self.sam2_dynamic_masks = None


    def get_flow(self, sintel_ckpt=False):
        print('precomputing flow...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        get_valid_flow_mask = OccMask(th=3.0)
        flow_net = load_RAFT() if sintel_ckpt else load_RAFT("third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        flow_net = flow_net.to(device)
        flow_net.eval()

        with torch.no_grad():
            chunk_size = 12
            flow_ij = []
            flow_ji = []
            num_pairs = len(self._ei)
            
            for i in tqdm(range(0, num_pairs, chunk_size)):
                end_idx = min(i + chunk_size, num_pairs)
                # build current batch image pairs in the loop
                batch_imgs = [
                    np.stack([self.imgs[idx] for idx in self._ei[i:end_idx]]),
                    np.stack([self.imgs[idx] for idx in self._ej[i:end_idx]])
                ]
                
                imgs_ij = [
                    torch.tensor(batch_imgs[0]).float().to(device),
                    torch.tensor(batch_imgs[1]).float().to(device)
                ]
                
                flow_ij.append(flow_net(imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                      imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                      iters=20, test_mode=True)[1])
                flow_ji.append(flow_net(imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                      imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                      iters=20, test_mode=True)[1])
                
                # clean the memory of the current batch
                del batch_imgs, imgs_ij
                torch.cuda.empty_cache()

            flow_ij = torch.cat(flow_ij, dim=0)
            flow_ji = torch.cat(flow_ji, dim=0)
            valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
            valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
            
        print('flow precomputed')
        # clean the flow net
        if flow_net is not None: 
            del flow_net
            torch.cuda.empty_cache()
            
        return flow_ij, flow_ji, valid_mask_i, valid_mask_j
    @torch.no_grad()
    def _parse_seq_from_imgpath(self, img_path: str):
        parts = img_path.split('/')
        if 'JPEGImages' in parts:
            i = parts.index('JPEGImages')
            if len(parts) > i + 2:
                return parts[i + 2]
        return os.path.basename(os.path.dirname(img_path))

    @torch.no_grad()
    def load_textregion_bg_track_masks(self, binarize_thr=127):
        """
        读取 TextRegion 的背景跟踪结果（全帧），返回: bg_masks [B, H, W] bool
        bg_masks[t] == True 表示 background
        """
        if not hasattr(self, "textregion_bg_track_dir") or self.textregion_bg_track_dir is None:
            print("[TR BG-Track] textregion_bg_track_dir missing")
            return None

        if not hasattr(self, "img_pathes") or self.img_pathes is None or len(self.img_pathes) == 0:
            print("[TR BG-Track] img_pathes missing/empty; cannot infer seq")

        seq = self._parse_seq_from_imgpath(self.img_pathes[0])
        masks_dir = os.path.join(self.textregion_bg_track_dir, seq, "masks")
        if not os.path.isdir(masks_dir):
            print(f"[TR BG-Track] masks_dir not found: {masks_dir}")
            return None

        H_img, W_img = self.imshape
        bg_list = []
        missing = 0

        # 假设你的帧命名就是 00000.png, 00001.png, ...
        for t in range(self.n_imgs):
            p = os.path.join(masks_dir, f"{t:05d}.png")
            if not os.path.exists(p):
                missing += 1
                bg_list.append(None)
                continue
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                missing += 1
                bg_list.append(None)
                continue

            # 统一尺寸
            if m.shape != (H_img, W_img):
                m = cv2.resize(m, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

            # 假设：>127 为 background（如果你实际相反，把 > 改成 <= 即可）
            bg = (m > binarize_thr)
            bg_list.append(torch.from_numpy(bg).bool())

        # 如果全缺就返回 None
        valid_cnt = sum(x is not None for x in bg_list)
        if valid_cnt == 0:
            print(f"[TR BG-Track] No valid masks under {masks_dir}")
            return None

        # 对缺失帧：用上一帧填充（或者全 False）
        last = None
        for i in range(len(bg_list)):
            if bg_list[i] is None:
                if last is None:
                    bg_list[i] = torch.zeros((H_img, W_img), dtype=torch.bool)
                else:
                    bg_list[i] = last.clone()
            last = bg_list[i]

        bg_masks = torch.stack(bg_list, dim=0)  # [B,H,W]
        print(f"[TR BG-Track] Loaded bg masks: valid={valid_cnt}/{self.n_imgs}, missing={missing}")
        return bg_masks
    @torch.no_grad()
    def load_textregion_first_frame(self, prefer_npy=True, bin_suffix="_bin"):
        """
        加载第一帧的textregion mask，用于过滤在背景区域的objects
        
        Returns:
            torch.Tensor or None: 第一帧的textregion mask (H_img, W_img)，背景=0，前景=1
        """
        if not hasattr(self, 'textregion_annotations_dir') or self.textregion_annotations_dir is None:
            print("[TextRegion Filter] No textregion_annotations_dir provided")
            return None
            
        if not hasattr(self, 'img_pathes') or self.img_pathes is None or len(self.img_pathes) == 0:
            print("[TextRegion Filter] img_pathes not found")
            return None
        
        # 解析第一帧的路径
        def _seq_and_stem(img_path: str):
            parts = img_path.split('/')
            if 'JPEGImages' in parts:
                i = parts.index('JPEGImages')
                if len(parts) <= i + 2:
                    return None, None
                seq = parts[i + 2]
                stem = os.path.splitext(parts[-1])[0]
                return seq, stem
            seq = os.path.basename(os.path.dirname(img_path))
            stem = os.path.splitext(os.path.basename(img_path))[0]
            return seq, stem
        
        img_path = self.img_pathes[0]  # 第一帧
        seq, stem = _seq_and_stem(img_path)
        
        if seq is None or stem is None:
            print(f"[TextRegion Filter] Cannot parse seq/stem from {img_path}")
            return None
        
        # TextRegion 二值文件路径
        seq_dir = os.path.join(self.textregion_annotations_dir, seq)
        npy_path = os.path.join(seq_dir, f"{stem}{bin_suffix}.npy")
        png_path = os.path.join(seq_dir, f"{stem}{bin_suffix}.png")
        
        # 读取二值 mask
        bin_mask = None
        if prefer_npy and os.path.exists(npy_path):
            try:
                bin_mask = np.load(npy_path).astype(np.uint8)
                print(f"[TextRegion Filter] Loaded {npy_path}")
            except Exception as e:
                print(f"[TextRegion Filter] Failed to load {npy_path}: {e}")
        
        if bin_mask is None and os.path.exists(png_path):
            m = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                bin_mask = (m > 127).astype(np.uint8)
                print(f"[TextRegion Filter] Loaded {png_path}")
        
        if bin_mask is None and not prefer_npy and os.path.exists(npy_path):
            try:
                bin_mask = np.load(npy_path).astype(np.uint8)
                print(f"[TextRegion Filter] Loaded {npy_path}")
            except Exception as e:
                print(f"[TextRegion Filter] Failed to load {npy_path}: {e}")
        
        if bin_mask is None:
            print(f"[TextRegion Filter] Mask not found for {seq}/{stem}")
            return None
        
        # 转换为torch tensor
        H_img, W_img = self.imshape
        if bin_mask.shape != (H_img, W_img):
            bin_mask = cv2.resize(bin_mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        
        textregion_mask = torch.from_numpy(bin_mask).bool()
        print(f"[TextRegion Filter] TextRegion mask loaded: {textregion_mask.sum().item()} foreground pixels")
        
        return textregion_mask
    @torch.no_grad()
    def filter_objects_by_textregion(self, object_global_attention, groups_hr, attention_threshold):
        """
        根据textregion first frame过滤objects:
        **核心逻辑**: 只过滤那些"高attention但在背景区域"的objects（噪声）
        保留那些在前景区域的objects，即使attention低
        
        Args:
            object_global_attention: dict, {obj_id: mean_attention}
            groups_hr: torch.Tensor, [B, H_img, W_img] region groups
            attention_threshold: float, 用于判断dynamic的阈值
            
        Returns:
            set: 需要过滤掉的object IDs
        """
        # 加载第一帧的textregion mask
        textregion_mask = self.load_textregion_first_frame()
        
        if textregion_mask is None:
            print("[TextRegion Filter] No textregion mask available, skipping filter")
            return set()
        
        device = groups_hr.device
        textregion_mask = textregion_mask.to(device)
        
        # 获取第一帧的region groups
        first_frame_groups = groups_hr[0]  # [H_img, W_img]
        
        # 统计每个object在第一帧中的像素
        filtered_objects = set()
        
        print(f"[TextRegion Filter] Using attention threshold: {attention_threshold:.4f}")
        
        for obj_id, global_att in object_global_attention.items():
            if obj_id == 0:  # 跳过背景
                continue
            
            # 找到该object在第一帧的所有像素
            obj_mask = (first_frame_groups == obj_id)  # [H_img, W_img]
            
            if obj_mask.sum() == 0:
                # 该object在第一帧不存在，跳过
                continue
            
            # 计算该object在textregion前景区域的像素比例
            # textregion_mask: 1=前景(objects), 0=背景(黑块)
            obj_in_foreground = obj_mask & textregion_mask
            obj_in_background = obj_mask & (~textregion_mask)
            
            total_pixels = obj_mask.sum().item()
            foreground_pixels = obj_in_foreground.sum().item()
            background_pixels = obj_in_background.sum().item()
            
            foreground_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0
            background_ratio = 1 - foreground_ratio
            
            # **关键修改**: 只过滤那些"高attention + 在背景区域"的objects
            # 这些才是真正的噪声（比如水面反光、背景纹理等）
            # 如果object在前景区域（foreground_ratio > 0.1），即使attention低也不过滤
            should_filter = False
            reason = ""
            
            if background_ratio > 0.9:  # 超过90%在背景区域
                if global_att > attention_threshold:  # 并且attention高于阈值
                    should_filter = True
                    reason = f"HIGH attention ({global_att:.4f}) but in background ({background_ratio*100:.1f}%)"
            
            if should_filter:
                filtered_objects.add(obj_id)
                print(f"[TextRegion Filter] Object {obj_id} filtered: {reason}")
            elif foreground_ratio > 0.1:  # 在前景区域的objects，打印信息但不过滤
                print(f"[TextRegion Filter] Object {obj_id} KEPT: "
                      f"attention={global_att:.4f}, "
                      f"in_foreground={foreground_pixels}/{total_pixels} ({foreground_ratio*100:.1f}%)")
        
        if len(filtered_objects) > 0:
            print(f"[TextRegion Filter] Filtered {len(filtered_objects)} objects (high attention + in background)")
        else:
            print(f"[TextRegion Filter] No objects filtered")
        
        return filtered_objects
    
    @torch.no_grad()
    def get_motion_mask_from_attns(self):
        """
        基于region-level的动态判断：
        1. 计算每个object在所有帧上的mean attention
        2. 用全局阈值判断object是否dynamic
        3. **新增**: 检查textregion first frame，过滤掉在背景区域的objects
        4. 将dynamic objects在各帧的region作为mask
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 检查是否有region groups
        if not hasattr(self, "region_groups") or len(self.region_groups) != self.n_imgs:
            print("[WARNING] region_groups not available, falling back to pixel-wise method")
            self._get_motion_mask_pixelwise()
            return
        
        # 获取attention maps - 确保在正确的设备上
        attns = self.get_atts()  # [B, Ht, Wt]
        if not isinstance(attns, torch.Tensor):
            attns = torch.stack(attns)
        attns = attns.to(device)  # 确保在正确的设备上
        
        B, Ht, Wt = attns.shape
        H_img, W_img = self.imshape
        
        print(f"[Region-level Dynamic Detection] Processing {B} frames...")
        
        # Step 1: 下采样 region_groups 到 token 级别
        # 确保所有region_groups都在同一个设备上
        groups_hr_list = []
        for rg in self.region_groups:
            if isinstance(rg, torch.Tensor):
                groups_hr_list.append(rg.to(device))
            else:
                groups_hr_list.append(torch.tensor(rg, device=device))
        
        groups_hr = torch.stack(groups_hr_list, dim=0).long()  # [B, H_img, W_img]
        groups_token = self._downsample_groups_to_tokens(groups_hr, H_img, W_img, patch=16)  # [B, Ht, Wt]
        
        # Step 2: 收集所有object在所有帧的attention值
        # 收集全局所有出现过的object IDs
        all_object_ids = set()
        for frame_idx in range(B):
            unique_ids = torch.unique(groups_token[frame_idx])
            all_object_ids.update(unique_ids[unique_ids != 0].cpu().tolist())
        
        all_object_ids = sorted(list(all_object_ids))
        print(f"[Region-level Dynamic Detection] Found {len(all_object_ids)} unique objects")
        
        # Step 3: 对每个object，计算其在所有帧上的mean attention
        object_global_attention = {}  # {obj_id: mean_attention_across_all_frames}
        
        for obj_id in all_object_ids:
            attention_values = []
            
            for frame_idx in range(B):
                # 找到该object在当前帧的位置 - 确保设备匹配
                obj_mask_token = (groups_token[frame_idx] == obj_id)  # [Ht, Wt], 已经在device上
                
                if obj_mask_token.sum() > 0:
                    # 计算该object在当前帧的mean attention
                    # attns[frame_idx]和obj_mask_token都在同一个device上
                    mean_att = attns[frame_idx][obj_mask_token].mean().item()
                    attention_values.append(mean_att)
            
            # 计算该object在所有出现帧的平均attention
            if len(attention_values) > 0:
                object_global_attention[obj_id] = np.mean(attention_values)
            else:
                object_global_attention[obj_id] = 0.0
        
        # Step 4: 使用自适应阈值计算threshold（提前计算）
        # 将所有object的global attention收集起来
        global_att_values = np.array(list(object_global_attention.values()))
        
        if len(global_att_values) == 0:
            print("[WARNING] No objects found, using empty dynamic masks")
            self.dynamic_masks = [torch.zeros((H_img, W_img), dtype=torch.bool, device=device) 
                                for _ in range(B)]
            self.init_dynamic_masks = [m.clone() for m in self.dynamic_masks]
            return
        
        # 使用 adaptive_multiotsu_variance 计算阈值
        threshold = adaptive_multiotsu_variance(global_att_values)
        print(f"[Region-level Dynamic Detection] Attention threshold: {threshold:.4f}")
        
        # ===== 新增：优先使用 TextRegion bg_track 来做 foreground/background object 划分 =====
        bg_track_masks = self.load_textregion_bg_track_masks()
        use_bg_track = isinstance(bg_track_masks, torch.Tensor)

        bg_object_ratio = {}  # {oid: avg_bg_ratio_across_frames}

        if use_bg_track:
            bg_track_masks = bg_track_masks.to(device)          # [B,H,W] bool
            first_frame_groups = groups_hr[0]                   # [H,W]
            print("[Region-level Dynamic Detection] Using TR bg-track masks for filtering")

            # 统计每个 object 在全帧中落在 background 的比例（按帧平均）
            for oid in list(object_global_attention.keys()):
                if oid == 0:
                    continue
                ratios = []
                for t in range(B):
                    obj_mask = (groups_hr[t] == oid)
                    denom = obj_mask.sum().item()
                    if denom == 0:
                        continue
                    bg_pixels = (obj_mask & bg_track_masks[t]).sum().item()
                    ratios.append(bg_pixels / max(denom, 1))
                bg_object_ratio[oid] = float(np.mean(ratios)) if len(ratios) > 0 else 1.0

            # 用 bg_ratio 划分 foreground / background candidates
            # 这里的阈值你可以调：0.9 表示“90%以上都在背景里”
            BG_RATIO_TH = 0.7
            foreground_object_ids_bg = {oid for oid, r in bg_object_ratio.items() if r < BG_RATIO_TH}
            background_candidates_bg = set(object_global_attention.keys()) - foreground_object_ids_bg

            print(f"[TR BG-Track] FG candidates={len(foreground_object_ids_bg)} "
                f"BG candidates={len(background_candidates_bg)} (BG_RATIO_TH={BG_RATIO_TH})")
        # ================== Step5-7 (BG-Track > cached valid-FG > all) ==================
        # 目标：
        # 1) 优先用 bg_track 全帧背景 mask 把 objects 分成 foreground / background candidates
        # 2) 若无 bg_track，再用 cached valid-FG（来自 validate_and_adjust_dynamic_map_with_gt）
        # 3) 若两者都无，则不做过滤，直接在全体 objects 上阈值

        cached_fg_valid = getattr(self, "tr_fg_valid_objects", set())
        use_cached_fg = isinstance(cached_fg_valid, set) and len(cached_fg_valid) > 0

        if use_bg_track:
            # bg_track 优先：foreground = 非背景占比高的 objects
            foreground_object_ids = set(oid for oid in foreground_object_ids_bg if oid in object_global_attention)
            background_candidates = set(object_global_attention.keys()) - foreground_object_ids
            print(f"[Region-level Dynamic Detection] Using BG-Track FG: "
                f"{len(foreground_object_ids)} foreground / {len(background_candidates)} background candidates")

        elif use_cached_fg:
            # 其次：用 cached valid-FG
            foreground_object_ids = set(oid for oid in cached_fg_valid if oid in object_global_attention)
            background_candidates = set(object_global_attention.keys()) - foreground_object_ids
            print(f"[Region-level Dynamic Detection] Using cached valid-FG: "
                f"{len(foreground_object_ids)} foreground / {len(background_candidates)} background candidates")

        else:
            # 最后：无任何先验，不过滤
            foreground_object_ids = set(object_global_attention.keys())
            background_candidates = set()
            print("[Region-level Dynamic Detection] No bg-track or cached valid-FG; "
                "skipping filtering and using global threshold on all objects")

        # —— Step 5：用“第一次全局阈值 threshold”剔除“高 attention 的背景噪声”
        # 注意：只有在我们真的定义了 background_candidates 的情况下才做（即 use_bg_track 或 use_cached_fg）
        noisy_background_objects = set()
        if len(background_candidates) > 0:
            noisy_background_objects = {oid for oid in background_candidates
                                        if object_global_attention.get(oid, 0.0) > threshold}

            for oid in noisy_background_objects:
                object_global_attention.pop(oid, None)

            # 同步把 noisy 从 background_candidates 移走（可选，但更干净）
            background_candidates -= noisy_background_objects

            print(f"[Region-level Dynamic Detection] Removed {len(noisy_background_objects)} noisy background objects")

        # —— Step 6：在“剔除噪声后的所有 objects（前景 + 非噪声背景）”上重算阈值
        global_att_values = np.array(list(object_global_attention.values()))
        if len(global_att_values) == 0:
            print("[WARNING] No objects remaining after background noise removal; using empty dynamic masks")
            self.dynamic_masks = [torch.zeros((H_img, W_img), dtype=torch.bool, device=device) for _ in range(B)]
            self.init_dynamic_masks = [m.clone() for m in self.dynamic_masks]
            return

        threshold_after_filter = adaptive_multiotsu_variance(global_att_values)
        print(f"[Region-level Dynamic Detection] Recalculated attention threshold after filtering: "
            f"{threshold_after_filter:.4f}")

        # —— Step 7：仅在“foreground objects”中确定哪些是 dynamic（使用 threshold_after_filter）
        dynamic_object_ids = set()

        # 保底 1：前景只有一个 object，直接判为 dynamic
        if len(foreground_object_ids) == 1:
            only_oid = next(iter(foreground_object_ids))
            dynamic_object_ids.add(only_oid)
            print(f"[Fallback] Only one foreground object ({only_oid}); mark as dynamic without thresholding.")
        else:
            for oid in foreground_object_ids:
                if object_global_attention.get(oid, 0.0) > threshold_after_filter:
                    dynamic_object_ids.add(oid)

            print(f"[Region-level Dynamic Detection] {len(dynamic_object_ids)}/{len(foreground_object_ids)} "
                f"foreground objects marked as dynamic")

            # 保底 2：若动态面积太小，则在“前景 objects 内”重算一次阈值并重筛
            MIN_MASK_RATIO = 0.002  # 0.2%，可调
            if len(dynamic_object_ids) > 0:
                total_pixels = float(H_img * W_img)
                ratios = []
                for t in range(B):
                    ghr = groups_hr[t]
                    dyn_pixels_t = 0.0
                    for oid in dynamic_object_ids:
                        dyn_pixels_t += (ghr == oid).sum().item()
                    ratios.append(dyn_pixels_t / (total_pixels + 1e-6))
                avg_dyn_ratio = float(np.mean(ratios))
            else:
                avg_dyn_ratio = 0.0

            if avg_dyn_ratio < MIN_MASK_RATIO:
                print(f"[Fallback] Dynamic area too small (avg {avg_dyn_ratio*100:.3f}%). "
                    f"Recomputing threshold WITHIN foreground objects only and re-selecting dynamics.")

                fg_att_values = np.array([object_global_attention.get(oid, 0.0) for oid in foreground_object_ids],
                                        dtype=np.float32)
                if len(fg_att_values) > 0:
                    threshold_fg_only = adaptive_multiotsu_variance(fg_att_values)
                else:
                    threshold_fg_only = threshold_after_filter

                print(f"[Fallback] Foreground-only Otsu threshold: {threshold_fg_only:.4f}")

                dynamic_object_ids = {
                    oid for oid in foreground_object_ids
                    if object_global_attention.get(oid, 0.0) > threshold_fg_only
                }
                print(f"[Fallback] Foreground-only selection -> {len(dynamic_object_ids)}/{len(foreground_object_ids)} dynamic")

        # ================== Step 5–7（替换结束） ==================

        print(f"[Region-level Dynamic Detection] {len(dynamic_object_ids)}/{len(foreground_object_ids)} foreground objects marked as dynamic")
        
        # 打印一些统计信息
        if len(object_global_attention) > 0:
            sorted_objs = sorted(object_global_attention.items(), key=lambda x: x[1], reverse=True)
            print(f"[Region-level Dynamic Detection] Top 5 most dynamic objects:")
            for obj_id, att in sorted_objs[:min(5, len(sorted_objs))]:
                status = "DYNAMIC" if obj_id in dynamic_object_ids else "static"
                print(f"  Object {obj_id}: attention={att:.4f} [{status}]")
        
        # Step 8: 生成每一帧的dynamic mask（基于高分辨率region_groups）
        self.dynamic_masks = []
        self.init_dynamic_masks = []
        
        for frame_idx in range(B):
            # 在高分辨率上生成mask
            group_hr = groups_hr[frame_idx]  # [H_img, W_img], 已经在device上
            
            # 初始化mask为False
            dynamic_mask = torch.zeros_like(group_hr, dtype=torch.bool)  # 自动继承device
            
            # 将所有dynamic objects在该帧的区域标记为True
            for obj_id in dynamic_object_ids:
                # group_hr和obj_id比较，都在同一个device上
                obj_mask_hr = (group_hr == obj_id)
                dynamic_mask |= obj_mask_hr
            
            self.dynamic_masks.append(dynamic_mask)
            self.init_dynamic_masks.append(dynamic_mask.clone())
        
        # Step 9: 可视化和统计
        print(f"[Region-level Dynamic Detection] Dynamic mask statistics:")
        for frame_idx in range(min(5, B)):  # 只打印前5帧
            mask_ratio = self.dynamic_masks[frame_idx].float().mean().item()
            print(f"  Frame {frame_idx}: {mask_ratio*100:.2f}% pixels marked as dynamic")
        
        # 保存object-level的判断结果，供后续分析使用
        self.dynamic_object_ids = dynamic_object_ids
        self.object_global_attention = object_global_attention

    
    # @torch.no_grad()
    # def get_motion_mask_from_attns(self):
    #     """
    #     基于region-level的动态判断：
    #     1. 计算每个object在所有帧上的mean attention
    #     2. 用全局阈值判断object是否dynamic
    #     3. **新增**: 检查textregion first frame，过滤掉在背景区域的objects
    #     4. 将dynamic objects在各帧的region作为mask
    #     """
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    #     # 检查是否有region groups
    #     if not hasattr(self, "region_groups") or len(self.region_groups) != self.n_imgs:
    #         print("[WARNING] region_groups not available, falling back to pixel-wise method")
    #         self._get_motion_mask_pixelwise()
    #         return
        
    #     # 获取attention maps - 确保在正确的设备上
    #     attns = self.get_atts()  # [B, Ht, Wt]
    #     if not isinstance(attns, torch.Tensor):
    #         attns = torch.stack(attns)
    #     attns = attns.to(device)  # 确保在正确的设备上
        
    #     B, Ht, Wt = attns.shape
    #     H_img, W_img = self.imshape
        
    #     print(f"[Region-level Dynamic Detection] Processing {B} frames...")
        
    #     # Step 1: 下采样 region_groups 到 token 级别
    #     # 确保所有region_groups都在同一个设备上
    #     groups_hr_list = []
    #     for rg in self.region_groups:
    #         if isinstance(rg, torch.Tensor):
    #             groups_hr_list.append(rg.to(device))
    #         else:
    #             groups_hr_list.append(torch.tensor(rg, device=device))
        
    #     groups_hr = torch.stack(groups_hr_list, dim=0).long()  # [B, H_img, W_img]
    #     groups_token = self._downsample_groups_to_tokens(groups_hr, H_img, W_img, patch=16)  # [B, Ht, Wt]
        
    #     # Step 2: 收集所有object在所有帧的attention值
    #     # 收集全局所有出现过的object IDs
    #     all_object_ids = set()
    #     for frame_idx in range(B):
    #         unique_ids = torch.unique(groups_token[frame_idx])
    #         all_object_ids.update(unique_ids[unique_ids != 0].cpu().tolist())
        
    #     all_object_ids = sorted(list(all_object_ids))
    #     print(f"[Region-level Dynamic Detection] Found {len(all_object_ids)} unique objects")
        
    #     # Step 3: 对每个object，计算其在所有帧上的mean attention
    #     object_global_attention = {}  # {obj_id: mean_attention_across_all_frames}
        
    #     for obj_id in all_object_ids:
    #         attention_values = []
            
    #         for frame_idx in range(B):
    #             # 找到该object在当前帧的位置 - 确保设备匹配
    #             obj_mask_token = (groups_token[frame_idx] == obj_id)  # [Ht, Wt], 已经在device上
                
    #             if obj_mask_token.sum() > 0:
    #                 # 计算该object在当前帧的mean attention
    #                 # attns[frame_idx]和obj_mask_token都在同一个device上
    #                 mean_att = attns[frame_idx][obj_mask_token].mean().item()
    #                 attention_values.append(mean_att)
            
    #         # 计算该object在所有出现帧的平均attention
    #         if len(attention_values) > 0:
    #             object_global_attention[obj_id] = np.mean(attention_values)
    #         else:
    #             object_global_attention[obj_id] = 0.0
        
    #     # Step 4: 使用自适应阈值计算threshold（提前计算）
    #     # 将所有object的global attention收集起来
    #     global_att_values = np.array(list(object_global_attention.values()))
        
    #     if len(global_att_values) == 0:
    #         print("[WARNING] No objects found, using empty dynamic masks")
    #         self.dynamic_masks = [torch.zeros((H_img, W_img), dtype=torch.bool, device=device) 
    #                             for _ in range(B)]
    #         self.init_dynamic_masks = [m.clone() for m in self.dynamic_masks]
    #         return
    #     # Step 4: 计算全局 attention 阈值（只算一次）
    #     threshold = adaptive_multiotsu_variance(global_att_values)
    #     print(f"[Region-level Dynamic Detection] Global Otsu threshold for attention: {threshold:.4f}")

    #     # Step 5: 前景筛选逻辑
    #     # 载入 textregion mask，只为了区分前景/背景（不再过滤高att背景）
    #     textregion_mask = self.load_textregion_first_frame()
    #     if textregion_mask is None:
    #         print("[Region-level Dynamic Detection] No textregion mask found, treating all objects as potential foreground")
    #         foreground_object_ids = set(object_global_attention.keys())
    #     else:
    #         # ★ 关键修复：把 mask 放到和 groups_hr 同一设备，并确保是 bool
    #         textregion_mask = textregion_mask.to(groups_hr.device, non_blocking=True)
    #         textregion_mask = textregion_mask.bool()

    #         first_frame_groups = groups_hr[0]  # [H_img, W_img]（已在 device 上）
    #         foreground_object_ids = set()
    #         for obj_id in object_global_attention.keys():
    #             if obj_id == 0:
    #                 continue
    #             obj_mask = (first_frame_groups == obj_id)  # bool, device 同 groups_hr
    #             if obj_mask.any():
    #                 # 用 logical_and，避免 & 因 dtype/设备导致的问题
    #                 fg_pixels = torch.logical_and(obj_mask, textregion_mask).float().sum()
    #                 total_pixels = obj_mask.float().sum()
    #                 fg_ratio = (fg_pixels / (total_pixels + 1e-6)).item()
    #                 if fg_ratio > 0.1:
    #                     foreground_object_ids.add(obj_id)

    #     print(f"[Region-level Dynamic Detection] {len(foreground_object_ids)} foreground objects detected")

    #     # Step 6: 仅在前景objects中判断 dynamic
    #     dynamic_object_ids = set()
    #     for obj_id in foreground_object_ids:
    #         if object_global_attention[obj_id] > threshold:
    #             dynamic_object_ids.add(obj_id)

    #     print(f"[Region-level Dynamic Detection] {len(dynamic_object_ids)}/{len(foreground_object_ids)} foreground objects marked as dynamic")

        
    #     # 打印一些统计信息
    #     if len(object_global_attention) > 0:
    #         sorted_objs = sorted(object_global_attention.items(), key=lambda x: x[1], reverse=True)
    #         print(f"[Region-level Dynamic Detection] Top 5 most dynamic objects:")
    #         for obj_id, att in sorted_objs[:min(5, len(sorted_objs))]:
    #             status = "DYNAMIC" if obj_id in dynamic_object_ids else "static"
    #             print(f"  Object {obj_id}: attention={att:.4f} [{status}]")
        
    #     # Step 8: 生成每一帧的dynamic mask（基于高分辨率region_groups）
    #     self.dynamic_masks = []
    #     self.init_dynamic_masks = []
        
    #     for frame_idx in range(B):
    #         # 在高分辨率上生成mask
    #         group_hr = groups_hr[frame_idx]  # [H_img, W_img], 已经在device上
            
    #         # 初始化mask为False
    #         dynamic_mask = torch.zeros_like(group_hr, dtype=torch.bool)  # 自动继承device
            
    #         # 将所有dynamic objects在该帧的区域标记为True
    #         for obj_id in dynamic_object_ids:
    #             # group_hr和obj_id比较，都在同一个device上
    #             obj_mask_hr = (group_hr == obj_id)
    #             dynamic_mask |= obj_mask_hr
            
    #         self.dynamic_masks.append(dynamic_mask)
    #         self.init_dynamic_masks.append(dynamic_mask.clone())
        
    #     # Step 9: 可视化和统计
    #     print(f"[Region-level Dynamic Detection] Dynamic mask statistics:")
    #     for frame_idx in range(min(5, B)):  # 只打印前5帧
    #         mask_ratio = self.dynamic_masks[frame_idx].float().mean().item()
    #         print(f"  Frame {frame_idx}: {mask_ratio*100:.2f}% pixels marked as dynamic")
        
    #     # 保存object-level的判断结果，供后续分析使用
    #     self.dynamic_object_ids = dynamic_object_ids
    #     self.object_global_attention = object_global_attention

    def _get_motion_mask_pixelwise(self):
        """
        原始的pixel-wise方法（作为fallback）
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dynamic_masks = []
        self.init_dynamic_masks = []
        
        attns = self.get_atts()  # [B, Ht, Wt]
        if not isinstance(attns, torch.Tensor):
            attns = torch.stack(attns)
        attns = attns.to(device)
        
        # Upsample to image resolution
        upsampled_attns = torch.nn.functional.interpolate(
            attns.unsqueeze(1),  # [B, 1, Ht, Wt]
            size=self.imshape,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H_img, W_img]
        
        # Threshold
        threshold = adaptive_multiotsu_variance(upsampled_attns.cpu().numpy())
        upsampled_mask = (upsampled_attns > threshold)
        
        for i in range(upsampled_mask.shape[0]):
            self.dynamic_masks.append(upsampled_mask[i].to(device))
            self.init_dynamic_masks.append(self.dynamic_masks[i].clone())
        
        print(f"[Pixel-wise Dynamic Detection] Using fallback pixel-wise method with threshold {threshold:.4f}")

    def _get_motion_mask_pixelwise(self):
        """
        原始的pixel-wise方法（作为fallback）
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dynamic_masks = []
        self.init_dynamic_masks = []
        
        attns = self.get_atts()  # [B, Ht, Wt]
        
        # Upsample to image resolution
        upsampled_attns = torch.nn.functional.interpolate(
            attns.unsqueeze(1),  # [B, 1, Ht, Wt]
            size=self.imshape,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H_img, W_img]
        
        # Threshold
        threshold = adaptive_multiotsu_variance(upsampled_attns.cpu().numpy())
        upsampled_mask = (upsampled_attns > threshold)
        
        for i in range(upsampled_mask.shape[0]):
            self.dynamic_masks.append(upsampled_mask[i].to(device))
            self.init_dynamic_masks.append(self.dynamic_masks[i].clone())
        
        print(f"[Pixel-wise Dynamic Detection] Using fallback pixel-wise method with threshold {threshold:.4f}")
        
    def get_motion_mask_from_pairs(self, view1, view2, pred1, pred2):
        assert self.is_symmetrized, 'only support symmetric case'
        symmetry_pairs_idx = [(i, i+len(self.edges)//2) for i in range(len(self.edges)//2)]
        intrinsics_i = []
        intrinsics_j = []
        R_i = []
        R_j = []
        T_i = []
        T_j = []
        depth_maps_i = []
        depth_maps_j = []
        for i, j in tqdm(symmetry_pairs_idx):
            new_view1 = {}
            new_view2 = {}
            for key in view1.keys():
                if isinstance(view1[key], list):
                    new_view1[key] = [view1[key][i], view1[key][j]]
                    new_view2[key] = [view2[key][i], view2[key][j]]
                elif isinstance(view1[key], torch.Tensor):
                    new_view1[key] = torch.stack([view1[key][i], view1[key][j]])
                    new_view2[key] = torch.stack([view2[key][i], view2[key][j]])
            new_view1['idx'] = [0, 1]
            new_view2['idx'] = [1, 0]
            new_pred1 = {}
            new_pred2 = {}
            for key in pred1.keys():
                if isinstance(pred1[key], list):
                    new_pred1[key] = [pred1[key][i], pred1[key][j]]
                elif isinstance(pred1[key], torch.Tensor):
                    new_pred1[key] = torch.stack([pred1[key][i], pred1[key][j]])
            for key in pred2.keys():
                if isinstance(pred2[key], list):
                    new_pred2[key] = [pred2[key][i], pred2[key][j]]
                elif isinstance(pred2[key], torch.Tensor):
                    new_pred2[key] = torch.stack([pred2[key][i], pred2[key][j]])
            pair_viewer = PairViewer(new_view1, new_view2, new_pred1, new_pred2, verbose=False)
            intrinsics_i.append(pair_viewer.get_intrinsics()[0])
            intrinsics_j.append(pair_viewer.get_intrinsics()[1])
            R_i.append(pair_viewer.get_im_poses()[0][:3, :3])
            R_j.append(pair_viewer.get_im_poses()[1][:3, :3])
            T_i.append(pair_viewer.get_im_poses()[0][:3, 3:])
            T_j.append(pair_viewer.get_im_poses()[1][:3, 3:])
            depth_maps_i.append(pair_viewer.get_depthmaps()[0])
            depth_maps_j.append(pair_viewer.get_depthmaps()[1])
        
        self.intrinsics_i = torch.stack(intrinsics_i).to(self.flow_ij.device)
        self.intrinsics_j = torch.stack(intrinsics_j).to(self.flow_ij.device)
        self.R_i = torch.stack(R_i).to(self.flow_ij.device)
        self.R_j = torch.stack(R_j).to(self.flow_ij.device)
        self.T_i = torch.stack(T_i).to(self.flow_ij.device)
        self.T_j = torch.stack(T_j).to(self.flow_ij.device)
        self.depth_maps_i = torch.stack(depth_maps_i).unsqueeze(1).to(self.flow_ij.device)
        self.depth_maps_j = torch.stack(depth_maps_j).unsqueeze(1).to(self.flow_ij.device)

        def safe_inv_or_pinv(matrix):
            try:
                # try to calculate the inverse matrix
                return torch.linalg.inv(matrix)
            except RuntimeError:
                # if the inverse matrix calculation fails, use the pseudo-inverse
                return torch.linalg.pinv(matrix)

        ego_flow_1_2, _ = self.depth_wrapper(self.R_i, self.T_i, self.R_j, self.T_j, 1 / (self.depth_maps_i + 1e-6), self.intrinsics_j, safe_inv_or_pinv(self.intrinsics_i))
        ego_flow_2_1, _ = self.depth_wrapper(self.R_j, self.T_j, self.R_i, self.T_i, 1 / (self.depth_maps_j + 1e-6), self.intrinsics_i, safe_inv_or_pinv(self.intrinsics_j))

        err_map_i = torch.norm(ego_flow_1_2[:, :2, ...] - self.flow_ij[:len(symmetry_pairs_idx)], dim=1)
        err_map_j = torch.norm(ego_flow_2_1[:, :2, ...] - self.flow_ji[:len(symmetry_pairs_idx)], dim=1)
        # normalize the error map for each pair
        err_map_i = (err_map_i - err_map_i.amin(dim=(1, 2), keepdim=True)) / (err_map_i.amax(dim=(1, 2), keepdim=True) - err_map_i.amin(dim=(1, 2), keepdim=True))
        err_map_j = (err_map_j - err_map_j.amin(dim=(1, 2), keepdim=True)) / (err_map_j.amax(dim=(1, 2), keepdim=True) - err_map_j.amin(dim=(1, 2), keepdim=True))
        self.dynamic_masks = [[] for _ in range(self.n_imgs)]
        self.init_dynamic_masks = [[] for _ in range(self.n_imgs)]

        for i, j in symmetry_pairs_idx:
            i_idx = self._ei[i]
            j_idx = self._ej[i]
            self.dynamic_masks[i_idx].append(err_map_i[i])
            self.dynamic_masks[j_idx].append(err_map_j[i])
        
        for i in range(self.n_imgs):
            self.dynamic_masks[i] = torch.stack(self.dynamic_masks[i]).mean(dim=0) > self.motion_mask_thre
            self.init_dynamic_masks[i] = self.dynamic_masks[i].detach().clone()

    def refine_motion_mask_w_sam2(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Save previous TF32 settings
        if device == 'cuda':
            prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            prev_allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
            # Enable TF32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        try:
            autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                frame_tensors = torch.from_numpy(np.array((self.imgs))).permute(0, 3, 1, 2).to(device)
                # frame_tensors = torch.from_numpy(np.array(self.imgs)).permute(0, 3, 1, 2).contiguous()  # keep on CPU
                inference_state = predictor.init_state(video_path=frame_tensors)
                mask_list = [self.dynamic_masks[i] for i in range(self.n_imgs)]
                
                ann_obj_id = 1
                self.sam2_dynamic_masks = [[] for _ in range(self.n_imgs)]
        
                # Process even frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 1:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )

                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 0:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
        
                # Process odd frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 0:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )

                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 1:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
        
                # Update dynamic masks
                for i in range(self.n_imgs):
                    self.sam2_dynamic_masks[i] = torch.from_numpy(self.sam2_dynamic_masks[i][0]).to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i].to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i] | self.sam2_dynamic_masks[i]
                # Clean up
                del predictor
        finally:
            # Restore previous TF32 settings
            if device == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
                torch.backends.cudnn.allow_tf32 = prev_allow_cudnn_tf32

    def _check_all_imgs_are_selected(self, msk):
        self.msk = torch.from_numpy(np.array(msk, dtype=bool)).to(self.device)
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'
        pass

    def preset_pose(self, known_poses, pose_msk=None, requires_grad=False):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        if known_poses.shape[-1] == 7: # xyz wxyz
            known_poses = [tum_to_pose_matrix(pose) for pose in known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)
        if len(known_poses) == self.n_imgs:
            if requires_grad:
                self.im_poses.requires_grad_(True)
            else:
                self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_intrinsics(self, known_intrinsics, msk=None):
        if isinstance(known_intrinsics, torch.Tensor) and known_intrinsics.ndim == 2:
            known_intrinsics = [known_intrinsics]
        for K in known_intrinsics:
            assert K.shape == (3, 3)
        self.preset_focal([K.diagonal()[:2].mean() for K in known_intrinsics], msk)
        if self.optimize_pp:
            self.preset_principal_point([K[:2, 2] for K in known_intrinsics], msk)

    def preset_focal(self, known_focals, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))
        if len(known_focals) == self.n_imgs:
            if requires_grad:
                self.im_focals.requires_grad_(True)
            else:
                self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        if self.shared_focal:
            log_focals = torch.stack([self.im_focals[0]] * self.n_imgs, dim=0)
        else:
            log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points_non_batch(self):
        return torch.stack([pp.new((W/2, H/2))+10*pp for pp, (H, W) in zip(self.im_pp, self.imshapes)])

    def get_principal_points_batch(self):
        return self._pp + 10 * self.im_pp

    def get_principal_points(self):
        if self.batchify:
            return self.get_principal_points_batch()
        else:
            return self.get_principal_points_non_batch()

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses_batch(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def get_im_poses_non_batch(self):  # cam to world
        cam2world = self._get_poses(torch.stack(list(self.im_poses)))
        return cam2world

    def get_im_poses(self):
        if self.batchify:
            return self.get_im_poses_batch()
        else:
            return self.get_im_poses_non_batch()

    def _set_depthmap_batch(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def _set_depthmap_non_batch(self, idx, depth, force=False):
        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def _set_depthmap(self, idx, depth, force=False):
        if self.batchify:
            return self._set_depthmap_batch(idx, depth, force)
        else:
            return self._set_depthmap_non_batch(idx, depth, force)
    
    def preset_depthmap(self, known_depthmaps, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, depth in zip(self._get_msk_indices(msk), known_depthmaps):
            if self.verbose:
                print(f' (setting depthmap #{idx})')
            self._no_grad(self._set_depthmap(idx, depth))

        if len(known_depthmaps) == self.n_imgs:
            if requires_grad:
                self.im_depthmaps.requires_grad_(True)
            else:
                self.im_depthmaps.requires_grad_(False)
    
    def _set_init_depthmap(self):
        depth_maps = self.get_depthmaps(raw=True)
        self.init_depthmap = [dm.detach().clone() for dm in depth_maps]

    def get_init_depthmaps(self, raw=False):
        res = self.init_depthmap
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_depthmaps_batch(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_depthmaps_non_batch(self):
        return [d.exp() for d in self.im_depthmaps]

    def get_depthmaps(self, raw=False):
        if self.batchify:
            return self.get_depthmaps_batch(raw)
        else:
            return self.get_depthmaps_non_batch()

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def depth_to_pts3d_partial(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps()

        # convert focal to (1,2,H,W) constant field
        def focal_ex(i): return focals[i][..., None, None].expand(1, *focals[i].shape, *self.imshapes[i])
        # get pointmaps in camera frame
        rel_ptmaps = [depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[i:i+1])[0] for i in range(im_poses.shape[0])]
        # project to world frame
        return [geotrf(pose, ptmap) for pose, ptmap in zip(im_poses, rel_ptmaps)]
    
    def get_pts3d_batch(self, raw=False, **kwargs):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_pts3d(self, raw=False, **kwargs):
        if self.batchify:
            return self.get_pts3d_batch(raw, **kwargs)
        else:
            return self.depth_to_pts3d_partial()

    def forward_batchify(self, epoch=9999):
        pw_poses = self.get_pw_poses()  # cam-to-world

        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        # camera temporal loss
        if self.temporal_smoothing_weight > 0:
            temporal_smoothing_loss = self.relative_pose_loss(self.get_im_poses()[:-1], self.get_im_poses()[1:]).sum()
        else:
            temporal_smoothing_loss = 0

        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch: # enable flow loss after certain epoch
            R_all, T_all = self.get_im_poses()[:,:3].split([3, 1], dim=-1)
            R1, T1 = R_all[self._ei], T_all[self._ei]
            R2, T2 = R_all[self._ej], T_all[self._ej]
            K_all = self.get_intrinsics()
            inv_K_all = torch.linalg.inv(K_all)
            K_1, inv_K_1 = K_all[self._ei], inv_K_all[self._ei]
            K_2, inv_K_2 = K_all[self._ej], inv_K_all[self._ej]
            depth_all = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            depth1, depth2 = depth_all[self._ei], depth_all[self._ej]
            disp_1, disp_2 = 1 / (depth1 + 1e-6), 1 / (depth2 + 1e-6)
            ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K_2, inv_K_1)
            ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K_1, inv_K_2)
            dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)
            dynamic_mask1, dynamic_mask2 = dynamic_masks_all[self._ei], dynamic_masks_all[self._ej]

            flow_loss_i = self.flow_loss_fn(ego_flow_1_2[:, :2, ...], self.flow_ij, ~dynamic_mask1, per_pixel_thre=self.pxl_thre)
            flow_loss_j = self.flow_loss_fn(ego_flow_2_1[:, :2, ...], self.flow_ji, ~dynamic_mask2, per_pixel_thre=self.pxl_thre)
            flow_loss = flow_loss_i + flow_loss_j
            print(f'flow loss: {flow_loss.item()}')
            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0: 
                flow_loss = 0
                self.flow_loss_flag = True
        else:    
            flow_loss = 0

        if self.depth_regularize_weight > 0:
            init_depthmaps = torch.stack(self.get_init_depthmaps(raw=False)).unsqueeze(1)
            depthmaps = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)
            depth_prior_loss = self.depth_regularizer(depthmaps, init_depthmaps, dynamic_masks_all)
        else:
            depth_prior_loss = 0

        loss = (li + lj) * 1 + self.temporal_smoothing_weight * temporal_smoothing_loss + \
                self.flow_loss_weight * flow_loss + self.depth_regularize_weight * depth_prior_loss

        return loss
    
    def forward_non_batchify(self, epoch=9999):

        # --(1) Perform the original pairwise 3D consistency loss (pairwise 3D consistency)--
        pw_poses = self.get_pw_poses()  # pair-wise poses (or adaptive poses)
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()   # 3D point clouds for each image
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0.0
        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # Transform the pairwise predictions to the world coordinate system
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            # Compute the distance loss between the projected point clouds and the predictions
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss += (li + lj)

        # Average the loss
        loss /= self.n_edges

        # --(2) Add temporal smoothing constraint between adjacent frames (temporal smoothing)--
        temporal_smoothing_loss = 0.0
        if self.temporal_smoothing_weight > 0:
            # Get the global poses (4x4) for all images
            im_poses = self.get_im_poses()  # shape: (n_imgs, 4, 4)
            # Stack the relative poses between adjacent frames and use the existing relative_pose_loss function
            rel_RT1, rel_RT2 = [], []
            for idx in range(self.n_imgs - 1):
                rel_RT1.append(im_poses[idx])
                rel_RT2.append(im_poses[idx + 1])
            if len(rel_RT1) > 0:
                rel_RT1 = torch.stack(rel_RT1, dim=0)  # shape: (n_imgs-1, 4, 4)
                rel_RT2 = torch.stack(rel_RT2, dim=0)
                # Compute the pose difference between adjacent frames
                temporal_smoothing_loss = self.relative_pose_loss(rel_RT1, rel_RT2).sum()
                loss += self.temporal_smoothing_weight * temporal_smoothing_loss

        # --(3) Add flow constraint (flow_loss), similar to forward_batchify--
        flow_loss = 0.0
        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch:
            # Iterate through each pair of images and compute the depth map and flow comparison
            im_poses = self.get_im_poses()   # (n_imgs, 4, 4)
            K_all = self.get_intrinsics()    # (n_imgs, 3, 3)
            inv_K_all = torch.linalg.inv(K_all)
            depthmaps = self.get_depthmaps(raw=False)  # list of depth maps (H, W)

            for e, (i, j) in enumerate(self.edges):
                # Get the rotation, translation, and intrinsics for the two frames
                R1 = im_poses[i][:3, :3].unsqueeze(0)  # shape: (1, 3, 3)
                T1 = im_poses[i][:3, 3].unsqueeze(-1).unsqueeze(0)  # (1, 3, 1)
                R2 = im_poses[j][:3, :3].unsqueeze(0)
                T2 = im_poses[j][:3, 3].unsqueeze(-1).unsqueeze(0)
                K1 = K_all[i].unsqueeze(0)     # (1, 3, 3)
                K2 = K_all[j].unsqueeze(0)
                inv_K1 = inv_K_all[i].unsqueeze(0)
                inv_K2 = inv_K_all[j].unsqueeze(0)

                # Construct disparity: disp = 1/depth
                depth1 = depthmaps[i].unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
                depth2 = depthmaps[j].unsqueeze(0).unsqueeze(1)
                disp_1 = 1.0 / (depth1 + 1e-6)
                disp_2 = 1.0 / (depth2 + 1e-6)

                # Compute "ego-motion flow" by projecting using DepthBasedWarping
                # Note that DepthBasedWarping expects batch dimension, so add unsqueeze(0)
                ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K2, inv_K1)
                ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K1, inv_K2)

                # Get the corresponding dynamic region masks (if any)
                dynamic_mask_i = self.dynamic_masks[i]  # shape: (H, W)
                dynamic_mask_j = self.dynamic_masks[j]

                # When computing flow loss, exclude or ignore dynamic regions
                flow_loss_i = self.flow_loss_fn(
                    ego_flow_1_2[0, :2, ...],   # shape: (2, H, W)
                    self.flow_ij[e],           # shape: (2, H, W),  i->j
                    ~dynamic_mask_i,           # mask: True = keep, False = ignore
                    per_pixel_thre=self.pxl_thre
                )
                flow_loss_j = self.flow_loss_fn(
                    ego_flow_2_1[0, :2, ...],
                    self.flow_ji[e],           # j->i
                    ~dynamic_mask_j,
                    per_pixel_thre=self.pxl_thre
                )
                flow_loss += (flow_loss_i + flow_loss_j)

            # Optional: handle cases where the flow loss is too large (e.g., early stop)
            # divide by the number of edges
            flow_loss /= self.n_edges
            print(f'flow loss: {flow_loss.item()}')
            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0:
                flow_loss = 0.0

            loss += self.flow_loss_weight * flow_loss

        # --(4) Add depth regularization (depth_prior_loss) to constrain the initial depth--
        if self.depth_regularize_weight > 0:
            init_depthmaps = self.get_init_depthmaps(raw=False)  # initial depth maps
            current_depthmaps = self.get_depthmaps(raw=False)     # current optimized depth maps
            depth_prior_loss = 0.0
            for i in range(self.n_imgs):
                # Apply constraints on static regions (ignore dynamic regions)
                # Make sure the shape has the batch dimension (B,1,H,W)
                depth_prior_loss += self.depth_regularizer(
                    current_depthmaps[i].unsqueeze(0).unsqueeze(1),
                    init_depthmaps[i].unsqueeze(0).unsqueeze(1),
                    self.dynamic_masks[i].unsqueeze(0).unsqueeze(1)
                )
            loss += self.depth_regularize_weight * depth_prior_loss

        return loss

    def forward(self, epoch=9999):
        if self.batchify:
            return self.forward_batchify(epoch)
        else:
            return self.forward_non_batchify(epoch)

    def relative_pose_loss(self, RT1, RT2):
        relative_RT = torch.matmul(torch.inverse(RT1), RT2)
        rotation_diff = relative_RT[:, :3, :3]
        translation_diff = relative_RT[:, :3, 3]

        # Frobenius norm for rotation difference
        rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(1, 2))

        # L2 norm for translation difference
        translation_loss = torch.norm(translation_diff, dim=1)

        # Combined loss (one can weigh these differently if needed)
        pose_loss = rotation_loss + translation_loss * self.translation_weight
        return pose_loss

def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img

def ordered_ratio(disp_a, disp_b, mask=None):
    ratio_a = torch.maximum(disp_a, disp_b) / \
        (torch.minimum(disp_a, disp_b)+1e-5)
    if mask is not None:
        ratio_a = ratio_a[mask]
    return ratio_a - 1
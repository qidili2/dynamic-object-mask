import json
import os
import os.path as osp
import glob
import shutil
from typing import Literal

import imageio.v3 as iio
import numpy as np
import roma
from loguru import logger as guru
from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import path_to_root  # noqa
from data.colmap import get_colmap_camera_params

# code from shape of motion
class iPhoneDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        start: int = 0,
        end: int = -1,
        factor: int = 1,
        split: Literal["train", "val"] = "train",
        depth_type: Literal[
            "midas",
            "depth_anything",
            "lidar",
            "depth_anything_colmap",
        ] = "depth_anything_colmap",
        camera_type: Literal["original", "refined"] = "refined",
        **_,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.training = split == "train"
        self.split = split
        self.factor = factor
        self.start = start
        self.end = end
        self.depth_type = depth_type
        self.camera_type = camera_type
        self.cache_dir = osp.join(data_dir, "flow3d_preprocessed", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Test if the current data has validation set.
        with open(osp.join(data_dir, "splits", "val.json")) as f:
            split_dict = json.load(f)
        self.has_validation = len(split_dict["frame_names"]) > 0

        # Load metadata.
        with open(osp.join(data_dir, "splits", f"{split}.json")) as f:
            split_dict = json.load(f)
        full_len = len(split_dict["frame_names"])
        end = min(end, full_len) if end > 0 else full_len
        self.end = end
        self.frame_names = split_dict["frame_names"][start:end]
        time_ids = [t for t in split_dict["time_ids"] if t >= start and t < end]
        self.time_ids = torch.tensor(time_ids) - start
        guru.info(f"{self.time_ids.min()=} {self.time_ids.max()=}")
        guru.info(f"{self.num_frames=}")
        with open(osp.join(data_dir, "extra.json")) as f:
            extra_dict = json.load(f)
        self.fps = float(extra_dict["fps"])

        # Load cameras.
        if self.camera_type == "original":
            Ks, w2cs = [], []
            for frame_name in self.frame_names:
                with open(osp.join(data_dir, "camera", f"{frame_name}.json")) as f:
                    camera_dict = json.load(f)
                focal_length = camera_dict["focal_length"]
                principal_point = camera_dict["principal_point"]
                Ks.append(
                    [
                        [focal_length, 0.0, principal_point[0]],
                        [0.0, focal_length, principal_point[1]],
                        [0.0, 0.0, 1.0],
                    ]
                )
                orientation = np.array(camera_dict["orientation"])
                position = np.array(camera_dict["position"])
                w2cs.append(
                    np.block(
                        [
                            [orientation, -orientation @ position[:, None]],
                            [np.zeros((1, 3)), np.ones((1, 1))],
                        ]
                    ).astype(np.float32)
                )
            self.Ks = torch.tensor(Ks)
            self.Ks[:, :2] /= factor
            self.w2cs = torch.from_numpy(np.array(w2cs))
        elif self.camera_type == "refined":
            Ks, w2cs = get_colmap_camera_params(
                osp.join(data_dir, "flow3d_preprocessed/colmap/sparse/"),
                [frame_name + ".png" for frame_name in self.frame_names],
            )
            self.Ks = torch.from_numpy(Ks[:, :3, :3].astype(np.float32))
            self.Ks[:, :2] /= factor
            self.w2cs = torch.from_numpy(w2cs.astype(np.float32))

        # Load depths.
        def load_depth(frame_name):
            if self.depth_type == "lidar":
                depth = np.load(
                    osp.join(
                        self.data_dir,
                        f"depth/{factor}x/{frame_name}.npy",
                    )
                )[..., 0]
            else:
                depth = np.load(
                    osp.join(
                        self.data_dir,
                        f"flow3d_preprocessed/aligned_{self.depth_type}/",
                        f"{factor}x/{frame_name}.npy",
                    )
                )
                depth[depth < 1e-3] = 1e-3
                depth = 1.0 / depth
            return depth

        self.depths = torch.from_numpy(
            np.array(
                [
                    load_depth(frame_name)
                    for frame_name in tqdm(
                        self.frame_names,
                        desc=f"Loading {self.split} depths",
                        leave=False,
                    )
                ],
                np.float32,
            )
        )
        max_depth_values_per_frame = self.depths.reshape(
            self.num_frames, -1
        ).max(1)[0]
        max_depth_value = max_depth_values_per_frame.median() * 2.5
        print("max_depth_value", max_depth_value)
        self.depths = torch.clamp(self.depths, 0, max_depth_value)

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index: int):
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": self.time_ids[index],
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # # (H, W).
            "depths": self.depths[index],
        }
        return data

if __name__ == "__main__":
    dirs = glob.glob("iphone/*/")
    dirs = sorted(dirs)
    
    for dir in dirs:
        data = iPhoneDataset(data_dir=dir)
        new_dir = osp.join(dir, 'prepare_all_stride10')
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir, exist_ok=True)
        
        max_frames = 10000
        total_frames = len(data)
        indices = list(range(0, total_frames, 10))
        if len(indices) > max_frames:
            indices = indices[:max_frames] 
        
        for subdir in ['rgb', 'masks', 'camera', 'depth', 'depth_npy']:
            os.makedirs(osp.join(new_dir, subdir), exist_ok=True)
            
        for idx in tqdm(indices, desc="Copying sampled frames"):
            frame_data = data[idx]
            frame_name = frame_data['frame_names']
            
            src_img = osp.join(dir, f"rgb/{data.factor}x/{frame_name}.png")
            dst_img = osp.join(new_dir, f"rgb/{frame_name}.png")
            shutil.copy(src_img, dst_img)
            
            src_mask = osp.join(dir, f"flow3d_preprocessed/track_anything/{data.factor}x/{frame_name}.png")
            dst_mask = osp.join(new_dir, f"masks/{frame_name}.png")
            shutil.copy(src_mask, dst_mask)
            
            camera_data = {
                'w2c': frame_data['w2cs'].numpy().tolist(),
                'K': frame_data['Ks'].numpy().tolist(),
            }
            with open(osp.join(new_dir, f"camera/{frame_name}.json"), 'w') as f:
                json.dump(camera_data, f, indent=2)
            
            depth = data.depths[idx].numpy()
            
            dst_depth = osp.join(new_dir, f"depth_npy/{frame_name}.npy")
            np.save(dst_depth, depth)
            
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis_path = osp.join(new_dir, f"depth/{frame_name}.png")
            iio.imwrite(depth_vis_path, depth_vis)
        
        def create_video(input_dir, output_path, fps=10):
            try:
                import os
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                cmd = f'/usr/bin/ffmpeg -y -framerate {fps} -pattern_type glob -i "{input_dir}/*.png" ' \
                      f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' \
                      f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p ' \
                      f'-movflags +faststart -b:v 5000k "{output_path}"'
                
                status = os.system(cmd)
                
                if status == 0:
                    guru.info(f"video saved to: {output_path}")
                    return True
                else:
                    guru.error(f"error when generating video: {status}")
                    return False
            except Exception as e:
                guru.error(f"error when generating video: {str(e)}")
                return False
        
        create_video(
            osp.join(new_dir, "rgb"),
            osp.join(new_dir, "rgb.mp4")
        )
        
        create_video(
            osp.join(new_dir, "masks"),
            osp.join(new_dir, "masks.mp4")
        )
        
        create_video(
            osp.join(new_dir, "depth"),
            osp.join(new_dir, "depth.mp4")
        )
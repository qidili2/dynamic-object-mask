# Load_and_Visualize_TAPVid_3D_samples.py

import os
import sys
import numpy as np
import cv2
import IPython
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import mediapy as media
import flow_vis
import scenepic as sp
from scipy.spatial.transform import Rotation as R
import imageio
import json
import os.path as osp

# Available examples for visualization
MINIVAL_FILES = {
    "adt": [
        "Apartment_release_decoration_seq138_5.npz",
        "Apartment_release_multiskeleton_party_seq117_4.npz",
        "Apartment_release_multiskeleton_party_seq121_1.npz",
        "Apartment_release_multiskeleton_party_seq121_2.npz",
        "Apartment_release_multiuser_cook_seq118_8.npz",
        "Apartment_release_multiuser_cook_seq144_4.npz",
        "Apartment_release_work_seq140_1.npz",
        "Lite_release_recognition_BirdHouseToy_seq030_6.npz",
    ],
}

# Data format explanations:
# images_jpeg_bytes: JPEG encoded video frames
# queries_xyt: Query points in (x, y, t) format
# tracks_xyz: 3D point tracks in world coordinates
# visibility: Binary visibility mask for each point
# intrinsics: Camera intrinsics (fx, fy, cx, cy)
# extrinsics_w2c: World-to-camera transformation matrices (optional)

def install_packages():
    """Install required packages and download necessary files"""
    # create download folder (if not exists)
    download_dir = 'tapvid3d_download'
    os.makedirs(download_dir, exist_ok=True)
    
    # check if tapvid3d_splits.py has been downloaded
    splits_file_path = os.path.join(download_dir, 'tapvid3d_splits.py')
    if not os.path.exists(splits_file_path):
        print('download tapvid3d_splits.py...', end='')
        os.system(f'wget https://raw.githubusercontent.com/google-deepmind/tapnet/main/tapnet/tapvid3d/splits/tapvid3d_splits.py -O {splits_file_path}')
        print('done')
    else:
        print('tapvid3d_splits.py already exists, skip download')

# only download necessary files, not repeat install packages
install_packages()

# add download folder to Python path
download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tapvid3d_download')
sys.path.append(download_dir)

def load_dataset_example(chosen_filename):
    """Load and parse contents of the dataset example file"""
    download_dir = 'tapvid3d_download'
    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, chosen_filename)
    
    try:
        if not os.path.exists(file_path):
            print(f'download dataset: {chosen_filename}')
            file_url = f'https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/minival_rc5/{chosen_filename}'
            result = os.system(f'wget {file_url} -O {file_path}')
            if result != 0:
                raise RuntimeError("download file failed")
            print("download done!")
        else:
            print(f'use downloaded file: {chosen_filename}')

        with open(file_path, 'rb') as in_f:
            in_npz = np.load(in_f)
            images_jpeg_bytes = in_npz['images_jpeg_bytes']
            queries_xyt = in_npz['queries_xyt']
            tracks_xyz = in_npz['tracks_XYZ']
            visibility = in_npz['visibility']
            intrinsics = in_npz['fx_fy_cx_cy']
            extrinsics_w2c = in_npz['extrinsics_w2c'] if 'extrinsics_w2c' in in_npz.files else None

        video = []
        for frame_bytes in images_jpeg_bytes:
            arr = np.frombuffer(frame_bytes, np.uint8)
            image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            video.append(image_rgb)
        video = np.stack(video, axis=0)

        print(f"In example {chosen_filename}:")
        print(f"  images_jpeg_bytes: {len(images_jpeg_bytes)} frames, each stored as JPEG bytes (and after decoding, the video shape: {video.shape})")
        print(f"  intrinsics: (fx, fy, cx, cy)={intrinsics}", intrinsics.dtype)
        print(f"  tracks_xyz: {tracks_xyz.shape}", tracks_xyz.dtype)
        print(f"  visibility: {visibility.shape}", visibility.dtype)
        print(f"  queries_xyt: {queries_xyt.shape}", queries_xyt.dtype)
        if extrinsics_w2c is not None:
            print(f"  extrinsics_w2c: {extrinsics_w2c.shape}", extrinsics_w2c.dtype)

        return video, tracks_xyz, visibility, intrinsics, extrinsics_w2c, queries_xyt

    except Exception as e:
        print(f"error when loading dataset example: {str(e)}")
        raise

def project_points_to_video_frame(camera_pov_points3d, camera_intrinsics, height, width):
    """Project 3d points to 2d image plane."""
    u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
    v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)

    f_u, f_v, c_u, c_v = camera_intrinsics

    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    # Mask of points that are in front of the camera and within image boundary
    masks = (camera_pov_points3d[..., 2] >= 1)
    masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
    return np.stack([u_d, v_d], axis=-1), masks

def plot_2d_tracks(video, points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False):
    """Visualize 2D point trajectories."""
    num_frames, num_points = points.shape[:2]

    # Precompute colormap for points
    color_map = matplotlib.colormaps.get_cmap('hsv')
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
    point_colors = np.zeros((num_points, 3))
    for i in range(num_points):
        point_colors[i] = np.array(color_map(cmap_norm(i)))[:3] * 255

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    frames = []
    for t in range(num_frames):
        frame = video[t].copy()

        # Draw tracks on the frame
        line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
        line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
        line_infront_cameras = infront_cameras[max(0, t - tracks_leave_trace) : t + 1]
        for s in range(line_tracks.shape[0] - 1):
            img = frame.copy()

            for i in range(num_points):
                if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
                elif show_occ and line_infront_cameras[s, i] and line_infront_cameras[s + 1, i]:  # occluded
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

            alpha = (s + 1) / (line_tracks.shape[0] - 1)
            frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

        # Draw end points on the frame
        for i in range(num_points):
            if visibles[t, i]:  # visible
                x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                cv2.circle(frame, (x, y), 2, point_colors[i], -1)
            elif show_occ and infront_cameras[t, i]:  # occluded
                x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                cv2.circle(frame, (x, y), 2, point_colors[i], 1)

        frames.append(frame)
    frames = np.stack(frames)
    return frames

def plot_3d_tracks(points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False):
    """Visualize 3D point trajectories."""
    num_frames, num_points = points.shape[0:2]

    color_map = matplotlib.colormaps.get_cmap('hsv')
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    if show_occ:
        x_min, x_max = np.min(points[infront_cameras, 0]), np.max(points[infront_cameras, 0])
        y_min, y_max = np.min(points[infront_cameras, 2]), np.max(points[infront_cameras, 2])
        z_min, z_max = np.min(points[infront_cameras, 1]), np.max(points[infront_cameras, 1])
    else:
        x_min, x_max = np.min(points[visibles, 0]), np.max(points[visibles, 0])
        y_min, y_max = np.min(points[visibles, 2]), np.max(points[visibles, 2])
        z_min, z_max = np.min(points[visibles, 1]), np.max(points[visibles, 1])

    interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    x_min = (x_min + x_max) / 2 - interval / 2
    x_max = x_min + interval
    y_min = (y_min + y_max) / 2 - interval / 2
    y_max = y_min + interval
    z_min = (z_min + z_max) / 2 - interval / 2
    z_max = z_min + interval

    frames = []
    for t in range(num_frames):
        fig = Figure(figsize=(6.4, 4.8))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.invert_zaxis()
        ax.view_init()

        for i in range(num_points):
            if visibles[t, i] or (show_occ and infront_cameras[t, i]):
                color = color_map(cmap_norm(i))
                line = points[max(0, t - tracks_leave_trace) : t + 1, i]
                ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=color, linewidth=1)
                end_point = points[t, i]
                ax.scatter(xs=end_point[0], ys=end_point[2], zs=end_point[1], color=color, s=3)

        fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
        fig.canvas.draw()
        frames.append(np.array(canvas.buffer_rgba())[..., :3])
    return np.stack(frames)


def create_axis(scene, n_lines=10, min_x=-2, max_x=2, min_y=-1.5, max_y=1.5, min_z=1, max_z=5):
    """Create axis for 3D visualization"""
    x_plane = scene.create_mesh("xplane")
    vert_start_XYZ = np.stack((min_x*np.ones(n_lines), min_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
    vert_end_XYZ = np.stack((min_x*np.ones(n_lines), max_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
    horiz_start_XYZ = np.stack((min_x*np.ones(n_lines), np.linspace(min_y,max_y,n_lines), min_z*np.ones(n_lines)), axis=-1)
    horiz_end_XYZ = np.stack((min_x*np.ones(n_lines), np.linspace(min_y,max_y,n_lines), max_z*np.ones(n_lines)), axis=-1)
    x_plane.add_lines(np.concatenate((vert_start_XYZ,horiz_start_XYZ), axis=0), np.concatenate((vert_end_XYZ,horiz_end_XYZ), axis=0), color=0.2*np.ones((3,1)))

    y_plane = scene.create_mesh("yplane")
    vert_start_XYZ = np.stack((min_x*np.ones(n_lines), max_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
    vert_end_XYZ = np.stack((max_x*np.ones(n_lines), max_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
    horiz_start_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), max_y * np.ones(n_lines), min_z*np.ones(n_lines)), axis=-1)
    horiz_end_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), max_y * np.ones(n_lines), max_z*np.ones(n_lines)), axis=-1)
    y_plane.add_lines(np.concatenate((vert_start_XYZ,horiz_start_XYZ), axis=0), np.concatenate((vert_end_XYZ,horiz_end_XYZ), axis=0), color=0.2*np.ones((3,1)))

    z_plane = scene.create_mesh("zplane")
    vert_start_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), min_y * np.ones(n_lines), max_z*np.ones(n_lines)), axis=-1)
    vert_end_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), max_y * np.ones(n_lines), max_z*np.ones(n_lines)), axis=-1)
    horiz_start_XYZ = np.stack((min_x * np.ones(n_lines), np.linspace(min_y,max_y,n_lines), max_z*np.ones(n_lines)), axis=-1)
    horiz_end_XYZ = np.stack((max_x * np.ones(n_lines), np.linspace(min_y,max_y,n_lines), max_z*np.ones(n_lines)), axis=-1)
    z_plane.add_lines(np.concatenate((vert_start_XYZ,horiz_start_XYZ), axis=0), np.concatenate((vert_end_XYZ,horiz_end_XYZ), axis=0), color=0.2*np.ones((3,1)))

    return x_plane, y_plane, z_plane

def get_interactive_3d_visualization(XYZ, h, w, fx, fy, cx, cy, framerate=15):
    """Generate interactive 3D visualization
    Args:
        XYZ: 3D point coordinates of shape [num_frames, num_points, 3]
        h: Image height
        w: Image width
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate
        cy: Principal point y coordinate
        framerate: Animation frame rate
    """
    n_frames, n_points = XYZ.shape[:2]
    fov_y = (np.arctan2(h/2, fy) * 180 / np.pi) * 2

    # compute colors
    cm = plt.get_cmap('hsv')
    Y = XYZ[0,:,1]
    XYZ = XYZ[:,np.argsort(Y),:]
    colors = cm(np.linspace(0,1,n_points))[:,:3]

    # create scene
    scene = sp.Scene()
    scene.framerate = framerate
    camera = sp.Camera(center=np.zeros(3), aspect_ratio=w/h, fov_y_degrees=fov_y, look_at=np.array([0.,0.,1.]), up_dir=np.array([0.,-1.,0.]))
    canvas = scene.create_canvas_3d(width=w, height=h, shading=sp.Shading(bg_color=sp.Colors.White), camera=camera)

    # create axis and frustrum
    x_plane, y_plane, z_plane = create_axis(scene)
    frustrum = scene.create_mesh("frustrum")
    frustrum.add_camera_frustum(camera, sp.Colors.Red, depth=0.5, thickness=0.002)

    # create track spheres
    spheres = scene.create_mesh("spheres")
    spheres.add_sphere(sp.Colors.White, transform=sp.Transforms.Scale(0.02))
    spheres.enable_instancing(XYZ[0], colors=colors)

    # create track trails
    lines_t = []
    for t in range(1, n_frames):
        start_XYZ = XYZ[t-1]
        end_XYZ = XYZ[t]
        start_colors = colors
        end_colors = colors
        mesh = scene.create_mesh(f"lines_{t}")
        mesh.add_lines(np.concatenate((start_XYZ, start_colors), axis=-1), np.concatenate((end_XYZ, end_colors), axis=-1))
        lines_t.append(mesh)

    # create scene frames
    for i in range(n_frames-1):
        frame = canvas.create_frame()
        frame.add_mesh(frustrum)
        for j in range(max(0, i-10), i):
            frame.add_mesh(lines_t[j])
        spheres_updated = scene.update_instanced_mesh("spheres", XYZ[i], colors=colors)
        frame.add_mesh(spheres_updated)
        frame.add_mesh(x_plane)
        frame.add_mesh(y_plane)
        frame.add_mesh(z_plane)

    scene.quantize_updates()

    # generate html
    SP_LIB = sp.js_lib_src()
    SP_SCRIPT = scene.get_script().replace(
        'window.onload = function()', 'function scenepic_main_function()'
    )
    HTML_string = (
        '<!DOCTYPE html>'
        '<html lang="en">'
        '<head>'
            '<meta charset="utf-8">'
            '<title>ScenePic </title>'
            f'<script>{SP_LIB}</script>'
            f'<script>{SP_SCRIPT} scenepic_main_function();</script>'
        '</head>'
        f'<body onload="scenepic_main_function()"></body>'
        '</html>'
    )
    html_object = IPython.display.HTML(HTML_string)
    IPython.display.display(html_object)
    print('Press PLAY ▶ to start animation')
    print(' - Drag with mouse to rotate')
    print(' - Use mouse-wheel for zoom')
    print(' - Shift to pan')
    print(' - Use camera button 📷 to restore camera view')

def plot_camera_trajectory(camera_rotations, camera_positions, plot3d_elev=30, plot3d_azim=10, resolution=(128, 128)):
    """Plot camera trajectory with optimized performance and lower resolution"""
    # set non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    num_frames = camera_positions.shape[0]

    # pre-calculate all data
    rotations = R.from_matrix(camera_rotations)
    camera_directions = rotations.apply(np.array([0, 0, -1]))

    # add 10% margin to coordinate range
    margin = 0.1
    x_min, x_max = np.min(camera_positions[..., 0]), np.max(camera_positions[..., 0])
    y_min, y_max = np.min(camera_positions[..., 1]), np.max(camera_positions[..., 1])
    z_min, z_max = np.min(camera_positions[..., 2]), np.max(camera_positions[..., 2])
    
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    z_margin = (z_max - z_min) * margin
    
    x_range = [x_min - x_margin, x_max + x_margin]
    y_range = [y_min - y_margin, y_max + y_margin]
    z_range = [z_min - z_margin, z_max + z_margin]

    # pre-calculate arrow length
    trajectory_length = np.sum(np.linalg.norm(np.diff(camera_positions, axis=0), axis=1))
    quiver_len = trajectory_length * 0.001

    # create and set figure (only create once)
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # pre-allocate frames array
    frames = np.zeros((num_frames, resolution[1], resolution[0], 3), dtype=np.uint8)

    try:
        for t in range(num_frames):
            # clear current frame
            ax.clear()
            
            # set view
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.view_init(elev=plot3d_elev, azim=plot3d_azim)

            # plot trajectory (only plot to current frame)
            if t > 0:
                ax.plot(camera_positions[:t+1, 0], 
                       camera_positions[:t+1, 1], 
                       camera_positions[:t+1, 2], 
                       'b-', linewidth=0.5)

            # plot camera position and direction
            ax.quiver(camera_positions[t, 0], 
                     camera_positions[t, 1], 
                     camera_positions[t, 2],
                     camera_directions[t, 0], 
                     camera_directions[t, 1], 
                     camera_directions[t, 2],
                     color='r', length=quiver_len, 
                     normalize=True,
                     arrow_length_ratio=0.05)

            # minimize labels and grid
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.grid(False)

            # render and get image data
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frames[t] = data.reshape(resolution[1], resolution[0], 4)[:, :, :3]

    finally:
        plt.close(fig)

    return frames

def main(choose_random=False, specific_example=None, num_frames=100, num_tracks=300, output_dir='results', enable_3d_viz=False, save_all=False):
    """Main function to run the visualization"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # create dataset sub-folders
        for dataset in MINIVAL_FILES.keys():
            os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
        
        # create data save directory
        example_name = specific_example.split('.')[0]  # remove .npz suffix
        
        # determine dataset type
        dataset_type = None
        for d_type in ["adt", "drivetrack", "pstudio"]:
            if specific_example in MINIVAL_FILES[d_type]:
                dataset_type = d_type
                break
        
        if dataset_type is None:
            raise ValueError(f"sample {specific_example} is not in any known dataset")
        
        # create sample directory in corresponding dataset sub-folder
        data_dir = os.path.join(output_dir, dataset_type, example_name)
        rgb_dir = os.path.join(data_dir, 'rgb')
        camera_dir = os.path.join(data_dir, 'camera')
        gif_dir = os.path.join(data_dir, 'gif')
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(camera_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

        # check if combined_visualization.gif exists
        combined_viz_path = os.path.join(gif_dir, 'combined_visualization.gif')
        if os.path.exists(combined_viz_path):
            print(f"skip {example_name}: combined_visualization.gif already exists")
            return

        # Validate inputs
        if not choose_random and specific_example is not None:
            if specific_example not in MINIVAL_FILES[dataset_type]:
                print(f"Warning: {specific_example} is not in the list of known examples")
        
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if num_tracks <= 0:
            raise ValueError("num_tracks must be positive")

        # Select example
        chosen_filename = random.choice(MINIVAL_FILES[dataset_type]) if choose_random else specific_example
        
        # Load dataset example
        video, tracks_xyz, visibility, intrinsics, extrinsics_w2c, queries_xyt = load_dataset_example(chosen_filename)

        # save original images and camera parameters
        print("save original data...")
        for frame_idx in range(len(video)):
            frame_name = f"{frame_idx:06d}"
            # save PNG image
            image_path = osp.join(rgb_dir, f"{frame_name}.png")
            cv2.imwrite(image_path, cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR))

            # save camera parameters
            if extrinsics_w2c is not None:
                camera_data = {
                    'w2c': extrinsics_w2c[frame_idx].tolist(),
                    'K': [
                        [intrinsics[0], 0, intrinsics[2]],
                        [0, intrinsics[1], intrinsics[3]],
                        [0, 0, 1]
                    ]
                }
                camera_path = osp.join(camera_dir, f"{frame_name}.json")
                with open(camera_path, 'w') as f:
                    json.dump(camera_data, f, indent=2)

        # downsample visualization data
        max_frames = 30
        if video.shape[0] > max_frames:
            # calculate stride
            stride = video.shape[0] // max_frames
            # ensure last frame is included
            viz_indices = list(range(0, video.shape[0] - 1, stride)) + [video.shape[0] - 1]
            
            # only downsample visualization data
            viz_video = video[viz_indices]
            viz_tracks_xyz = tracks_xyz[viz_indices]
            viz_visibility = visibility[viz_indices]
            viz_extrinsics_w2c = extrinsics_w2c[viz_indices] if extrinsics_w2c is not None else None
        else:
            viz_video = video
            viz_tracks_xyz = tracks_xyz
            viz_visibility = visibility
            viz_extrinsics_w2c = extrinsics_w2c

        # limit number of tracks
        if viz_tracks_xyz.shape[1] > num_tracks:
            track_indices = np.random.choice(viz_tracks_xyz.shape[1], num_tracks, replace=False)
            viz_tracks_xyz = viz_tracks_xyz[:, track_indices]
            viz_visibility = viz_visibility[:, track_indices]

        # Sort points by height
        sorted_indices = np.argsort(viz_tracks_xyz[0, ..., 1])
        viz_tracks_xyz = viz_tracks_xyz[:, sorted_indices]
        viz_visibility = viz_visibility[:, sorted_indices]

        # Project to 2D
        tracks_xy, infront_cameras = project_points_to_video_frame(
            viz_tracks_xyz, intrinsics, viz_video.shape[1], viz_video.shape[2])

        # Generate visualizations
        print("generate 2D visualization...")
        video2d_viz = plot_2d_tracks(viz_video, tracks_xy, viz_visibility, infront_cameras, show_occ=True)

        # resize video
        print("generate visualization...")
        resized_video = media.resize_video(viz_video, (360, 480))
        resized_video2d_viz = media.resize_video(video2d_viz, (360, 480))
        
        # Generate 3D visualization if enabled and save if save_all is True
        if enable_3d_viz and save_all:
            print("generate 3D visualization...")
            video3d_viz = plot_3d_tracks(viz_tracks_xyz, viz_visibility, infront_cameras, show_occ=True)
            resized_video3d_viz = media.resize_video(video3d_viz, (360, 480))

        # Generate camera trajectory if available
        if viz_extrinsics_w2c is not None:
            print("generate camera trajectory visualization...")
            viz_extrinsics_c2w = np.linalg.inv(viz_extrinsics_w2c)
            extrinsics_plot_video = plot_camera_trajectory(
                camera_rotations=viz_extrinsics_c2w[:, :3, :3],
                camera_positions=viz_extrinsics_c2w[:, :3, -1],
                resolution=(128, 128)
            )
            resized_extrinsics_plot_video = media.resize_video(extrinsics_plot_video, (360, 480))

        if dataset_type in ['adt', 'drivetrack']:  # only these datasets have camera pose
            # create three parts combined visualization
            combined_viz = np.concatenate([
                resized_video,
                resized_video2d_viz,
                resized_extrinsics_plot_video
            ], axis=2)
        else:
            print("skip camera trajectory visualization (pstudio dataset has no camera pose)")
            # only create two parts combined visualization
            combined_viz = np.concatenate([
                resized_video,
                resized_video2d_viz
            ], axis=2)

        # save visualization results
        if save_all:
            print("save separate visualization results...")
            imageio.mimsave(
                osp.join(gif_dir, '2d_visualization.gif'),
                resized_video2d_viz.astype(np.uint8),
                fps=12,
                loop=0,
                optimize=True
            )
            
            if enable_3d_viz:
                imageio.mimsave(
                    osp.join(gif_dir, '3d_visualization.gif'),
                    resized_video3d_viz.astype(np.uint8),
                    fps=12,
                    loop=0,
                    optimize=True
                )

            if dataset_type in ['adt', 'drivetrack']:
                imageio.mimsave(
                    osp.join(gif_dir, 'camera_trajectory.gif'),
                    resized_extrinsics_plot_video.astype(np.uint8),
                    fps=12,
                    loop=0,
                    optimize=True
                )
            imageio.mimsave(
                osp.join(gif_dir, 'rgb_video.gif'),
                resized_video.astype(np.uint8),
                fps=12,
                loop=0,
                optimize=True
            )

        # save combined visualization
        print("save combined visualization...")
        imageio.mimsave(
            osp.join(gif_dir, 'combined_visualization.gif'),
            combined_viz.astype(np.uint8),
            fps=12,
            loop=0,
            optimize=True
        )

        print(f"all data saved to {data_dir} folder")

    except Exception as e:
        print(f"error when generating visualization: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration parameters
    NUM_FRAMES = float('inf')
    NUM_TRACKS = float('inf')
    OUTPUT_DIR = '.'
    ENABLE_3D_VIZ = False
    SAVE_ALL = True

    # iterate over all datasets and samples
    for dataset_type in MINIVAL_FILES.keys():
        print(f"\nprocessing dataset: {dataset_type}")
        for example in MINIVAL_FILES[dataset_type]:
            print(f"\nprocessing sample: {example}")
            try:
                main(
                    choose_random=False,
                    specific_example=example,
                    num_frames=NUM_FRAMES,
                    num_tracks=NUM_TRACKS,
                    output_dir=OUTPUT_DIR,
                    enable_3d_viz=ENABLE_3D_VIZ,
                    save_all=SAVE_ALL
                )
            except KeyboardInterrupt:
                print("\nuser interrupt processing")
                sys.exit(0)
            except Exception as e:
                print(f"error when processing sample {example}: {str(e)}")
                continue
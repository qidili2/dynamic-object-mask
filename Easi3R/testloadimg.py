import os
from dust3r.utils.image import load_images

# 使用完全相同的参数
dir_path = "data/davis/DAVIS/JPEGImages/480p/surf"
filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
filelist.sort()

print("Testing load_images with exact same parameters...")
try:
    imgs = load_images(filelist, size=512, verbose=True)
    print("SUCCESS: load_images worked without tracking")
except Exception as e:
    print(f"FAILED: {e}")
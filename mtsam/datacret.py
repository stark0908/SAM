import os
import h5py
import numpy as np
from PIL import Image
import cv2

# -------- LOAD MAT FILE --------
mat_file = h5py.File("nyu_depth_v2_labeled.mat", 'r')

images = mat_file['images']   # (N, 3, 640, 480)
depths = mat_file['depths']   # (N, 640, 480)
labels = mat_file['labels']   # (N, 640, 480)

N = images.shape[0]

print("Total samples:", N)

# -------- OUTPUT DIR --------
base_dir = "nyuv2_full"

for sub in ['rgb', 'depth', 'seg40', 'normals']:
    os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

# -------- NORMALS FUNCTION --------
def depth_to_normals(depth):
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

    normal = np.dstack((-dx, -dy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)

    return (normal + 1) / 2  # normalize to [0,1]

# -------- EXTRACTION --------
for i in range(N):

    # -------- RGB --------
    img = np.array(images[i])              # (3, 640, 480)
    img = np.transpose(img, (1, 2, 0))     # (640, 480, 3)
    img = np.rot90(img, k=1)               # → (480, 640, 3)
    img = img.astype(np.uint8)
    Image.fromarray(img).save(f"{base_dir}/rgb/{i}.png")

    # -------- DEPTH --------
    depth = np.array(depths[i]).astype(np.float32)  # (640, 480)
    depth = np.rot90(depth, k=1)                    # → (480, 640)
    np.save(f"{base_dir}/depth/{i}.npy", depth)

    # -------- SEGMENTATION (40-class) --------
    seg = np.array(labels[i])            # uint16
    seg = np.rot90(seg, k=1)             # → (480, 640)
    seg = seg.astype(np.uint8)           # safe (values ~0–40)
    Image.fromarray(seg).save(f"{base_dir}/seg40/{i}.png")

    # -------- NORMALS --------
    normals = depth_to_normals(depth)    # use rotated depth
    normals = (normals * 255).astype(np.uint8)
    Image.fromarray(normals).save(f"{base_dir}/normals/{i}.png")

    if i % 100 == 0:
        print(f"Processed: {i}/{N}")

print("✅ Done: RGB + Depth + Seg40 + Normals (correct orientation)")

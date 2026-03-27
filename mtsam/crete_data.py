import os
import h5py
import numpy as np
from PIL import Image
import cv2
from scipy.io import loadmat

# -------- LOAD DATA --------
mat_file = h5py.File("nyu_depth_v2_labeled.mat", 'r')
images = mat_file['images']
depths = mat_file['depths']

# -------- LOAD SPLITS --------
splits = loadmat("nyuv2-meta-data/splits.mat")

train_ids = splits['trainNdxs'].squeeze() - 1
test_ids  = splits['testNdxs'].squeeze() - 1

# create val split
val_size = int(0.1 * len(train_ids))
val_ids = train_ids[:val_size]
train_ids = train_ids[val_size:]

# -------- DIR SETUP --------
base_dir = "nyuv2"

def create_dirs(split):
    for sub in ['rgb', 'depth', 'seg', 'normals']:
        os.makedirs(os.path.join(base_dir, split, sub), exist_ok=True)

for s in ['train', 'val', 'test']:
    create_dirs(s)

# -------- NORMALS --------
def depth_to_normals(depth):
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

    normal = np.dstack((-dx, -dy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)

    return (normal + 1) / 2

# -------- BUILD LABEL MAP --------
def build_label_map(label_dir):
    label_map = {}

    for fname in os.listdir(label_dir):
        if fname.endswith(".png"):
            idx = int(fname.split('_')[-1].split('.')[0])
            label_map[idx] = fname

    return label_map

train_label_dir = "nyuv2-meta-data/train_labels_13"
test_label_dir  = "nyuv2-meta-data/test_labels_13"

train_label_map = build_label_map(train_label_dir)
test_label_map  = build_label_map(test_label_dir)

# -------- PROCESS FUNCTION --------
def process_split(indices, split, label_dir, label_map):

    valid_count = 0

    for i in indices:

        label_file = label_map.get(i + 1)  # MATLAB → Python index fix

        if label_file is None:
            continue  # skip sample completely

        # -------- RGB --------
        img = np.array(images[i])
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(img).save(f"{base_dir}/{split}/rgb/{valid_count}.png")

        # -------- DEPTH --------
        depth = np.array(depths[i]).astype(np.float32)
        np.save(f"{base_dir}/{split}/depth/{valid_count}.npy", depth)

        # -------- SEG --------
        label_path = os.path.join(label_dir, label_file)
        seg = Image.open(label_path)
        seg = np.array(seg, dtype=np.uint8)
        Image.fromarray(seg).save(f"{base_dir}/{split}/seg/{valid_count}.png")

        # -------- NORMALS --------
        normals = depth_to_normals(depth)
        normals = (normals * 255).astype(np.uint8)
        Image.fromarray(normals).save(f"{base_dir}/{split}/normals/{valid_count}.png")

        if valid_count % 100 == 0:
            print(f"{split}: {valid_count}")

        valid_count += 1

# -------- RUN --------
process_split(train_ids, "train", train_label_dir, train_label_map)
process_split(val_ids,   "val",   train_label_dir, train_label_map)
process_split(test_ids,  "test",  test_label_dir,  test_label_map)

print("✅ Dataset ready (correct + paper-aligned)")

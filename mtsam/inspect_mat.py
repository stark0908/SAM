import h5py
import numpy as np

FILE_PATH = "/home/Stark/Downloads/nyu_depth_v2_labeled.mat"

f = h5py.File(FILE_PATH, 'r')

print("\n" + "="*60)
print("🔑 ROOT KEYS")
print("="*60)
for key in f.keys():
    print(key)

# --------------------------------------------------
# 1. FULL STRUCTURE INSPECTION
# --------------------------------------------------
print("\n" + "="*60)
print("📦 FULL DATASET STRUCTURE")
print("="*60)

def inspect(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}")
        print(f"  shape: {obj.shape}")
        print(f"  dtype: {obj.dtype}")
        print(f"  type : Dataset")
        print("-"*40)
    elif isinstance(obj, h5py.Group):
        print(f"{name}/ (Group)")
        print("-"*40)

f.visititems(inspect)

# --------------------------------------------------
# 2. CORE DATASETS CHECK
# --------------------------------------------------
print("\n" + "="*60)
print("📊 CORE DATA SHAPES")
print("="*60)

images = f['images']
depths = f['depths']
labels = f['labels']

print("images shape :", images.shape)
print("depths shape :", depths.shape)
print("labels shape :", labels.shape)

# --------------------------------------------------
# 3. SLICING TEST (VERY IMPORTANT)
# --------------------------------------------------
print("\n" + "="*60)
print("🔍 SLICING TEST")
print("="*60)

img0 = np.array(images[:, :, :, 0])
depth0 = np.array(depths[:, :, 0])
label0 = np.array(labels[:, :, 0])

print("img0 shape   :", img0.shape)
print("depth0 shape :", depth0.shape)
print("label0 shape :", label0.shape)

# --------------------------------------------------
# 4. VALUE INSPECTION
# --------------------------------------------------
print("\n" + "="*60)
print("📈 VALUE RANGES")
print("="*60)

print("RGB min/max   :", img0.min(), img0.max())
print("Depth min/max :", depth0.min(), depth0.max())
print("Label unique (first 20):", np.unique(label0)[:20])

# --------------------------------------------------
# 5. MULTIPLE SAMPLE CHECK
# --------------------------------------------------
print("\n" + "="*60)
print("🔄 MULTI-SAMPLE CONSISTENCY CHECK")
print("="*60)

for i in [0, 10, 100, 500]:
    if i >= images.shape[3]:
        continue

    img = np.array(images[:, :, :, i])
    depth = np.array(depths[:, :, i])

    print(f"\nSample {i}")
    print("  img shape   :", img.shape)
    print("  depth shape :", depth.shape)
    print("  img range   :", img.min(), img.max())
    print("  depth range :", depth.min(), depth.max())

# --------------------------------------------------
# 6. LABEL DISTRIBUTION CHECK
# --------------------------------------------------
print("\n" + "="*60)
print("🏷️ LABEL DISTRIBUTION (sample)")
print("="*60)

all_labels = []
for i in range(50):  # check first 50 images
    lbl = np.array(labels[:, :, i])
    all_labels.append(np.unique(lbl))

unique_labels = np.unique(np.concatenate(all_labels))
print("Unique labels (first 50 images):", unique_labels[:50])
print("Total unique labels:", len(unique_labels))

# --------------------------------------------------
# 7. CLASS NAMES DECODING
# --------------------------------------------------
print("\n" + "="*60)
print("🧾 CLASS NAMES (DECODING MATLAB REFERENCES)")
print("="*60)

if 'names' in f:
    try:
        names_ref = f['names'][0]
        names = []

        for ref in names_ref:
            obj = f[ref]
            name = ''.join(chr(c) for c in obj[:])
            names.append(name)

        print("First 20 class names:")
        for i, n in enumerate(names[:20]):
            print(f"{i}: {n}")

        print("\nTotal classes:", len(names))

    except Exception as e:
        print("Error decoding names:", e)

# --------------------------------------------------
# 8. MEMORY TYPE CHECK
# --------------------------------------------------
print("\n" + "="*60)
print("🧠 DATA ACCESS TYPE CHECK")
print("="*60)

print("Type of images slice:", type(images[:, :, :, 0]))
print("Type after np.array:", type(np.array(images[:, :, :, 0])))

print("\nDone ✅")

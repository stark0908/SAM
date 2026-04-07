import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
from mtsam import MTSam
from image_encoder import ImageEncoderViT
from mask_decoder import TaskDecoder
from transformer import TwoWayTransformer
import torch.nn.functional as F

# ---------------------------
# DATASET (STRICT seg13)
# ---------------------------
class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, target_size=(480, 640)):
        self.root_dir = root_dir
        self.target_size = target_size

        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.seg_dir = os.path.join(root_dir, 'seg13')   # ✅ STRICT
        self.normals_dir = os.path.join(root_dir, 'normals')

        self.files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
        print(f"Total samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        base = os.path.splitext(fname)[0]

        resize_size = (self.target_size[1], self.target_size[0])

        # -------- RGB --------
        img = Image.open(os.path.join(self.rgb_dir, fname)).convert('RGB')
        img = img.resize(resize_size, Image.BILINEAR)
        img = np.array(img)

        # -------- DEPTH (.npy) --------
        depth = np.load(os.path.join(self.depth_dir, base + '.npy')).astype(np.float32)
        if depth.max() > 100:
            depth = depth / 1000.0

        depth = Image.fromarray(depth)
        depth = depth.resize(resize_size, Image.BILINEAR)
        depth = np.array(depth, dtype=np.float32)

        # -------- SEG (13-class) --------
        seg = Image.open(os.path.join(self.seg_dir, fname))
        seg = seg.resize(resize_size, Image.NEAREST)
        seg = np.array(seg, dtype=np.int64)

        # -------- NORMALS --------
        normals = Image.open(os.path.join(self.normals_dir, fname))
        normals = normals.resize(resize_size, Image.BILINEAR)
        normals = np.array(normals, dtype=np.float32) / 255.0

        # -------- TO TENSOR --------
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        seg = torch.from_numpy(seg).long()
        normals = torch.from_numpy(normals).permute(2, 0, 1).float()

        return {
            'image': img,
            'depth': depth,
            'seg': seg,
            'normals': normals
        }

# ---------------------------
# SPLIT (REPRODUCIBLE)
# ---------------------------
def create_splits(dataset, train_ratio=0.8, seed=42):
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Train: {train_size}, Val: {val_size}")
    return train_set, val_set

# ---------------------------
# LOSS
# ---------------------------
def get_loss_function(task_idx, pred, target):
    if task_idx == 0:
        return F.cross_entropy(pred, target, ignore_index=255)
    elif task_idx == 1:
        return F.mse_loss(pred, target)
    elif task_idx == 2:
        return F.mse_loss(pred, target)

# ---------------------------
# TRAINING
# ---------------------------
def train_mtsam_on_nyuv2(data_dir, batch_size=4, num_epochs=100, lr=1e-4, seed=42):

    embed_dim = 256
    image_size = 1024
    num_tasks = 3

    task_channels = [13, 1, 3]   # ✅ STRICT 13 CLASS

    # -------- MODEL --------
    image_encoder = ImageEncoderViT(
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
        num_tasks=num_tasks,
        out_chans=embed_dim,
        window_size=14,
    )

    task_decoders = nn.ModuleList([
        TaskDecoder(
            transformer_dim=embed_dim,
            num_channels=ch,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=1024,
                num_heads=4,
            ),
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ) for ch in task_channels
    ])

    model = MTSam(image_encoder=image_encoder, task_decoders=task_decoders)

    # freeze backbone
    model.image_encoder.freeze_w0()

    # -------- DATA --------
    dataset = NYUv2Dataset(data_dir)
    train_set, val_set = create_splits(dataset, seed=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    for i in range(1,5):
        mask = dataset[i]["segmentation"]
        print(torch.unique(mask))



    # -------- OPTIMIZER --------
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):

            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            segs = batch['seg'].to(device)
            normals = batch['normals'].to(device)

            optimizer.zero_grad()
            losses = []

            for task_idx, target in enumerate([segs, depths, normals]):

                batched_input = [{'image': images[i]} for i in range(images.size(0))]
                outputs = model(batched_input, task_idx=task_idx)
                pred = outputs['masks']

                if pred.shape[-2:] != target.shape[-2:]:
                    pred = F.interpolate(
                        pred,
                        size=target.shape[-2:],
                        mode='bilinear' if task_idx > 0 else 'nearest',
                        align_corners=False if task_idx > 0 else None
                    )

                loss = get_loss_function(task_idx, pred, target)
                losses.append(loss)

            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}")

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:

                images = batch['image'].to(device)
                depths = batch['depth'].to(device)
                segs = batch['seg'].to(device)
                normals = batch['normals'].to(device)

                losses = []

                for task_idx, target in enumerate([segs, depths, normals]):

                    batched_input = [{'image': images[i]} for i in range(images.size(0))]
                    pred = model(batched_input, task_idx=task_idx)['masks']

                    if pred.shape[-2:] != target.shape[-2:]:
                        pred = F.interpolate(pred, size=target.shape[-2:])

                    losses.append(get_loss_function(task_idx, pred, target))

                val_loss += sum(losses).item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_mtsam_on_nyuv2(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        seed=args.seed
    )

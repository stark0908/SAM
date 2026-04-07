import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
from mtsam import MTSam
from image_encoder import ImageEncoderViT
from mask_decoder import TaskDecoder
from transformer import TwoWayTransformer
import torch.nn.functional as F

TASK_NAMES = ["Segmentation", "Depth", "Normals"]

# ---------------------------
# LOGGING HELPER
# ---------------------------
def log(msg, symbol="•"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {symbol} {msg}")

def log_section(title):
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)

def log_task_losses(tag, losses_dict, epoch=None):
    prefix = f"Epoch {epoch:>3d} | " if epoch is not None else ""
    parts = [f"{prefix}{tag}"]
    for task_name, val in losses_dict.items():
        parts.append(f"{task_name}: {val:.4f}")
    parts.append(f"Total: {sum(losses_dict.values()):.4f}")
    print("  |  ".join(parts))


# ---------------------------
# DATASET
# ---------------------------
class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, target_size=(480, 640)):
        self.root_dir = root_dir
        self.target_size = target_size

        self.rgb_dir     = os.path.join(root_dir, 'rgb')
        self.depth_dir   = os.path.join(root_dir, 'depth')
        self.seg_dir     = os.path.join(root_dir, 'seg13')
        self.normals_dir = os.path.join(root_dir, 'normals')

        log_section("Dataset Loading")
        for name, path in [
            ("RGB",     self.rgb_dir),
            ("Depth",   self.depth_dir),
            ("Seg13",   self.seg_dir),
            ("Normals", self.normals_dir),
        ]:
            exists = os.path.isdir(path)
            status = "✓" if exists else "✗ MISSING"
            log(f"{name:<10} {path}  [{status}]", symbol="  ")

        self.files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
        log(f"Found {len(self.files)} samples", symbol="✓")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        base  = os.path.splitext(fname)[0]
        resize_size = (self.target_size[1], self.target_size[0])  # (W, H) for PIL

        # RGB
        img = Image.open(os.path.join(self.rgb_dir, fname)).convert('RGB')
        img = img.resize(resize_size, Image.BILINEAR)
        img = np.array(img)

        # Depth (.npy)
        depth = np.load(os.path.join(self.depth_dir, base + '.npy')).astype(np.float32)
        if depth.max() > 100:
            depth = depth / 1000.0
        depth = Image.fromarray(depth)
        depth = depth.resize(resize_size, Image.BILINEAR)
        depth = np.array(depth, dtype=np.float32)

        # Seg (13-class)
        seg = Image.open(os.path.join(self.seg_dir, fname))
        seg = seg.resize(resize_size, Image.NEAREST)
        seg = np.array(seg, dtype=np.int64)

        # Normals
        normals = Image.open(os.path.join(self.normals_dir, fname))
        normals = normals.resize(resize_size, Image.BILINEAR)
        normals = np.array(normals, dtype=np.float32) / 255.0

        img     = torch.from_numpy(img).permute(2, 0, 1).float()
        depth   = torch.from_numpy(depth).unsqueeze(0).float()
        seg     = torch.from_numpy(seg).long()
        normals = torch.from_numpy(normals).permute(2, 0, 1).float()

        return {'image': img, 'depth': depth, 'seg': seg, 'normals': normals}


# ---------------------------
# SPLIT
# ---------------------------
def create_splits(dataset, train_ratio=0.8, seed=42):
    log_section("Dataset Split")
    total      = len(dataset)
    train_size = int(train_ratio * total)
    val_size   = total - train_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    log(f"Train : {train_size} samples  ({train_ratio*100:.0f}%)")
    log(f"Val   : {val_size} samples  ({(1-train_ratio)*100:.0f}%)")
    log(f"Seed  : {seed}")
    return train_set, val_set


# ---------------------------
# LOSS
# ---------------------------
def get_loss_function(task_idx, pred, target):
    if task_idx == 0:   # Segmentation
        return F.cross_entropy(pred, target, ignore_index=255)
    elif task_idx == 1: # Depth
        return F.mse_loss(pred, target)
    elif task_idx == 2: # Normals
        return F.mse_loss(pred, target)


# ---------------------------
# ONE EPOCH
# ---------------------------
def run_epoch(model, loader, optimizer, device, epoch, num_epochs, phase="Train"):
    is_train = (phase == "Train")
    model.train() if is_train else model.eval()

    task_totals = {name: 0.0 for name in TASK_NAMES}
    num_batches = len(loader)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:>3d}/{num_epochs} [{phase:<5}]",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in pbar:
            images  = batch['image'].to(device)
            targets = [
                batch['seg'].to(device),      # task 0
                batch['depth'].to(device),    # task 1
                batch['normals'].to(device),  # task 2
            ]

            if is_train:
                optimizer.zero_grad()

            losses = []
            for task_idx, (target, task_name) in enumerate(zip(targets, TASK_NAMES)):
                batched_input = [{'image': images[i]} for i in range(images.size(0))]
                pred = model(batched_input, task_idx=task_idx)['masks']

                if pred.shape[-2:] != target.shape[-2:]:
                    pred = F.interpolate(
                        pred,
                        size=target.shape[-2:],
                        mode='bilinear' if task_idx > 0 else 'nearest',
                        align_corners=False if task_idx > 0 else None,
                    )

                loss = get_loss_function(task_idx, pred, target)
                losses.append(loss)
                task_totals[task_name] += loss.item()

            total_loss = sum(losses)

            if is_train:
                total_loss.backward()
                optimizer.step()

            # Live tqdm postfix: show each task loss
            pbar.set_postfix({
                "seg":  f"{losses[0].item():.3f}",
                "dep":  f"{losses[1].item():.3f}",
                "nor":  f"{losses[2].item():.3f}",
                "tot":  f"{total_loss.item():.3f}",
            })

    # Average over batches
    avg_losses = {name: task_totals[name] / num_batches for name in TASK_NAMES}
    return avg_losses


# ---------------------------
# TRAINING
# ---------------------------
def train_mtsam_on_nyuv2(data_dir, batch_size=4, num_epochs=100, lr=1e-4, seed=42, gpu=1):

    log_section("Configuration")
    log(f"Data dir    : {data_dir}")
    log(f"Batch size  : {batch_size}")
    log(f"Epochs      : {num_epochs}")
    log(f"LR          : {lr}")
    log(f"Seed        : {seed}")
    log(f"GPU         : {gpu}")

    embed_dim  = 256
    image_size = 1024
    num_tasks  = 3
    task_channels = [14, 1, 3]  # seg=14 classes, depth=1, normals=3

    # -------- MODEL --------
    log_section("Model Construction")
    log("Building ImageEncoderViT ...")
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

    log("Building TaskDecoders ...")
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
    for name, ch in zip(TASK_NAMES, task_channels):
        log(f"  {name:<15} → {ch} output channel(s)")

    model = MTSam(image_encoder=image_encoder, task_decoders=task_decoders)
    model.image_encoder.freeze_w0()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total params    : {total_params:,}")
    log(f"Trainable params: {trainable_params:,}")
    log(f"Frozen params   : {total_params - trainable_params:,}")

    # -------- DATA --------
    dataset = NYUv2Dataset(data_dir)

    log_section("Segmentation Label Sanity Check")
    for i in range(min(5, len(dataset))):
        mask    = dataset[i]["seg"]
        unique  = torch.unique(mask).tolist()
        log(f"Sample {i}: unique seg labels = {unique}")

    train_set, val_set = create_splits(dataset, seed=seed)

    log_section("DataLoaders")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True, persitant_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persitant_workers=True)
    log(f"Train loader: {len(train_loader)} batches")
    log(f"Val   loader: {len(val_loader)} batches")

    # -------- OPTIMIZER --------
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    log_section("Device")
    log(f"Using: {device}")
    if device.type == 'cuda':
        log(f"GPU name : {torch.cuda.get_device_name(device)}")
        log(f"VRAM     : {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    model.to(device)

    # -------- BEST CHECKPOINT TRACKING --------
    best_val_total = float('inf')

    # -------- TRAIN LOOP --------
    log_section("Training Start")
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_losses = run_epoch(model, train_loader, optimizer, device, epoch, num_epochs, phase="Train")
        # Validate
        val_losses   = run_epoch(model, val_loader,   optimizer, device, epoch, num_epochs, phase="Val")

        scheduler.step()

        epoch_time  = time.time() - epoch_start
        val_total   = sum(val_losses.values())
        train_total = sum(train_losses.values())

        # Per-task summary line
        log_task_losses("Train", train_losses, epoch=epoch)
        log_task_losses("Val  ", val_losses,   epoch=epoch)

        # LR + timing
        current_lr = scheduler.get_last_lr()[0]
        log(
            f"Epoch {epoch:>3d} | LR: {current_lr:.2e}  "
            f"Time: {epoch_time:.1f}s  "
            f"Elapsed: {(time.time()-train_start)/60:.1f}min",
            symbol="→"
        )

        # Best model checkpoint
        if val_total < best_val_total:
            best_val_total = val_total
            log(f"★  New best val loss: {best_val_total:.4f} — saving checkpoint", symbol="★")
            torch.save(model.state_dict(), "best_mtsam.pth")

        print()  # blank line between epochs for readability

    log_section("Training Complete")
    log(f"Total time : {(time.time()-train_start)/60:.1f} min")
    log(f"Best val   : {best_val_total:.4f}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   required=True)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--num_epochs', type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--gpu',        type=int,   default=1)
    args = parser.parse_args()

    train_mtsam_on_nyuv2(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        seed=args.seed,
        gpu=args.gpu,
    )

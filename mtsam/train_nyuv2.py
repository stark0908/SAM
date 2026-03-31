import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
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

class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(480, 640)):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = (480, 640)  # (H, W)

        # NYUv2 structure (assuming processed PNG format)
        # You may need to adjust paths based on your dataset preparation
        self.image_dir = os.path.join(root_dir, split, 'rgb')
        self.depth_dir = os.path.join(root_dir, split, 'depth')
        self.seg_dir = os.path.join(root_dir, split, 'seg')
        self.normals_dir = os.path.join(root_dir, split, 'normals')

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Get list of files
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])

        print(f"Found {len(self.image_files)} images in {split} set")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]

        # ✅ FIX: PIL expects (W, H)
        resize_size = (self.target_size[1], self.target_size[0])

        # Load RGB image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = image.resize(resize_size, Image.BILINEAR)
        image = np.array(image)

        # Load depth map
        depth_path = os.path.join(self.depth_dir, f"{base_name}.png")
        if os.path.exists(depth_path):
            depth = Image.open(depth_path)
            depth = depth.resize(resize_size, Image.BILINEAR)
            depth = np.array(depth, dtype=np.float32)

            if depth.max() > 100:
                depth = depth / 1000.0
        else:
            depth = np.zeros(self.target_size, dtype=np.float32)

        # Load segmentation
        seg_path = os.path.join(self.seg_dir, f"{base_name}.png")
        if os.path.exists(seg_path):
            seg = Image.open(seg_path)
            seg = seg.resize(resize_size, Image.NEAREST)
            seg = np.array(seg, dtype=np.int64)
        else:
            seg = np.zeros(self.target_size, dtype=np.int64)

        # Load normals
        normals_path = os.path.join(self.normals_dir, f"{base_name}.png")
        if os.path.exists(normals_path):
            normals = Image.open(normals_path)
            normals = normals.resize(resize_size, Image.BILINEAR)
            normals = np.array(normals, dtype=np.float32) / 255.0
        else:
            # ✅ FIX: correct (H, W, 3)
            normals = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        seg = torch.from_numpy(seg).long()
        normals = torch.from_numpy(normals).permute(2, 0, 1).float()

        return {
            'image': image,
            'depth': depth,
            'seg': seg,
            'normals': normals
        }

def get_loss_function(task_idx, pred, target):
    if task_idx == 0:  # Semantic segmentation
        return F.cross_entropy(pred, target, ignore_index=255)
    elif task_idx == 1:  # Depth estimation
        return F.mse_loss(pred, target)
    elif task_idx == 2:  # Surface normals
        return F.mse_loss(pred, target)
    else:
        raise ValueError(f"Unknown task index: {task_idx}")

def train_mtsam_on_nyuv2(data_dir, batch_size=4, num_epochs=100, lr=1e-4, save_dir='./checkpoints'):
    # Model configuration
    embed_dim = 256
    image_size = 1024
    patch_size = 16
    num_tasks = 3
    task_channels = [13, 1, 3]  # seg classes, depth, normals

    # Initialize model
    image_encoder = ImageEncoderViT(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
        num_tasks=num_tasks,
        out_chans=embed_dim,
        window_size=14,
    )

    task_decoders = nn.ModuleList()
    for channels in task_channels:
        task_decoders.append(TaskDecoder(
            transformer_dim=embed_dim,
            num_channels=channels,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=1024,
                num_heads=4,
            ),
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ))

    model = MTSam(
        image_encoder=image_encoder,
        task_decoders=task_decoders,
    )

    # Freeze the main weights, only train ToRA parameters
    model.image_encoder.freeze_w0()

    # Dataset and DataLoader
    train_dataset = NYUv2Dataset(root_dir=data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = NYUv2Dataset(root_dir=data_dir, split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Training loop
    num_epochs = 100
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            segs = batch['seg'].to(device)
            normals = batch['normals'].to(device)

            optimizer.zero_grad()

            # Train each task
            task_losses = []
            for task_idx, target in enumerate([segs, depths, normals]):
                # Create batched input for the model - split batch into individual images
                batched_input = [{'image': images[i]} for i in range(images.size(0))]

                outputs = model(batched_input, task_idx=task_idx)
                pred_masks = outputs['masks']

                # Resize predictions to target size if needed
                if pred_masks.shape[-2:] != target.shape[-2:]:
                    pred_masks = F.interpolate(pred_masks, size=target.shape[-2:], mode='bilinear' if task_idx > 0 else 'nearest', align_corners=False if task_idx > 0 else None)

                loss = get_loss_function(task_idx, pred_masks, target)
                task_losses.append(loss)

            # Combine losses (you might want to weight them differently)
            total_batch_loss = sum(task_losses)
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device)
                segs = batch['seg'].to(device)
                normals = batch['normals'].to(device)

                task_losses = []
                for task_idx, target in enumerate([segs, depths, normals]):
                    batched_input = [{'image': images[i]} for i in range(images.size(0))]
                    outputs = model(batched_input, task_idx=task_idx)
                    pred_masks = outputs['masks']

                    if pred_masks.shape[-2:] != target.shape[-2:]:
                        pred_masks = F.interpolate(pred_masks, size=target.shape[-2:], mode='bilinear' if task_idx > 0 else 'nearest', align_corners=False if task_idx > 0 else None)

                    loss = get_loss_function(task_idx, pred_masks, target)
                    task_losses.append(loss)

                val_loss += sum(task_losses).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MTSAM on NYUv2 Dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to NYUv2 dataset directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    train_mtsam_on_nyuv2(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )

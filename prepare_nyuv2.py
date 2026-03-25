#!/usr/bin/env python3
"""
NYUv2 Dataset Preparation Script

This script helps prepare the NYUv2 dataset for training MTSAM.
NYUv2 contains RGB images, depth maps, semantic segmentation labels, and surface normals.

Dataset Structure Expected:
nyuv2/
├── train/
│   ├── rgb/          # RGB images
│   ├── depth/         # Depth maps
│   ├── seg/           # Semantic segmentation (13 classes)
│   └── normals/       # Surface normals
├── val/
│   ├── rgb/
│   ├── depth/
│   ├── seg/
│   └── normals/
└── test/
    ├── rgb/
    ├── depth/
    ├── seg/
    └── normals/

Usage:
1. Download NYUv2 dataset from: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Extract and organize the data as shown above
3. Run this script to verify the dataset structure
4. Update the path in train_nyuv2.py

Alternatively, you can use preprocessed versions from:
- https://github.com/ankurhanda/nyuv2-meta-data
- https://github.com/charlesCXK/TorchSemiSeg
"""

import os
import argparse
from pathlib import Path

def verify_dataset_structure(root_dir):
    """Verify that the dataset has the expected structure."""
    required_dirs = [
        'train/rgb', 'train/depth', 'train/seg', 'train/normals',
        'val/rgb', 'val/depth', 'val/seg', 'val/normals',
        'test/rgb', 'test/depth', 'test/seg', 'test/normals'
    ]

    root_path = Path(root_dir)

    print(f"Verifying dataset structure in: {root_dir}")

    all_good = True
    for dir_path in required_dirs:
        full_path = root_path / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {full_path}")
            all_good = False
        else:
            # Count files in directory
            files = list(full_path.glob('*'))
            print(f"✅ {full_path}: {len(files)} files")

    if all_good:
        print("\n✅ Dataset structure verification passed!")
        print("You can now run train_nyuv2.py")
    else:
        print("\n❌ Dataset structure verification failed!")
        print("Please organize your data according to the expected structure.")

    return all_good

def create_dataset_structure(root_dir):
    """Create the expected directory structure."""
    dirs_to_create = [
        'train/rgb', 'train/depth', 'train/seg', 'train/normals',
        'val/rgb', 'val/depth', 'val/seg', 'val/normals',
        'test/rgb', 'test/depth', 'test/seg', 'test/normals'
    ]

    root_path = Path(root_dir)

    for dir_path in dirs_to_create:
        full_path = root_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {full_path}")

    print(f"\nDirectory structure created in: {root_dir}")
    print("Now place your NYUv2 data files in the appropriate directories.")

def main():
    parser = argparse.ArgumentParser(description='NYUv2 Dataset Preparation')
    parser.add_argument('--root_dir', type=str, default='./nyuv2',
                        help='Root directory for the dataset')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing dataset structure')
    parser.add_argument('--create_structure', action='store_true',
                        help='Create the expected directory structure')

    args = parser.parse_args()

    if args.verify:
        verify_dataset_structure(args.root_dir)
    elif args.create_structure:
        create_dataset_structure(args.root_dir)
    else:
        print("NYUv2 Dataset Preparation Script")
        print("=================================")
        print()
        print("Expected dataset structure:")
        print("nyuv2/")
        print("├── train/")
        print("│   ├── rgb/          # RGB images (.png)")
        print("│   ├── depth/         # Depth maps (.png)")
        print("│   ├── seg/           # Semantic segmentation (.png)")
        print("│   └── normals/       # Surface normals (.png)")
        print("├── val/")
        print("│   └── ...")
        print("└── test/")
        print("    └── ...")
        print()
        print("Commands:")
        print("  python prepare_nyuv2.py --create_structure --root_dir /path/to/nyuv2")
        print("  python prepare_nyuv2.py --verify --root_dir /path/to/nyuv2")
        print()
        print("Download NYUv2 from: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")

if __name__ == '__main__':
    main()
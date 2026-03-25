# Training MTSAM on NYUv2 Dataset

This repository contains the Multi-Task Segment Anything Model (MTSAM) implementation and training code for the NYUv2 dataset.

## Overview

MTSAM is a multi-task fine-tuning of the Segment Anything Model (SAM) that can perform multiple vision tasks simultaneously:
- Semantic Segmentation (13 classes)
- Depth Estimation
- Surface Normal Estimation

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare NYUv2 Dataset

Download the NYUv2 dataset from the official source:
- [NYUv2 Depth Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

Or use preprocessed versions from:
- [NYUv2 Meta Data](https://github.com/ankurhanda/nyuv2-meta-data)
- [Torch Semi-Seg](https://github.com/charlesCXK/TorchSemiSeg)

Organize the dataset in the following structure:

```
nyuv2/
├── train/
│   ├── rgb/          # RGB images (.png)
│   ├── depth/         # Depth maps (.png)
│   ├── seg/           # Semantic segmentation (.png)
│   └── normals/       # Surface normals (.png)
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
```

Use the preparation script to verify your dataset structure:

```bash
python prepare_nyuv2.py --verify --root_dir /path/to/nyuv2
```

## Training

Run the training script:

```bash
python train_nyuv2.py --data_dir /path/to/nyuv2 --batch_size 4 --num_epochs 100 --lr 1e-4
```

### Training Arguments

- `--data_dir`: Path to the NYUv2 dataset directory (required)
- `--batch_size`: Batch size for training (default: 4)
- `--num_epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)

### Multi-GPU Training

The code automatically detects and uses CUDA if available. For multi-GPU training, use:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_nyuv2.py --data_dir /path/to/nyuv2
```

## Model Architecture

### MTSAM Components

1. **Image Encoder (ToRA-integrated ViT)**: Processes input images with task-specific adaptations
2. **Task Decoders**: Separate decoders for each task (segmentation, depth, normals)
3. **ToRA (Tensorized low-Rank Adaptation)**: Parameter-efficient fine-tuning method

### Key Features

- **Multi-Task Learning**: Simultaneous training on multiple vision tasks
- **Parameter Efficiency**: Only ToRA parameters are trained, main ViT weights are frozen
- **Task-Specific Processing**: Each task has its own decoder optimized for the output format

## Evaluation

After training, you can evaluate the model using the test set. The training script saves:
- Best model checkpoint (`best_model.pth`)
- Regular checkpoints every 10 epochs (`checkpoint_epoch_X.pth`)

## Results

The model outputs:
- **Semantic Segmentation**: 13-class probability maps
- **Depth Estimation**: Depth maps in meters
- **Surface Normals**: 3-channel normal maps

## Citation

If you use this code, please cite the MTSAM paper:

```
@article{mtsam2024,
  title={Multi-Task Fine-Tuning for Segment Anything Model},
  author={...},
  journal={...},
  year={2024}
}
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Dataset not found**: Ensure correct path and file structure
3. **Import errors**: Install all dependencies from `requirements.txt`

### Performance Tips

- Use larger batch sizes if you have more GPU memory
- The model benefits from longer training (100+ epochs)
- Monitor validation loss to avoid overfitting</content>
<parameter name="filePath">/home/Stark/Transformers/SAM/README.md
# Multi-Task SAM (MTSAM)

This folder contains the Multi-Task Segment Anything Model implementation that performs simultaneous learning for:
- Semantic Segmentation (13 classes)
- Depth Estimation  
- Surface Normal Estimation

## Files

- `mtsam.py` - Main MTSAM model class
- `image_encoder.py` - Vision Transformer with ToRA integration
- `mask_decoder.py` - TaskDecoder for multi-channel outputs
- `transformer.py` - Two-way transformer for feature-prompt interaction
- `common.py` - Common utilities (MLPBlock, LayerNorm2d, etc.)
- `test_mtsam.py` - Test script for MTSAM model
- `train_nyuv2.py` - Training script for NYUv2 dataset
- `prepare_nyuv2.py` - Dataset preparation utilities

## Key Features

- **Multi-task learning**: Single model for multiple vision tasks simultaneously
- **Parameter efficient**: Uses ToRA (Tensorized low-Rank Adaptation) for fine-tuning
- **Task-specific decoders**: Each task has its own optimized decoder
- **Frozen backbone**: Main ViT weights frozen, only ToRA parameters trained

## Training on NYUv2

### Setup Dataset
```bash
python prepare_nyuv2.py --create_structure --root_dir /path/to/nyuv2
python prepare_nyuv2.py --verify --root_dir /path/to/nyuv2
```

### Train Model
```bash
python train_nyuv2.py --data_dir /path/to/nyuv2 --batch_size 4 --num_epochs 100
```

### Training Arguments
- `--data_dir`: Path to NYUv2 dataset (required)
- `--batch_size`: Batch size (default: 4)
- `--num_epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--save_dir`: Checkpoint directory (default: ./checkpoints)

## Testing

```bash
python test_mtsam.py
```

This creates a model and runs a forward pass with random inputs across all three tasks.
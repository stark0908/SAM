# SAM Implementations

Complete implementations of Segment Anything Model variants in this workspace.

## 📁 Folder Structure

### `original_sam/`
**Original Segment Anything Model** - Single-task prompt-based segmentation

- **Features**: Point/box/mask prompts, standard architecture, single output
- **Usage**: `cd original_sam && python test_sam.py`
- **Files**: sam.py, prompt_encoder.py, mask_decoder.py, image_encoder.py, transformer.py, common.py, test_sam.py

### `mtsam/`
**Multi-Task SAM** - Simultaneous multi-task learning

- **Tasks**: Semantic segmentation (13 classes), depth estimation, surface normals
- **Features**: ToRA parameter efficiency, task-specific decoders, frozen backbone
- **Usage**: 
  ```bash
  cd mtsam
  python test_mtsam.py
  python train_nyuv2.py --data_dir /path/to/nyuv2
  ```
- **Files**: mtsam.py, test_mtsam.py, train_nyuv2.py, prepare_nyuv2.py, image_encoder.py, mask_decoder.py, transformer.py, common.py

## 🚀 Quick Start

### Original SAM
```bash
cd original_sam
python test_sam.py
```

### MTSAM on NYUv2
```bash
cd mtsam
python prepare_nyuv2.py --create_structure --root_dir /path/to/nyuv2
python train_nyuv2.py --data_dir /path/to/nyuv2 --batch_size 4 --num_epochs 100
```

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- torch >= 1.9.0
- torchvision >= 0.10.0
- numpy
- Pillow

## 📚 Documentation

- [MTSAM Architecture](MTSAM_Architecture.md) - Detailed breakdown of MTSAM components
- Each subfolder has its own README.md with additional details

## 🔗 References

**Original SAM Paper:**
```
Kirillov et al., "Segment Anything" (2023)
```

**MTSAM:**
Multi-Task Fine-Tuning for Segment Anything Model using ToRA
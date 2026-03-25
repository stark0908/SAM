# Original SAM Implementation

This folder contains the original Segment Anything Model (SAM) implementation, reverted from the multi-task version (MTSAM).

## Files

- `sam.py` - Main SAM model class
- `image_encoder.py` - Vision Transformer image encoder
- `prompt_encoder.py` - Encoder for prompts (points, boxes, masks)
- `mask_decoder.py` - Decoder for generating masks
- `transformer.py` - Two-way transformer for prompt-image interaction
- `common.py` - Common utilities and building blocks
- `test_sam.py` - Test script for the SAM model

## Usage

```python
from sam import Sam
from image_encoder import ImageEncoderViT
from prompt_encoder import PromptEncoder
from mask_decoder import MaskDecoder
from transformer import TwoWayTransformer

# Initialize components
image_encoder = ImageEncoderViT(...)
prompt_encoder = PromptEncoder(...)
mask_decoder = MaskDecoder(...)

# Create SAM model
sam = Sam(image_encoder, prompt_encoder, mask_decoder)

# Run inference
batched_input = [{
    "image": image_tensor,
    "point_coords": point_coords,
    "point_labels": point_labels
}]
outputs = sam(batched_input, multimask_output=True)
```

## Key Differences from MTSAM

- **Single Task**: Only performs segmentation, not multi-task learning
- **Prompt-Based**: Uses explicit prompts (points, boxes, masks) instead of task tokens
- **Standard Architecture**: Follows the original SAM paper implementation
- **No ToRA**: Does not include the ToRA parameter-efficient fine-tuning

## Testing

Run the test script:

```bash
cd original_sam
python test_sam.py
```

This will create a simple SAM model and run a forward pass with random inputs.
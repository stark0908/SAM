# Segment Anything Model (SAM) Architecture Guide

This document breaks down the custom implementation of the Segment Anything Model (SAM) provided in this workspace. It explicitly details what each component inside each file is responsible for, complete with functional examples.

---

## 1. `common.py`
This module contains standard neural network building blocks that are shared across different parts of the architecture.

* **`MLPBlock`**
  * **What it does**: A basic Multi-Layer Perceptron (feed-forward network) with two linear projection layers separated by an activation function (GELU by default).
  * **Example**: Inside the ViT blocks and the Mask Decoder transformer blocks, after the attention mechanism aggregates information from different tokens, the `MLPBlock` acts independently on each token to encode non-linear combinations of features.
* **`LayerNorm2d`**
  * **What it does**: Native PyTorch `nn.LayerNorm` normalizes over the last dimension. However, convolutional feature maps are ordered as `(Batch, Channels, Height, Width)`. This custom class computes the channel-wise mean and variance without needing to permute the tensor back and forth.
  * **Example**: Used in the Image Encoder neck to normalize the `256x64x64` spatial image feature map after the 1x1 and 3x3 downsampling convolutions.

---

## 2. `transformer.py`
This module contains the sophisticated two-way attention mechanism responsible for deeply mixing the prompt tokens with the image features in the mask decoder.

* **`Attention`**
  * **What it does**: A standard scaled dot-product Multi-Head Attention module. It supports passing distinct queries (`q`), keys (`k`), and values (`v`). It also accepts a `downsample_rate` to internally reduce the dimensionality of the embeddings, making the cross-attention significantly faster.
* **`TwoWayAttentionBlock`**
  * **What it does**: A single layer block inside the decoder that acts as a refinement step. It executes four sequential operations:
    1. **Self-attention** on prompt tokens (how do the points relate to each other?).
    2. **Cross-attention** from tokens to the image (prompts extract specific feature contexts from the image map).
    3. **MLP** on the updated tokens.
    4. **Cross-attention** from the image back to the tokens (image map gets updated with context about what the user clicked on).
* **`TwoWayTransformer`**
  * **What it does**: Stacks multiple `TwoWayAttentionBlock`s together. It orchestrates the flow of data: flattening the `64x64` image map into a sequence of $4096$ tokens, passing them through the blocks alongside positional encodings, and applying a final cross-attention step to fully prepare the queries for mask generation.

---

## 3. `image_encoder.py`
This module implements a Vision Transformer (ViT) customized for high-resolution images ($1024 \times 1024$), heavily inspired by Masked Autoencoders (MAE).

* **`PatchEmbed`**
  * **What it does**: Uses a 2D convolution with a large kernel (`16x16`) and large stride (`16`) to "chunk" the image into non-overlapping patches, flattening them into the embedding dimension space.
  * **Example**: A `3 x 1024 x 1024` image turns into a `64 x 64` grid of vectors.
* **`window_partition` / `window_unpartition`**
  * **What it does**: Standard global attention on a $64 \times 64$ ($4096$ tokens) grid has a quadratic complexity $O(N^2)$ making it extremely memory intensive. These utilities partition the grid into small, localized $14 \times 14$ windows so attention is only computed locally. `window_unpartition` rebuilds the full tensor after attention finishes.
* **`AttentionWin`**
  * **What it does**: Standard ViT multi-head attention that acts on the localized windows. It is also designed to inject *relative positional embeddings*, helping the network understand distance offsets within the spatial window.
* **`Block`**
  * **What it does**: Combines `AttentionWin`, layer normalization, window partitioning logic, and an `MLPBlock` into a single, cohesive Transformer block.
* **`ImageEncoderViT`**
  * **What it does**: The massive backbone of SAM. It embeds patches, adds absolute positional encodings, runs them through the blocks (mixing windowed blocks for efficiency and a few global blocks to transfer information across the whole image), and finishes with a convolutional "neck" to output the highly semantic `256 x 64 x 64` visual representation.

---

## 4. `prompt_encoder.py`
Translates user instructions (sparse clicks/boxes or a dense mask) into $256$-dimensional mathematical representations.

* **`PositionEmbeddingRandom`**
  * **What it does**: Maps $2D$ normalized pixel coordinates $(x, y)$ into a high-dimensional vector space using random Gaussian Fourier features. 
  * **Example**: It takes a user click at `(500, 500)`, maps it to `(0.488, 0.488)`, projects it against a random matrix, and applies `sin`/`cos` to output a unique $128$-dim spatial signature.
* **`PromptEncoder`**
  * **What it does**: Hub for parsing all inputs. 
    1. **Points**: Applies `PositionEmbeddingRandom` and adds a learned "foreground" or "background" state vector.
    2. **Boxes**: Treats a box as two points (top-left, bottom-right), encodes them spatially, and adds learned "top-left" / "bottom-right" state vectors.
    3. **Masks**: Takes an existing dense mask image (`1024x1024`) and cascades it through convolutions (`mask_downscaling`) to downscale it by 4x.
    4. **Missing Prompts**: Automatically applies placeholders (`not_a_point_embed`, `no_mask_embed`) so the decoder always receives consistent tensor shapes even when no prompts are provided.

---

## 5. `mask_decoder.py`
A very fast, lightweight module that predicts the actual pixel masks directly from the embeddings.

* **`MLP`**
  * **What it does**: A customizable multi-layer perceptron built to handle an arbitrary number of hidden layers (unlike `common.MLPBlock` which is fixed to 2 layers).
* **`MaskDecoder`**
  * **What it does**: 
    1. Creates learned placeholder tokens (`mask_tokens` to represent predicted masks, and an `iou_token` to represent the confidence score).
    2. Concatenates these placeholders with the user's sparse prompt embeddings and sends them through the `TwoWayTransformer`.
    3. **`output_upscaling`**: Applies Transpose Convolutions to scale the `64x64` image features back up to `256x256`.
    4. **Hypernetwork multiplication**: Uses `output_hypernetworks_mlps` to convert the refined `mask_tokens` into a set of dynamic weights. It then computes the dot-product between these dynamic weights and the upscaled image features to predict the final spatial mask.
    5. **`iou_prediction_head`**: Predicts an Intersection-over-Union (IoU) scalar from the `iou_token` to tell you how confident the model is in its mask prediction.

---

## 6. `sam.py`
The top-level coordination class.

* **`Sam`**
  * **What it does**: The user-facing module that owns the `ImageEncoderViT`, `PromptEncoder`, and `MaskDecoder`. 
  * **`preprocess`**: Zero-pads a raw input image up to `1024x1024` to handle arbitrarily sized photos, and normalizes it using standard RGB ImageNet values.
  * **`forward`**: Implements the main forward pass logic. It first passes the image batch to the image encoder. It then iterates through each image and its corresponding prompts, evaluates the prompt encoder, and evaluates the mask decoder to stream the final masks into an output dictionary.

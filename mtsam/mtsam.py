import torch
from torch import nn
from typing import Any, Dict, List, Tuple
import torch.nn.functional as F

from image_encoder import ImageEncoderViT
from mask_decoder import TaskDecoder

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

class MTSam(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        task_decoders: nn.ModuleList,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.task_decoders = task_decoders
        self.pe_layer = PositionEmbeddingRandom(128)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        task_idx: int,
    ) -> Dict[str, torch.Tensor]:
        
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        # Image embeddings specific to task_idx using ToRA
        image_embeddings = self.image_encoder(input_images, task_idx=task_idx)

        # Generate dense PE
        b, c, h, w = image_embeddings.shape
        image_pe = self.pe_layer((h, w)).unsqueeze(0).expand(b, -1, -1, -1)

        masks, iou_predictions = self.task_decoders[task_idx](
            image_embeddings=image_embeddings,
            image_pe=image_pe,
        )

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
        }

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def load_pretrained(self, ckpt_path: str) -> None:
        """
        Loads the pretrained SAM weights where applicable.
        Maps the image_encoder directly and mask_decoder to each task_decoder.
        Handles shape mismatches gracefully.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
            
        mapped_state_dict = {}
        for key, value in state_dict.items():
            # Image Encoder
            if key.startswith("image_encoder."):
                mapped_state_dict[key] = value
                
            # Mask Decoder -> Task Decoders
            elif key.startswith("mask_decoder."):
                sub_key = key.replace("mask_decoder.", "")
                for i in range(len(self.task_decoders)):
                    task_key = f"task_decoders.{i}.{sub_key}"
                    
                    if sub_key == "mask_tokens.weight":
                        # Original: (4, 256), Target: (2, 256)
                        mapped_state_dict[task_key] = value[:2, :]
                    elif sub_key.startswith("output_hypernetworks_mlps.0."):
                        # Original is ModuleList, Target is single MLP
                        new_key = task_key.replace("output_hypernetworks_mlps.0.", "output_hypernetworks_mlps.")
                        mapped_state_dict[new_key] = value
                    else:
                        mapped_state_dict[task_key] = value

        # Dynamic shape matching filter
        model_state_dict = self.state_dict()
        filtered_state_dict = {}
        for k, v in mapped_state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded SAM weights from {ckpt_path}.")
        print(f"Matched {len(filtered_state_dict)} parameters out of {len(state_dict)} in checkpoint.")

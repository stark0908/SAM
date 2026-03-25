import torch
from torch import Tensor
from torch import nn
from typing import Tuple, Type
from common import LayerNorm2d
from transformer import TwoWayTransformer

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        num_channels: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_channels = num_channels

        self.iou_token = nn.Embedding(1, transformer_dim)
        
        # E_t: task embeddings for image features
        self.task_embeddings = nn.Parameter(torch.zeros(num_channels, transformer_dim))
        
        # F_P: task-specific prompt tokens replacing sparse prompts
        self.task_prompt_tokens = nn.Parameter(torch.randn(num_channels, transformer_dim))
        
        self.mask_tokens = nn.Embedding(2, transformer_dim) # token for masks
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, 1, iou_head_depth)

    def forward(
        self,
        image_embeddings: Tensor, # B, C, H, W
        image_pe: Tensor, # B, C, H, W
    ) -> Tuple[Tensor, Tensor]:
        b, c, h, w = image_embeddings.shape
        
        # Broad-cast sum: E_t + image_embeddings
        E_t = self.task_embeddings.view(self.num_channels, c, 1, 1)
        # We expand image_embeddings to (B, N_t, C, H, W) and add E_t
        src = image_embeddings.unsqueeze(1) + E_t.unsqueeze(0)
        # flatten batch and channel: (B * N_t, C, H, W)
        src = src.view(b * self.num_channels, c, h, w)
        
        pos_src = image_pe.repeat(1, self.num_channels, 1, 1).view(b * self.num_channels, c, h, w)
        
        F_p = self.task_prompt_tokens.unsqueeze(0).expand(b, -1, -1).reshape(b * self.num_channels, 1, -1)
        
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight[0:1]], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(b * self.num_channels, -1, -1)
        
        tokens = torch.cat((output_tokens, F_p), dim=1)
        
        hs, src = self.transformer(src, pos_src, tokens)
        
        iou_token_out = hs[:, 0, :]
        mask_token_out = hs[:, 1, :]
        
        src = src.transpose(1, 2).view(b * self.num_channels, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        
        hyper_in = self.output_hypernetworks_mlps(mask_token_out)
        
        b_n, c_up, h_up, w_up = upscaled_embedding.shape
        masks = (hyper_in.unsqueeze(1) @ upscaled_embedding.view(b_n, c_up, h_up * w_up)).view(b_n, 1, h_up, w_up)
        
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        masks = masks.view(b, self.num_channels, h_up, w_up)
        iou_pred = iou_pred.view(b, self.num_channels)
        
        return masks, iou_pred

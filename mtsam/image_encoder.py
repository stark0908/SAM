import torch
import torch.nn as nn
from typing import Optional, Tuple
from common import MLPBlock, LayerNorm2d

class ToRA(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_tasks: int, p: int = 8, q: int = 8, v: int = 8) -> None:
        super().__init__()
        self.U1 = nn.Parameter(torch.randn(out_dim, p))
        self.U2 = nn.Parameter(torch.randn(in_dim, q))
        self.U3 = nn.Parameter(torch.randn(num_tasks, v))
        self.G = nn.Parameter(torch.zeros(p, q, v))
        nn.init.normal_(self.U1)
        nn.init.normal_(self.U2)
        nn.init.normal_(self.U3)
        nn.init.zeros_(self.G)
        
    def forward(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        U3_t = self.U3[task_idx]
        G_t = torch.einsum('pqv,v->pq', self.G, U3_t)
        delta_W_t = torch.einsum('op,pq,iq->oi', self.U1, G_t, self.U2)
        return torch.einsum('...i,oi->...o', x, delta_W_t)

class ToRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_tasks: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.tora = ToRA(in_features, out_features, num_tasks)
        
    def forward(self, x: torch.Tensor, task_idx: Optional[int] = None) -> torch.Tensor:
        out = self.linear(x)
        if task_idx is not None:
            out = out + self.tora(x, task_idx)
        return out

class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_tasks: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        window_size: int = 0,
        use_rel_pos: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionWin(
            dim,
            num_heads=num_heads,
            num_tasks=num_tasks,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            window_size=(window_size, window_size) if window_size > 0 else None,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x: torch.Tensor, task_idx: Optional[int] = None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        
        # Handle window partitioning if necessary
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x, task_idx)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x, task_idx)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class AttentionWin(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_tasks: int = 1,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = ToRALinear(dim, dim * 3, num_tasks=num_tasks, bias=qkv_bias)
        self.proj = ToRALinear(dim, dim, num_tasks=num_tasks)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos and window_size is not None:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor, task_idx: Optional[int] = None) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x, task_idx).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        if self.use_rel_pos:
            pass

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x, task_idx)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_tasks: int = 1,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                num_tasks=num_tasks,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                window_size=window_size if i not in global_attn_indexes else 0,
                use_rel_pos=use_rel_pos,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor, task_idx: Optional[int] = None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x, task_idx)

        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

    def freeze_w0(self):
        # Freezes the main weights while only allowing ToRA parameters to train.
        for name, param in self.named_parameters():
            if 'tora' not in name and 'norm' not in name and 'pos_embed' not in name and 'neck' not in name:
                param.requires_grad = False

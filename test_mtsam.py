import torch
from mtsam import MTSam
from image_encoder import ImageEncoderViT
from mask_decoder import TaskDecoder
from transformer import TwoWayTransformer
from torch import nn

def test():
    embed_dim = 256
    image_size = 1024
    patch_size = 16
    
    num_tasks = 3
    # E.g. 13-classes segmentation, 1-class depth, 3-class normals
    task_channels = [13, 1, 3]

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
    
    mtsam = MTSam(
        image_encoder=image_encoder,
        task_decoders=task_decoders,
    )
    
    print("MTSAM Model initialized. Running forward pass...")
    
    image = torch.rand(3, 1024, 1024)
    batched_input = [{"image": image}]
    
    with torch.no_grad():
        for task_idx in range(num_tasks):
            outputs = mtsam(batched_input, task_idx=task_idx)
            print(f"Task {task_idx} ({task_channels[task_idx]} channels) Outputs:")
            print(f"Masks shape: {outputs['masks'].shape}")
            print(f"IoU shape: {outputs['iou_predictions'].shape}")

if __name__ == '__main__':
    test()

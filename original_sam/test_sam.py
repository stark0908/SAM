import torch
from sam import Sam
from image_encoder import ImageEncoderViT
from prompt_encoder import PromptEncoder
from mask_decoder import MaskDecoder
from transformer import TwoWayTransformer

def test():
    embed_dim = 256
    image_size = 1024
    patch_size = 16
    image_embedding_size = image_size // patch_size

    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    image_encoder = ImageEncoderViT(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim, # use 256 directly for quick test
        depth=2,
        num_heads=4,
        out_chans=embed_dim,
        window_size=14,
    )
    
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=embed_dim,
            mlp_dim=1024,
            num_heads=4,
        ),
        transformer_dim=embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )
    
    print("Model initialized. Running forward pass...")
    
    image = torch.rand(3, 1024, 1024)
    point_coords = torch.randint(low=0, high=1024, size=(1, 2, 2)).float()
    point_labels = torch.randint(low=0, high=2, size=(1, 2)).float()
    
    batched_input = [
        {
            "image": image,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }
    ]
    
    with torch.no_grad():
        outputs = sam(batched_input, multimask_output=True)
    
    print(f"Masks shape: {outputs[0]['masks'].shape}")
    print(f"IoU shape: {outputs[0]['iou_predictions'].shape}")

if __name__ == '__main__':
    test()

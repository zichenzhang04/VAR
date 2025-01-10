# 1. Download checkpoints and build models
import torchvision.transforms as T
from models import VQVAE, build_vae_var
import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
from PIL import Image
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw
import matplotlib.pyplot as plt
# disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
# disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt):
    os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt):
    os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        # hard-coded VQVAE hyperparameters
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

# Directory containing tactile images
tactile_folder = "./tactile"
tactile_images = [osp.join(tactile_folder, f) for f in os.listdir(
    tactile_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Image preprocessing: resizing to 256x256 and converting to tensor
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),  # Convert image to (C, H, W) and normalize to [0, 1]
])

# Function to reconstruct and visualize images
def visualize_reconstruction(image_paths, vae):
    i = 0
    for img_path in image_paths:
        # Load and preprocess tactile image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(
            0).to(device)  # Add batch dimension

        # Pass through VQ-VAE to reconstruct
        with torch.no_grad():
            reconstructed_img_tensor = vae(
                img_tensor)[0].cpu()  # Reconstructed image

        # Convert tensors to images
        original_img = T.ToPILImage()(img_tensor.squeeze(0))
        reconstructed_img = T.ToPILImage()(reconstructed_img_tensor.squeeze(0))

        # Visualize original and reconstructed images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original {img_path}")
        axes[0].axis("off")
        axes[1].imshow(reconstructed_img)
        axes[1].set_title(f"Reconstructed {img_path}")
        axes[1].axis("off")
        # Save the combined figure as one image
        fig.savefig(f"combined_image_{i}.png", bbox_inches='tight')  # Save figure
        i += 1


# Run visualization for all tactile images
visualize_reconstruction(tactile_images, vae)

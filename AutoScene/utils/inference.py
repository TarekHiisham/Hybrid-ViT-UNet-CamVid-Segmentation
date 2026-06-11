"""
AutoScene – Inference Utilities
================================
Handles model loading and prediction.

Key design principle:
  • All models receive a 384×384 tensor as input (internal resize).
  • Output logits are at 384×384.
  • We THEN resize the predicted mask to the ORIGINAL image dimensions
    so the overlay is always pixel-accurate to the uploaded content.
"""

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.vitunet import ViTUNet

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

IMG_SIZE   = 384   # fixed training resolution
NUM_CLASSES = 32

_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
])


# ──────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_custom_model(weights_path: str) -> torch.nn.Module:
    """Load ViTUNet from a .pth checkpoint."""
    device = get_device()
    model = ViTUNet(
        img_size=IMG_SIZE,
        in_channels=3,
        embed_dim=96,
        patch_size=6,
        n_heads=6,
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    # state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_pretrained_model(weights_path: str) -> torch.nn.Module:
    """Load pretrained U-Net (ResNet-34 backbone) from a .pth checkpoint."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch is not installed. "
            "Run: pip install segmentation-models-pytorch"
        )
    device = get_device()
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,          # weights loaded from checkpoint
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_attention_type="scse",
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

@torch.no_grad()
def predict(model: torch.nn.Module,
            image_rgb: np.ndarray) -> np.ndarray:
    """
    Run segmentation on a single RGB image.

    Args:
        model     : loaded PyTorch model
        image_rgb : uint8 numpy array [H, W, 3]  (original size)

    Returns:
        mask      : int32 numpy array [H_orig, W_orig]  – class indices
                    already resized back to the ORIGINAL image dimensions.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    device = get_device()

    # 1. Pre-process: resize to 384×384 and convert to tensor
    augmented = _TRANSFORM(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0).to(device)   # (1, 3, 384, 384)

    # 2. Forward pass → logits at 384×384
    logits = model(tensor)   # (1, 32, 384, 384)

    # 3. Resize logits to ORIGINAL resolution before argmax
    #    This preserves boundary accuracy at the original scale.
    if (orig_h, orig_w) != (IMG_SIZE, IMG_SIZE):
        logits = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

    # 4. Argmax → class mask at original size
    mask = logits.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int32)
    return mask   # [orig_h, orig_w]

# Architecture – ViT-UNet

## Overview

This project introduces a hybrid segmentation architecture that combines a **Vision Transformer (ViT) encoder** with a **U-Net decoder**. The motivation is to leverage the global attention mechanism of transformers for feature extraction, while using the proven U-Net skip-connection design for precise spatial reconstruction — a combination well-suited for dense prediction tasks like semantic segmentation.

---

## Full Pipeline

```
Input Image (any resolution)
        │
        ▼
  Resize to 384×384 (internal preprocessing)
        │
        ▼
┌─────────────────────────────────────────┐
│           ViT Encoder                   │
│                                         │
│  PatchEmbedding (patch_size=6)          │
│  → tokens: (B, 4096, 96)               │
│                                         │
│  PositionalEmbedding                    │
│                                         │
│  Stage 1: TransformerEncoderLayer ×6   ──── skip₁ (B, 96,  64, 64)
│  DownSample → (B, 2048, 192)           │
│                                         │
│  Stage 2: TransformerEncoderLayer ×4   ──── skip₂ (B, 192, 32, 32)
│  DownSample → (B, 512, 384)            │
│                                         │
│  Stage 3: TransformerEncoderLayer ×2   ──── skip₃ (B, 384, 16, 16)
│  DownSample → (B, 128, 768)            │
│                                         │
│  Bottleneck: (B, 768, 8, 8)            │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│           U-Net Decoder                 │
│                                         │
│  UpSample → cat(skip₃) → Conv+BN+ReLU │
│  UpSample → cat(skip₂) → Conv+BN+ReLU │
│  UpSample → cat(skip₁) → Conv+BN+ReLU │
│  Conv 1×1 → 32 class logits            │
│                                         │
│  Bilinear interpolate → original size  │
└─────────────────────────────────────────┘
        │
        ▼
  Output Mask (original resolution, 32 classes)
```

---

## Encoder – ViTransformerEncoder

### Patch Embedding
The input image is tokenized using a `Conv2d` with `kernel_size=patch_size=6` and `stride=6`, producing non-overlapping patches.

- Input: `(B, 3, 384, 384)`
- Output tokens: `(B, 4096, 96)` — 4096 patches, each embedded into 96 dimensions

### Positional Embedding
A learnable positional encoding tensor of shape `(1, n_patches, embed_dim)` is added to the patch tokens, initialized with truncated normal (`std=0.02`).

### Encoder Stages
Three progressive stages, each consisting of a shared `TransformerEncoderLayer` applied repeatedly, followed by a `DownSample` block.

| Stage | Layers | Input dim | Output dim | Spatial size |
|-------|--------|-----------|------------|--------------|
| 1     | ×6     | 96        | 192        | 64×64 → 32×32 |
| 2     | ×4     | 192       | 384        | 32×32 → 16×16 |
| 3     | ×2     | 384       | 768        | 16×16 → 8×8  |

**Key detail — weight sharing:** Each stage uses a single `TransformerEncoderLayer` instance applied `num_layers` times in a loop. This reduces parameter count while still benefiting from iterative refinement.

### TransformerEncoderLayer config
```python
nn.TransformerEncoderLayer(
    d_model     = embed_dim,
    nhead       = 6,
    dim_feedforward = embed_dim * 2,
    batch_first = True,
    norm_first  = True,   # Pre-LN for training stability
)
```

### DownSample Block
Spatial downsampling via `Conv2d(stride=2)` + `BatchNorm2d` + `GELU`, doubling the channel dimension at each step.

```python
Conv2d(C, 2C, kernel_size=3, stride=2, padding=1)
→ BatchNorm2d(2C)
→ GELU
```

After each stage, the feature map is reshaped from sequence format `(B, N, C)` back to spatial `(B, C, H, W)` and saved as a **skip connection**.

---

## Decoder – ViTUNetDecoder

The decoder mirrors the encoder's three stages in reverse, using transposed convolutions for upsampling and concatenating the corresponding encoder skip features.

### UpSample Block
```python
ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
```

### Stage structure (repeated ×3)
```
UpSample(C → C/2)
Concatenate with skip (→ C channels again)
Conv2d(C, C/2, 3×3) + BatchNorm2d + ReLU
```

### Output head
```python
Conv2d(C/8, 32, kernel_size=1)   # 32 = number of CamVid classes
```

### Final interpolation
After the decoder head, bilinear interpolation resizes the `32×384×384` logit map back to the **original input resolution**. This is the key step that makes the app resolution-agnostic.

---

## Training Configuration

| Hyperparameter     | Value              |
|--------------------|--------------------|
| Input resolution   | 384 × 384          |
| Patch size         | 6                  |
| Embedding dim      | 96                 |
| Attention heads    | 6                  |
| Optimizer          | Adam               |
| Learning rate      | 1e-4               |
| Loss function      | CrossEntropyLoss   |
| Epochs             | 30                 |
| Batch size         | 4                  |

### Data augmentation (training only)
```python
A.Resize(384, 384)
A.HorizontalFlip(p=0.5)
A.ColorJitter(brightness=0.1, p=0.5)
A.ToFloat(max_value=255.0)
```

---

## Comparison Model – Pretrained U-Net

For benchmarking, a U-Net with a **ResNet-34 backbone** (pretrained on ImageNet) was fine-tuned on CamVid using the `segmentation_models_pytorch` library.

```python
smp.Unet(
    encoder_name          = "resnet34",
    encoder_weights       = "imagenet",
    in_channels           = 3,
    classes               = 32,
    decoder_attention_type = "scse",   # Squeeze-and-Excitation attention
)
```

Same optimizer, loss, and training schedule as the custom model, to ensure a fair comparison.

---

## Resolution Handling at Inference

Both models were trained at 384×384. At inference time:

1. Input image/frame is resized to 384×384 for the forward pass.
2. Output logits `(1, 32, 384, 384)` are upsampled via bilinear interpolation to the **original image dimensions** before argmax.
3. The segmentation mask and overlay are rendered at the original resolution.

This ensures the app works correctly on any input size without quality loss.

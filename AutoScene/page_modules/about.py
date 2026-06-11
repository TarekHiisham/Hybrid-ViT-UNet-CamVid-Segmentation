"""
AutoScene – About Page
"""

import streamlit as st


def show():
    st.title("ℹ️ About AutoScene")

    st.markdown("""
    ## What is AutoScene?

    AutoScene is a semantic segmentation platform built for autonomous driving scenes.
    It was developed as an advanced deep learning project comparing a **custom-designed
    ViT-UNet** architecture against a **pretrained U-Net** with a ResNet-34 backbone.

    ---

    ## Architecture

    ### ViT-UNet (Custom Model)
    A Vision Transformer encoder coupled with a U-Net-style decoder featuring skip connections.

    ```
    Input Image
        │
        ▼
    PatchEmbedding (patch_size=6)
        │
        ▼
    PositionalEmbedding
        │
        ├──▶ EncoderStage 1 (6 layers, dim=96)  ──┐ skip 1
        │        │ DownSample
        ├──▶ EncoderStage 2 (4 layers, dim=192) ──┤ skip 2
        │        │ DownSample
        └──▶ EncoderStage 3 (2 layers, dim=384) ──┘ skip 3
                 │ DownSample
                 ▼
            Bottleneck (dim=768, spatial=8×8)
                 │
        ┌────────┘ + skip 3
        ▼
    UpSample → Conv → BN → ReLU
        │
        ┌────────────── + skip 2
        ▼
    UpSample → Conv → BN → ReLU
        │
        ┌────────────── + skip 1
        ▼
    UpSample → Conv → BN → ReLU
        │
        ▼
    Conv 1×1 → 32 classes
        │
        ▼
    Bilinear Interpolation → Original Resolution
    ```

    ### Pretrained U-Net (ResNet-34 backbone)
    Uses `segmentation_models_pytorch` with scSE attention in the decoder.
    The encoder (ResNet-34) was initialised with ImageNet weights before fine-tuning on CamVid.

    ---

    ## Dataset – CamVid

    | Split  | Images |
    |--------|--------|
    | Train  | 369    |
    | Val    | 100    |
    | Test   | 232    |

    - **32 semantic classes** (Sky, Road, Car, Pedestrian, Tree, Building, …)
    - Input resolution during training: **384 × 384**
    - Inference is size-agnostic: images/videos are processed at their original resolution.

    ---

    ## Resolution Handling

    During training all images were resized to **384 × 384**.  
    At inference time AutoScene:
    1. Resizes the input to 384 × 384 internally for the forward pass.
    2. Upsamples the logits **back to the original image/video resolution** before argmax.
    3. Renders the overlay at the original resolution — no quality loss from double-rescaling.

    """)

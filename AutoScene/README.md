# 🚗 AutoScene – Autonomous Scene Segmentation

A Streamlit application for semantic segmentation of autonomous driving scenes,
comparing a **custom ViT-UNet** architecture against a **Pretrained U-Net** (ResNet-34),
both trained on the **CamVid dataset** (32 classes).

---

## Project Structure

```
AutoScene/
├── app.py                    # Streamlit entry point
├── requirements.txt
│
├── models/
│   └── vitunet.py            # ViT Encoder + U-Net Decoder architecture
│
├── utils/
│   ├── camvid_classes.py     # 32-class palette, overlay blending
│   ├── inference.py          # Model loaders + predict() at original resolution
│   ├── model_cache.py        # Streamlit-cached model loading + selector UI
│   └── video_processing.py   # Video frame sampling and output writing
│
├── pages/
│   ├── home.py               # Landing page
│   ├── image_seg.py          # Image segmentation page
│   ├── video_seg.py          # Video segmentation page
│   ├── class_legend.py       # CamVid class colour legend
│   └── about.py              # Architecture & project info
│
├── weights/
    ├── custom/               
    └── pretrained/          

---

## Quick Start

### 1. Create environment

```bash
conda create -n autoscene python=3.10 -y
conda activate autoscene
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For **GPU acceleration** (recommended), replace the `torch` line in
`requirements.txt` with a CUDA wheel from https://pytorch.org/get-started/locally/

### 3. Run the app

```bash
streamlit run app.py
```

---

## Features

| Feature | Details |
|---------|---------|
| **Image segmentation** | Upload any image (any resolution) → instant segmentation overlay |
| **Model selection** | Switch between ViT-UNet and Pretrained U-Net per session |
| **Overlay control** | Adjustable opacity slider (0.1 – 0.9) |
| **Mask-only view** | Toggle to see pure segmentation mask or 3-panel comparison |
| **Class breakdown** | Per-image class detection with pixel percentages |
| **Download** | Export overlay PNG and mask PNG |
| **Video segmentation** | Upload MP4/AVI/MOV → get annotated video back |
| **Frame sampling** | Process every N-th frame for speed control |
| **Class legend** | Visual palette of all 32 CamVid classes |
| **Original resolution** | All outputs rendered at the input image/video resolution |

---

## Resolution Handling

Training used **384 × 384** images. At inference:

1. Input is resized to 384 × 384 internally.
2. Logits are upsampled back to the **original resolution** before argmax.
3. The overlay is rendered at the original size — pixel-perfect at any input size.

---

## Checkpoint Format

Both checkpoints are saved as dictionaries:

```python
{
    'epoch': ...,
    'model_state_dict': ...,   # ← used by the loader
    'optimizer_state_dict': ...,
    'train_loss': ...,
    'val_loss':   ...,
    ...
}
```

The loaders also accept a raw `state_dict` (just the weights tensor dict) for compatibility.

"""
AutoScene – Shared Model Cache
================================
Centralises model loading with Streamlit caching
so weights are only read from disk once per session.
"""

import os
import streamlit as st

CUSTOM_WEIGHTS_DIR    = "weights/custom"
PRETRAINED_WEIGHTS_DIR = "weights/pretrained"


def list_weight_files(directory: str) -> list[str]:
    """Return sorted list of .pth files in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory) if f.endswith(".pth")
    )


@st.cache_resource(show_spinner="Loading ViT-UNet model …")
def get_custom_model(weights_filename: str):
    from utils.inference import load_custom_model
    path = os.path.join(CUSTOM_WEIGHTS_DIR, weights_filename)
    return load_custom_model(path)


@st.cache_resource(show_spinner="Loading Pretrained U-Net model …")
def get_pretrained_model(weights_filename: str):
    from utils.inference import load_pretrained_model
    path = os.path.join(PRETRAINED_WEIGHTS_DIR, weights_filename)
    return load_pretrained_model(path)


def model_selector(sidebar: bool = True) -> tuple:
    """
    Render model-choice UI.
    Returns (model_object, model_label_str) or (None, None) if no weights found.
    """
    container = st.sidebar if sidebar else st

    container.subheader("🤖 Model Selection")

    model_choice = container.radio(
        "Choose a model",
        ["ViT-UNet (Custom)", "Pretrained U-Net (ResNet-34)"],
        key="model_radio",
    )

    model = None
    label = None

    if model_choice == "ViT-UNet (Custom)":
        files = list_weight_files(CUSTOM_WEIGHTS_DIR)
        if not files:
            container.warning(
                f"No `.pth` files found in `{CUSTOM_WEIGHTS_DIR}/`.\n\n"
                "Place your trained checkpoint there and refresh."
            )
            return None, None
        chosen = container.selectbox("Checkpoint", files, key="custom_ckpt")
        model  = get_custom_model(chosen)
        label  = f"ViT-UNet · {chosen}"

    else:
        files = list_weight_files(PRETRAINED_WEIGHTS_DIR)
        if not files:
            container.warning(
                f"No `.pth` files found in `{PRETRAINED_WEIGHTS_DIR}/`.\n\n"
                "Place your pretrained checkpoint there and refresh."
            )
            return None, None
        chosen = container.selectbox("Checkpoint", files, key="pretrained_ckpt")
        model  = get_pretrained_model(chosen)
        label  = f"Pretrained U-Net · {chosen}"

    return model, label

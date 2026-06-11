"""
AutoScene – Image Segmentation Page
"""

import io
import numpy as np
from PIL import Image
import streamlit as st

from utils.model_cache import model_selector
from utils.inference import predict
from utils.camvid_classes import mask_to_color, overlay_mask, PALETTE, CAMVID_CLASSES, NUM_CLASSES


def show():
    st.title("🖼️ Image Segmentation")
    st.caption("Upload a scene image — the model segments it and overlays the result at the original resolution.")

    # ── Model picker (sidebar) ─────────────────────────────────────────────
    model, model_label = model_selector(sidebar=True)

    # ── Upload control ─────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG / BMP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    # ── Overlay settings ───────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("🎨 Display Options")
        alpha = st.slider(
            "Overlay opacity", min_value=0.1, max_value=0.9,
            value=0.45, step=0.05,
            help="How strongly the segmentation colours show over the original image."
        )
        show_mask_only = st.checkbox("Show mask only (no original)", value=False)

    # ── Run segmentation ───────────────────────────────────────────────────
    if uploaded is not None and model is not None:
        # Load image, keep original size
        pil_img   = Image.open(uploaded).convert("RGB")
        img_array = np.array(pil_img)          # uint8 [H, W, 3]
        orig_h, orig_w = img_array.shape[:2]

        st.markdown(
            f"**Uploaded:** `{uploaded.name}` · "
            f"{orig_w}×{orig_h} px · "
            f"Model: _{model_label}_"
        )

        with st.spinner("Segmenting …"):
            mask = predict(model, img_array)   # [orig_h, orig_w] at original size

        color_mask = mask_to_color(mask)       # [orig_h, orig_w, 3]
        blended    = overlay_mask(img_array, mask, alpha=alpha)

        # ── Display ────────────────────────────────────────────────────────
        if show_mask_only:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(img_array, use_container_width=True)
            with col2:
                st.subheader("Segmentation Mask")
                st.image(color_mask, use_container_width=True)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Original")
                st.image(img_array, use_container_width=True)
            with col2:
                st.subheader("Segmentation Mask")
                st.image(color_mask, use_container_width=True)
            with col3:
                st.subheader("Overlay")
                st.image(blended, use_container_width=True)

        # ── Class breakdown ────────────────────────────────────────────────
        with st.expander("📊 Detected classes in this image"):
            unique_classes = np.unique(mask)
            cols = st.columns(4)
            for i, cls_idx in enumerate(unique_classes):
                if cls_idx < len(CAMVID_CLASSES):
                    name = CAMVID_CLASSES[cls_idx][0]
                    r, g, b = PALETTE[cls_idx]
                    pixel_count = int(np.sum(mask == cls_idx))
                    pct = 100 * pixel_count / (orig_h * orig_w)
                    cols[i % 4].markdown(
                        f'<span style="'
                        f'background:rgb({r},{g},{b});'
                        f'padding:2px 8px;border-radius:4px;'
                        f'color:{"#000" if (int(r)+int(g)+int(b)) > 380 else "#fff"}'
                        f'">{name}</span> {pct:.1f}%',
                        unsafe_allow_html=True,
                    )

        # ── Download buttons ───────────────────────────────────────────────
        st.divider()
        dl_col1, dl_col2 = st.columns(2)

        def pil_to_bytes(arr):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()

        with dl_col1:
            st.download_button(
                "⬇️ Download overlay",
                data=pil_to_bytes(blended),
                file_name="autoscene_overlay.png",
                mime="image/png",
            )
        with dl_col2:
            st.download_button(
                "⬇️ Download mask",
                data=pil_to_bytes(color_mask),
                file_name="autoscene_mask.png",
                mime="image/png",
            )

    elif uploaded is not None and model is None:
        st.warning("Please place model weights in the appropriate folder and refresh.")
    else:
        st.info("⬆️ Upload an image to get started.")

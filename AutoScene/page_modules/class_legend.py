"""
AutoScene – Class Legend Page
"""

import numpy as np
import streamlit as st
from PIL import Image
import io

from utils.camvid_classes import CAMVID_CLASSES, PALETTE, NUM_CLASSES


def color_swatch(r, g, b, size=40) -> bytes:
    """Return a small solid-color PNG as bytes."""
    arr = np.full((size, size, 3), [r, g, b], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def show():
    st.title("CamVid Class Legend")
    st.caption(
        "AutoScene was trained on the CamVid dataset with 32 semantic classes. "
        "Each class is assigned a unique colour for visualization."
    )

    st.divider()

    # Build a full-palette strip
    palette_strip = np.zeros((60, NUM_CLASSES * 30, 3), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        palette_strip[:, i*30:(i+1)*30] = PALETTE[i]
    st.image(palette_strip, caption="All 32 class colours", use_container_width=True)

    st.divider()

    # Table of classes
    cols = st.columns([1, 3, 2])
    cols[0].markdown("**Colour**")
    cols[1].markdown("**Class Name**")
    cols[2].markdown("**RGB**")

    seen_names = set()
    for idx, cls_info in enumerate(CAMVID_CLASSES):
        name = cls_info[0]
        r, g, b = cls_info[1], cls_info[2], cls_info[3]

        # De-duplicate display (notebook had some repeated names)
        display_name = name if name not in seen_names else f"{name} ({idx})"
        seen_names.add(name)

        cols = st.columns([1, 3, 2])
        swatch = color_swatch(r, g, b, size=28)
        cols[0].image(swatch, width=28)
        cols[1].write(f"**{idx}** · {display_name}")
        cols[2].write(f"`({r}, {g}, {b})`")

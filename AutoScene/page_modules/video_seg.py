"""
AutoScene – Video Segmentation Page
"""

import os
import tempfile
import numpy as np
import streamlit as st

from utils.model_cache import model_selector
from utils.video_processing import process_video, get_video_info


def show():
    st.title("🎬 Video Segmentation")
    st.caption(
        "Upload a driving video. AutoScene samples frames, runs segmentation "
        "on each, and exports a full annotated video with the overlay "
        "at the original resolution."
    )

    # ── Model picker ───────────────────────────────────────────────────────
    model, model_label = model_selector(sidebar=True)

    # ── Upload ─────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a video (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    # ── Settings ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("🎨 Display Options")
        alpha = st.slider(
            "Overlay opacity", 0.1, 0.9, 0.45, 0.05,
            help="Segmentation colour strength over the original frame."
        )
        st.subheader("⚙️ Processing Options")
        sample_every = st.slider(
            "Sample every N frames", 1, 10, 1,
            help="1 = process every frame (slow). 2 = every other frame (faster). "
                 "Higher values speed up processing but reduce temporal smoothness."
        )

    # ── Run ────────────────────────────────────────────────────────────────
    if uploaded is not None and model is not None:

        # Save upload to a temp file so OpenCV can open it
        suffix = os.path.splitext(uploaded.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        info = get_video_info(tmp_path)
        effective_fps = info["fps"] / max(sample_every, 1)

        st.markdown(
            f"**Uploaded:** `{uploaded.name}` · "
            f"{info['width']}×{info['height']} px · "
            f"{info['total_frames']} frames · "
            f"{info['fps']:.1f} fps · "
            f"Model: _{model_label}_"
        )
        st.info(
            f"Processing **{max(1, info['total_frames'] // sample_every)}** frames "
            f"(every {sample_every} frame(s)) → output at **{effective_fps:.1f} fps**."
        )

        if st.button("▶️ Run Video Segmentation", type="primary"):
            progress_bar = st.progress(0, text="Starting …")
            status_text  = st.empty()
            total_frames = max(1, info["total_frames"] // sample_every)

            def _cb(current, total):
                pct = min(current / total, 1.0)
                progress_bar.progress(pct, text=f"Frame {current}/{total}")
                status_text.caption(f"Processing frame {current} of {total} …")

            with st.spinner("Segmenting video …"):
                output_path = process_video(
                    video_path=tmp_path,
                    model=model,
                    alpha=alpha,
                    sample_every=sample_every,
                    progress_callback=_cb,
                )

            progress_bar.progress(1.0, text="Done ✅")
            status_text.empty()
            st.success("Video segmentation complete!")

            # ── Preview & download ─────────────────────────────────────────
            st.subheader("Segmented Video Preview")
            with open(output_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes, format="video/mp4")

            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download segmented video",
                    data=f.read(),
                    file_name="autoscene_segmented.mp4",
                    mime="video/mp4",
                )

        # Cleanup temp input file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    elif uploaded is not None and model is None:
        st.warning("Please place model weights in the appropriate folder and refresh.")
    else:
        st.info("⬆️ Upload a video to get started.")

    # ── Tips ──────────────────────────────────────────────────────────────
    with st.expander("💡 Tips for faster processing"):
        st.markdown("""
        - **Sample every 2–4 frames** for a good speed/quality trade-off on long clips.
        - Using **CUDA** (GPU) speeds up inference significantly. The app auto-detects it.
        - Short clips (< 30 s) process quickly even at sample_every = 1.
        - The output video preserves the **original resolution** of your upload.
        """)
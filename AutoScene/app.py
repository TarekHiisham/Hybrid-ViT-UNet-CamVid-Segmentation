"""
AutoScene – Autonomous Scene Segmentation
==========================================
Main Streamlit entry point.
Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AutoScene",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.image("assets/logo.png", use_container_width=True) if __import__("os").path.exists("assets/logo.png") else None
st.sidebar.title("🚗 AutoScene")
st.sidebar.caption("Autonomous Scene Segmentation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home",
     "🖼️ Image Segmentation",
     "🎬 Video Segmentation",
     "📊 Class Legend",
     "ℹ️ About"],
    label_visibility="collapsed",
)

st.sidebar.divider()
# ── Page routing ───────────────────────────────────────────────────────────
if page == "🏠 Home":
    from page_modules import home
    home.show()

elif page == "🖼️ Image Segmentation":
    from page_modules import image_seg
    image_seg.show()

elif page == "🎬 Video Segmentation":
    from page_modules import video_seg
    video_seg.show()

elif page == "📊 Class Legend":
    from page_modules import class_legend
    class_legend.show()

elif page == "ℹ️ About":
    from page_modules import about
    about.show()

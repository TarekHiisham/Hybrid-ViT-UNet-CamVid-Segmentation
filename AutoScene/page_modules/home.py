import streamlit as st


def show():
    st.title("AutoScene")
    st.subheader("Autonomous Scene Segmentation Platform")

    st.markdown("""
    AutoScene applies semantic segmentation to real-world driving scenes using two models:

    | Model | Architecture | Backbone |
    |-------|-------------|---------|
    | **ViT-UNet** | Custom ViT Encoder + U-Net Decoder | Trained from scratch |
    | **Pretrained U-Net** | U-Net | ResNet-34 (ImageNet) |

    Both models were trained on the **CamVid** dataset with **32 semantic classes**.

    ---
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### Image Segmentation\nUpload a scene image and get instant segmentation with an overlay view.")

    with col2:
        st.info("### Video Segmentation\nUpload a driving video. AutoScene samples frames, segments them, and returns a full annotated video.")

    with col3:
        st.info("### Class Legend\nExplore all 32 CamVid classes and their assigned colors.")

    st.markdown("---")
    st.markdown("**Getting started** → place your `.pth` weight files in `weights/custom/` or `weights/pretrained/`, then choose a page from the sidebar.")

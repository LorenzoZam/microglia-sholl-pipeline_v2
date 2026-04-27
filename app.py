import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_image_coordinates import streamlit_image_coordinates
from morphology_features import math_morphology_pipeline, calculate_sholl_profile

st.set_page_config(page_title="Sholl Analysis App", layout="wide", page_icon="🔬")
sns.set_theme(style="ticks", context="talk", palette="colorblind")

st.title("🔬 Automated Sholl Analysis Pipeline")
st.markdown("Try out the full morphometric extraction pipeline natively on your browser. Upload a 2D image of a single cell (e.g. microglia, neuron) to trace its skeleton and calculate intersections.")

uploaded_file = st.file_uploader("Upload Image (TIFF/PNG/JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image bytes accurately
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
         st.error("Error loading image. Is it a valid 8-bit image?")
         st.stop()

    st.sidebar.header("Pipeline Settings")
    h_val = st.sidebar.slider("Denoising Strength (h)", min_value=1, max_value=30, value=11, step=1)
    step_size = st.sidebar.number_input("Sholl Step Size (px)", min_value=1, max_value=50, value=10)
    
    st.markdown("### Step 1: Preprocessing & Skeletonization")
    col1, col2 = st.columns(2)
    
    # Process Image using the exact same backend function from the main pipeline
    mask, skeleton, dt = math_morphology_pipeline(img_gray, h_val)
    
    with col1:
        st.image(img_gray, caption="Original Image", use_container_width=True)
    with col2:
        # Create a red skeleton overlay
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[skeleton > 0] = [255, 0, 0]
        st.image(overlay, caption=f"Skeletonized Trace (h={h_val})", use_container_width=True)

    st.markdown("### Step 2: Soma Selection")
    st.info("Click directly on the image below to select the centroid (soma). The Sholl graph will calculate instantly.")
    
    # Render the interactive image component
    value = streamlit_image_coordinates(overlay, key="soma_selector")
    
    if value is not None:
        sx, sy = value["x"], value["y"]
        st.success(f"✅ Soma Coordinates Registered: (X: {sx}, Y: {sy})")
        
        st.markdown("### Step 3: Sholl Profile Results")
        with st.spinner("Extracting morphometrics..."):
            # Calculate Sholl using the core algorithm
            radii, intersections = calculate_sholl_profile(skeleton, (sx, sy), step_size)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(radii, intersections, marker='o', linewidth=3, color='teal', markersize=6)
            ax.fill_between(radii, 0, intersections, color='teal', alpha=0.2)
            ax.set_xlabel("Distance from Soma (px)", fontweight='bold')
            ax.set_ylabel("Number of Intersections", fontweight='bold')
            ax.set_title("Single Cell Sholl Curve", fontweight='bold', pad=15)
            sns.despine()
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            
            # Simple metrics
            max_r = max(radii) if len(radii) > 0 else 0
            max_int = max(intersections) if len(intersections) > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Radius (px)", max_r)
            c2.metric("Max Intersections", max_int)
            c3.metric("Total Intersections", sum(intersections))
            st.balloons()

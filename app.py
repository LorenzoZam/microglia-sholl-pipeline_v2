import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_image_coordinates import streamlit_image_coordinates
from skimage.morphology import skeletonize, remove_small_objects

st.set_page_config(page_title="Sholl Analysis App", layout="wide", page_icon="🔬")
sns.set_theme(style="ticks", context="talk", palette="colorblind")

@st.cache_data
def process_pipeline(img_gray, h_val):
    # Denoise
    denoised = cv2.fastNlMeansDenoising(img_gray, None, h=h_val, templateWindowSize=7, searchWindowSize=21)
    
    # Tophat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    
    # Binarize
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean small fragments
    cleaned = remove_small_objects(binary.astype(bool), min_size=3)
    
    # Skeletonize
    skel = skeletonize(cleaned)
    return (cleaned.astype(np.uint8) * 255), (skel.astype(np.uint8) * 255)

@st.cache_data
def calc_sholl(skeleton, center, step_size):
    y_coords, x_coords = np.where(skeleton > 0)
    if len(y_coords) == 0:
        return [], []
        
    cx, cy = center
    dists = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    max_dist = int(np.max(dists))
    
    radii = list(range(step_size, max_dist + step_size, step_size))
    intersections = []
    
    for r in radii:
        mask = np.zeros_like(skeleton)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, 1)
        
        intersect = cv2.bitwise_and(skeleton, mask)
        _, labeled, _, _ = cv2.connectedComponentsWithStats(intersect, connectivity=8)
        
        intersections.append(labeled - 1 if labeled > 0 else 0)
        
    return radii, intersections

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
    
    mask, skeleton = process_pipeline(img_gray, h_val)
    
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
            radii, intersections = calc_sholl(skeleton, (sx, sy), step_size)
            
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

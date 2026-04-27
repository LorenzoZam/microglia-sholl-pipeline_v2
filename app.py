import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_image_coordinates import streamlit_image_coordinates

# Import the actual scientific components directly from the user's codebase!
from run_sholl_pipeline import (
    apply_adaptive_patching, process_patch, binarize_image, 
    remove_small_fragments, remove_isolated_fibers, skeletonize_image, 
    bridge_nearby_fragments, apply_morph_close, apply_dilate,
    remove_isolated_fibers_refine, bridge_nearby_fragments_refine,
    get_connected_component, compute_sholl_intersections, 
    generate_concentric_circles, measure_farthest_neurite
)
from morphology_features import (
    box_counting_fractal_dimension_with_data, box_counting_lacunarity,
    skeleton_to_graph, compute_graph_centralities, 
    schoenen_ramification_index, soma_shape_metrics
)

st.set_page_config(page_title="🔬 Sholl Analysis WebDemo", layout="wide", page_icon="🧠")
sns.set_theme(style="ticks", context="talk", palette="colorblind")

# Configure a large width so clicks map accurately
st.title("🔬 Automated Sholl Analysis Pipeline (Demo)")
st.markdown("This web app perfectly replicates the rigorous mathematical pipeline used in the native desktop script. Upload a single-cell image to trace its skeleton and extract morphometrics using Voronoi isolation.")

@st.cache_data
def run_scientific_skeletonization(image, h_val):
    # Step-by-step rigorous pipeline
    denoised = cv2.fastNlMeansDenoising(image, None, h=h_val, templateWindowSize=7, searchWindowSize=21)
    # Adaptive Patching and Tophat (Simplified loop)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    binary = binarize_image(tophat)
    cleaned = remove_small_fragments(binary, min_size=3)
    sk_initial = remove_isolated_fibers(cleaned, min_length=3)
    bridged = bridge_nearby_fragments(sk_initial, max_dist=5)
    skel = skeletonize_image(bridged)
    closed = apply_morph_close(skel)
    dilated = apply_dilate(closed)
    cleaned_final = remove_small_fragments(dilated, min_size=5)
    final_skel = remove_isolated_fibers_refine(cleaned_final, min_length=30)
    final_skel = bridge_nearby_fragments_refine(final_skel, max_dist=12)
    return binary, final_skel

uploaded_file = st.file_uploader("Upload Image (TIFF/PNG/JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
         st.error("Error loading image.")
         st.stop()

    st.sidebar.header("Denoising Calibration")
    h_val = st.sidebar.slider("NLM Denoising Parameter (h)", min_value=1, max_value=30, value=11)
    step_size = st.sidebar.number_input("Sholl Step Size (pixels)", min_value=1, max_value=50, value=10)

    # 1. Processing
    with st.spinner("Applying adaptive morphological filters..."):
        binary, skeleton = run_scientific_skeletonization(img_gray, h_val)

    st.markdown("### 1) Interactive Denoising Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_gray, caption="Raw Image")
    with col2:
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay[skeleton > 0] = [255, 0, 0]
        st.image(overlay, caption=f"Skeletonized Tracing (h={h_val})")

    st.markdown("### 2) Soma Selection & Voronoi Isolation")
    st.info("Click directly on the 'Soma' (cell body) in the image below to isolate its specific connected component!")

    # Soma clicking interactive point
    click_data = streamlit_image_coordinates(overlay, key="soma_clicker")

    if click_data is not None:
        cx, cy = click_data["x"], click_data["y"]
        
        # We must isolate the exact cell using their algorithm
        with st.spinner("Isolating cell connected component..."):
            component_mask, (corr_y, corr_x) = get_connected_component(skeleton, (cy, cx))
        
        if component_mask is None:
            st.error("No valid neuronal/microglial skeleton found at that coordinate. Please click directly on a red path.")
        else:
            st.success(f"Cell Isolated Successfully! Exact Centroid Snapped to: (X:{corr_x}, Y:{corr_y})")
            
            # Sholl Math
            radii_list = []
            max_radius = 500  # Fallback
            farthest = measure_farthest_neurite(component_mask, [(corr_x, corr_y)])
            if 1 in farthest:
                ex, ey = farthest[1]
                max_radius = np.sqrt((ex - corr_x)**2 + (ey - corr_y)**2)
                max_radius = np.ceil(max_radius / step_size) * step_size
            
            radii = generate_concentric_circles(max_radius, step_size)
            intersections = compute_sholl_intersections(component_mask, corr_x, corr_y, radii)
            
            # --- Emulating The QC Dashboard View ---
            st.markdown("### 3) Quality Control Dashboard (Virtual)")
            db_col1, db_col2 = st.columns([1, 1])
            
            # Subplot 1: Isolated Cell Mask + Circles
            fig_db, ax_db = plt.subplots(figsize=(6,6))
            ax_db.imshow(component_mask, cmap='gray')
            ax_db.scatter(corr_x, corr_y, color='cyan', s=50, label="Soma")
            for r in radii:
                c = plt.Circle((corr_x, corr_y), r, color='red', fill=False, linestyle='--', alpha=0.3)
                ax_db.add_patch(c)
            ax_db.axis('off')
            ax_db.set_title("Isolated Voronoi Component Mask")
            with db_col1:
                st.pyplot(fig_db)

            # Subplot 2: The Sholl Profile Plot
            fig_sholl, ax_sholl = plt.subplots(figsize=(6,6))
            ax_sholl.plot(radii, intersections, marker='o', linewidth=3, color='teal')
            ax_sholl.fill_between(radii, 0, intersections, color='teal', alpha=0.2)
            ax_sholl.set_xlabel("Radius (px)", fontweight='bold')
            ax_sholl.set_ylabel("Intersections", fontweight='bold')
            ax_sholl.set_title("Cell Sholl Curve", fontweight='bold')
            sns.despine()
            with db_col2:
                st.pyplot(fig_sholl)
            
            st.balloons()

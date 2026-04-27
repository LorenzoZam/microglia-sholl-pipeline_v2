import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_image_coordinates import streamlit_image_coordinates

# Import the actual scientific components directly from the user's codebase!
from run_sholl_pipeline import (
    apply_adaptive_patching, process_patch, apply_tophat, binarize_image, 
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

st.title("🔬 Automated Sholl Analysis Pipeline (Demo)")
st.markdown("This web app perfectly replicates the rigorous mathematical pipeline used in the native desktop script. Upload a single-cell image to trace its skeleton and extract morphometrics using Voronoi isolation.")

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if 'soma_points' not in st.session_state:
    st.session_state.soma_points = []
if 'ui_mode' not in st.session_state:
    st.session_state.ui_mode = 'selecting'
if 'last_click1' not in st.session_state:
    st.session_state.last_click1 = None
if 'last_click2' not in st.session_state:
    st.session_state.last_click2 = None

def undo_last():
    if st.session_state.soma_points:
        st.session_state.soma_points.pop()

def clear_all():
    st.session_state.soma_points = []
    
def accept_all():
    st.session_state.ui_mode = 'qc'
    
def restart():
    st.session_state.soma_points = []
    st.session_state.ui_mode = 'selecting'

# -------------------------------------------------------------
# CORE ALGORITHM
# -------------------------------------------------------------
@st.cache_data
def run_scientific_skeletonization(image, h_val):
    # Strictly matched from main() in run_sholl_pipeline.py
    global_var = np.var(image)
    processed_image = apply_adaptive_patching(image, global_var)
    
    # NLM happens here equivalent to preview_denoising
    denoised = cv2.fastNlMeansDenoising(processed_image, None, h=h_val, templateWindowSize=7, searchWindowSize=21)
    
    den_th_image = apply_tophat(denoised)
    binary = binarize_image(den_th_image)
    frag_filtered = remove_small_fragments(binary)
    closed = apply_morph_close(frag_filtered)
    dilated = apply_dilate(closed)
    skel = skeletonize_image(dilated)
    bridged = bridge_nearby_fragments(skel)
    refined = remove_isolated_fibers_refine(bridged)
    bridge_refined = bridge_nearby_fragments_refine(refined)
    skeleton = remove_isolated_fibers(bridge_refined)
    
    return processed_image, binary, skeleton

# -------------------------------------------------------------
# MAIN APP BODY
# -------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Image (TIFF/PNG/JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
         st.error("Error loading image.")
         st.stop()

    st.sidebar.header("Denoising Calibration")
    h_val = st.sidebar.slider("NLM Parameter (h)", min_value=1, max_value=30, value=11, disabled=(st.session_state.ui_mode == 'qc'))
    step_size = st.sidebar.number_input("Sholl Step Size (px)", min_value=1, max_value=50, value=4)

    with st.spinner("Applying rigorous morphological filters..."):
        processed_image, binary, skeleton = run_scientific_skeletonization(img_gray, h_val)

    if st.session_state.ui_mode == 'selecting':
        st.markdown("### Step 2: Interactive Cell Detection")
        st.info("Click directly on ANY of the two images below to register a Soma coordinate. You can accumulate multiple cells exactly like in Matplotlib!")
        
        # Prepare Interactive Overlays
        ov_left = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        ov_right = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        ov_right[skeleton > 0] = [255, 0, 0] # Skeleton overlay
        
        # Draw all historic clicks
        for idx, (cx, cy) in enumerate(st.session_state.soma_points, start=1):
            cv2.circle(ov_left, (int(cx), int(cy)), 10, (0, 255, 255), -1)
            cv2.circle(ov_right, (int(cx), int(cy)), 10, (0, 255, 255), -1)
            cv2.putText(ov_left, str(idx), (int(cx)+15, int(cy)+15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(ov_right, str(idx), (int(cx)+15, int(cy)+15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Scale display images to avoid scrollbars without using CSS overrides that break coordinate mapping
        DISPLAY_WIDTH = 700
        scale_ratio = min(1.0, DISPLAY_WIDTH / float(ov_left.shape[1])) # Scale down only if larger than 700px
        new_dim = (int(ov_left.shape[1] * scale_ratio), int(ov_left.shape[0] * scale_ratio))
        
        ov_left_small = cv2.resize(ov_left, new_dim, interpolation=cv2.INTER_AREA)
        ov_right_small = cv2.resize(ov_right, new_dim, interpolation=cv2.INTER_AREA)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Processed Image**")
            val1 = streamlit_image_coordinates(ov_left_small, key="img1")
        with col2:
            st.markdown("**Skeletonized Tracing**")
            val2 = streamlit_image_coordinates(ov_right_small, key="img2")
            
        # Detect new clicks
        if val1 is not None and val1 != st.session_state.last_click1:
            real_x = int(val1['x'] / scale_ratio)
            real_y = int(val1['y'] / scale_ratio)
            st.session_state.soma_points.append((real_x, real_y))
            st.session_state.last_click1 = val1
            st.rerun()
        if val2 is not None and val2 != st.session_state.last_click2:
            real_x = int(val2['x'] / scale_ratio)
            real_y = int(val2['y'] / scale_ratio)
            st.session_state.soma_points.append((real_x, real_y))
            st.session_state.last_click2 = val2
            st.rerun()
            
        # Replicated Matplotlib Menu
        b1, b2, b3 = st.columns([1,1,2])
        b1.button("Undo Last ↺", on_click=undo_last)
        b2.button("Clear All ✗", on_click=clear_all)
        b3.button(f"Accept Somas ({len(st.session_state.soma_points)}) & Continue ✓", on_click=accept_all, type="primary")

    elif st.session_state.ui_mode == 'qc':
        st.markdown("### Step 3: Quality Control Dashboard")
        st.button("← Back to Soma Selection", on_click=restart)
        
        if not st.session_state.soma_points:
            st.warning("No cells selected.")
            
        farthest_endpoints = measure_farthest_neurite(skeleton, st.session_state.soma_points)
        all_cells_sholl_data = []  # Accumulate ALL intersection data for the final plot
        
        for idx, (raw_x, raw_y) in enumerate(st.session_state.soma_points, start=1):
            st.divider()
            st.subheader(f"🧠 Analysis for Cell {idx}")
            
            # 1. Isolation using inverse tuple convention (Y, X) for spatial geometry
            comp_mask, (corr_x, corr_y) = get_connected_component(skeleton, (raw_y, raw_x))
            
            if comp_mask is None:
                st.error(f"Soma {idx} Invalid: Unconnected Background Space.")
                continue
                
            # 2. Extract Sholl Max Radii logic
            radii_list = []
            max_radius = 500
            
            farthest = farthest_endpoints.get(idx)
            if farthest:
                ex, ey = farthest
                max_radius = np.sqrt((ex - corr_x)**2 + (ey - corr_y)**2)
                max_radius = np.ceil(max_radius / step_size) * step_size
                
            additional_r = np.arange(max_radius, max_radius + (3 * step_size), step_size)
            base_r = generate_concentric_circles(max_radius, step_size)
            radii = np.unique(np.concatenate([base_r, additional_r]))
            
            intersections = compute_sholl_intersections(comp_mask, corr_x, corr_y, radii)
            
            # --- 3. EXHAUSTIVE MORPHOMETRIC EXTRACTION ---
            with st.spinner(f"Extracting advanced morphometrics for Cell {idx}..."):
                fd, _log_sizes, _log_counts = box_counting_fractal_dimension_with_data(comp_mask)
                lac = box_counting_lacunarity(comp_mask)
                import networkx as nx
                G = skeleton_to_graph(comp_mask)
                betw, clos = compute_graph_centralities(G)
                sri = schoenen_ramification_index(intersections, radii, (corr_x, corr_y), comp_mask)
                soma_area, soma_circ = soma_shape_metrics(binary, (corr_x, corr_y))
            
            # --- 4. EXACT PIPELINE DASHBOARD REPLICATION ---
            # Metric Card
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Fractal Dim", f"{fd:.3f}" if not np.isnan(fd) else "N/A")
            m2.metric("Lacunarity", f"{lac:.2f}" if not np.isnan(lac) else "N/A")
            m3.metric("Ramification", f"{sri:.2f}" if not np.isnan(sri) else "N/A")
            m4.metric("Centrality", f"{betw:.3f}" if not np.isnan(betw) else "N/A")
            m5.metric("Soma Area", f"{soma_area:.0f} px" if not np.isnan(soma_area) else "N/A")
            m6.metric("Circularity", f"{soma_circ:.2f}" if not np.isnan(soma_circ) else "N/A")

            db_col1, db_col2, db_col3 = st.columns([1.2, 1, 1])
            
            # Panel A: Voronoi Mask + Circles (Exact Matplotlib style with RGB overlay and Cropping)
            fig_mask, ax1 = plt.subplots(figsize=(6,6))
            
            if len(img_gray.shape) == 2:
                overlay_panel = np.stack([img_gray]*3, axis=-1).copy()
            else:
                overlay_panel = img_gray.copy()
            
            skel_bool = np.asarray(comp_mask, dtype=bool)
            overlay_panel[skel_bool] = [0, 255, 0] # Green skeleton
            
            from morphology_features import _circle_coords
            for r in radii:
                rr, cc = _circle_coords(corr_y, corr_x, int(r), overlay_panel.shape[:2])
                overlay_panel[rr, cc] = [255, 60, 60] # Red circles
                
            # Draw soma
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    rr, cc_s = corr_y + dr, corr_x + dc
                    if 0 <= rr < overlay_panel.shape[0] and 0 <= cc_s < overlay_panel.shape[1]:
                        if dr*dr + dc*dc <= 9:
                            overlay_panel[rr, cc_s] = [255, 255, 0]
                            
            # Crop around soma
            crop_r = int(max(radii) * 1.3) if len(radii) > 0 else 100
            r0 = max(0, corr_y - crop_r)
            r1 = min(overlay_panel.shape[0], corr_y + crop_r)
            c0 = max(0, corr_x - crop_r)
            c1 = min(overlay_panel.shape[1], corr_x + crop_r)
            
            ax1.imshow(overlay_panel[r0:r1, c0:c1])
            ax1.axis('off')
            ax1.set_title("A) Sholl Back-trace & Voronoi Mask", fontsize=10, fontweight='bold')
            db_col1.pyplot(fig_mask)
            
            # Panel D: Fractal log-log
            fig_frac, ax_d = plt.subplots(figsize=(6,6))
            if len(_log_sizes) >= 2:
                ax_d.scatter(_log_sizes, _log_counts, c='steelblue', s=40)
                try:
                    coeffs = np.polyfit(_log_sizes, _log_counts, 1)
                    x_fit = np.linspace(_log_sizes.min(), _log_sizes.max(), 50)
                    ax_d.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2, label=f'D = {fd:.3f}')
                    ax_d.legend(fontsize=10)
                except:
                    pass
            ax_d.set_xlabel("log(1/s)", fontsize=9)
            ax_d.set_ylabel("log N(s)", fontsize=9)
            ax_d.set_title("D) Fractal Dimension", fontsize=10, fontweight='bold')
            db_col2.pyplot(fig_frac)
            
            # Panel E: Sholl Curve
            fig_curve, ax2 = plt.subplots(figsize=(6,6))
            ax2.plot(radii, intersections, marker='o', linewidth=3, color='teal')
            ax2.fill_between(radii, 0, intersections, color='teal', alpha=0.2)
            ax2.set_xlabel("Radius (px)", fontweight='bold')
            ax2.set_ylabel("Intersections", fontweight='bold')
            ax2.set_title("E) Arborization Profile", fontsize=10, fontweight='bold')
            sns.despine()
            db_col3.pyplot(fig_curve)
            
            # Update accumulator for final plot
            import pandas as pd
            for r, inters in zip(radii, intersections):
                all_cells_sholl_data.append({
                    "Radius (um)": r * 0.56, # Assuming hardcoded typical conversion factor mentioned before
                    "Radius (px)": r,
                    "Intersections": inters,
                    "Cell": f"Cell {idx}"
                })
            
            plt.close('all')
            
        if all_cells_sholl_data:
            st.divider()
            st.markdown("### 📈 Global Pipeline Report")
            df_sholl = pd.DataFrame(all_cells_sholl_data)
            
            pcol1, pcol2 = st.columns([1, 3])
            palette_opt = pcol1.selectbox("🎨 Select Palette (App Only Feature)", ["viridis", "husl", "magma", "Set2", "colorblind", "flare", "mako"])
            conv_factor = pcol1.number_input("Conversion Factor (um/px)", min_value=0.1, max_value=5.0, value=0.56, step=0.01)
            
            df_sholl["Radius"] = df_sholl["Radius (px)"] * conv_factor
            
            fig_global, ax_glob = plt.subplots(figsize=(10, 6))
            
            # Plot individual cell traces
            sns.lineplot(data=df_sholl, x='Radius', y='Intersections', hue='Cell', alpha=0.4, linewidth=1.5, palette=palette_opt, ax=ax_glob)
            # Overlay aggregated mean curve (like the original script)
            sns.lineplot(data=df_sholl, x='Radius', y='Intersections', color='black', linewidth=3, errorbar=None, label="Mean Trend", ax=ax_glob)
            
            ax_glob.set_xlabel("Distance from Soma (µm)", fontweight='bold', fontsize=12)
            ax_glob.set_ylabel("Number of Intersections", fontweight='bold', fontsize=12)
            ax_glob.set_title("Aggregated Sholl Morphometric Analysis", fontweight='bold', fontsize=14)
            sns.despine()
            
            pcol2.pyplot(fig_global)
            
        st.balloons()

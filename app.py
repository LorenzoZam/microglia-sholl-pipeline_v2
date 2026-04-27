import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

# Import the actual scientific components directly from the user's codebase
from run_sholl_pipeline import (
    apply_adaptive_patching, apply_tophat, binarize_image,
    remove_small_fragments, remove_isolated_fibers, skeletonize_image,
    bridge_nearby_fragments, apply_morph_close, apply_dilate,
    remove_isolated_fibers_refine, bridge_nearby_fragments_refine,
    get_connected_component, compute_sholl_intersections,
    generate_concentric_circles, measure_farthest_neurite
)
from morphology_features import (
    box_counting_fractal_dimension_with_data, box_counting_lacunarity,
    skeleton_to_graph, compute_graph_centralities,
    schoenen_ramification_index, soma_shape_metrics, _circle_coords
)

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="🔬 Sholl Analysis WebDemo", layout="wide", page_icon="🧠")
sns.set_theme(style="ticks", context="talk", palette="colorblind")

st.title("🔬 Automated Sholl Analysis Pipeline (Demo)")
st.markdown(
    "This web app perfectly replicates the rigorous mathematical pipeline used in the "
    "native desktop script. Upload a single-cell image to trace its skeleton and extract "
    "morphometrics using Voronoi isolation."
)

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "soma_points" not in st.session_state:
    st.session_state.soma_points = []
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = "selecting"
if "last_click1" not in st.session_state:
    st.session_state.last_click1 = None
if "last_click2" not in st.session_state:
    st.session_state.last_click2 = None


def undo_last():
    if st.session_state.soma_points:
        st.session_state.soma_points.pop()


def clear_all():
    st.session_state.soma_points = []


def accept_all():
    st.session_state.ui_mode = "qc"


def restart():
    st.session_state.soma_points = []
    st.session_state.ui_mode = "selecting"


# -------------------------------------------------------------
# CORE ALGORITHM (cached)
# -------------------------------------------------------------
@st.cache_data
def run_scientific_skeletonization(image, h_val):
    """Strictly matched pipeline from main() in run_sholl_pipeline.py."""
    global_var = np.var(image)
    processed_image = apply_adaptive_patching(image, global_var)
    denoised = cv2.fastNlMeansDenoising(
        processed_image, None, h=h_val, templateWindowSize=7, searchWindowSize=21
    )
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
# FILE UPLOAD
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Image (TIFF/PNG/JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        st.error("Error loading image.")
        st.stop()

    # =========================================================
    # STEP 2 — INTERACTIVE CELL DETECTION
    # =========================================================
    if st.session_state.ui_mode == "selecting":
        st.markdown("### Step 2: Interactive Cell Detection")
        st.info(
            "Click directly on ANY of the two images below to register a Soma coordinate. "
            "You can accumulate multiple cells exactly like in Matplotlib!"
        )

        # ── Controls (rendered BELOW images via st.container trick) ──────────
        img_container   = st.container()   # images go here
        ctrl_container  = st.container()   # controls go here (visually below)

        with ctrl_container:
            st.markdown("---")
            st.markdown("#### 🎛️ Denoising Calibration")
            c1, c2 = st.columns(2)
            h_val = c1.slider(
                "NLM Parameter (h)", min_value=1, max_value=30, value=11
            )
            step_size = int(c2.number_input(
                "Sholl Step Size (px)", min_value=1, max_value=50, value=4
            ))

        # Run pipeline with current h_val
        with st.spinner("Applying rigorous morphological filters..."):
            processed_image, binary, skeleton = run_scientific_skeletonization(
                img_gray, h_val
            )

        # ── Build overlay images ──────────────────────────────────────────────
        # Left: patch-wise denoised image (colour)
        ov_left  = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        # Right: pure skeleton on black background (no underlying image)
        ov_right = np.zeros((*processed_image.shape, 3), dtype=np.uint8)
        ov_right[skeleton > 0] = [255, 255, 255]   # white skeleton on black

        # Draw registered somas
        LIME_GREEN = (50, 205, 50)
        for idx, (cx, cy) in enumerate(st.session_state.soma_points, start=1):
            cv2.circle(ov_left,  (int(cx), int(cy)), 5, LIME_GREEN, -1)
            cv2.circle(ov_right, (int(cx), int(cy)), 5, LIME_GREEN, -1)
            cv2.putText(ov_left,  str(idx), (int(cx)+10, int(cy)+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LIME_GREEN, 2, cv2.LINE_AA)
            cv2.putText(ov_right, str(idx), (int(cx)+10, int(cy)+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LIME_GREEN, 2, cv2.LINE_AA)

        # Scale down to avoid scrollbars while preserving coordinate fidelity
        DISPLAY_WIDTH = 700
        scale_ratio = min(1.0, DISPLAY_WIDTH / float(ov_left.shape[1]))
        new_dim = (
            int(ov_left.shape[1] * scale_ratio),
            int(ov_left.shape[0] * scale_ratio),
        )
        ov_left_small  = cv2.resize(ov_left,  new_dim, interpolation=cv2.INTER_AREA)
        ov_right_small = cv2.resize(ov_right, new_dim, interpolation=cv2.INTER_AREA)

        # ── Render images ─────────────────────────────────────────────────────
        with img_container:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Processed Image**")
                val1 = streamlit_image_coordinates(ov_left_small,  key="img1")
            with col2:
                st.markdown("**Skeletonized Tracing**")
                val2 = streamlit_image_coordinates(ov_right_small, key="img2")

        # Handle new clicks (back-project scaled coordinates)
        if val1 is not None and val1 != st.session_state.last_click1:
            st.session_state.soma_points.append((
                int(val1["x"] / scale_ratio),
                int(val1["y"] / scale_ratio),
            ))
            st.session_state.last_click1 = val1
            st.rerun()
        if val2 is not None and val2 != st.session_state.last_click2:
            st.session_state.soma_points.append((
                int(val2["x"] / scale_ratio),
                int(val2["y"] / scale_ratio),
            ))
            st.session_state.last_click2 = val2
            st.rerun()

        # ── Action buttons ────────────────────────────────────────────────────
        b1, b2, b3 = st.columns([1, 1, 2])
        b1.button("Undo Last ↺", on_click=undo_last)
        b2.button("Clear All ✗",  on_click=clear_all)
        b3.button(
            f"Accept Somas ({len(st.session_state.soma_points)}) & Continue ✓",
            on_click=accept_all,
            type="primary",
        )

    # =========================================================
    # STEP 3 — QC DASHBOARD
    # =========================================================
    elif st.session_state.ui_mode == "qc":
        # step_size must be available in this branch too (sidebar-free design)
        step_size_ctrl = st.container()
        with step_size_ctrl:
            step_size = int(st.number_input(
                "Sholl Step Size (px) — locked after acceptance",
                min_value=1, max_value=50, value=4, disabled=True
            ))

        st.markdown("### Step 3: Quality Control Dashboard")
        st.button("← Back to Soma Selection", on_click=restart)

        if not st.session_state.soma_points:
            st.warning("No cells selected.")
            st.stop()

        with st.spinner("Running pipeline..."):
            _, processed_image_qc = cv2.imencode(".png", img_gray)   # dummy to satisfy cache
            processed_image, binary, skeleton = run_scientific_skeletonization(img_gray, 11)

        # ── Compute farthest endpoints for ALL somas upfront ─────────────────
        farthest_endpoints = measure_farthest_neurite(skeleton, st.session_state.soma_points)

        # ── Global Topology Viewer ────────────────────────────────────────────
        st.markdown("#### 🌍 Global Topology Viewer")
        st.info("Full-field map: white skeleton, green soma markers, red Sholl rings.")

        if len(img_gray.shape) == 2:
            glob_ov = np.zeros((*img_gray.shape, 3), dtype=np.uint8)  # black background
        else:
            glob_ov = np.zeros((*img_gray.shape[:2], 3), dtype=np.uint8)
        glob_ov[skeleton > 0] = [255, 255, 255]   # white skeleton

        LIME_GREEN = (50, 205, 50)
        for idx, (raw_x, raw_y) in enumerate(st.session_state.soma_points, start=1):
            result = get_connected_component(skeleton, (raw_y, raw_x))
            if result[0] is None:
                continue
            _, (cx, cy) = result
            # Compute radii for this cell
            m_rad = 500
            ft = farthest_endpoints.get(idx)
            if ft:
                m_rad = np.ceil(
                    np.sqrt((ft[0] - cx) ** 2 + (ft[1] - cy) ** 2) / step_size
                ) * step_size
            base_r    = generate_concentric_circles(m_rad, step_size)
            add_r     = np.arange(m_rad, m_rad + 3 * step_size, step_size)
            rs        = np.unique(np.concatenate([base_r, add_r]))
            for r in rs:
                rr, cc = _circle_coords(cy, cx, int(r), glob_ov.shape[:2])
                glob_ov[rr, cc] = [255, 60, 60]   # red rings
            cv2.circle(glob_ov, (cx, cy), 5, LIME_GREEN, -1)
            cv2.putText(glob_ov, str(idx), (cx + 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LIME_GREEN, 2, cv2.LINE_AA)

        fig_g, ax_g = plt.subplots(figsize=(12, 10))
        ax_g.imshow(glob_ov)
        ax_g.axis("off")
        ax_g.set_title("Global Skeleton Map with Sholl Rings", fontsize=12, fontweight="bold")
        st.pyplot(fig_g)
        plt.close(fig_g)

        # ── Per-cell analysis loop ────────────────────────────────────────────
        all_cells_sholl_data = []

        for idx, (raw_x, raw_y) in enumerate(st.session_state.soma_points, start=1):
            st.divider()
            st.subheader(f"🧠 Analysis for Cell {idx}")

            comp_mask, (corr_x, corr_y) = get_connected_component(skeleton, (raw_y, raw_x))
            if comp_mask is None:
                st.error(f"Soma {idx}: no connected skeleton found at this location.")
                continue

            # Compute radii
            max_radius = 500
            farthest = farthest_endpoints.get(idx)
            if farthest:
                ex, ey = farthest
                max_radius = np.ceil(
                    np.sqrt((ex - corr_x) ** 2 + (ey - corr_y) ** 2) / step_size
                ) * step_size

            base_r       = generate_concentric_circles(max_radius, step_size)
            additional_r = np.arange(max_radius, max_radius + 3 * step_size, step_size)
            radii        = np.unique(np.concatenate([base_r, additional_r]))

            intersections = compute_sholl_intersections(comp_mask, corr_x, corr_y, radii)

            # Morphometric extraction
            with st.spinner(f"Extracting morphometrics for Cell {idx}..."):
                fd, _log_sizes, _log_counts = box_counting_fractal_dimension_with_data(comp_mask)
                lac                         = box_counting_lacunarity(comp_mask)
                G                           = skeleton_to_graph(comp_mask)
                betw, clos                  = compute_graph_centralities(G)
                sri                         = schoenen_ramification_index(
                    intersections, radii, (corr_x, corr_y), comp_mask
                )
                soma_area, soma_circ        = soma_shape_metrics(binary, (corr_x, corr_y))

            # ── Metric tiles ──────────────────────────────────────────────────
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Fractal Dim",   f"{fd:.3f}"       if not np.isnan(fd)        else "N/A")
            m2.metric("Lacunarity",    f"{lac:.2f}"      if not np.isnan(lac)       else "N/A")
            m3.metric("Ramification",  f"{sri:.2f}"      if not np.isnan(sri)       else "N/A")
            m4.metric("Centrality",    f"{betw:.3f}"     if not np.isnan(betw)      else "N/A")
            m5.metric("Soma Area",     f"{soma_area:.0f} px" if not np.isnan(soma_area) else "N/A")
            m6.metric("Circularity",   f"{soma_circ:.2f}" if not np.isnan(soma_circ) else "N/A")

            db_col1, db_col2, db_col3 = st.columns([1.2, 1, 1])

            # Panel A — Backtrace overlay (RGB, cropped)
            if len(img_gray.shape) == 2:
                overlay_panel = np.stack([img_gray] * 3, axis=-1).copy()
            else:
                overlay_panel = img_gray.copy()
            overlay_panel[np.asarray(comp_mask, dtype=bool)] = [0, 255, 0]
            for r in radii:
                rr, cc = _circle_coords(corr_y, corr_x, int(r), overlay_panel.shape[:2])
                overlay_panel[rr, cc] = [255, 60, 60]
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    rr, cc_s = corr_y + dr, corr_x + dc
                    if 0 <= rr < overlay_panel.shape[0] and 0 <= cc_s < overlay_panel.shape[1]:
                        if dr * dr + dc * dc <= 9:
                            overlay_panel[rr, cc_s] = [255, 255, 0]
            crop_r = int(max(radii) * 1.3) if len(radii) > 0 else 100
            r0 = max(0, corr_y - crop_r);  r1 = min(overlay_panel.shape[0], corr_y + crop_r)
            c0 = max(0, corr_x - crop_r);  c1 = min(overlay_panel.shape[1], corr_x + crop_r)
            fig_bt, ax_bt = plt.subplots(figsize=(6, 6))
            ax_bt.imshow(overlay_panel[r0:r1, c0:c1])
            ax_bt.axis("off")
            ax_bt.set_title("A) Sholl Back-trace & Voronoi Mask", fontsize=10, fontweight="bold")
            db_col1.pyplot(fig_bt)

            # Panel D — Fractal log-log
            fig_frac, ax_d = plt.subplots(figsize=(6, 6))
            if len(_log_sizes) >= 2:
                ax_d.scatter(_log_sizes, _log_counts, c="steelblue", s=40)
                try:
                    coeffs = np.polyfit(_log_sizes, _log_counts, 1)
                    x_fit  = np.linspace(_log_sizes.min(), _log_sizes.max(), 50)
                    ax_d.plot(x_fit, np.polyval(coeffs, x_fit), "r-", linewidth=2,
                              label=f"D = {fd:.3f}")
                    ax_d.legend(fontsize=10)
                except Exception:
                    pass
            ax_d.set_xlabel("log(1/s)", fontsize=9)
            ax_d.set_ylabel("log N(s)", fontsize=9)
            ax_d.set_title("D) Fractal Dimension", fontsize=10, fontweight="bold")
            db_col2.pyplot(fig_frac)

            # Panel E — Sholl curve
            fig_curve, ax2 = plt.subplots(figsize=(6, 6))
            ax2.plot(radii, intersections, marker="o", linewidth=3, color="teal")
            ax2.fill_between(radii, 0, intersections, color="teal", alpha=0.2)
            ax2.set_xlabel("Radius (px)", fontweight="bold")
            ax2.set_ylabel("Intersections",   fontweight="bold")
            ax2.set_title("E) Arborization Profile", fontsize=10, fontweight="bold")
            sns.despine()
            db_col3.pyplot(fig_curve)

            plt.close("all")

            # Accumulate for global report
            for r, inters in zip(radii, intersections):
                all_cells_sholl_data.append({
                    "Radius (px)":  r,
                    "Intersections": inters,
                    "Cell": f"Cell {idx}",
                })

        # ── Global Pipeline Report ────────────────────────────────────────────
        if all_cells_sholl_data:
            st.divider()
            st.markdown("### 📈 Global Pipeline Report")
            df_sholl = pd.DataFrame(all_cells_sholl_data)
            n_cells  = df_sholl["Cell"].nunique()

            pcol1, pcol2 = st.columns([1, 3])
            palette_opt   = pcol1.selectbox(
                "🎨 Select Palette",
                ["colorblind", "husl", "viridis", "magma", "Set2", "flare", "mako"],
            )
            conv_factor = float(pcol1.number_input(
                "Conversion Factor (µm/px)",
                min_value=0.01, max_value=10.0, value=0.56, step=0.01,
            ))
            show_individual = pcol1.toggle(
                "Show individual cell curves", value=False,
                help="Off = show only mean ± CI; On = show one trace per cell"
            )
            df_sholl["Radius (µm)"] = df_sholl["Radius (px)"] * conv_factor

            fig_global, ax_glob = plt.subplots(figsize=(10, 6))

            if show_individual:
                # Individual traces + mean on top
                sns.lineplot(
                    data=df_sholl, x="Radius (µm)", y="Intersections",
                    hue="Cell", alpha=0.55, linewidth=1.2,
                    palette=palette_opt, ax=ax_glob,
                )
                sns.lineplot(
                    data=df_sholl, x="Radius (µm)", y="Intersections",
                    color="black", linewidth=2.5, errorbar=None,
                    label="Mean", ax=ax_glob,
                )
            else:
                # PRIMARY MODE: mean ± 95 % CI, individual curves hidden
                # Build a neutral colour from the chosen palette
                palette_colors = sns.color_palette(palette_opt, n_colors=1)
                mean_color = palette_colors[0]
                sns.lineplot(
                    data=df_sholl, x="Radius (µm)", y="Intersections",
                    color=mean_color, linewidth=2.5, errorbar=("ci", 95),
                    err_kws={"alpha": 0.25}, ax=ax_glob,
                    label=f"Mean ± 95% CI  (n={n_cells})",
                )

            ax_glob.set_xlabel("Distance from Soma (µm)", fontweight="bold", fontsize=12)
            ax_glob.set_ylabel("Number of Intersections",  fontweight="bold", fontsize=12)
            ax_glob.set_title(
                "Aggregated Sholl Morphometric Analysis", fontweight="bold", fontsize=14
            )
            ax_glob.legend(fontsize=10, frameon=False)
            sns.despine()
            pcol2.pyplot(fig_global)
            plt.close(fig_global)

            csv = df_sholl.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Sholl Data as CSV",
                data=csv,
                file_name="sholl_analysis_results.csv",
                mime="text/csv",
                help="Download the full intersection table ready for GraphPad / Excel.",
            )

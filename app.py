import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from streamlit_image_coordinates import streamlit_image_coordinates

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
    schoenen_ramification_index, soma_shape_metrics, _circle_coords,
    generate_qc_dashboard
)

# ─────────────────────────────────────────────────────────────
# CONFIG CONSTANTS  (7.2)
# ─────────────────────────────────────────────────────────────
DISPLAY_WIDTH_PX   = 700
DEFAULT_MAX_RADIUS = 500
CROP_MARGIN_FACTOR = 1.3
DEFAULT_UM_PER_PX  = 0.56
SOMA_GREEN         = (50, 205, 50)

# Biological reference ranges for flag coloring  (5.2)
BIO_RANGES = {
    "fd":        (0.7,  1.8),
    "lac":       (1.0,  10.0),
    "betw":      (0.0,  0.5),
    "clos":      (0.0,  0.3),
    "sri":       (0.5,  20.0),
    "soma_area": (50.0, 2000.0),
    "soma_circ": (0.03, 1.0),
}

def _flag_color(value, lo, hi):
    if np.isnan(value): return "red"
    return "green" if lo <= value <= hi else "darkorange"

def _metric_badge(col, label, value, fmt, key, unit=""):
    """Render st.metric with a colored caption flag.  (5.2)"""
    lo, hi = BIO_RANGES.get(key, (None, None))
    disp = (fmt.format(value) + unit) if not np.isnan(value) else "N/A"
    col.metric(label, disp)
    if lo is not None and not np.isnan(value):
        color = _flag_color(value, lo, hi)
        col.markdown(
            f"<div style='font-size:0.72rem;color:{color};margin-top:-14px'>"
            f"{'✔ normal' if color=='green' else '⚠ outside ref'}</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & HEADER  (5.1)
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔬 MicroSholl", layout="wide", page_icon="🧠")
sns.set_theme(style="ticks", context="talk", palette="colorblind")

st.markdown("""
<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
     padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.2rem;">
  <h1 style="color:#7ee8fa;margin:0;font-size:2.1rem;">🧠 MicroSholl</h1>
  <p style="color:#cfd9df;margin:0.3rem 0 0;font-size:1rem;">
    Advanced Microglia Morphology Analysis &nbsp;·&nbsp; v2.0
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_defaults = dict(
    soma_points=[], ui_mode="selecting",
    last_click1=None, last_click2=None,
    rejected_cells=set(), chosen_sample=None,
    step_size=4, um_per_px=DEFAULT_UM_PER_PX,
    confirm_restart=False,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def undo_last():
    if st.session_state.soma_points:
        st.session_state.soma_points.pop()

def clear_all():
    st.session_state.soma_points = []

def accept_all(step_size_val, um_per_px_val):
    """Persist user parameters before switching to QC.  (1.2, 2.2)"""
    st.session_state.step_size = step_size_val
    st.session_state.um_per_px = um_per_px_val
    st.session_state.ui_mode   = "qc"

def _do_restart():
    for k, v in _defaults.items():
        st.session_state[k] = v if not isinstance(v, (list, set, dict)) else type(v)()
    st.session_state.confirm_restart = False

# ─────────────────────────────────────────────────────────────
# PROGRESS INDICATOR  (1.1)
# ─────────────────────────────────────────────────────────────
_step_labels = ["① Load Image", "② Select Somas", "③ QC & Analysis", "④ Export"]

# Determine current step: "selecting" without an image loaded = still on Step 1
_has_image = (st.session_state.chosen_sample is not None
              or st.session_state.get("last_uploaded") is not None)
if st.session_state.ui_mode == "qc":
    _step_idx = 2
elif st.session_state.ui_mode == "selecting" and _has_image:
    _step_idx = 1
else:
    _step_idx = 0

def _step_html(label, active):
    bg  = "#7ee8fa" if active else "#2c5364"
    col = "#0f2027" if active else "#cfd9df"
    fw  = "bold"    if active else "normal"
    return (f"<span style='background:{bg};color:{col};font-weight:{fw};"
            f"padding:4px 12px;border-radius:20px;font-size:0.85rem;margin:2px'>"
            f"{label}</span>")

st.markdown(
    "<div style='margin-bottom:1rem'>" +
    " &nbsp;›&nbsp; ".join(_step_html(l, i == _step_idx) for i, l in enumerate(_step_labels)) +
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# CORE ALGORITHM (cached)  (2.1 — accepts tmpl/search window)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def run_scientific_skeletonization(image, h_val, tmpl_win=7, search_win=21):
    global_var      = np.var(image)
    processed_image = apply_adaptive_patching(image, global_var)
    denoised        = cv2.fastNlMeansDenoising(processed_image, None, h=h_val,
                                               templateWindowSize=tmpl_win,
                                               searchWindowSize=search_win)
    den_th_image    = apply_tophat(denoised)
    binary          = binarize_image(den_th_image)
    frag_filtered   = remove_small_fragments(binary)
    closed          = apply_morph_close(frag_filtered)
    dilated         = apply_dilate(closed)
    skel            = skeletonize_image(dilated)
    bridged         = bridge_nearby_fragments(skel)
    refined         = remove_isolated_fibers_refine(bridged)
    bridge_refined  = bridge_nearby_fragments_refine(refined)
    skeleton        = remove_isolated_fibers(bridge_refined)
    return processed_image, binary, skeleton

# ─────────────────────────────────────────────────────────────
# SAMPLE IMAGE HELPERS
# ─────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(__file__).parent / "sample_images"

def _list_samples():
    if not SAMPLE_DIR.exists():
        return []
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(f.name for f in SAMPLE_DIR.iterdir() if f.suffix.lower() in exts)

# ─────────────────────────────────────────────────────────────
# IMAGE SOURCE — two tabs
# ─────────────────────────────────────────────────────────────
tab_sample, tab_upload = st.tabs(["🖼️ Try a Sample Image", "📤 Upload Your Own"])

img_gray = None

with tab_sample:
    samples = _list_samples()
    if not samples:
        st.info("No sample images found. Add `.tif` / `.png` files to the `sample_images/` folder in the repo.")
    else:
        st.markdown("Select one of the bundled microscopy images to try the pipeline instantly:")
        thumb_cols = st.columns(min(len(samples), 5))
        for col, name in zip(thumb_cols, samples):
            path  = SAMPLE_DIR / name
            thumb = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if thumb is not None:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                h, w  = thumb.shape[:2]
                scale = 150 / max(h, w)
                thumb_small = cv2.resize(thumb, (int(w * scale), int(h * scale)))
                col.image(thumb_small, caption=name, use_container_width=True)
                if col.button("Use", key=f"btn_{name}"):
                    for _k, _v in _defaults.items():
                        st.session_state[_k] = _v if not isinstance(_v, (list, set, dict)) else type(_v)()
                    st.session_state.chosen_sample = name
                    st.rerun()

        chosen = st.session_state.chosen_sample or samples[0]
        img_color = cv2.imread(str(SAMPLE_DIR / chosen), cv2.IMREAD_COLOR)
        if img_color is not None:
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
            st.success(f"✅ Loaded: **{chosen}**  ({img_gray.shape[1]} × {img_gray.shape[0]} px)")
        else:
            img_gray = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Image (TIFF/PNG/JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_color  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_color is None:
            st.error("Error loading image.")
            st.stop()
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        img_gray  = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        if st.session_state.get("last_uploaded") != uploaded_file.name:
            for _k, _v in _defaults.items():
                st.session_state[_k] = _v if not isinstance(_v, (list, set, dict)) else type(_v)()
            st.session_state["last_uploaded"] = uploaded_file.name

# Guard: nothing selected yet  (6.1)
if img_gray is None:
    st.info("👆 Please select a sample image or upload your own to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# STEP 2 — INTERACTIVE CELL DETECTION
# ─────────────────────────────────────────────────────────────
if st.session_state.ui_mode == "selecting":
    st.markdown("### Step 2: Interactive Cell Detection")
    st.caption(
        "👆 Click on any **cell body (soma)** in either image below to register it. "
        "The skeleton panel (right) helps you verify the tracing quality. "
        "You can accumulate multiple cells."
    )

    img_container  = st.container()   # images rendered here
    ctrl_container = st.container()   # controls rendered visually below

    with ctrl_container:
        st.markdown("---")
        st.markdown("#### 🎛️ Denoising & Analysis Parameters")
        c1, c2, c3 = st.columns(3)
        h_val       = c1.slider("NLM Parameter (h)", min_value=1, max_value=30, value=11)
        step_size_w = int(c2.number_input("Sholl Step Size (px)", min_value=1, max_value=50, value=st.session_state.step_size))
        um_per_px_w = float(c3.number_input(
            "Pixel size (µm/px)", min_value=0.01, max_value=10.0,
            value=st.session_state.um_per_px, step=0.01,
            help="Microscope calibration factor — used for µm axis in the final plot."
        ))

        with st.expander("⚙️ Advanced Pipeline Settings", expanded=False):
            adv1, adv2 = st.columns(2)
            tmpl_win   = adv1.slider("NLM Template Window (px)", 5, 15, 7, step=2)
            search_win = adv2.slider("NLM Search Window (px)", 11, 31, 21, step=2)
            st.caption("ℹ️ Changing these clears the processing cache.")

    with st.spinner("Applying rigorous morphological filters..."):
        processed_image, binary, skeleton = run_scientific_skeletonization(
            img_gray, h_val, tmpl_win, search_win
        )

    # Build overlays
    ov_left  = img_color.copy()  # Show the raw fluorescent image
    ov_right = np.zeros((*processed_image.shape, 3), dtype=np.uint8)
    ov_right[skeleton > 0] = [255, 255, 255]

    for idx, (cx, cy) in enumerate(st.session_state.soma_points, start=1):
        cv2.circle(ov_left,  (int(cx), int(cy)), 5, SOMA_GREEN, -1)
        cv2.circle(ov_right, (int(cx), int(cy)), 5, SOMA_GREEN, -1)
        cv2.putText(ov_left,  str(idx), (int(cx)+10, int(cy)+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SOMA_GREEN, 2, cv2.LINE_AA)
        cv2.putText(ov_right, str(idx), (int(cx)+10, int(cy)+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SOMA_GREEN, 2, cv2.LINE_AA)

    scale_ratio    = min(1.0, DISPLAY_WIDTH_PX / float(ov_left.shape[1]))
    new_dim        = (int(ov_left.shape[1] * scale_ratio), int(ov_left.shape[0] * scale_ratio))
    ov_left_small  = cv2.resize(ov_left,  new_dim, interpolation=cv2.INTER_AREA)
    ov_right_small = cv2.resize(ov_right, new_dim, interpolation=cv2.INTER_AREA)

    with img_container:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Fluorescent Image**")
            val1 = streamlit_image_coordinates(ov_left_small,  key="img1", use_column_width="auto")
        with col2:
            st.markdown("**Skeleton Tracing**")
            val2 = streamlit_image_coordinates(ov_right_small, key="img2", use_column_width="auto")

    # --- FIX 6.2: dedup by (x,y) tuple, not full dict ---
    if val1 is not None:
        coord1 = (val1["x"], val1["y"])
        if coord1 != st.session_state.last_click1:
            st.session_state.soma_points.append(
                (int(val1["x"] / scale_ratio), int(val1["y"] / scale_ratio)))
            st.session_state.last_click1 = coord1
            st.rerun()
    if val2 is not None:
        coord2 = (val2["x"], val2["y"])
        if coord2 != st.session_state.last_click2:
            st.session_state.soma_points.append(
                (int(val2["x"] / scale_ratio), int(val2["y"] / scale_ratio)))
            st.session_state.last_click2 = coord2
            st.rerun()

    b1, b2, b3 = st.columns([1, 1, 2])
    b1.button("Undo Last ↺", on_click=undo_last)
    b2.button("Clear All ✗",  on_click=clear_all)
    b3.button(
        f"Accept Somas ({len(st.session_state.soma_points)}) & Continue ✓",
        on_click=accept_all, args=(step_size_w, um_per_px_w), type="primary",
    )

# ─────────────────────────────────────────────────────────────
# STEP 3 — QC DASHBOARD
# ─────────────────────────────────────────────────────────────
elif st.session_state.ui_mode == "qc":
    # --- FIX 1.2: Read persisted step_size (not hardcoded 4) ---
    step_size   = st.session_state.step_size
    conv_factor = st.session_state.um_per_px

    st.markdown("### Step 3: Quality Control Dashboard")

    # --- FIX 1.3: confirmation before back ---
    if st.session_state.confirm_restart:
        st.warning("⚠️ This will clear all QC data. Are you sure?")
        rb1, rb2 = st.columns(2)
        if rb1.button("Yes, restart", type="primary"):
            _do_restart()
            st.rerun()
        if rb2.button("Cancel"):
            st.session_state.confirm_restart = False
            st.rerun()
    else:
        if st.button("← Back to Soma Selection"):
            st.session_state.confirm_restart = True
            st.rerun()

    if not st.session_state.soma_points:
        st.warning("No cells selected.")
        st.stop()

    with st.spinner("Running pipeline..."):
        processed_image, binary, skeleton = run_scientific_skeletonization(img_gray, 11)

    farthest_endpoints = measure_farthest_neurite(skeleton, st.session_state.soma_points)

    # ── Global Topology Viewer (compact) ────────────────────────
    with st.expander("🌍 Global Topology Viewer", expanded=True):
        st.caption("Full-field map: white skeleton · green soma markers · red Sholl rings")

        glob_ov = np.zeros((*img_gray.shape[:2], 3), dtype=np.uint8)
        glob_ov[skeleton > 0] = [255, 255, 255]

        for idx, (raw_x, raw_y) in enumerate(st.session_state.soma_points, start=1):
            result = get_connected_component(skeleton, (raw_y, raw_x))
            if result[0] is None:
                continue
            _, (cx, cy) = result
            m_rad = DEFAULT_MAX_RADIUS
            ft    = farthest_endpoints.get(idx)
            if ft:
                m_rad = np.ceil(np.sqrt((ft[0]-cx)**2 + (ft[1]-cy)**2) / step_size) * step_size
            rs = np.unique(np.concatenate([
                generate_concentric_circles(m_rad, step_size),
                np.arange(m_rad, m_rad + 3*step_size, step_size)
            ]))
            for r in rs:
                rr, cc = _circle_coords(cy, cx, int(r), glob_ov.shape[:2])
                glob_ov[rr, cc] = [255, 60, 60]
            cv2.circle(glob_ov, (cx, cy), 5, SOMA_GREEN, -1)
            cv2.putText(glob_ov, str(idx), (cx+10, cy+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, SOMA_GREEN, 2, cv2.LINE_AA)

        fig_g, ax_g = plt.subplots(figsize=(10, 5))
        ax_g.imshow(glob_ov);  ax_g.axis("off")
        ax_g.set_title("Global Skeleton Map with Sholl Rings", fontsize=11, fontweight="bold")
        fig_g.tight_layout(pad=0.5)
        st.pyplot(fig_g);  plt.close(fig_g)

    # ── Per-cell analysis loop ────────────────────────────────
    all_cells_sholl_data = []
    summary_rows = []

    for idx, (raw_x, raw_y) in enumerate(st.session_state.soma_points, start=1):
        st.divider()

        comp_mask, (corr_x, corr_y) = get_connected_component(skeleton, (raw_y, raw_x))
        if comp_mask is None:
            st.warning(f"Cell {idx}: No skeleton found near click — consider re-clicking closer to a branch.")
            continue

        max_radius = DEFAULT_MAX_RADIUS
        farthest   = farthest_endpoints.get(idx)
        if farthest:
            ex, ey     = farthest
            max_radius = np.ceil(np.sqrt((ex-corr_x)**2 + (ey-corr_y)**2) / step_size) * step_size

        base_r       = generate_concentric_circles(max_radius, step_size)
        additional_r = np.arange(max_radius, max_radius + 3*step_size, step_size)
        radii        = np.unique(np.concatenate([base_r, additional_r]))

        intersections = compute_sholl_intersections(comp_mask, corr_x, corr_y, radii)

        with st.spinner(f"Extracting morphometrics for Cell {idx}..."):
            fd, _log_sizes, _log_counts = box_counting_fractal_dimension_with_data(comp_mask)
            lac                         = box_counting_lacunarity(comp_mask)
            G                           = skeleton_to_graph(comp_mask)
            betw, clos                  = compute_graph_centralities(G)
            sri                         = schoenen_ramification_index(intersections, radii, (corr_x, corr_y), comp_mask)
            soma_area, soma_circ        = soma_shape_metrics(binary, (corr_x, corr_y))

        # ── Cell header with styled banner ────────────────────
        st.markdown(
            f"<div style='background:linear-gradient(90deg,#1a1a2e,#16213e,#0f3460);"
            f"padding:0.6rem 1.2rem;border-radius:8px;margin-bottom:0.8rem'>"
            f"<span style='color:#e0e0e0;font-size:1.15rem;font-weight:600'>"
            f"🧠 Cell {idx}</span>"
            f"<span style='color:#7ee8fa;font-size:0.85rem;margin-left:1.2rem'>"
            f"Soma at ({corr_x}, {corr_y})"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        # ── Metric cards with biological reference flags ──────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        _metric_badge(m1, "Fractal Dim",  fd,        "{:.3f}", "fd")
        _metric_badge(m2, "Lacunarity",   lac,       "{:.2f}", "lac")
        _metric_badge(m3, "Ramification", sri,       "{:.2f}", "sri")
        _metric_badge(m4, "Betweenness",  betw,      "{:.3f}", "betw")
        _metric_badge(m5, "Soma Area",    soma_area, "{:.0f}", "soma_area", " px")
        _metric_badge(m6, "Circularity",  soma_circ, "{:.2f}", "soma_circ")

        # ── 5-panel QC dashboard (no redundant Panel F) ───────
        metrics_dict = dict(
            FD=fd, Lac=lac, Betw=betw, Clos=clos,
            SRI=sri, Area=soma_area, Circ=soma_circ,
            log_inv_sizes=_log_sizes, log_counts=_log_counts,
        )
        with st.spinner(f"Rendering QC dashboard for Cell {idx}..."):
            fig_qc = generate_qc_dashboard(
                img_gray, binary, skeleton, comp_mask,
                (corr_x, corr_y), radii, intersections,
                metrics_dict, G, idx,
                streamlit_mode=True
            )
        st.pyplot(fig_qc)
        plt.close(fig_qc)

        # ── Accept / Reject toggle ────────────────────────────
        accepted = st.checkbox(
            f"✅  Include Cell {idx} in Global Report",
            value=(idx not in st.session_state.rejected_cells),
            key=f"accept_cell_{idx}",
        )
        if accepted:
            st.session_state.rejected_cells.discard(idx)
            for r, inters in zip(radii, intersections):
                all_cells_sholl_data.append({
                    "Radius (px)": r, "Intersections": inters, "Cell": f"Cell {idx}",
                })
            summary_rows.append({
                "Cell": f"Cell {idx}",
                "Fractal Dim": round(fd, 4)        if not np.isnan(fd)        else float("nan"),
                "Lacunarity":  round(lac, 4)       if not np.isnan(lac)       else float("nan"),
                "Ramification":round(sri, 4)       if not np.isnan(sri)       else float("nan"),
                "Betweenness": round(betw, 4)      if not np.isnan(betw)      else float("nan"),
                "Soma Area px":round(soma_area, 1) if not np.isnan(soma_area) else float("nan"),
                "Circularity": round(soma_circ, 4) if not np.isnan(soma_circ) else float("nan"),
            })
        else:
            st.session_state.rejected_cells.add(idx)
            st.caption(f"⚠️ Cell {idx} excluded from Global Report.")

    # ── FIX 4.1: Per-cell summary table ───────────────────────
    if summary_rows:
        st.divider()
        st.markdown("#### 📋 Accepted Cells — Metrics Summary")
        df_summary = pd.DataFrame(summary_rows).set_index("Cell")
        st.dataframe(
            df_summary.style.background_gradient(
                subset=["Fractal Dim", "Lacunarity"], cmap="RdYlGn"
            ).format(na_rep="N/A", precision=3),
            use_container_width=True,
        )
        csv_metrics = df_summary.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Metrics CSV",
            data=csv_metrics,
            file_name="sholl_morphometrics.csv",
            mime="text/csv",
        )

    # ── Global Pipeline Report ────────────────────────────────
    if all_cells_sholl_data:
        st.divider()
        st.markdown("### 📈 Global Pipeline Report")
        df_sholl = pd.DataFrame(all_cells_sholl_data)
        n_cells  = df_sholl["Cell"].nunique()

        pcol1, pcol2 = st.columns([1, 3])
        palette_opt     = pcol1.selectbox("🎨 Palette",
            ["colorblind", "husl", "viridis", "magma", "Set2", "flare", "mako"])
        show_individual = pcol1.toggle("Show individual cell curves", value=False,
            help="Off = Mean ± SEM only; On = one trace per cell + mean")

        df_sholl["Radius (µm)"] = df_sholl["Radius (px)"] * conv_factor

        fig_global, ax_glob = plt.subplots(figsize=(10, 6))
        if show_individual:
            sns.lineplot(data=df_sholl, x="Radius (µm)", y="Intersections",
                hue="Cell", alpha=0.55, linewidth=1.2, palette=palette_opt, ax=ax_glob)
            sns.lineplot(data=df_sholl, x="Radius (µm)", y="Intersections",
                color="black", linewidth=2.5, errorbar=None, label="Mean", ax=ax_glob)
        else:
            mean_color = sns.color_palette(palette_opt, n_colors=1)[0]
            sns.lineplot(data=df_sholl, x="Radius (µm)", y="Intersections",
                color=mean_color, linewidth=2.5, errorbar="se",
                err_kws={"alpha": 0.25}, ax=ax_glob,
                label=f"Mean ± SEM  (n={n_cells})")

        ax_glob.set_xlabel(f"Distance from Soma (µm)  [1 px = {conv_factor} µm]",
                           fontweight="bold", fontsize=12)
        ax_glob.set_ylabel("Number of Intersections",  fontweight="bold", fontsize=12)
        ax_glob.set_title("Aggregated Sholl Morphometric Analysis", fontweight="bold", fontsize=14)
        ax_glob.legend(fontsize=10, frameon=False)
        sns.despine()
        pcol2.pyplot(fig_global);  plt.close(fig_global)

        csv = df_sholl.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sholl Data as CSV",
            data=csv, file_name="sholl_analysis_results.csv", mime="text/csv",
            help="Download the full intersection table ready for GraphPad / Excel.",
        )


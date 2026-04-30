import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
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
# CONFIG CONSTANTS
# ─────────────────────────────────────────────────────────────
DISPLAY_WIDTH_PX   = 550
DEFAULT_MAX_RADIUS = 500
CROP_MARGIN_FACTOR = 1.3
DEFAULT_UM_PER_PX  = 0.56
SOMA_GREEN         = (50, 205, 50)

# Biological reference ranges for flag coloring
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
    """Render st.metric with a colored caption flag."""
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
# PAGE CONFIG & HEADER
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔬 MicroSholl Batch", layout="wide", page_icon="🧠")
sns.set_theme(style="ticks", context="talk", palette="colorblind")

st.markdown("""
<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
     padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.2rem;">
  <h1 style="color:#7ee8fa;margin:0;font-size:2.1rem;">🧠 MicroSholl Batch</h1>
  <p style="color:#cfd9df;margin:0.3rem 0 0;font-size:1rem;">
    High-Throughput Microglia Morphology Analysis &nbsp;·&nbsp; v2.0
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TEMP BATCH DIR
# ─────────────────────────────────────────────────────────────
BATCH_TEMP_DIR = Path(__file__).parent / ".tmp_batch"
if not BATCH_TEMP_DIR.exists():
    BATCH_TEMP_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_defaults = dict(
    workflow_mode="Single Image",
    experiment_queue=[], 
    current_queue_idx=0,
    master_sholl_data=[], 
    master_metrics=[],
    available_groups=["Control", "Treatment"],
    # Per-image state
    soma_points=[], ui_mode="queue_setup",
    last_click1=None, last_click2=None,
    rejected_cells=set(),
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
    st.session_state.step_size = step_size_val
    st.session_state.um_per_px = um_per_px_val
    st.session_state.ui_mode   = "qc"

def _do_restart():
    for k, v in _defaults.items():
        st.session_state[k] = v if not isinstance(v, (list, set, dict)) else type(v)()
    if BATCH_TEMP_DIR.exists():
        shutil.rmtree(BATCH_TEMP_DIR)
        BATCH_TEMP_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PROGRESS INDICATOR
# ─────────────────────────────────────────────────────────────
_step_labels = ["① Setup Queue", "② Select Somas", "③ Cell QC", "④ Global Report"]

if st.session_state.ui_mode == "export":
    _step_idx = 3
elif st.session_state.ui_mode == "qc":
    _step_idx = 2
elif st.session_state.ui_mode == "selecting":
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
# CORE ALGORITHM (cached)
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

SAMPLE_DIR = Path(__file__).parent / "sample_images"

def _list_samples():
    if not SAMPLE_DIR.exists(): return []
    return sorted(f.name for f in SAMPLE_DIR.iterdir() if f.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})

# ─────────────────────────────────────────────────────────────
# STEP 1: QUEUE SETUP
# ─────────────────────────────────────────────────────────────
if st.session_state.ui_mode == "queue_setup":
    w1, w2, w3 = st.columns([1, 2, 1])
    workflow_choice = w2.radio(
        "🧠 Select Analysis Mode:", 
        options=["Single Image (Quick)", "Batch Mode (Experiment Groups)"], 
        index=0 if st.session_state.get("workflow_mode", "Single Image") == "Single Image" else 1,
        horizontal=True
    )
    
    new_mode = "Single Image" if "Single Image" in workflow_choice else "Batch Mode"
    if new_mode != st.session_state.workflow_mode:
        st.session_state.workflow_mode = new_mode
        st.session_state.experiment_queue = []
        st.rerun()

    st.markdown("---")

    if st.session_state.workflow_mode == "Batch Mode":
        st.markdown("### Step 1: Experiment Builder")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Define Experimental Groups")
            new_group = st.text_input("New Group Name", "")
            if st.button("➕ Add Group") and new_group and new_group not in st.session_state.available_groups:
                st.session_state.available_groups.append(new_group)
                st.rerun()
            st.write("**Active Groups:**")
            for g in st.session_state.available_groups:
                st.markdown(f"- `{g}`")
            st.markdown("---")
            target_group = st.selectbox("Assign images to group:", st.session_state.available_groups)
        img_col = col2
    else:
        st.markdown("### Step 1: Select an Image")
        target_group = "Single_Run"
        img_col = st.container()

    with img_col:
        if st.session_state.workflow_mode == "Batch Mode":
            st.markdown("#### Add Images to Queue")
            
        tab_sample, tab_upload = st.tabs(["🖼️ Try a Sample Image", "📤 Upload Your Own"])
        
        with tab_sample:
            samples = _list_samples()
            if not samples:
                st.info("No sample images found.")
            else:
                if st.session_state.workflow_mode == "Single Image":
                    st.markdown("Select one of the bundled microscopy images to try the pipeline instantly:")
                
                thumb_cols = st.columns(min(len(samples), 5))
                for col, name in zip(thumb_cols, samples):
                    path = SAMPLE_DIR / name
                    thumb = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if thumb is not None:
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                        h, w = thumb.shape[:2]
                        scale = 150 / max(h, w)
                        thumb_small = cv2.resize(thumb, (int(w * scale), int(h * scale)))
                        col.image(thumb_small, caption=name, use_container_width=True)
                        
                        btn_label = "Use" if st.session_state.workflow_mode == "Single Image" else "➕ Add"
                        if col.button(btn_label, key=f"btn_{name}"):
                            st.session_state.experiment_queue.append({
                                "type": "sample",
                                "filename": name,
                                "path": str(path),
                                "group": target_group
                            })
                            if st.session_state.workflow_mode == "Single Image":
                                st.session_state.ui_mode = "selecting"
                                st.session_state.current_queue_idx = 0
                            else:
                                st.success(f"Added {name} to {target_group}")
                            st.rerun()
                
        with tab_upload:
            is_multiple = (st.session_state.workflow_mode == "Batch Mode")
            uploaded_files = st.file_uploader("Upload Images", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=is_multiple)
            if uploaded_files:
                if not is_multiple:
                    uploaded_files = [uploaded_files]
                btn_label = "Start Analysis" if st.session_state.workflow_mode == "Single Image" else "➕ Add Uploads to Queue"
                if st.button(btn_label, type="primary"):
                    for f in uploaded_files:
                        path = BATCH_TEMP_DIR / f.name
                        with open(path, "wb") as out_f:
                            out_f.write(f.read())
                        st.session_state.experiment_queue.append({
                            "type": "upload",
                            "filename": f.name,
                            "path": str(path),
                            "group": target_group
                        })
                    if st.session_state.workflow_mode == "Single Image":
                        st.session_state.ui_mode = "selecting"
                        st.session_state.current_queue_idx = 0
                    else:
                        st.success(f"Added {len(uploaded_files)} images to {target_group}")
                    st.rerun()
                
    if st.session_state.workflow_mode == "Batch Mode":
        st.divider()
        st.markdown("#### Current Queue Workload")
        if not st.session_state.experiment_queue:
            st.info("Queue is empty. Add images above.")
        else:
            df_queue = pd.DataFrame(st.session_state.experiment_queue)[["filename", "group", "type"]]
            st.dataframe(df_queue, use_container_width=True)
            rc1, rc2 = st.columns([1, 4])
            if rc1.button("🗑️ Clear Queue"):
                st.session_state.experiment_queue = []
                st.rerun()
            if rc2.button("🚀 Start Batch Analysis", type="primary"):
                st.session_state.ui_mode = "selecting"
                st.session_state.current_queue_idx = 0
                st.rerun()
    st.stop()


# ─────────────────────────────────────────────────────────────
# LOAD CURRENT IMAGE (For Selecting and QC steps)
# ─────────────────────────────────────────────────────────────
if st.session_state.ui_mode in ["selecting", "qc"]:
    if st.session_state.current_queue_idx >= len(st.session_state.experiment_queue):
        st.session_state.ui_mode = "export"
        st.rerun()
        
    current_item = st.session_state.experiment_queue[st.session_state.current_queue_idx]
    img_color = cv2.imread(current_item["path"], cv2.IMREAD_COLOR)
    img_gray  = cv2.imread(current_item["path"], cv2.IMREAD_GRAYSCALE)
    if img_color is None or img_gray is None:
        st.error(f"Failed to load {current_item['filename']}.")
        st.stop()
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    
    st.markdown(f"### Image {st.session_state.current_queue_idx + 1} of {len(st.session_state.experiment_queue)}")
    st.markdown(f"**Filename:** `{current_item['filename']}` &nbsp;|&nbsp; **Group:** `{current_item['group']}`")

# ─────────────────────────────────────────────────────────────
# STEP 2 — INTERACTIVE CELL DETECTION
# ─────────────────────────────────────────────────────────────
if st.session_state.ui_mode == "selecting":
    st.caption("👆 Click on somas in either image. Accumulate multiple cells.")
    
    img_container  = st.container()
    ctrl_container = st.container()

    with ctrl_container:
        st.markdown("---")
        st.markdown("#### 🎛️ Denoising & Analysis Parameters")
        c1, c2, c3 = st.columns(3)
        h_val       = c1.slider("NLM Parameter (h)", min_value=1, max_value=20, value=9)
        step_size_w = int(c2.number_input("Sholl Step Size (px)", min_value=1, max_value=50, value=st.session_state.step_size))
        um_per_px_w = float(c3.number_input("Pixel size (µm/px)", min_value=0.01, max_value=10.0, value=st.session_state.um_per_px, step=0.01))

        with st.expander("⚙️ Advanced Pipeline Settings", expanded=False):
            adv1, adv2 = st.columns(2)
            tmpl_win   = adv1.slider("NLM Template Window (px)", 5, 15, 7, step=2)
            search_win = adv2.slider("NLM Search Window (px)", 11, 31, 21, step=2)

    with st.spinner("Applying rigorous morphological filters..."):
        processed_image, binary, skeleton = run_scientific_skeletonization(img_gray, h_val, tmpl_win, search_win)

    ov_left  = img_color.copy()
    ov_right = np.zeros((*processed_image.shape, 3), dtype=np.uint8)
    ov_right[skeleton > 0] = [255, 255, 255]

    for idx, (cx, cy) in enumerate(st.session_state.soma_points, start=1):
        cv2.circle(ov_left,  (int(cx), int(cy)), 5, SOMA_GREEN, -1)
        cv2.circle(ov_right, (int(cx), int(cy)), 5, SOMA_GREEN, -1)
        cv2.putText(ov_left,  str(idx), (int(cx)+10, int(cy)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, SOMA_GREEN, 2, cv2.LINE_AA)
        cv2.putText(ov_right, str(idx), (int(cx)+10, int(cy)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, SOMA_GREEN, 2, cv2.LINE_AA)

    scale_ratio    = min(1.0, DISPLAY_WIDTH_PX / float(ov_left.shape[1]))
    new_dim        = (int(ov_left.shape[1] * scale_ratio), int(ov_left.shape[0] * scale_ratio))
    ov_left_small  = cv2.resize(ov_left,  new_dim, interpolation=cv2.INTER_AREA)
    ov_right_small = cv2.resize(ov_right, new_dim, interpolation=cv2.INTER_AREA)

    with img_container:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Fluorescent Image**")
            val1 = streamlit_image_coordinates(ov_left_small,  key="img1")
        with col2:
            st.markdown("**Skeleton Tracing**")
            val2 = streamlit_image_coordinates(ov_right_small, key="img2")

    if val1 is not None:
        coord1 = (val1["x"], val1["y"])
        if coord1 != st.session_state.last_click1:
            st.session_state.soma_points.append((int(val1["x"] / scale_ratio), int(val1["y"] / scale_ratio)))
            st.session_state.last_click1 = coord1
            st.rerun()
    if val2 is not None:
        coord2 = (val2["x"], val2["y"])
        if coord2 != st.session_state.last_click2:
            st.session_state.soma_points.append((int(val2["x"] / scale_ratio), int(val2["y"] / scale_ratio)))
            st.session_state.last_click2 = coord2
            st.rerun()

    b1, b2, b3 = st.columns([1, 1, 2])
    b1.button("Undo Last ↺", on_click=undo_last)
    b2.button("Clear All ✗",  on_click=clear_all)
    if not st.session_state.soma_points:
        if b3.button("Skip Image ⏭️"):
            st.session_state.current_queue_idx += 1
            st.session_state.soma_points = []
            st.session_state.rejected_cells = set()
            st.rerun()
    else:
        b3.button(f"Process Somas ({len(st.session_state.soma_points)}) ✓", on_click=accept_all, args=(step_size_w, um_per_px_w), type="primary")

# ─────────────────────────────────────────────────────────────
# STEP 3 — QC DASHBOARD
# ─────────────────────────────────────────────────────────────
elif st.session_state.ui_mode == "qc":
    step_size   = st.session_state.step_size
    conv_factor = st.session_state.um_per_px

    st.markdown("### Step 3: Quality Control Dashboard")

    if st.button("← Back to Soma Selection"):
        st.session_state.ui_mode = "selecting"
        st.rerun()

    with st.spinner("Running pipeline..."):
        processed_image, binary, skeleton = run_scientific_skeletonization(img_gray, 11)

    farthest_endpoints = measure_farthest_neurite(skeleton, st.session_state.soma_points)

    with st.expander("🌍 Global Topology Viewer", expanded=True):
        glob_ov = np.zeros((*img_gray.shape[:2], 3), dtype=np.uint8)
        glob_ov[skeleton > 0] = [255, 255, 255]

        for idx, (raw_x, raw_y) in enumerate(st.session_state.soma_points, start=1):
            result = get_connected_component(skeleton, (raw_y, raw_x))
            if result[0] is None: continue
            _, (cx, cy) = result
            m_rad = DEFAULT_MAX_RADIUS
            ft    = farthest_endpoints.get(idx)
            if ft:
                m_rad = np.ceil(np.sqrt((ft[0]-cx)**2 + (ft[1]-cy)**2) / step_size) * step_size
            rs = np.unique(np.concatenate([generate_concentric_circles(m_rad, step_size), np.arange(m_rad, m_rad + 3*step_size, step_size)]))
            for r in rs:
                rr, cc = _circle_coords(cy, cx, int(r), glob_ov.shape[:2])
                glob_ov[rr, cc] = [255, 60, 60]
            cv2.circle(glob_ov, (cx, cy), 5, SOMA_GREEN, -1)
            cv2.putText(glob_ov, str(idx), (cx+10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, SOMA_GREEN, 2, cv2.LINE_AA)

        fig_g, ax_g = plt.subplots(figsize=(10, 5))
        ax_g.imshow(glob_ov);  ax_g.axis("off")
        ax_g.set_title("Global Skeleton Map with Sholl Rings", fontsize=11, fontweight="bold")
        fig_g.tight_layout(pad=0.5)
        st.pyplot(fig_g);  plt.close(fig_g)

    local_sholl_data = []
    local_summary_rows = []

    for idx, (raw_x, raw_y) in enumerate(st.session_state.soma_points, start=1):
        st.divider()
        comp_mask, (corr_x, corr_y) = get_connected_component(skeleton, (raw_y, raw_x))
        if comp_mask is None:
            st.warning(f"Cell {idx}: No skeleton found near click.")
            continue

        max_radius = DEFAULT_MAX_RADIUS
        farthest   = farthest_endpoints.get(idx)
        if farthest:
            ex, ey     = farthest
            max_radius = np.ceil(np.sqrt((ex-corr_x)**2 + (ey-corr_y)**2) / step_size) * step_size

        radii = np.unique(np.concatenate([generate_concentric_circles(max_radius, step_size), np.arange(max_radius, max_radius + 3*step_size, step_size)]))
        intersections = compute_sholl_intersections(comp_mask, corr_x, corr_y, radii)

        with st.spinner(f"Extracting morphometrics for Cell {idx}..."):
            fd, _log_sizes, _log_counts = box_counting_fractal_dimension_with_data(comp_mask)
            lac                         = box_counting_lacunarity(comp_mask)
            G                           = skeleton_to_graph(comp_mask)
            betw, clos                  = compute_graph_centralities(G)
            sri                         = schoenen_ramification_index(intersections, radii, (corr_x, corr_y), comp_mask)
            soma_area, soma_circ        = soma_shape_metrics(binary, (corr_x, corr_y))

        st.markdown(
            f"<div style='background:linear-gradient(90deg,#1a1a2e,#16213e,#0f3460);padding:0.6rem 1.2rem;border-radius:8px;margin-bottom:0.8rem'>"
            f"<span style='color:#e0e0e0;font-size:1.15rem;font-weight:600'>🧠 Cell {idx}</span>"
            f"<span style='color:#7ee8fa;font-size:0.85rem;margin-left:1.2rem'>Soma at ({corr_x}, {corr_y})</span></div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        _metric_badge(m1, "Fractal Dim",  fd,        "{:.3f}", "fd")
        _metric_badge(m2, "Lacunarity",   lac,       "{:.2f}", "lac")
        _metric_badge(m3, "Ramification", sri,       "{:.2f}", "sri")
        _metric_badge(m4, "Betweenness",  betw,      "{:.3f}", "betw")
        _metric_badge(m5, "Soma Area",    soma_area, "{:.0f}", "soma_area", " px")
        _metric_badge(m6, "Circularity",  soma_circ, "{:.2f}", "soma_circ")

        metrics_dict = dict(FD=fd, Lac=lac, Betw=betw, Clos=clos, SRI=sri, Area=soma_area, Circ=soma_circ, log_inv_sizes=_log_sizes, log_counts=_log_counts)
        with st.spinner(f"Rendering QC dashboard for Cell {idx}..."):
            fig_qc = generate_qc_dashboard(img_gray, binary, skeleton, comp_mask, (corr_x, corr_y), radii, intersections, metrics_dict, G, idx, streamlit_mode=True)
        st.pyplot(fig_qc)
        plt.close(fig_qc)

        accepted = st.checkbox(f"✅ Include Cell {idx}", value=(idx not in st.session_state.rejected_cells), key=f"accept_cell_{idx}")
        if accepted:
            st.session_state.rejected_cells.discard(idx)
            for r, inters in zip(radii, intersections):
                local_sholl_data.append({
                    "Radius (px)": r, "Intersections": inters, "Cell": f"{current_item['filename']}_C{idx}",
                    "Group": current_item['group'], "Image": current_item['filename']
                })
            local_summary_rows.append({
                "Cell": f"{current_item['filename']}_C{idx}",
                "Group": current_item['group'],
                "Image": current_item['filename'],
                "Fractal Dim": round(fd, 4) if not np.isnan(fd) else float("nan"),
                "Lacunarity":  round(lac, 4) if not np.isnan(lac) else float("nan"),
                "Ramification":round(sri, 4) if not np.isnan(sri) else float("nan"),
                "Betweenness": round(betw, 4) if not np.isnan(betw) else float("nan"),
                "Soma Area px":round(soma_area, 1) if not np.isnan(soma_area) else float("nan"),
                "Circularity": round(soma_circ, 4) if not np.isnan(soma_circ) else float("nan"),
            })
        else:
            st.session_state.rejected_cells.add(idx)

    st.divider()
    btn_lbl = "💾 Finish and View Report" if st.session_state.get("workflow_mode") == "Single Image" else "💾 Save Cells & Next Image"
    if st.button(btn_lbl, type="primary"):
        st.session_state.master_sholl_data.extend(local_sholl_data)
        st.session_state.master_metrics.extend(local_summary_rows)
        
        st.session_state.current_queue_idx += 1
        st.session_state.soma_points = []
        st.session_state.rejected_cells = set()
        
        if st.session_state.current_queue_idx >= len(st.session_state.experiment_queue):
            st.session_state.ui_mode = "export"
        else:
            st.session_state.ui_mode = "selecting"
        st.rerun()

# ─────────────────────────────────────────────────────────────
# STEP 4 — GLOBAL PIPELINE REPORT (BATCH AGGREGATION)
# ─────────────────────────────────────────────────────────────
elif st.session_state.ui_mode == "export":
    st.markdown("### Step 4: Batch Global Report")
    
    if st.button("← Back to Setup (Restart)", on_click=_do_restart):
        st.rerun()

    if not st.session_state.master_sholl_data:
        st.warning("No data was collected.")
        st.stop()

    df_sholl = pd.DataFrame(st.session_state.master_sholl_data)
    df_metrics = pd.DataFrame(st.session_state.master_metrics)
    conv_factor = st.session_state.um_per_px
    df_sholl["Radius (µm)"] = df_sholl["Radius (px)"] * conv_factor

    st.markdown("#### 📊 Comparative Sholl Curve")
    pcol1, pcol2 = st.columns([1, 3])
    palette_opt = pcol1.selectbox("🎨 Palette", ["colorblind", "husl", "Set2", "viridis", "flare"])
    show_individual = pcol1.toggle("Show individual cell curves", value=False)

    fig_global, ax_glob = plt.subplots(figsize=(10, 6))
    if show_individual:
        sns.lineplot(data=df_sholl, x="Radius (µm)", y="Intersections", hue="Group", units="Cell", estimator=None, alpha=0.3, linewidth=1.0, palette=palette_opt, ax=ax_glob)
        sns.lineplot(data=df_sholl, x="Radius (µm)", y="Intersections", hue="Group", linewidth=2.5, errorbar=None, palette=palette_opt, ax=ax_glob)
    else:
        sns.lineplot(data=df_sholl, x="Radius (µm)", y="Intersections", hue="Group", linewidth=2.5, errorbar="se", palette=palette_opt, ax=ax_glob)

    ax_glob.set_xlabel(f"Distance from Soma (µm)  [1 px = {conv_factor} µm]", fontweight="bold")
    ax_glob.set_ylabel("Number of Intersections",  fontweight="bold")
    ax_glob.set_title("Grouped Sholl Profiles", fontweight="bold")
    ax_glob.legend(title="Group", frameon=False)
    sns.despine()
    pcol2.pyplot(fig_global)
    plt.close(fig_global)

    st.markdown("#### 📋 Morphometric Comparisons")
    if len(df_metrics) > 0 and "Group" in df_metrics.columns:
        fig_m, axes = plt.subplots(1, 4, figsize=(16, 4))
        metrics_to_plot = ["Fractal Dim", "Ramification", "Soma Area px", "Lacunarity"]
        for ax, metric in zip(axes, metrics_to_plot):
            sns.boxplot(data=df_metrics, x="Group", y=metric, palette=palette_opt, ax=ax, width=0.5)
            sns.stripplot(data=df_metrics, x="Group", y=metric, color="black", alpha=0.5, ax=ax)
            ax.set_title(metric)
            ax.set_xlabel("")
        sns.despine()
        fig_m.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)

    st.divider()
    col_csv1, col_csv2 = st.columns(2)
    col_csv1.download_button(
        "📥 Download Master Sholl CSV",
        data=df_sholl.to_csv(index=False).encode("utf-8"),
        file_name="master_sholl_intersections.csv",
        mime="text/csv"
    )
    col_csv2.download_button(
        "📥 Download Master Metrics CSV",
        data=df_metrics.to_csv(index=False).encode("utf-8"),
        file_name="master_morphometrics.csv",
        mime="text/csv"
    )

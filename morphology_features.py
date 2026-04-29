"""
morphology_features.py — Multi-dimensional morphometric analysis for microglia.

This module provides functions for computing advanced morphological descriptors
beyond classical Sholl analysis, including:

  * Fractal Dimension (D) and Lacunarity (lam) via box-counting
  * Graph-theory metrics (Betweenness / Closeness Centrality) from skeletons
  * Schoenen Ramification Index (Nm / Np)
  * Soma shape descriptors (Area, Circularity)
  * Quality-control back-trace overlay images
  * Interactive QC dashboard for per-cell visual inspection

References:
  - Karperien et al. (2013) — FracLac for ImageJ
  - Morrison et al. (2017) — Graph-based neuronal morphometry
  - Schoenen (1982) — Ramification Index
  - Sholl (1953) — Dendritic field analysis

Dependencies: numpy, cv2, networkx, skimage, matplotlib
"""

import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
#  Helper: tight bounding-box crop with margin
# ---------------------------------------------------------------------------

def _crop_to_content(mask, margin=5):
    """Crop a binary mask to its bounding box + margin."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return mask  # empty
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    rmax = min(mask.shape[0], rmax + margin + 1)
    cmin = max(0, cmin - margin)
    cmax = min(mask.shape[1], cmax + margin + 1)
    return mask[rmin:rmax, cmin:cmax]


# ---------------------------------------------------------------------------
#  1. Fractal Dimension — box-counting
# ---------------------------------------------------------------------------

def box_counting_fractal_dimension(binary_mask):
    """
    Estimate the fractal dimension D of a binary structure via box-counting.

    The mask is first cropped to its tight bounding box to avoid bias from
    large empty margins.  D is the slope of log N(s) vs log(1/s).

    Parameters
    ----------
    binary_mask : np.ndarray (bool or uint8)
        2-D binary image of the structure (e.g. skeleton or component mask).

    Returns
    -------
    float
        Estimated fractal dimension D.  Returns np.nan on failure.
    tuple
        (log_inv_sizes, log_counts, D) for QC plotting.  Use
        ``box_counting_fractal_dimension_with_data`` for this.

    Scientific Rationale
    --------------------
    Box-counting is the standard algorithm implemented in FracLac (Karperien,
    2013).  Fractal dimension captures the space-filling complexity of the
    microglial arbor:  ramified microglia have higher D (~1.4–1.5) compared to
    amoeboid forms (~1.1–1.2).
    """
    result = box_counting_fractal_dimension_with_data(binary_mask)
    return result[0]


def box_counting_fractal_dimension_with_data(binary_mask):
    """
    Same as ``box_counting_fractal_dimension`` but also returns log-log data
    for QC plotting.

    Returns
    -------
    tuple (D, log_inv_sizes, log_counts)
        D : float — fractal dimension (or NaN)
        log_inv_sizes : np.ndarray — log(1/s)
        log_counts : np.ndarray — log(N(s))
    """
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.sum() == 0:
        return (np.nan, np.array([]), np.array([]))

    # Crop to bounding box for unbiased counting
    mask = _crop_to_content(mask, margin=2)

    # Pad to square power-of-2
    size = max(mask.shape)
    p = int(np.ceil(np.log2(max(size, 4))))
    side = 2 ** p
    padded = np.zeros((side, side), dtype=bool)
    padded[:mask.shape[0], :mask.shape[1]] = mask

    # Box sizes: powers of 2 from 2 up to side // 2
    box_sizes = 2 ** np.arange(1, p)
    counts = np.zeros(len(box_sizes), dtype=int)

    for idx, s in enumerate(box_sizes):
        n_blocks_y = side // s
        n_blocks_x = side // s
        reshaped = padded[:n_blocks_y * s, :n_blocks_x * s].reshape(
            n_blocks_y, s, n_blocks_x, s
        )
        counts[idx] = int(np.any(reshaped, axis=(1, 3)).sum())

    # Remove zero-count entries
    valid = counts > 0
    if valid.sum() < 2:
        return (np.nan, np.array([]), np.array([]))

    log_sizes = np.log(1.0 / box_sizes[valid])
    log_counts = np.log(counts[valid].astype(float))

    coeffs = np.polyfit(log_sizes, log_counts, 1)
    D = float(coeffs[0])
    return (D, log_sizes, log_counts)


# ---------------------------------------------------------------------------
#  2. Lacunarity — box-counting (sliding box)
# ---------------------------------------------------------------------------

def box_counting_lacunarity(binary_mask, box_sizes=None):
    """
    Estimate mean lacunarity lam of a binary structure via the gliding-box
    method.  The mask is cropped to its bounding box before computation.

    For each box size *s*, a sliding window counts the mass (number of
    foreground pixels) in each position.  Lacunarity at scale s is:

        lam(s) = var(mass) / mean(mass)^2 + 1

    The returned value is the arithmetic mean of lam across scales.

    Parameters
    ----------
    binary_mask : np.ndarray (bool or uint8)
        2-D binary image.
    box_sizes : list[int] or None
        Box sizes to evaluate.  If None, uses [2, 4, 8, 16, 32, 64] clipped
        to fit the cropped image.

    Returns
    -------
    float
        Mean lacunarity lam.  Returns np.nan if computation fails.

    Scientific Rationale
    --------------------
    Lacunarity measures spatial heterogeneity of the structure.  Low lam
    indicates a homogeneous, space-filling pattern (ramified microglia);
    high lam indicates clustering or gaps (reactive/amoeboid microglia).
    """
    mask = np.asarray(binary_mask, dtype=np.float32)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.float32)

    if mask.sum() == 0:
        return np.nan

    # --- FIX: crop to bounding box to avoid inflated values ---
    bool_mask = mask > 0
    rows = np.any(bool_mask, axis=1)
    cols = np.any(bool_mask, axis=0)
    if not rows.any():
        return np.nan
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    margin = 2
    rmin = max(0, rmin - margin)
    rmax = min(mask.shape[0], rmax + margin + 1)
    cmin = max(0, cmin - margin)
    cmax = min(mask.shape[1], cmax + margin + 1)
    mask = mask[rmin:rmax, cmin:cmax]

    h, w = mask.shape
    if box_sizes is None:
        box_sizes = [s for s in [2, 4, 8, 16, 32, 64] if s < min(h, w)]

    if not box_sizes:
        return np.nan

    lacunarities = []
    for s in box_sizes:
        integral = cv2.integral(mask)
        y_max = h - s + 1
        x_max = w - s + 1
        if y_max <= 0 or x_max <= 0:
            continue
        box_sums = (
            integral[s:s + y_max, s:s + x_max]
            - integral[:y_max, s:s + x_max]
            - integral[s:s + y_max, :x_max]
            + integral[:y_max, :x_max]
        )
        mu = box_sums.mean()
        if mu == 0:
            continue
        var = box_sums.var()
        lac_s = var / (mu ** 2) + 1.0
        lacunarities.append(lac_s)

    return float(np.mean(lacunarities)) if lacunarities else np.nan


# ---------------------------------------------------------------------------
#  3. Skeleton -> Graph conversion + centrality metrics
# ---------------------------------------------------------------------------

def skeleton_to_graph(skeleton_mask):
    """
    Convert a binary skeleton to a NetworkX graph.

    Skeleton pixels are classified as:
      - **Endpoints**: exactly 1 neighbour (degree 1)
      - **Junctions**: >=3 neighbours
      - **Slab pixels**: exactly 2 neighbours (form edges between nodes)

    The resulting graph has junction/endpoint nodes connected by edges whose
    weight equals the branch length in pixels.

    Parameters
    ----------
    skeleton_mask : np.ndarray (bool or uint8)
        Binary skeleton image (1-pixel wide).

    Returns
    -------
    networkx.Graph
        Undirected graph representing the skeleton topology.
    """
    skel = np.asarray(skeleton_mask, dtype=bool)
    if skel.sum() == 0:
        return nx.Graph()

    # Re-skeletonize to guarantee 1-pixel width
    skel = skeletonize(skel)

    # Classify each pixel by its 8-connected neighbour count
    coords = np.argwhere(skel)
    coord_set = set(map(tuple, coords))

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]

    def _neighbours(r, c):
        return [(r + dr, c + dc) for dr, dc in neighbor_offsets
                if (r + dr, c + dc) in coord_set]

    # Identify node pixels (endpoints + junctions)
    node_pixels = {}  # pixel -> node_id
    node_id = 0
    for r, c in coords:
        n = len(_neighbours(r, c))
        if n != 2:  # endpoint (1) or junction (>=3) or isolated (0)
            node_pixels[(r, c)] = node_id
            node_id += 1

    G = nx.Graph()
    for px, nid in node_pixels.items():
        G.add_node(nid, pos=px)

    # Trace branches between node pixels
    visited_edges = set()
    for start_px, start_id in node_pixels.items():
        for nb in _neighbours(*start_px):
            if nb in node_pixels:
                other_id = node_pixels[nb]
                edge = tuple(sorted((start_id, other_id)))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    G.add_edge(start_id, other_id, weight=1)
            else:
                prev = start_px
                curr = nb
                length = 1
                while curr not in node_pixels:
                    nbs = _neighbours(*curr)
                    nbs = [n for n in nbs if n != prev]
                    if not nbs:
                        break
                    length += 1
                    prev = curr
                    curr = nbs[0]
                if curr in node_pixels:
                    other_id = node_pixels[curr]
                    edge = tuple(sorted((start_id, other_id)))
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        G.add_edge(start_id, other_id, weight=length)

    return G


def compute_graph_centralities(G):
    """
    Compute mean Betweenness and Closeness Centrality for a skeleton graph.

    Parameters
    ----------
    G : networkx.Graph
        Skeleton graph (from ``skeleton_to_graph``).

    Returns
    -------
    tuple[float, float]
        (mean_betweenness, mean_closeness).  Returns (nan, nan) for empty
        or trivial graphs.

    Scientific Rationale
    --------------------
    Betweenness centrality identifies critical branch points whose removal
    would disconnect large portions of the arbor.
    Closeness centrality reflects how quickly signals could spread from any
    point to the rest.
    """
    if G.number_of_nodes() < 2:
        return (np.nan, np.nan)

    betw = nx.betweenness_centrality(G, weight='weight')
    clos = nx.closeness_centrality(G, distance='weight')

    mean_betw = float(np.mean(list(betw.values())))
    mean_clos = float(np.mean(list(clos.values())))
    return mean_betw, mean_clos


# ---------------------------------------------------------------------------
#  4. Schoenen Ramification Index
# ---------------------------------------------------------------------------

def schoenen_ramification_index(intersections, radii, soma_point,
                                 skeleton_mask):
    """
    Compute the Schoenen Ramification Index  SRI = Nm / Np.

    Parameters
    ----------
    intersections : array-like
        Sholl intersection counts at each radius.
    radii : array-like
        Corresponding radii.
    soma_point : tuple (x, y)
        Soma coordinates (x, y) -- note: column, row order.
    skeleton_mask : np.ndarray (bool or uint8)
        Binary skeleton of the cell component.

    Returns
    -------
    float
        Ramification index.  Returns np.nan if Np == 0.

    Scientific Rationale
    --------------------
    Nm (maximum intersections) reflects peak branching complexity.
    Np (number of primary branches) is the count of skeleton pixels directly
    adjacent to the soma, approximating the first-order processes.
    SRI normalises complexity by the number of primary branches, enabling
    cross-cell comparison (Schoenen, 1982).
    """
    intersections = np.asarray(intersections)
    radii = np.asarray(radii)
    if intersections.size == 0:
        return np.nan

    Nm = float(intersections.max())

    # Count primary branches: skeleton pixels in 8-neighbourhood of soma
    skel = np.asarray(skeleton_mask, dtype=bool)
    sx, sy = int(soma_point[0]), int(soma_point[1])
    h, w = skel.shape

    Np = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            r, c = sy + dr, sx + dc
            if 0 <= r < h and 0 <= c < w and skel[r, c]:
                Np += 1

    if Np == 0:
        # --- FIX: expand fallback search up to a radius outside the soma ---
        ys, xs = np.where(skel)
        if len(xs) > 0:
            dists = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
            
            # Since soma might be hollowed/large, use ~1.5x the first Sholl radius to find branches
            search_r = float(radii[0] * 1.5) if len(radii) > 0 else 12.0
            search_r_inner = float(radii[0] * 0.5) if len(radii) > 0 else 4.0
            
            Np = int(np.sum((dists > search_r_inner) & (dists <= search_r)))

    if Np == 0:
        # Last resort: use the first non-zero Sholl intersection count
        non_zero = intersections[intersections > 0]
        if len(non_zero) > 0:
            Np = int(non_zero[0])

    return float(Nm / Np) if Np > 0 else np.nan


# ---------------------------------------------------------------------------
#  5. Soma shape metrics
# ---------------------------------------------------------------------------

MIN_SOMA_AREA_PX = 10  # regions smaller than this are artifacts, not somata

def soma_shape_metrics(binary_image, soma_point, search_radius=30):
    """
    Compute area and circularity of the soma region.

    The soma is identified as the largest connected foreground region near
    ``soma_point`` (filtered to area >= MIN_SOMA_AREA_PX).

    Parameters
    ----------
    binary_image : np.ndarray (uint8)
        Binary image (from binarization step), NOT the skeleton.
    soma_point : tuple (x, y)
        Soma coordinates (col, row).
    search_radius : int
        Radius around soma_point in which to search for a region.

    Returns
    -------
    tuple[float, float]
        (soma_area_px, soma_circularity).
        Circularity = 4*pi*Area / Perimeter^2.  Perfect circle = 1.0.
        Clamped to [0, 1]. Returns (nan, nan) if no region is found.

    Scientific Rationale
    --------------------
    Soma area increases and circularity approaches 1.0 during microglial
    activation (ramified -> amoeboid transition), making these two simple
    scalar descriptors powerful classifiers of activation state.
    """
    binary = np.asarray(binary_image, dtype=np.uint8)
    if binary.max() <= 1:
        binary = binary * 255

    sx, sy = int(soma_point[0]), int(soma_point[1])
    h, w = binary.shape

    # Crop a region around the soma for efficiency
    y0 = max(0, sy - search_radius)
    y1 = min(h, sy + search_radius)
    x0 = max(0, sx - search_radius)
    x1 = min(w, sx + search_radius)
    crop = binary[y0:y1, x0:x1]

    # --- FIX: Isolate the soma body from its dendrites ---
    # Apply morphological opening to detach thin branches, leaving the thick core
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened_crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)

    labeled = label(opened_crop > 0)
    regions = regionprops(labeled)
    if not regions:
        return (np.nan, np.nan)

    # --- FIX: filter tiny regions and prefer largest nearby ---
    soma_local = (sy - y0, sx - x0)
    candidates = []
    for reg in regions:
        if reg.area < MIN_SOMA_AREA_PX:
            continue  # skip pixel-scale fragments
        cy, cx = reg.centroid
        d = np.sqrt((cy - soma_local[0]) ** 2 + (cx - soma_local[1]) ** 2)
        candidates.append((d, reg))

    if not candidates:
        # All regions were too small — fall back to closest regardless of size
        best_region = min(regions, key=lambda r: np.sqrt(
            (r.centroid[0] - soma_local[0]) ** 2 +
            (r.centroid[1] - soma_local[1]) ** 2
        ))
    else:
        # Among regions within a reasonable distance, pick the largest
        # "Reasonable" = within 2x the distance of the closest one
        candidates.sort(key=lambda x: x[0])
        min_dist = candidates[0][0]
        close_candidates = [
            (d, r) for d, r in candidates if d <= max(min_dist * 2, 10)
        ]
        best_region = max(close_candidates, key=lambda x: x[1].area)[1]

    area = float(best_region.area)
    perimeter = float(best_region.perimeter)
    if perimeter == 0:
        return (area, np.nan)

    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
    # --- FIX: clamp circularity to [0, 1] ---
    circularity = float(np.clip(circularity, 0.0, 1.0))
    return (area, circularity)


# ---------------------------------------------------------------------------
#  6. QC back-trace overlay (file-based)
# ---------------------------------------------------------------------------

def generate_backtrace_overlay(original_image, skeleton_mask, soma_point,
                                radii, output_path, intersections=None):
    """
    Generate and save a quality-control overlay image.

    Composites:
      - Original image (grayscale background)
      - Skeleton pixels (green)
      - Sholl concentric circles (red)
      - Soma position (yellow marker)
      - Optional: intersection counts annotated on circles

    Parameters
    ----------
    original_image : np.ndarray
        Original grayscale microscopy image.
    skeleton_mask : np.ndarray (bool or uint8)
        Binary skeleton (component mask for this cell).
    soma_point : tuple (x, y)
        Soma coordinates.
    radii : array-like
        Sholl radii.
    output_path : str
        File path to save the overlay PNG.
    intersections : array-like or None
        If provided, annotate each circle with its count.

    Returns
    -------
    None
    """
    if len(original_image.shape) == 2:
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original_image.copy()

    skel_bool = np.asarray(skeleton_mask, dtype=bool)
    overlay[skel_bool, 0] = 0    # B
    overlay[skel_bool, 1] = 255  # G
    overlay[skel_bool, 2] = 0    # R (in BGR)

    sx, sy = int(soma_point[0]), int(soma_point[1])

    for i, r in enumerate(radii):
        cv2.circle(overlay, (sx, sy), int(r), (0, 0, 255), 1)
        if intersections is not None and i < len(intersections):
            # Distribute text around the circles to prevent overlapping
            angle = (i % 12) * (2 * np.pi / 12)
            txt_x = int(sx + r * np.cos(angle))
            txt_y = int(sy - r * np.sin(angle))
            cv2.putText(overlay, str(int(intersections[i])),
                        (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.circle(overlay, (sx, sy), 4, (0, 255, 255), -1)
    cv2.imwrite(output_path, overlay)


# ---------------------------------------------------------------------------
#  7. Interactive QC Dashboard
# ---------------------------------------------------------------------------

def _flag_color(value, lo, hi):
    """Return green/orange/red for normal/edge/anomalous values."""
    if np.isnan(value):
        return 'red'
    if lo <= value <= hi:
        return 'green'
    return 'darkorange'


def generate_qc_dashboard(original_image, binary_image, skeleton_mask,
                           cell_skeleton, soma_point, radii, intersections,
                           metrics, graph, soma_id, output_path=None,
                           streamlit_mode=False):
    """
    Display an interactive multi-panel QC dashboard for one cell.

    Panels:
      A — Backtrace overlay (original + skeleton + Sholl circles)
      B — Soma region (zoomed crop with detected contour)
      C — Skeleton graph (with nodes coloured by type)
      D — Fractal log-log plot
      E — Metric card with colour-coded flags

    Parameters
    ----------
    original_image : np.ndarray
        Original grayscale microscopy image.
    binary_image : np.ndarray
        Binary image from binarization step.
    skeleton_mask : np.ndarray
        Full skeleton mask.
    cell_skeleton : np.ndarray
        Connected-component skeleton for this cell.
    soma_point : tuple (x, y)
        Soma coordinates.
    radii : list/array
        Sholl radii used.
    intersections : list/array
        Intersection counts per radius.
    metrics : dict
        Keys: 'FD', 'Lac', 'Betw', 'Clos', 'SRI', 'Area', 'Circ',
              'log_inv_sizes', 'log_counts'.
    graph : networkx.Graph
        Skeleton graph for this cell.
    soma_id : int
        Cell identifier.
    output_path : str or None
        If provided, save the dashboard as a PNG image.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Shared helpers ───────────────────────────────────────────────────
    sx, sy = int(soma_point[0]), int(soma_point[1])

    # Pre-compute crop bounds (used by Panels A, B, C)
    crop_r = int(max(radii) * 1.3) if len(radii) > 0 else 100
    r0 = max(0, sy - crop_r)
    r1 = min(original_image.shape[0], sy + crop_r)
    c0 = max(0, sx - crop_r)
    c1 = min(original_image.shape[1], sx + crop_r)

    area_val = metrics.get('Area', np.nan)
    circ_val = metrics.get('Circ', np.nan)
    fd_val   = metrics.get('FD', np.nan)
    sri_val  = metrics.get('SRI', np.nan)
    log_inv  = metrics.get('log_inv_sizes', np.array([]))
    log_cnt  = metrics.get('log_counts', np.array([]))

    # ── Build overlay for Panel A ────────────────────────────────────────
    if len(original_image.shape) == 2:
        overlay = np.stack([original_image] * 3, axis=-1).copy()
    else:
        overlay = original_image.copy()
    overlay[np.asarray(cell_skeleton, dtype=bool)] = [0, 255, 0]
    for r in radii:
        rr, cc = _circle_coords(sy, sx, int(r), overlay.shape[:2])
        overlay[rr, cc] = [255, 60, 60]
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            rr, cc_s = sy + dr, sx + dc
            if 0 <= rr < overlay.shape[0] and 0 <= cc_s < overlay.shape[1]:
                if dr*dr + dc*dc <= 9:
                    overlay[rr, cc_s] = [255, 255, 0]

    # ── Build soma zoom for Panel B ──────────────────────────────────────
    sr = 35
    y0 = max(0, sy - sr);  y1 = min(binary_image.shape[0], sy + sr)
    x0 = max(0, sx - sr);  x1 = min(binary_image.shape[1], sx + sr)
    soma_crop = binary_image[y0:y1, x0:x1].copy()
    soma_u8   = (soma_crop > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(soma_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    soma_rgb = np.stack([soma_crop] * 3, axis=-1)
    if soma_rgb.max() <= 1:
        soma_rgb = (soma_rgb * 255).astype(np.uint8)
    cv2.drawContours(soma_rgb, contours, -1, (0, 255, 255), 1)
    local_sy, local_sx = sy - y0, sx - x0
    cv2.circle(soma_rgb, (local_sx, local_sy), 2, (255, 255, 0), -1)

    area_str = f"{area_val:.0f}" if not np.isnan(area_val) else "N/A"
    circ_str = f"{circ_val:.3f}" if not np.isnan(circ_val) else "N/A"

    # ── Build skeleton graph visual for Panel C ──────────────────────────
    skel_vis = np.zeros((*cell_skeleton.shape[:2], 3), dtype=np.uint8)
    skel_vis[np.asarray(cell_skeleton, dtype=bool)] = [100, 100, 100]
    skel_crop = skel_vis[r0:r1, c0:c1].copy()

    # =====================================================================
    #  STREAMLIT MODE — compact 5-panel layout (no redundant metric card)
    # =====================================================================
    if streamlit_mode:
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f"QC Dashboard  —  Soma {soma_id}", fontsize=13,
                     fontweight='bold', color='#333')
        gs = GridSpec(2, 6, figure=fig, hspace=0.32, wspace=0.55)

        # Row 1: A (2 cols) | B (2 cols) | C (2 cols)
        ax_a = fig.add_subplot(gs[0, 0:2])
        ax_a.imshow(overlay[r0:r1, c0:c1]);  ax_a.axis('off')
        ax_a.set_title("A) Sholl Back-trace", fontsize=10, fontweight='bold')

        ax_b = fig.add_subplot(gs[0, 2:4])
        ax_b.imshow(soma_rgb);  ax_b.axis('off')
        ax_b.set_title(f"B) Soma  |  Area={area_str} px  Circ={circ_str}",
                        fontsize=10, fontweight='bold')

        ax_c = fig.add_subplot(gs[0, 4:6])
        ax_c.imshow(skel_crop);  ax_c.axis('off')
        if graph.number_of_nodes() > 0:
            betw_dict = nx.betweenness_centrality(graph, weight='weight') \
                if graph.number_of_nodes() >= 2 else {}
            for nid, data in graph.nodes(data=True):
                pos = data.get('pos', None)
                if pos is None:
                    continue
                nr, nc = pos
                pr, pc = nr - r0, nc - c0
                if 0 <= pr < skel_crop.shape[0] and 0 <= pc < skel_crop.shape[1]:
                    degree = graph.degree(nid)
                    b = betw_dict.get(nid, 0)
                    size = max(3, int(b * 40))
                    color = 'cyan' if degree == 1 else 'red'
                    ax_c.plot(pc, pr, 'o', color=color, markersize=size, alpha=0.8)
        ax_c.set_title(f"C) Skeleton Graph  |  {graph.number_of_nodes()} nodes  "
                        f"{graph.number_of_edges()} edges",
                        fontsize=10, fontweight='bold')

        # Row 2: D (3 cols) | E (3 cols) — wider panels for charts
        ax_d = fig.add_subplot(gs[1, 0:3])
        if len(log_inv) >= 2:
            ax_d.scatter(log_inv, log_cnt, c='steelblue', s=40, zorder=3)
            try:
                coeffs = np.polyfit(log_inv, log_cnt, 1)
                x_fit  = np.linspace(log_inv.min(), log_inv.max(), 50)
                ax_d.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2,
                          label=f'D = {fd_val:.3f}')
                ax_d.legend(fontsize=10, loc='lower right')
            except np.linalg.LinAlgError:
                ax_d.text(0.5, 0.5, "Fit failed", ha='center', va='center',
                          transform=ax_d.transAxes, fontsize=12, color='red')
        else:
            ax_d.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                      transform=ax_d.transAxes, fontsize=12, color='red')
        ax_d.set_xlabel("log(1/s)", fontsize=9)
        ax_d.set_ylabel("log N(s)", fontsize=9)
        ax_d.set_title("D) Fractal Dimension (box-counting)", fontsize=10, fontweight='bold')
        ax_d.grid(True, alpha=0.3)

        ax_e = fig.add_subplot(gs[1, 3:6])
        radii_arr = np.asarray(radii)
        inter_arr = np.asarray(intersections)
        ax_e.plot(radii_arr, inter_arr, 'o-', color='steelblue', linewidth=1.5, markersize=4)
        ax_e.fill_between(radii_arr, 0, inter_arr, alpha=0.15, color='steelblue')
        ax_e.set_xlabel("Radius (px)", fontsize=9)
        ax_e.set_ylabel("Intersections", fontsize=9)
        sri_str = f"SRI={sri_val:.2f}" if not np.isnan(sri_val) else "SRI=N/A"
        ax_e.set_title(f"E) Sholl Profile  |  {sri_str}", fontsize=10, fontweight='bold')
        ax_e.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig

    # =====================================================================
    #  DESKTOP MODE — original 6-panel layout with interactive buttons
    # =====================================================================
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"QC Dashboard  ---  Soma {soma_id}", fontsize=14,
                 fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(overlay[r0:r1, c0:c1])
    ax_a.set_title("A) Sholl Back-trace", fontsize=10, fontweight='bold')
    ax_a.axis('off')

    # Panel B
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(soma_rgb)
    ax_b.set_title(f"B) Soma  |  Area={area_str} px  Circ={circ_str}",
                    fontsize=10, fontweight='bold')
    ax_b.axis('off')

    # Panel C
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(skel_crop)
    if graph.number_of_nodes() > 0:
        betw_dict = nx.betweenness_centrality(graph, weight='weight') \
            if graph.number_of_nodes() >= 2 else {}
        for nid, data in graph.nodes(data=True):
            pos = data.get('pos', None)
            if pos is None:
                continue
            nr, nc = pos
            pr, pc = nr - r0, nc - c0
            if 0 <= pr < skel_crop.shape[0] and 0 <= pc < skel_crop.shape[1]:
                degree = graph.degree(nid)
                b = betw_dict.get(nid, 0)
                size = max(3, int(b * 40))
                color = 'cyan' if degree == 1 else 'red'
                ax_c.plot(pc, pr, 'o', color=color, markersize=size, alpha=0.8)
    ax_c.set_title(f"C) Skeleton Graph  |  {graph.number_of_nodes()} nodes  "
                    f"{graph.number_of_edges()} edges",
                    fontsize=10, fontweight='bold')
    ax_c.axis('off')

    # Panel D
    ax_d = fig.add_subplot(gs[1, 0])
    if len(log_inv) >= 2:
        ax_d.scatter(log_inv, log_cnt, c='steelblue', s=40, zorder=3)
        try:
            coeffs = np.polyfit(log_inv, log_cnt, 1)
            x_fit = np.linspace(log_inv.min(), log_inv.max(), 50)
            ax_d.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2,
                      label=f'D = {fd_val:.3f}')
            ax_d.legend(fontsize=10, loc='lower right')
        except np.linalg.LinAlgError:
            ax_d.text(0.5, 0.5, "Fit failed (SVD Error)", ha='center', va='center',
                      transform=ax_d.transAxes, fontsize=12, color='red')
    else:
        ax_d.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                  transform=ax_d.transAxes, fontsize=12, color='red')
    ax_d.set_xlabel("log(1/s)", fontsize=9)
    ax_d.set_ylabel("log N(s)", fontsize=9)
    ax_d.set_title("D) Fractal Dimension (box-counting)", fontsize=10, fontweight='bold')
    ax_d.grid(True, alpha=0.3)

    # Panel E
    ax_e = fig.add_subplot(gs[1, 1])
    radii_arr = np.asarray(radii)
    inter_arr = np.asarray(intersections)
    ax_e.plot(radii_arr, inter_arr, 'o-', color='steelblue', linewidth=1.5, markersize=4)
    ax_e.fill_between(radii_arr, 0, inter_arr, alpha=0.15, color='steelblue')
    ax_e.set_xlabel("Radius (px)", fontsize=9)
    ax_e.set_ylabel("Intersections", fontsize=9)
    sri_str = f"SRI={sri_val:.2f}" if not np.isnan(sri_val) else "SRI=N/A"
    ax_e.set_title(f"E) Sholl Profile  |  {sri_str}", fontsize=10, fontweight='bold')
    ax_e.grid(True, alpha=0.3)

    # Panel F (desktop only — metric summary card)
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis('off')
    metric_rows = [
        ("Fractal Dimension", fd_val, 0.7, 1.8),
        ("Lacunarity", metrics.get('Lac', np.nan), 1.0, 10.0),
        ("Betweenness C.", metrics.get('Betw', np.nan), 0.0, 0.5),
        ("Closeness C.", metrics.get('Clos', np.nan), 0.0, 0.3),
        ("Ramification Idx", sri_val, 0.5, 20.0),
        ("Soma Area (px)", area_val, 50.0, 2000.0),
        ("Soma Circularity", circ_val, 0.03, 1.0),
    ]
    y_pos = 0.92
    ax_f.text(0.05, y_pos + 0.06, "F) Metric Summary",
              fontsize=11, fontweight='bold', transform=ax_f.transAxes)
    for name, val, lo, hi in metric_rows:
        color = _flag_color(val, lo, hi)
        val_str = f"{val:.4f}" if not np.isnan(val) else "N/A"
        marker = "o" if color == 'green' else ("!" if color == 'darkorange' else "X")
        ax_f.text(0.05, y_pos, f"[{marker}]  {name}:  {val_str}",
                  fontsize=10, color=color, transform=ax_f.transAxes,
                  fontfamily='monospace')
        y_pos -= 0.12

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig

    # Add interactive Accept/Reject buttons to fix Windows input() crash
    ax_acc = plt.axes([0.65, 0.02, 0.08, 0.05])
    ax_rej = plt.axes([0.75, 0.02, 0.08, 0.05])
    ax_all = plt.axes([0.85, 0.02, 0.08, 0.05])
    
    btn_acc = Button(ax_acc, 'Accept', color='lightgreen', hovercolor='palegreen')
    btn_rej = Button(ax_rej, 'Reject', color='salmon', hovercolor='lightcoral')
    btn_all = Button(ax_all, 'Accept All', color='lightblue', hovercolor='skyblue')
    
    response_container = {'choice': 'n'}  # default to reject if window is closed without clicking
    
    def on_acc(event):
        response_container['choice'] = 'y'
        plt.close(fig)
        
    def on_rej(event):
        response_container['choice'] = 'n'
        plt.close(fig)
        
    def on_all(event):
        response_container['choice'] = 'q'
        plt.close(fig)
        
    btn_acc.on_clicked(on_acc)
    btn_rej.on_clicked(on_rej)
    btn_all.on_clicked(on_all)
    
    # Block script until one button is clicked or window is explicitly closed
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except Exception:
        pass
    plt.show(block=True)

    return response_container['choice']


def _circle_coords(cy, cx, radius, shape):
    """Generate (row, col) arrays for circle perimeter pixels (Bresenham)."""
    angles = np.linspace(0, 2 * np.pi, max(360, radius * 6))
    rr = (cy + radius * np.sin(angles)).astype(int)
    cc = (cx + radius * np.cos(angles)).astype(int)
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return rr[mask], cc[mask]

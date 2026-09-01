import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops, shannon_entropy
from scipy.spatial import distance
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.restoration import estimate_sigma
import concurrent.futures
from matplotlib.widgets import Slider, Button
# Multi-dimensional morphometric feature extraction
from morphology_features import (
    box_counting_fractal_dimension,
    box_counting_fractal_dimension_with_data,
    box_counting_lacunarity,
    skeleton_to_graph,
    compute_graph_centralities,
    schoenen_ramification_index,
    soma_shape_metrics,
    generate_backtrace_overlay,
    generate_qc_dashboard,
)
from provenance import build_manifest, write_manifest
from pipeline_config import (
    DEFAULT_SHOLL_STEP_PX,
    NLM_BASELINE_SIGMA_FACTOR,
    nlm_baseline_from_sigma,
    nlm_review_range,
)

# Sholl Analysis Pipeline for Neuronal Morphology from Microscopy Images
# This script implements an automated Sholl analysis workflow for quantifying dendritic arborization in neuronal or microglial cells from grayscale microscopy images.
# The pipeline employs advanced image preprocessing techniques to enhance contrast and reduce noise, followed by skeletonization and interactive soma selection.
# Sholl intersections are computed individually for each cell to avoid overlap, ensuring accurate morphological profiling.
# Outputs include CSV files with intersection data, intermediate images, and visualization plots.
# Key innovations: Adaptive CLAHE patching for uneven illumination, multi-stage denoising with user validation, and connected component isolation for multi-cell images.
# References: Sholl (1953) for concentric circle analysis; CLAHE (Pizer et al., 1987); Skeletonization (Lee et al., 1994).

def gaussian_weight_mask(size):
    """
    Generate a Gaussian weight mask for blending overlapping image patches.
    
    This function creates a 2D Gaussian kernel normalized to [0,1], used for seamless blending in adaptive patching.
    Blending reduces artifacts at patch boundaries, improving overall image quality for downstream skeletonization.
    
    Parameters:
    - size (int): The size of the square mask (e.g., patch size).
    
    Returns:
    - np.ndarray: A 2D Gaussian mask of shape (size, size).
    
    Scientific Rationale: Gaussian weighting mimics natural image gradients, minimizing discontinuities in contrast-enhanced images (Gonzalez & Woods, 2018).
    """
    size = int(size)
    sigma = size / 4.0 
    k = cv2.getGaussianKernel(size, sigma)
    gaussian = k @ k.T
    return gaussian / np.max(gaussian)

def process_patch(patch, global_var):
    """
    Apply adaptive Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image patch.
    
    CLAHE enhances local contrast while preventing over-amplification of noise. Clip limits are dynamically adjusted based on patch variance and entropy,
    adapting to regions with varying illumination or texture (e.g., dense vs. sparse neuronal structures).
    
    Parameters:
    - patch (np.ndarray): Grayscale image patch.
    - global_var (float): Global image variance for normalization.
    
    Returns:
    - np.ndarray: CLAHE-enhanced patch.
    
    Scientific Rationale: Adaptive CLAHE addresses uneven staining in microscopy images, improving skeleton fidelity (Zuiderveld, 1994).
    Thresholds are empirically derived from image statistics to balance enhancement and artifact suppression.
    """
    patch_var = np.var(patch)
    patch_entropy = shannon_entropy(patch)

    # Empirical thresholds for adaptive CLAHE
    low_var_thresh = 0.7 * global_var
    high_entropy_thresh = 5
    low_entropy_thresh = 3.5

    if patch_var < low_var_thresh and patch_entropy < low_entropy_thresh:
        clip = 6
    elif patch_entropy < low_entropy_thresh:
        clip = 4.8
    elif patch_entropy > high_entropy_thresh:
        clip = 3.6
    elif patch_var < 1.2 * global_var:
        clip = 2.8
    elif patch_var < 2.0 * global_var:
        clip = 2
    else:
        clip = 1.8

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(patch)

def get_entropy_map(image, radius=5):
    """
    Compute a local entropy map of the image.
    
    Entropy quantifies texture complexity, guiding adaptive patch sizing in CLAHE application.
    
    Parameters:
    - image (np.ndarray): Grayscale image.
    - radius (int): Disk radius for local entropy calculation.
    
    Returns:
    - np.ndarray: Entropy map.
    
    Scientific Rationale: Entropy provides a computational criterion for local
    adaptation in heterogeneous images (Haralick et al., 1973); performance is
    dataset-dependent.
    """
    return entropy(image, disk(radius))

def apply_adaptive_patching(image, global_var, entropy_thresh=4, stride_ratio=0.5):
    """
    Apply adaptive CLAHE patching with blending across the entire image.
    
    The image is divided into overlapping patches of variable size (84x84 or 168x168) based on local entropy.
    Patches are processed with CLAHE and blended using Gaussian weights to avoid seams.
    
    Parameters:
    - image (np.ndarray): Input grayscale image.
    - global_var (float): Global variance for adaptive thresholding.
    - entropy_thresh (float): Threshold for patch size selection.
    - stride_ratio (float): Overlap ratio for patching (0.5 = 50% overlap).
    
    Returns:
    - np.ndarray: Contrast-enhanced image.
    
    Scientific Rationale: Patch-based CLAHE with blending corrects for illumination gradients in microscopy, preserving fine neuronal details (Reza, 2004).
    Concurrent processing accelerates computation for large images.
    """
    h, w = image.shape
    if h < 8 or w < 8:
        raise ValueError("Input image must be at least 8 × 8 pixels")
    min_patch_size = 8  # CLAHE tile size

    # Calculate padding needed to make both dimensions multiples of min_patch_size
    pad_h = (min_patch_size - (h % min_patch_size)) % min_patch_size
    pad_w = (min_patch_size - (w % min_patch_size)) % min_patch_size

    # Pad the image with reflection to avoid border artifacts
    padded_image = np.pad(
        image,
        ((0, pad_h), (0, pad_w)),
        mode='reflect'
    )
    ph, pw = padded_image.shape

    reconstructed = np.zeros_like(padded_image, dtype=np.float32)
    weight_map = np.zeros_like(padded_image, dtype=np.float32)

    def process_and_blend(y, x, patch_size):
        # Always extract a patch of patch_size x patch_size
        y0 = min(y, ph - patch_size)
        x0 = min(x, pw - patch_size)
        patch = padded_image[y0:y0+patch_size, x0:x0+patch_size]
        processed = process_patch(patch, global_var)
        weight = gaussian_weight_mask(patch_size)
        return (y0, x0, patch_size, processed, weight)

    tasks = []
    stride = int(84 * stride_ratio)
    for y in range(0, ph, stride):
        for x in range(0, pw, stride):
            # Always use full patch size for entropy calculation
            entropy_size = min(168, ph, pw)
            y0 = max(0, min(y, ph - entropy_size))
            x0 = max(0, min(x, pw - entropy_size))
            patch = padded_image[y0:y0+entropy_size, x0:x0+entropy_size]
            patch_entropy = shannon_entropy(patch)
            requested_size = 84 if patch_entropy < entropy_thresh else 168
            patch_size = min(requested_size, ph, pw)
            tasks.append((y, x, patch_size))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_and_blend, y, x, patch_size) for y, x, patch_size in tasks]
        for future in concurrent.futures.as_completed(futures):
            y0, x0, patch_size, processed, weight = future.result()
            reconstructed[y0:y0+patch_size, x0:x0+patch_size] += processed * weight
            weight_map[y0:y0+patch_size, x0:x0+patch_size] += weight

    weight_map[weight_map == 0] = 1.0
    final_image = reconstructed / weight_map

    # Crop back to original size
    final_image = final_image[:h, :w]
    return np.clip(final_image, 0, 255).astype(np.uint8)

def denoise_image(image, global_var):
    """
    Perform user-guided denoising using Non-local Means (NLM).
    
    Multiple denoising levels (h values) are previewed via skeletonization,
    allowing the user to choose a dataset-appropriate noise reduction level.
    
    Parameters:
    - image (np.ndarray): Input image.
    - global_var (float): Global variance (for potential future adaptation).
    
    Returns:
    - np.ndarray: Denoised image.
    
    Scientific Rationale: NLM denoising preserves edges in biological images, crucial for accurate skeleton tracing (Buades et al., 2005).
    User review helps assess whether denoising is appropriate for the image.
    """
    # User preview for denoising
    denoised, h_used = preview_denoising(image)
    print(f"User selected h={h_used}")
    return denoised

def apply_tophat(image):
    """
    Apply morphological top-hat transform to enhance bright structures.
    
    Top-hat filtering highlights small bright features (e.g., fine dendrites) against a darker background.
    
    Parameters:
    - image (np.ndarray): Grayscale image.
    
    Returns:
    - np.ndarray: Filtered image.
    
    Scientific Rationale: Top-hat enhances contrast for thin structures in microscopy, aiding binarization (Serra, 1982).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def binarize_image(image):
    """
    Binarize the image using Otsu's automatic thresholding.
    
    Converts grayscale to binary, separating foreground (neuronal structures) from background.
    
    Parameters:
    - image (np.ndarray): Grayscale image.
    
    Returns:
    - np.ndarray: Binary image.
    
    Scientific Rationale: Otsu's method maximizes inter-class variance and is
    commonly used as a starting point for bimodal intensity distributions
    (Otsu, 1979); suitability remains dataset-dependent.
    """
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def remove_small_fragments(binary_image, min_size=3):
    """
    Remove small connected components from the binary image.
    
    Eliminates noise or debris smaller than min_size pixels.
    
    Parameters:
    - binary_image (np.ndarray): Binary image.
    - min_size (int): Minimum area in pixels.
    
    Returns:
    - np.ndarray: Cleaned binary image.
    
    Scientific Rationale: Fragment removal reduces false positives in skeletonization, improving morphological accuracy (Haralick & Shapiro, 1992).
    """
    bool_img = binary_image.astype(bool)
    cleaned = remove_small_objects(bool_img, min_size=min_size)
    return (cleaned.astype(np.uint8)) * 255

def remove_isolated_fibers(binary_image, min_length=3):
    """
    Remove short isolated fibers from the skeleton.
    
    Filters skeleton components by area (length proxy).
    
    Parameters:
    - binary_image (np.ndarray): Binary skeleton.
    - min_length (int): Minimum area.
    
    Returns:
    - np.ndarray: Refined skeleton.
    
    Scientific Rationale: Eliminates spurious branches, ensuring skeletons represent true neuronal arborization (Sorzano et al., 2015).
    """
    skel = skeletonize(binary_image > 0)
    labeled = label(skel)
    output = np.zeros_like(binary_image)

    for region in regionprops(labeled):
        if region.area >= min_length:
            output[labeled == region.label] = 255

    return output

def remove_isolated_fibers_refine(binary_image, min_length=30):
    """
    Further refine skeleton by removing longer isolated fibers.
    
    Higher threshold for advanced cleaning.
    
    Parameters:
    - binary_image (np.ndarray): Skeleton image.
    - min_length (int): Minimum area.
    
    Returns:
    - np.ndarray: Highly refined skeleton.
    
    Scientific Rationale: Multi-stage filtering balances sensitivity and specificity in complex morphologies.
    """
    skel = skeletonize(binary_image > 0)
    labeled = label(skel)
    output = np.zeros_like(binary_image)

    for region in regionprops(labeled):
        if region.area >= min_length:
            output[labeled == region.label] = 255

    return output

def apply_morph_close(image):
    """
    Apply morphological closing to fill small holes.
    
    Connects nearby structures by filling gaps.
    
    Parameters:
    - image (np.ndarray): Binary image.
    
    Returns:
    - np.ndarray: Closed image.
    
    Scientific Rationale: Closing preserves topology in fragmented neuronal images (Serra, 1982).
    """
    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_dilate(image):
    """
    Dilate the image to connect nearby structures.
    
    Expands foreground pixels.
    
    Parameters:
    - image (np.ndarray): Binary image.
    
    Returns:
    - np.ndarray: Dilated image.
    
    Scientific Rationale: Dilation bridges minor gaps in dendrites, enhancing connectivity (Haralick & Shapiro, 1992).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(image, kernel, iterations=1)

def skeletonize_image(image):
    """
    Skeletonize the binary image using morphological thinning.
    
    Reduces structures to 1-pixel wide skeletons.
    
    Parameters:
    - image (np.ndarray): Binary image.
    
    Returns:
    - np.ndarray: Skeletonized image.
    
    Scientific Rationale: Skeletonization enables quantitative analysis of branching patterns (Lee et al., 1994).
    """
    binary_bool = (image > 0)
    skeleton = skeletonize(binary_bool)
    return (skeleton.astype(np.uint8)) * 255

def bridge_nearby_fragments(binary_image, max_dist=5):
    """
    Bridge close skeleton endpoints to connect fragments.
    
    Draws lines between nearby endpoints if they belong to different components.
    
    Parameters:
    - binary_image (np.ndarray): Skeleton image.
    - max_dist (int): Maximum distance for bridging.
    
    Returns:
    - np.ndarray: Bridged skeleton.
    
    Scientific Rationale: Fragment bridging reconstructs incomplete dendrites, improving Sholl accuracy in noisy images (Meijering, 2010).
    """
    from skimage.draw import line

    skel = skeletonize(binary_image > 0)
    endpoints = find_endpoints(skel)
    endpoints = np.array(endpoints)

    if len(endpoints) < 2:
        return binary_image  # No connections possible

    # Find all pairs of close endpoints
    dists = distance.cdist(endpoints, endpoints)
    np.fill_diagonal(dists, np.inf)
    pairs = np.argwhere(dists < max_dist)

    # Map components
    labeled, _ = label(binary_image > 0, connectivity=2, return_num=True)

    for i, j in pairs:
        y1, x1 = endpoints[i][1], endpoints[i][0]
        y2, x2 = endpoints[j][1], endpoints[j][0]
        if labeled[y1, x1] != labeled[y2, x2]:
            rr, cc = line(y1, x1, y2, x2)
            binary_image[rr, cc] = 255

    return binary_image

def bridge_nearby_fragments_refine(binary_image, max_dist=12):
    """
    Further bridge fragments with a larger distance threshold.
    
    Extended bridging for robust reconstruction.
    
    Parameters:
    - binary_image (np.ndarray): Skeleton image.
    - max_dist (int): Maximum distance.
    
    Returns:
    - np.ndarray: Refined bridged skeleton.
    
    Scientific Rationale: Multi-threshold bridging adapts to varying image resolutions.
    """
    from skimage.draw import line

    skel = skeletonize(binary_image > 0)
    endpoints = find_endpoints(skel)
    endpoints = np.array(endpoints)

    if len(endpoints) < 2:
        return binary_image  # No connections possible

    # Find all pairs of close endpoints
    dists = distance.cdist(endpoints, endpoints)
    np.fill_diagonal(dists, np.inf)
    pairs = np.argwhere(dists < max_dist)

    # Map components
    labeled, _ = label(binary_image > 0, connectivity=2, return_num=True)

    for i, j in pairs:
        y1, x1 = endpoints[i][1], endpoints[i][0]
        y2, x2 = endpoints[j][1], endpoints[j][0]
        if labeled[y1, x1] != labeled[y2, x2]:
            rr, cc = line(y1, x1, y2, x2)
            binary_image[rr, cc] = 255

    return binary_image

def preprocess_image(image):  # Optional step for preprocessing pipeline visualization
    """
    Complete preprocessing pipeline with intermediate visualization.
    
    Applies full sequence: CLAHE, denoising, top-hat, binarization, morphological operations, skeletonization, and bridging.
    Displays steps for validation.
    
    Parameters:
    - image (np.ndarray): Input grayscale image.
    
    Returns:
    - tuple: Final skeleton and intermediate images.
    
    Scientific Rationale: Visualization aids in troubleshooting preprocessing, ensuring pipeline reliability (Pang et al., 2011).
    """
    # Complete image preprocessing pipeline
    global_var = np.var(image)
    print(f"global_var value: {global_var}")

    # Advanced image processing
    processed_image = apply_adaptive_patching(image, global_var)
    denoised_image = denoise_image(processed_image, global_var)
    den_th_image = apply_tophat(denoised_image)
    binary = binarize_image(den_th_image)
    frag_filtered = remove_small_fragments(binary)
    closed = apply_morph_close(frag_filtered)
    dilated = apply_dilate(closed)
    skel = skeletonize_image(dilated)
    bridged = bridge_nearby_fragments(skel)
    refined = remove_isolated_fibers_refine(bridged)
    bridge_refined = bridge_nearby_fragments_refine(refined)
    skeleton = remove_isolated_fibers(bridge_refined)

    # Show intermediate processing steps (optional)
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title("Patch & Adjustment")
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(den_th_image, cmap='gray')
    plt.title("Top-Hat")
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(binary, cmap='gray')
    plt.title("Binarized")
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.imshow(closed, cmap='gray')
    plt.title("Closing")
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(dilated, cmap='gray')
    plt.title("Dilate")
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(skel, cmap='gray')
    plt.title("Skeleton")
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(skeleton, cmap='gray')
    plt.title("Skeleton bridge")
    plt.axis('off')

    plt.tight_layout()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except Exception:
        pass
    plt.show()

    return skeleton, processed_image, den_th_image, binary

def get_connected_component(skeleton, soma_point):
    """
    Extract the connected component of the skeleton containing the soma.
    
    Uses 8-connectivity labeling to isolate individual neuronal arbors, preventing overlap in multi-cell images.
    
    Parameters:
    - skeleton (np.ndarray): Skeletonized image.
    - soma_point (tuple): (y, x) coordinates of soma.
    
    Returns:
    - tuple: Binary mask of component and corrected soma coordinates.
    
    Scientific Rationale: Component isolation restricts Sholl measurement to
    the selected connected structure. Its fidelity depends on segmentation and
    connectivity quality (Wearne et al., 2005).
    """
    labeled_skel, num_labels = label(skeleton, connectivity=2, return_num=True)
    y, x = soma_point
    soma_label = labeled_skel[y, x]
    if soma_label == 0:
        skel_points = np.column_stack(np.where(skeleton > 0))
        if len(skel_points) == 0:
            return None, (x, y)
        dists = distance.cdist([(y, x)], skel_points)
        nearest_idx = np.argmin(dists)
        y, x = skel_points[nearest_idx]
        soma_label = labeled_skel[y, x]
        if soma_label == 0:
            return None, (x, y)
    component = (labeled_skel == soma_label)
    return component, (x, y)

def find_endpoints(skeleton, component_mask=None):
    """
    Identify endpoints in the skeleton (pixels with one neighbor).
    
    Endpoints indicate branch tips, used for radius estimation.
    
    Parameters:
    - skeleton (np.ndarray): Skeleton image.
    - component_mask (np.ndarray): Optional mask to restrict search.
    
    Returns:
    - list: List of (x, y) endpoint coordinates.
    
    Scientific Rationale: Endpoint detection quantifies arbor extent for Sholl circle generation (Sholl, 1953).
    """
    endpoints = []
    skeleton = (skeleton > 0).astype(np.float32)
    
    height, width = skeleton.shape
    for r, c in zip(*np.where(skeleton)):
        # 3x3 window around the selected pixel (r, c)
        window = skeleton[max(0, r-1):min(height, r+2),
                          max(0, c-1):min(width, c+2)]
        # Count the number of neighbors
        neighbors = np.sum(window) - 1  # Subtract 1 to exclude the center pixel
        
        if neighbors == 1 and (component_mask is None or component_mask[r, c]):
            endpoints.append((c, r))
    
    return endpoints

def measure_farthest_neurite(skeleton, soma_points):
    """
    For each soma, find the farthest endpoint in its connected component.
    
    Determines maximum radius for Sholl analysis.
    
    Parameters:
    - skeleton (np.ndarray): Skeleton image.
    - soma_points (list): List of (x, y) soma coordinates.
    
    Returns:
    - dict: Farthest endpoints per soma ID.
    
    Scientific Rationale: Farthest neurite defines the extent of concentric circles, ensuring comprehensive coverage (Ristanović et al., 2006).
    """
    farthest_endpoints = {}
    for i, (soma_x, soma_y) in enumerate(soma_points, start=1):
        component_mask, (soma_x, soma_y) = get_connected_component(skeleton, (soma_y, soma_x))
        if component_mask is None:
            farthest_endpoints[i] = None
            continue
        endpoints = find_endpoints(skeleton, component_mask)
        if not endpoints:
            farthest_endpoints[i] = None
            continue
        # endpoints are (x, y), so use (soma_x, soma_y)
        dists = distance.cdist([(soma_x, soma_y)], endpoints)
        max_distance_idx = np.argmax(dists)
        farthest_endpoint = endpoints[max_distance_idx]
        farthest_endpoints[i] = farthest_endpoint
    return farthest_endpoints

def generate_concentric_circles(max_radius, step_size):
    """
    Generate radii for Sholl concentric circles.
    
    Creates evenly spaced radii up to max_radius.
    
    Parameters:
    - max_radius (float): Maximum radius.
    - step_size (float): Spacing between radii.
    
    Returns:
    - np.ndarray: Array of radii.
    
    Scientific Rationale: Concentric circles quantify branching density at increasing distances (Sholl, 1953).
    """
    return np.arange(step_size, max_radius + step_size, step_size)

def compute_sholl_intersections(skeleton, soma_x, soma_y, radii):
    """
    Compute Sholl intersections for given radii.
    
    Counts skeleton edges that cross each radius.
    
    Parameters:
    - skeleton (np.ndarray): Skeleton image (or component mask).
    - soma_x, soma_y (float): Soma coordinates.
    - radii (np.ndarray): Radii for analysis.
    
    Returns:
    - np.ndarray: Intersection counts.
    
    Scientific Rationale: Intersection counting measures dendritic complexity, a key metric in neurobiology (Ristanović et al., 2006).
    """
    skel = np.asarray(skeleton, dtype=bool)
    intersections = np.zeros(len(radii), dtype=int)
    if not skel.any():
        return intersections

    yy, xx = np.indices(skel.shape)
    radial_distance = np.hypot(xx - soma_x, yy - soma_y)

    # Visit every undirected 8-connected skeleton edge exactly once. An edge
    # crosses a circle when its endpoints lie on opposite sides. Counting
    # edges avoids the orientation-dependent over-counting caused by counting
    # every skeleton pixel in a radial band.
    forward_offsets = ((0, 1), (1, -1), (1, 0), (1, 1))
    edge_distances = []
    height, width = skel.shape
    for dy, dx in forward_offsets:
        y0 = slice(max(0, -dy), min(height, height - dy))
        x0 = slice(max(0, -dx), min(width, width - dx))
        y1 = slice(max(0, dy), min(height, height + dy))
        x1 = slice(max(0, dx), min(width, width + dx))
        connected = skel[y0, x0] & skel[y1, x1]
        if connected.any():
            edge_distances.append(
                (radial_distance[y0, x0][connected],
                 radial_distance[y1, x1][connected])
            )

    for index, radius in enumerate(np.asarray(radii, dtype=float)):
        count = 0
        for d0, d1 in edge_distances:
            # Half-open convention assigns a vertex lying exactly on a circle
            # once and avoids double-counting its adjacent radial edge.
            count += int(np.count_nonzero(
                ((d0 < radius) & (d1 >= radius)) |
                ((d1 < radius) & (d0 >= radius))
            ))
        intersections[index] = count
    return intersections

def get_filename_without_extension(file_path):
    """
    Extract filename without extension.
    
    Used for output naming.
    
    Parameters:
    - file_path (str): Full path.
    
    Returns:
    - str: Basename without extension.
    """
    base_name = os.path.basename(file_path)
    name_without_extension, _ = os.path.splitext(base_name)
    return name_without_extension

def select_image_file():
    """
    Open file dialog for image selection.
    
    Supports multiple file selection for Batch Processing.
    
    Returns:
    - tuple: Selected file paths.
    """
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title="Select one or more image files for Batch Processing",
        filetypes=[("Image files", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*")]
    )
    root.destroy()
    return file_paths

def get_output_dir(file_path):
    """
    Create output directory based on input filename.
    
    Organizes results per image.
    
    Parameters:
    - file_path (str): Input path.
    
    Returns:
    - str: Output directory path.
    """
    base_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(base_dir, f"{base_name}_sholl_output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    """
    Main Sholl analysis workflow (Batch Processing Ready).
    
    Orchestrates preprocessing, soma selection, intersection calculation, and output generation.
    Includes visualization, per-image CSVs, and a global Master CSV export.
    """
    step_size = DEFAULT_SHOLL_STEP_PX
    extra_circles = 3

    file_paths = select_image_file()
    if not file_paths:
        print("No files selected. Exiting.")
        return

    # Master tracking array for ALL files
    master_sholl_data = []
    total_accepted = 0
    total_rejected = 0

    # Retrieve base directory from the first file to save the Master CSV
    base_master_dir = os.path.dirname(file_paths[0])

    for file_index, file_path in enumerate(file_paths, start=1):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        filename_prefix = get_filename_without_extension(file_path)
        output_dir = get_output_dir(file_path)
        
        print("\n" + "#"*60)
        print(f"PROCESSING IMAGE {file_index}/{len(file_paths)}: {filename_prefix}")
        print("#"*60)

        # Preprocess image up to processed_image
        global_var = np.var(image)
        print(f"  > Global Variance: {global_var:.2f}")
        processed_image = apply_adaptive_patching(image, global_var)
        denoised, h_used = preview_denoising(processed_image)
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

        # Show processed image and skeleton side by side for selection
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.canvas.manager.set_window_title(f"Soma Selection - {filename_prefix}")
        
        ax1.imshow(processed_image, cmap='gray')
        ax1.set_title("Processed Image")
        ax1.axis('off')
        
        ax2.imshow(skeleton, cmap='gray')
        ax2.set_title("Skeletonized Image")
        ax2.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.921)
        plt.suptitle("Click on EITHER image to select soma locations.", fontsize=13, fontweight='bold')
        
        soma_points_raw = []
        ui_elements = {'markers': [], 'texts': []}

        def update_markers():
            # Remove old markers and text
            for m in ui_elements['markers'] + ui_elements['texts']:
                m.remove()
            ui_elements['markers'].clear()
            ui_elements['texts'].clear()

            # Draw new ones on both axes
            for idx, (x, y) in enumerate(soma_points_raw, start=1):
                for ax in (ax1, ax2):
                    m = ax.scatter(x, y, color='cyan', s=40, edgecolors='black', zorder=5)
                    t = ax.text(x + 5, y + 5, str(idx), color='cyan', fontsize=12, fontweight='bold', zorder=5)
                    ui_elements['markers'].append(m)
                    ui_elements['texts'].append(t)

            fig.canvas.draw_idle()

        def onclick(event):
            # Ensure click is inside one of the two image axes
            if event.inaxes in (ax1, ax2):
                soma_points_raw.append((event.xdata, event.ydata))
                update_markers()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # UI Buttons at the bottom
        ax_undo = plt.axes([0.3, 0.02, 0.1, 0.05])
        btn_undo = Button(ax_undo, 'Undo Last', color='lightcoral', hovercolor='salmon')

        ax_clear = plt.axes([0.45, 0.02, 0.1, 0.05])
        btn_clear = Button(ax_clear, 'Clear All', color='khaki', hovercolor='gold')

        ax_acc = plt.axes([0.6, 0.02, 0.15, 0.05])
        btn_acc = Button(ax_acc, 'Accept & Continue', color='lightgreen', hovercolor='palegreen')

        def on_undo(event):
            if soma_points_raw:
                soma_points_raw.pop()
                update_markers()
        btn_undo.on_clicked(on_undo)

        def on_clear(event):
            soma_points_raw.clear()
            update_markers()
        btn_clear.on_clicked(on_clear)

        def on_accept(event):
            plt.close(fig)
        btn_acc.on_clicked(on_accept)

        try:
            plt.get_current_fig_manager().window.state('zoomed')
        except Exception:
            pass
        
        # Block script until window is closed (Accept & Continue pressed or manual close)
        plt.show(block=True)
        
        # Cleanup
        fig.canvas.mpl_disconnect(cid)
        soma_points = soma_points_raw

        if not soma_points:
            print(f"  > No somas selected for {filename_prefix}. Skipping Image...")
            continue

        soma_points = [(int(x), int(y)) for x, y in soma_points]

        # Save intermediate images
        cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_gray_scale_image.png"), image)
        cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_processed_image.png"), processed_image)
        cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_den_th_image.png"), den_th_image)
        cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_binary.png"), binary)
        cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_skeleton.png"), skeleton)
        write_manifest(
            os.path.join(output_dir, f"{filename_prefix}_run_manifest.json"),
            build_manifest(file_path, {
                "nlm_h": h_used,
                "nlm_template_window_px": 7,
                "nlm_search_window_px": 21,
                "sholl_step_px": step_size,
                "extra_circles": extra_circles,
                "top_hat_kernel_px": 35,
                "minimum_fragment_px": 3,
                "bridge_distances_px": [5, 12],
            }),
        )

        # Visualization of soma and radii
        plt.imshow(skeleton, cmap='gray')
        
        for i, (sx, sy) in enumerate(soma_points, start=1):
            plt.scatter(sx, sy, color='cyan', s=20, label=f"Soma {i}")
            plt.text(sx, sy, str(i), color='white', fontsize=12, ha='center', va='center')
        
        farthest_endpoints = measure_farthest_neurite(skeleton, soma_points)
        
        for i, (soma_x, soma_y) in enumerate(soma_points, start=1):
            component_mask, (soma_x, soma_y) = get_connected_component(skeleton, (soma_y, soma_x))
            plt.scatter(soma_x, soma_y, color='blue', s=20, label="Corrected soma")
            farthest_endpoint = farthest_endpoints.get(i)
            if farthest_endpoint:
                ex, ey = farthest_endpoint
                max_radius = np.sqrt((ex - soma_x) ** 2 + (ey - soma_y) ** 2)
                
                max_radius = np.ceil(max_radius / step_size) * step_size
                additional_radii = np.arange(max_radius, max_radius + extra_circles * step_size, step_size)
                
                radii = generate_concentric_circles(max_radius, step_size)
                radii = np.unique(np.concatenate([radii, additional_radii]))
                
                for r in radii:
                    circle = plt.Circle((soma_x, soma_y), r, color='r', fill=False, linestyle='--', alpha=0.5)
                    plt.gca().add_patch(circle)
            else:
                print(f"Warning: No endpoints found for Soma {i}")
            
        plt.title(f"Soma Detection & Sholl Masks - {filename_prefix}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_soma_and_radii.png"))
        plt.close()

        # Generate Intersection Output and Morphometrics
        print(f"  > Starting Morphometric extraction for {filename_prefix}...")
        
        sholl_data = [] # local per-image collector
        accepted_cells = 0
        rejected_cells = 0
        accept_all_remaining = False

        for i, (soma_x, soma_y) in enumerate(soma_points, start=1):
            # 1. Connected component isolation (for this cell)
            component_mask, (soma_x, soma_y) = get_connected_component(skeleton, (soma_y, soma_x))
            if component_mask is None:
                print(f"  >> Skipping Soma {i}: Isolated component not found.")
                continue

            # 2. Get bounding radii based on farthest endpoints
            farthest_endpoint = farthest_endpoints.get(i)
            if farthest_endpoint:
                ex, ey = farthest_endpoint
                max_radius = np.sqrt((ex - soma_x) ** 2 + (ey - soma_y) ** 2)
                max_radius = np.ceil(max_radius / step_size) * step_size
                additional_radii = np.arange(
                    max_radius, max_radius + extra_circles * step_size, step_size
                )
                radii = generate_concentric_circles(max_radius, step_size)
                radii = np.unique(np.concatenate([radii, additional_radii]))
            else:
                radii = []

            # 3. Base Sholl Intersections
            intersections = compute_sholl_intersections(
                component_mask, soma_x, soma_y, radii
            )

            # 4. Advanced Morphometrics Extractor
            # Box-counting features
            fd, _log_sizes, _log_counts = box_counting_fractal_dimension_with_data(component_mask)
            lac = box_counting_lacunarity(component_mask)
            
            # Graph-theory features
            G = skeleton_to_graph(component_mask)
            betw, clos = compute_graph_centralities(G)
            
            # Ramification and Soma shapes
            sri = schoenen_ramification_index(
                intersections, radii, (soma_x, soma_y), component_mask
            )
            soma_area, soma_circ = soma_shape_metrics(binary, (soma_x, soma_y))

            # Visualization Data (Already log-transformed by the extractor)
            log_inv_sizes = _log_sizes
            log_counts = _log_counts

            # 5. QC back-trace overlay image (always saved for audit)
            backtrace_path = os.path.join(
                output_dir, f"{filename_prefix}_backtrace_soma_{i}.png"
            )
            generate_backtrace_overlay(
                image, component_mask, (soma_x, soma_y), radii,
                backtrace_path, intersections=intersections
            )

            # 6. Interactive QC Dashboard — skip if auto-accepting
            if not accept_all_remaining:
                metrics_dict = {
                    'FD': fd, 'Lac': lac, 'Betw': betw, 'Clos': clos,
                    'SRI': sri, 'Area': soma_area, 'Circ': soma_circ,
                    'log_inv_sizes': log_inv_sizes, 'log_counts': log_counts,
                }
                dashboard_path = os.path.join(
                    output_dir, f"{filename_prefix}_qc_dashboard_soma_{i}.png"
                )
                response = generate_qc_dashboard(
                    original_image=image,
                    binary_image=binary,
                    skeleton_mask=skeleton,
                    cell_skeleton=component_mask,
                    soma_point=(soma_x, soma_y),
                    radii=radii,
                    intersections=intersections,
                    metrics=metrics_dict,
                    graph=G,
                    soma_id=i,
                    output_path=dashboard_path,
                )

                if response == 'q':
                    cell_accepted = True
                    accept_all_remaining = True
                elif response == 'n':
                    cell_accepted = False
                    accept_all_remaining = False
                else:
                    cell_accepted = True
                    accept_all_remaining = False
            else:
                # Silently accept if auto-accepting
                cell_accepted = True

            if not cell_accepted:
                rejected_cells += 1
                total_rejected += 1
                # Rename backtrace to flag it as rejected
                rejected_path = os.path.join(
                    output_dir,
                    f"{filename_prefix}_REJECTED_backtrace_soma_{i}.png"
                )
                if os.path.exists(backtrace_path):
                    os.replace(backtrace_path, rejected_path)
                print(f"  >> Soma {i} REJECTED (excluded from CSV)")
                continue  # skip adding to CSV

            # Append rows for the CURRENT cell first
            accepted_cells += 1
            total_accepted += 1
            if accept_all_remaining:
                print(f"  >> Soma {i} auto-accepted")
            else:
                print(f"  >> Soma {i} ACCEPTED")
                
            for radius, count in zip(radii, intersections):
                sholl_data.append([
                    i, radius, count,
                    fd, lac, betw, clos, sri, soma_area, soma_circ
                ])
                # Master array injects Image_Name at index 0
                master_sholl_data.append([
                    filename_prefix, i, radius, count,
                    fd, lac, betw, clos, sri, soma_area, soma_circ
                ])

        # Save extended Sholl + morphometric Per-Image CSV
        columns_local = [
            'Soma_ID', 'Radius', 'Intersections',
            'Fractal_Dimension', 'Lacunarity',
            'Betweenness_Centrality', 'Closeness_Centrality',
            'Ramification_Index', 'Soma_Area', 'Soma_Circularity'
        ]
        if sholl_data:
            df = pd.DataFrame(sholl_data, columns=columns_local)
            df.to_csv(os.path.join(output_dir, f"sholl_intersections_{filename_prefix}.csv"), index=False)
            print(f"  > Saved Local CSV for {filename_prefix}")
        else:
            print(f"  > No accepted cells to save for {filename_prefix}")

    # =========================================================================
    # GLOBAL MASTER EXPORT
    # =========================================================================
    if master_sholl_data:
        columns_master = [
            'Image_Name', 'Soma_ID', 'Radius', 'Intersections',
            'Fractal_Dimension', 'Lacunarity',
            'Betweenness_Centrality', 'Closeness_Centrality',
            'Ramification_Index', 'Soma_Area', 'Soma_Circularity'
        ]
        master_df = pd.DataFrame(master_sholl_data, columns=columns_master)
        master_out_path = os.path.join(base_master_dir, "MASTER_Sholl_Metrics_Batch.csv")
        master_df.to_csv(master_out_path, index=False)
        
        print("\n" + "="*60)
        print(f"✅ BATCH ANALYSIS COMPILED SUCCESSFULLY")
        print("="*60)
        print(f"  Total Images Processed: {len(file_paths)}")
        print(f"  Master CSV Saved to:    {master_out_path}")
        print(f"  Total Accepted Cells:   {total_accepted}")
        print(f"  Total Rejected Cells:   {total_rejected}")
        print("="*60 + "\n")
    else:
        print("\n❌ BATCH ANALYSIS FINISHED: No cells accepted from any image.")

def preview_denoising(image):
    """
    Preview denoising effects using an interactive Matplotlib Slider GUI.
    
    A dataset-calibrated starting ``h`` value is derived from the wavelet noise
    estimate. The user must review and may fine-tune this heuristic using a
    live-updating slider before accepting it.
    
    Parameters:
    - image (np.ndarray): Input grayscale image.
    
    Returns:
    - tuple: Selected denoised image and h value.
    
    Calibration status: the 0.7 multiplier was derived from manual review of
    six accepted Iba1 benchmark regions from the included example dataset. It
    is not a universally validated optimum and should be reviewed per dataset.
    """
    # 1. Estimate the base noise level of the image.
    sigma_est = estimate_sigma(image, channel_axis=None)
    baseline_h = nlm_baseline_from_sigma(sigma_est)
    print(
        f"Noise estimate: sigma={sigma_est:.2f} -> dataset-calibrated "
        f"starting h={baseline_h} ({NLM_BASELINE_SIGMA_FACTOR:g} × sigma)"
    )
    # 2. Pre-compute the manual-review range for a responsive UI.
    h_range = nlm_review_range(baseline_h)
    min_h = h_range[0]
    max_h = h_range[-1]
    
    print(f"Pre-computing skeletons for h values: {h_range} to ensure zero-lag UI...")
    precomputed_skeletons = {}
    precomputed_denoised = {}

    for h in h_range:
        print(f"  Pre-calculating h={h}...", end="\r")
        denoised = cv2.fastNlMeansDenoising(image, None, h=h, templateWindowSize=7, searchWindowSize=21)
        den_th = apply_tophat(denoised)
        binary = binarize_image(den_th)
        frag_filtered = remove_small_fragments(binary)
        closed = apply_morph_close(frag_filtered)
        dilated = apply_dilate(closed)
        skel = skeletonize_image(dilated)
        bridged = bridge_nearby_fragments(skel)
        refined = remove_isolated_fibers_refine(bridged)
        bridge_refined = bridge_nearby_fragments_refine(refined)
        skeleton = remove_isolated_fibers(bridge_refined)
        
        precomputed_skeletons[h] = skeleton
        precomputed_denoised[h] = denoised
        
    print(f"  Pre-calculation complete. Opening User Interface...")

    # Set up the Matplotlib figure and Grid
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title('Interactive Denoising Validation (Zero-Lag Precomputed)')
    
    # Define layout
    ax_orig = plt.axes([0.05, 0.25, 0.40, 0.65])
    ax_skel = plt.axes([0.50, 0.25, 0.45, 0.65])
    
    # Original Image Panel
    ax_orig.imshow(image, cmap='gray')
    ax_orig.set_title("Original Input Image", fontweight='bold')
    ax_orig.axis('off')

    # Skeleton Preview Panel
    ax_skel.set_title(f"Skeleton Preview (Live Update)", fontweight='bold')
    ax_skel.axis('off')
    
    # Initialize imshow with vmin=0, vmax=255 so the binary skeleton is white on black!
    skel_img_plot = ax_skel.imshow(np.zeros_like(image), cmap='gray', vmin=0, vmax=255)
    
    # Variables to track state
    current_h = [baseline_h]
    
    def update_skeleton(h_val):
        h = max(int(h_val), 1)
        # Snap to closest pre-computed h to prevent KeyErrors
        h = min(h_range, key=lambda x: abs(x - h))
        
        # INSTANT 0ms lookup instead of heavy algorithmic computation
        skel_img_plot.set_data(precomputed_skeletons[h])
        ax_skel.set_title(f"Skeleton Preview (h={h})", fontweight='bold')
        fig.canvas.draw_idle()
        return h

    current_h[0] = update_skeleton(baseline_h)

    # 3. Interactive UI controls
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.05])
    slider = Slider(
        ax=ax_slider,
        label='Denoising Strength (h)',
        valmin=min_h,
        valmax=max_h,
        valinit=baseline_h,
        valstep=1,
        color='steelblue'
    )
    
    def on_slider_change(val):
        current_h[0] = update_skeleton(val)

    slider.on_changed(on_slider_change)

    # Accept Button
    ax_button = plt.axes([0.8, 0.02, 0.15, 0.06])
    button = Button(ax_button, 'Accept & Continue', color='lightgreen', hovercolor='palegreen')
    
    def on_accept(event):
        plt.close(fig)

    button.on_clicked(on_accept)
    
    fig.text(0.02, 0.04, "Instructions: Move the slider to instantly swap pre-computed noise reductions.\nClick 'Accept & Continue' when the skeleton looks clean and continuous.", fontsize=10, fontstyle='italic')
    
    # Block script execution until window is explicitly closed
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except Exception:
        pass
    plt.show(block=True)
    
    print(f"User selected h={current_h[0]}. Proceeding with analysis...")
    
    # Immediately return the cached Heavy NLM matrix so no further loop computes
    return precomputed_denoised[current_h[0]], current_h[0]
if __name__ == "__main__":    
    main()

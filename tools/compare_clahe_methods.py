"""Run a method-label-hidden CLAHE preprocessing comparison.

This utility compares the production adaptive patch-wise CLAHE strategy with a
single-setting whole-image CLAHE comparator. It is designed as an internal,
qualitative ablation study. Visual preference does not establish segmentation
accuracy or biological validity.

For every prepared single-channel 8-bit TIFF, the reviewer:

1. selects one region of interest on the raw image before seeing candidates;
2. compares two A/B candidates with matched display and downstream settings;
3. records Prefer A, Prefer B, Equivalent, Neither, or Exclude.

Method positions are deterministic and counterbalanced across the input batch.
Only masked decisions and masked QC panels are written during collection.
Method identities, descriptive counts, and labeled filenames are generated
after every source image in the configured batch has a completed record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import matplotlib
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Support both ``python -m tools.compare_clahe_methods`` and direct execution
# from the repository's ``tools`` directory (including the VS Code Run button).
if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib.widgets import Button
from skimage.measure import shannon_entropy
from skimage.restoration import estimate_sigma

from pipeline_config import nlm_baseline_from_sigma
from run_sholl_pipeline import (
    apply_adaptive_patching,
    apply_dilate,
    apply_morph_close,
    apply_tophat,
    binarize_image,
    bridge_nearby_fragments,
    bridge_nearby_fragments_refine,
    remove_isolated_fibers,
    remove_isolated_fibers_refine,
    remove_small_fragments,
    skeletonize_image,
)


TOOL_VERSION = "1.1"
DEFAULT_RANDOM_SEED = 1729
FIXED_CLIP_LIMITS = (1.8, 2.0, 2.8, 3.6, 4.8, 6.0)
TILE_GRID_SIZE = (8, 8)
METHOD_ADAPTIVE = "adaptive_patchwise"
METHOD_FIXED = "single_setting_whole_image"
ALLOWED_CHOICES = {
    "candidate_a",
    "candidate_b",
    "tie",
    "neither",
    "excluded",
}

MASKED_CSV_FIELDS = (
    "Source_File",
    "Relative_Path",
    "Source_Folder",
    "Source_SHA256",
    "Reviewer_ID",
    "Reviewed_UTC",
    "Status",
    "Exclusion_Stage",
    "ROI_X_Min_px",
    "ROI_X_Max_px",
    "ROI_Y_Min_px",
    "ROI_Y_Max_px",
    "Image_Width_px",
    "Image_Height_px",
    "Fixed_Clip_Limit",
    "Fixed_Clip_Selection",
    "Tile_Grid",
    "Adaptive_Sigma_Estimate",
    "Common_NLM_h",
    "Common_NLM_h_Source",
    "Candidate_Cache_SHA256",
    "Random_Seed",
    "Displayed_Method_Labels",
    "Reviewer_Choice",
)

UNBLINDED_CSV_FIELDS = MASKED_CSV_FIELDS + (
    "Randomization_Block_Index",
    "Candidate_A_Method",
    "Candidate_B_Method",
    "Preferred_Method",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_input_directory() -> Path:
    """Open a native directory chooser, importing tkinter only when needed."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    selected = filedialog.askdirectory(
        title="Select prepared single-channel Iba1 TIFF directory"
    )
    root.destroy()
    if not selected:
        raise RuntimeError("No input directory was selected.")
    return Path(selected)


def discover_images(inputs: list[Path]) -> tuple[list[Path], Path]:
    """Resolve TIFF inputs and return them in a stable order."""
    if not inputs:
        inputs = [select_input_directory()]

    images: list[Path] = []
    for item in inputs:
        resolved = item.resolve()
        if resolved.is_dir():
            images.extend(
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            )
        elif resolved.is_file() and resolved.suffix.lower() in {".tif", ".tiff"}:
            images.append(resolved)
        else:
            raise ValueError(f"Not a TIFF file or directory: {item}")

    images = sorted(set(images), key=lambda path: str(path).lower())
    if not images:
        raise RuntimeError("No TIFF images were found.")

    directory_inputs = [item.resolve() for item in inputs if item.resolve().is_dir()]
    common_root = directory_inputs[0] if len(directory_inputs) == 1 else Path.cwd()
    return images, common_root


def build_input_manifest(
    images: list[Path],
    common_root: Path,
) -> list[dict[str, str]]:
    """Hash the configured source images and reject duplicate content."""
    manifest: list[dict[str, str]] = []
    seen_hashes: dict[str, Path] = {}
    for image_path in images:
        source_hash = sha256_file(image_path)
        if source_hash in seen_hashes:
            raise ValueError(
                "Duplicate source content detected; each source image must "
                f"appear once. {image_path} duplicates {seen_hashes[source_hash]}."
            )
        seen_hashes[source_hash] = image_path
        try:
            relative_path = str(image_path.relative_to(common_root))
        except ValueError:
            relative_path = str(image_path)
        manifest.append(
            {
                "source_file": image_path.name,
                "relative_path": relative_path,
                "source_folder": image_path.parent.name,
                "source_sha256": source_hash,
            }
        )
    return manifest


def load_uint8_grayscale(path: Path) -> np.ndarray:
    """Load a prepared TIFF and reject inputs that change the comparison."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            "Expected a single-channel 8-bit TIFF, got "
            f"shape={image.shape}, dtype={image.dtype}: {path}"
        )
    if min(image.shape) < 8:
        raise ValueError(f"Image must be at least 8 x 8 pixels: {path}")
    return image


def apply_fixed_clahe(
    image: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int] = TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply one CLAHE configuration across the complete image."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Whole-image CLAHE expects a single-channel uint8 image.")
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit must be finite and positive.")
    if len(tile_grid_size) != 2 or any(int(value) <= 0 for value in tile_grid_size):
        raise ValueError("tile_grid_size must contain two positive integers.")

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(map(int, tile_grid_size)),
    )
    return clahe.apply(image)


def whole_image_clip_from_stats(image: np.ndarray) -> float:
    """Apply production clip thresholds once to whole-image statistics.

    The resulting comparator has one setting within each image. It is a
    convenience for internal exploration, not a replacement for locking a
    comparator on a separate tuning set.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            "Whole-image clip selection expects a single-channel uint8 image."
        )

    global_var = float(np.var(image))
    image_entropy = float(shannon_entropy(image))
    if image_entropy < 3.5:
        return 4.8
    if image_entropy > 5.0:
        return 3.6
    if global_var > 0:
        return 2.8
    return 1.8


def skeleton_for_common_h(enhanced_image: np.ndarray, h_value: int) -> np.ndarray:
    """Run the production post-CLAHE chain with one shared NLM strength."""
    denoised = cv2.fastNlMeansDenoising(
        enhanced_image,
        None,
        h=int(h_value),
        templateWindowSize=7,
        searchWindowSize=21,
    )
    top_hat = apply_tophat(denoised)
    binary = binarize_image(top_hat)
    fragments_removed = remove_small_fragments(binary)
    closed = apply_morph_close(fragments_removed)
    dilated = apply_dilate(closed)
    skeleton = skeletonize_image(dilated)
    skeleton = bridge_nearby_fragments(skeleton)
    skeleton = remove_isolated_fibers_refine(skeleton)
    skeleton = bridge_nearby_fragments_refine(skeleton)
    return remove_isolated_fibers(skeleton)


def write_roi_record(
    path: Path,
    roi_bounds: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> None:
    """Persist the pre-exposure ROI so an interrupted review cannot reselect it."""
    payload = {
        "roi_bounds": list(map(int, roi_bounds)),
        "image_height_px": int(image_shape[0]),
        "image_width_px": int(image_shape[1]),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_roi_record(
    path: Path,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Load and validate a previously fixed ROI."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("image_height_px") != int(image_shape[0])
        or payload.get("image_width_px") != int(image_shape[1])
    ):
        raise RuntimeError(f"Saved ROI dimensions do not match the image: {path}")
    bounds = payload.get("roi_bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise RuntimeError(f"Saved ROI record is invalid: {path}")
    x_min, x_max, y_min, y_max = map(int, bounds)
    if (
        x_min < 0
        or y_min < 0
        or x_max <= x_min
        or y_max <= y_min
        or x_max > image_shape[1]
        or y_max > image_shape[0]
    ):
        raise RuntimeError(f"Saved ROI bounds are invalid: {path}")
    return x_min, x_max, y_min, y_max


def write_candidate_cache(
    path: Path,
    metadata: dict,
    adaptive_image: np.ndarray,
    fixed_image: np.ndarray,
    adaptive_skeleton: np.ndarray,
    fixed_skeleton: np.ndarray,
) -> None:
    """Atomically cache the exact candidates displayed to the reviewer."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.array(json.dumps(metadata)),
            adaptive_image=adaptive_image,
            fixed_image=fixed_image,
            adaptive_skeleton=np.asarray(adaptive_skeleton, dtype=np.uint8),
            fixed_skeleton=np.asarray(fixed_skeleton, dtype=np.uint8),
        )
    temporary.replace(path)


def load_candidate_cache(
    path: Path,
    source_hash: str,
    image_shape: tuple[int, int],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load exact cached candidates and validate their identity and shape."""
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "metadata_json",
            "adaptive_image",
            "fixed_image",
            "adaptive_skeleton",
            "fixed_skeleton",
        }
        if set(cache.files) != required:
            raise RuntimeError(f"Candidate cache schema is invalid: {path}")
        metadata = json.loads(str(cache["metadata_json"].item()))
        arrays = {
            name: np.array(cache[name], copy=True)
            for name in required
            if name != "metadata_json"
        }
    if metadata.get("source_sha256") != source_hash:
        raise RuntimeError(f"Candidate cache source hash does not match: {path}")
    if any(array.shape != image_shape for array in arrays.values()):
        raise RuntimeError(f"Candidate cache shape does not match: {path}")
    if arrays["adaptive_image"].dtype != np.uint8:
        raise RuntimeError(f"Adaptive candidate dtype is invalid: {path}")
    if arrays["fixed_image"].dtype != np.uint8:
        raise RuntimeError(f"Whole-image candidate dtype is invalid: {path}")
    if arrays["adaptive_skeleton"].dtype != np.uint8:
        raise RuntimeError(f"Adaptive skeleton dtype is invalid: {path}")
    if arrays["fixed_skeleton"].dtype != np.uint8:
        raise RuntimeError(f"Whole-image skeleton dtype is invalid: {path}")
    enhanced = {
        METHOD_ADAPTIVE: arrays["adaptive_image"],
        METHOD_FIXED: arrays["fixed_image"],
    }
    skeletons = {
        METHOD_ADAPTIVE: arrays["adaptive_skeleton"],
        METHOD_FIXED: arrays["fixed_skeleton"],
    }
    return metadata, enhanced, skeletons


def build_candidate_mappings(
    source_hashes: list[str],
    random_seed: int,
    reviewer_id: str,
) -> dict[str, tuple[str, str, int]]:
    """Build deterministic, reviewer-specific, counterbalanced A/B mappings."""
    if not reviewer_id.strip():
        raise ValueError("reviewer_id must not be empty.")
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("source_hashes must be unique.")

    def ordering_key(source_hash: str) -> str:
        token = (
            f"{int(random_seed)}:{reviewer_id}:{source_hash}:batch-order"
        ).encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    ordered_hashes = sorted(source_hashes, key=ordering_key)
    start_token = f"{int(random_seed)}:{reviewer_id}:start-side".encode("utf-8")
    adaptive_first_at_even = hashlib.sha256(start_token).digest()[0] % 2 == 0

    mappings: dict[str, tuple[str, str, int]] = {}
    for block_index, source_hash in enumerate(ordered_hashes):
        adaptive_first = (block_index % 2 == 0) == adaptive_first_at_even
        methods = (
            (METHOD_ADAPTIVE, METHOD_FIXED)
            if adaptive_first
            else (METHOD_FIXED, METHOD_ADAPTIVE)
        )
        mappings[source_hash] = (methods[0], methods[1], block_index)
    return mappings


def overlay_skeleton(image: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """Overlay a green skeleton on an unchanged uint8 grayscale image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb[np.asarray(skeleton).astype(bool)] = (0, 255, 70)
    return rgb


def select_review_region(
    raw_image: np.ndarray,
    image_name: str,
) -> tuple[int, int, int, int] | None:
    """Let the reviewer define one ROI before displaying either candidate."""
    height, width = raw_image.shape
    fig, axis = plt.subplots(figsize=(11, 8))
    fig.canvas.manager.set_window_title("Select CLAHE comparison region")
    axis.imshow(raw_image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(
        "Select a representative region on the raw image",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel(
        "Use zoom/pan, disable the toolbar tool, then select Use current view."
    )
    axis.set_xticks([])
    axis.set_yticks([])

    state: dict[str, tuple[int, int, int, int] | bool | None] = {
        "bounds": None,
        "confirmed": False,
        "excluded": False,
    }
    status = fig.text(
        0.02,
        0.025,
        "This ROI is fixed after candidate exposure. Exclude the image if it "
        "is not suitable.",
        fontsize=10,
    )

    def confirm_region(_event) -> None:
        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()
        x_min = max(0, int(np.floor(min(x_limits))))
        x_max = min(width, int(np.ceil(max(x_limits))))
        y_min = max(0, int(np.floor(min(y_limits))))
        y_max = min(height, int(np.ceil(max(y_limits))))
        if x_max - x_min < 8 or y_max - y_min < 8:
            status.set_text("The selected view is too small; zoom out and try again.")
            fig.canvas.draw_idle()
            return
        state["bounds"] = (x_min, x_max, y_min, y_max)
        state["confirmed"] = True
        plt.close(fig)

    def exclude_image(_event) -> None:
        state["excluded"] = True
        state["confirmed"] = True
        plt.close(fig)

    exclude_axis = fig.add_axes([0.62, 0.015, 0.14, 0.05])
    button_axis = fig.add_axes([0.77, 0.015, 0.21, 0.05])
    exclude_button = Button(
        exclude_axis,
        "Exclude before A/B",
        color="#E5E5E5",
    )
    confirm_button = Button(
        button_axis,
        "Use current view",
        color="#D8E8D4",
        hovercolor="#BED8B8",
    )
    exclude_button.on_clicked(exclude_image)
    confirm_button.on_clicked(confirm_region)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.91, bottom=0.1)
    fig.suptitle(f"CLAHE benchmark ROI: {image_name}", fontsize=13)
    try:
        fig.canvas.manager.window.state("zoomed")
    except Exception:
        pass
    plt.show(block=True)

    if state["excluded"]:
        return None
    if not state["confirmed"] or state["bounds"] is None:
        plt.close(fig)
        raise RuntimeError(
            "ROI window was closed without confirming a field of view."
        )
    return state["bounds"]


def _crop(
    image: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x_min, x_max, y_min, y_max = roi_bounds
    return image[y_min:y_max, x_min:x_max]


def review_pair(
    raw_image: np.ndarray,
    enhanced_by_method: dict[str, np.ndarray],
    skeleton_by_method: dict[str, np.ndarray],
    image_name: str,
    method_order: tuple[str, str],
    roi_bounds: tuple[int, int, int, int],
    common_h: int,
    show_labels: bool,
) -> str:
    """Collect one paired preference without allowing post-exposure ROI edits."""
    method_a, method_b = method_order
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.canvas.manager.set_window_title("CLAHE A/B comparison")

    raw_crop = _crop(raw_image, roi_bounds)
    axes[0, 0].imshow(raw_crop, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Raw reference", loc="left", fontweight="bold")

    for column, (label, method) in enumerate(
        zip(("A", "B"), (method_a, method_b)),
        start=1,
    ):
        enhanced_crop = _crop(enhanced_by_method[method], roi_bounds)
        skeleton_crop = _crop(skeleton_by_method[method], roi_bounds)
        enhanced_title = f"Candidate {label}: enhanced"
        skeleton_title = f"Candidate {label}: matched skeleton"
        if show_labels:
            enhanced_title += f" | {method}"
            skeleton_title += f" | {method}"
        axes[0, column].imshow(
            enhanced_crop,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[0, column].set_title(enhanced_title, loc="left")
        axes[1, column].imshow(overlay_skeleton(raw_crop, skeleton_crop))
        axes[1, column].set_title(skeleton_title, loc="left")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.02,
        0.95,
        "Review criterion\n\n"
        "Prefer the candidate whose skeleton follows visible Iba1 processes\n"
        "while avoiding isolated or background-derived structure.\n\n"
        "A denser skeleton is not automatically better.\n\n"
        f"Both candidates use NLM h={common_h} and identical downstream settings.",
        va="top",
        ha="left",
        fontsize=11,
        transform=axes[1, 0].transAxes,
    )
    for axis in axes.flat:
        if axis is not axes[1, 0]:
            axis.set_xticks([])
            axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    state: dict[str, str | bool | None] = {
        "choice": None,
        "confirmed": False,
    }

    def record_choice(choice: str):
        def callback(_event) -> None:
            state["choice"] = choice
            state["confirmed"] = True
            plt.close(fig)

        return callback

    button_specs = (
        (0.20, "Prefer A", "#CFE2F3", "candidate_a"),
        (0.34, "Equivalent", "#E5E5E5", "tie"),
        (0.48, "Prefer B", "#CFE2F3", "candidate_b"),
        (0.62, "Neither", "#F4CCCC", "neither"),
        (0.76, "Exclude image", "#E5E5E5", "excluded"),
    )
    buttons = []
    for left, label, color, choice in button_specs:
        button_axis = fig.add_axes([left, 0.012, 0.115, 0.045])
        button = Button(button_axis, label, color=color)
        button.on_clicked(record_choice(choice))
        buttons.append(button)

    title = (
        "Unblinded debug comparison"
        if show_labels
        else "Method-label-hidden paired comparison"
    )
    fig.suptitle(
        f"{title}: {image_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.92,
        bottom=0.08,
        wspace=0.04,
        hspace=0.12,
    )
    try:
        fig.canvas.manager.window.state("zoomed")
    except Exception:
        pass
    plt.show(block=True)

    if not state["confirmed"]:
        plt.close(fig)
        raise RuntimeError(
            "A/B comparison window was closed without a recorded decision."
        )
    return str(state["choice"])


def preferred_method(
    reviewer_choice: str,
    method_order: tuple[str, str],
) -> str:
    """Unmask one recorded A/B choice."""
    if reviewer_choice == "candidate_a":
        return method_order[0]
    if reviewer_choice == "candidate_b":
        return method_order[1]
    if reviewer_choice in {"tie", "neither", "excluded"}:
        return reviewer_choice
    raise ValueError(f"Unknown reviewer choice: {reviewer_choice}")


def save_masked_qc(
    output_path: Path,
    raw_image: np.ndarray,
    enhanced_by_method: dict[str, np.ndarray],
    skeleton_by_method: dict[str, np.ndarray],
    method_order: tuple[str, str],
    roi_bounds: tuple[int, int, int, int],
    image_name: str,
    common_h: int,
    reviewer_choice: str,
    show_labels: bool,
) -> None:
    """Save an A/B-labeled panel without revealing methods during collection."""
    raw_crop = _crop(raw_image, roi_bounds)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(raw_crop, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Raw reference", loc="left", fontweight="bold")

    for column, (label, method) in enumerate(
        zip(("A", "B"), method_order),
        start=1,
    ):
        enhanced_crop = _crop(enhanced_by_method[method], roi_bounds)
        skeleton_crop = _crop(skeleton_by_method[method], roi_bounds)
        enhanced_title = f"Candidate {label}: enhanced"
        skeleton_title = f"Candidate {label}: matched skeleton"
        if show_labels:
            enhanced_title += f" | {method}"
            skeleton_title += f" | {method}"
        axes[0, column].imshow(
            enhanced_crop,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[0, column].set_title(enhanced_title, loc="left")
        axes[1, column].imshow(overlay_skeleton(raw_crop, skeleton_crop))
        axes[1, column].set_title(skeleton_title, loc="left")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.02,
        0.95,
        f"Recorded response\n\n{reviewer_choice}\n\n"
        f"Common downstream NLM h={common_h}",
        va="top",
        ha="left",
        fontsize=11,
        transform=axes[1, 0].transAxes,
    )
    for axis in axes.flat:
        if axis is not axes[1, 0]:
            axis.set_xticks([])
            axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f"Masked CLAHE comparison record: {image_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.92,
        bottom=0.03,
        wspace=0.04,
        hspace=0.10,
    )
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_records(
    path: Path,
    records: list[dict],
    fieldnames: tuple[str, ...],
) -> None:
    """Atomically write comparison records."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def validate_masked_records(
    records: list[dict],
    reviewer_id: str,
    expected_hashes: set[str],
    expected_random_seed: int | None = None,
    expected_show_labels: bool | None = None,
) -> None:
    """Reject incompatible, duplicate, or corrupted resume records."""
    observed_hashes: set[str] = set()
    for row_number, record in enumerate(records, start=2):
        source_hash = record.get("Source_SHA256", "")
        choice = record.get("Reviewer_Choice", "")
        status = record.get("Status", "")
        exclusion_stage = record.get("Exclusion_Stage", "")
        if record.get("Reviewer_ID") != reviewer_id:
            raise ValueError(
                f"Masked CSV row {row_number} has a different Reviewer_ID."
            )
        if source_hash not in expected_hashes:
            raise ValueError(
                f"Masked CSV row {row_number} is not in the configured manifest."
            )
        if source_hash in observed_hashes:
            raise ValueError(
                f"Duplicate Source_SHA256 in masked CSV row {row_number}."
            )
        if choice not in ALLOWED_CHOICES:
            raise ValueError(
                f"Unknown Reviewer_Choice in masked CSV row {row_number}: {choice}"
            )
        expected_status = "excluded" if choice == "excluded" else "reviewed"
        if status != expected_status:
            raise ValueError(
                f"Inconsistent Status and Reviewer_Choice in row {row_number}."
            )
        if exclusion_stage not in {"", "pre_exposure", "post_exposure"}:
            raise ValueError(
                f"Unknown Exclusion_Stage in masked CSV row {row_number}."
            )
        if choice == "excluded" and exclusion_stage not in {
            "pre_exposure",
            "post_exposure",
        }:
            raise ValueError(
                f"Excluded row {row_number} must record its exclusion stage."
            )
        if choice != "excluded" and exclusion_stage:
            raise ValueError(
                f"Reviewed row {row_number} cannot have an exclusion stage."
            )
        try:
            width = int(record["Image_Width_px"])
            height = int(record["Image_Height_px"])
            random_seed = int(record["Random_Seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid image or seed field in masked CSV row {row_number}."
            ) from error
        if width < 8 or height < 8:
            raise ValueError(
                f"Invalid image dimensions in masked CSV row {row_number}."
            )
        if expected_random_seed is not None and random_seed != expected_random_seed:
            raise ValueError(
                f"Random_Seed changed in masked CSV row {row_number}."
            )
        displayed_labels = record.get("Displayed_Method_Labels", "")
        if isinstance(displayed_labels, bool):
            displayed_labels_value = displayed_labels
        elif displayed_labels in {"True", "False"}:
            displayed_labels_value = displayed_labels == "True"
        else:
            raise ValueError(
                f"Invalid Displayed_Method_Labels in row {row_number}."
            )
        if (
            expected_show_labels is not None
            and displayed_labels_value != bool(expected_show_labels)
        ):
            raise ValueError(
                f"Displayed_Method_Labels changed in row {row_number}."
            )

        if exclusion_stage == "pre_exposure":
            roi_fields = (
                record.get("ROI_X_Min_px", ""),
                record.get("ROI_X_Max_px", ""),
                record.get("ROI_Y_Min_px", ""),
                record.get("ROI_Y_Max_px", ""),
            )
            if any(value != "" for value in roi_fields):
                raise ValueError(
                    f"Pre-exposure exclusion row {row_number} must have no ROI."
                )
            observed_hashes.add(source_hash)
            continue

        try:
            x_min = int(record["ROI_X_Min_px"])
            x_max = int(record["ROI_X_Max_px"])
            y_min = int(record["ROI_Y_Min_px"])
            y_max = int(record["ROI_Y_Max_px"])
            common_h = int(record["Common_NLM_h"])
            clip_limit = float(record["Fixed_Clip_Limit"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid numeric field in masked CSV row {row_number}."
            ) from error
        if (
            x_min < 0
            or y_min < 0
            or x_max <= x_min
            or y_max <= y_min
            or x_max > width
            or y_max > height
        ):
            raise ValueError(f"Invalid ROI bounds in masked CSV row {row_number}.")
        if (
            common_h < 1
            or common_h > 20
            or not np.isfinite(clip_limit)
            or clip_limit <= 0
        ):
            raise ValueError(
                f"Invalid preprocessing value in masked CSV row {row_number}."
            )
        cache_hash = record.get("Candidate_Cache_SHA256", "")
        if len(cache_hash) != 64:
            raise ValueError(
                f"Invalid candidate cache hash in masked CSV row {row_number}."
            )
        try:
            int(cache_hash, 16)
        except ValueError as error:
            raise ValueError(
                f"Invalid candidate cache hash in masked CSV row {row_number}."
            ) from error
        observed_hashes.add(source_hash)


def load_masked_records(
    path: Path,
    reviewer_id: str,
    expected_hashes: set[str],
    expected_random_seed: int | None = None,
    expected_show_labels: bool | None = None,
) -> list[dict]:
    """Read and validate resumable masked records."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MASKED_CSV_FIELDS:
            raise ValueError(
                "Masked review CSV schema does not match this tool version."
            )
        records = list(reader)
    validate_masked_records(
        records,
        reviewer_id,
        expected_hashes,
        expected_random_seed,
        expected_show_labels,
    )
    return records


def validate_candidate_cache_files(
    records: list[dict],
    candidate_cache_directory: Path,
) -> None:
    """Verify cached candidate files referenced by completed review rows."""
    for record in records:
        if record.get("Exclusion_Stage") == "pre_exposure":
            continue
        source_hash = record["Source_SHA256"]
        cache_path = candidate_cache_directory / f"{source_hash[:12]}.npz"
        if not cache_path.exists():
            raise RuntimeError(f"Missing candidate cache: {cache_path}")
        observed_hash = sha256_file(cache_path)
        if observed_hash != record.get("Candidate_Cache_SHA256"):
            raise RuntimeError(f"Candidate cache hash changed: {cache_path}")


def unblind_records(
    masked_records: list[dict],
    mappings: dict[str, tuple[str, str, int]],
) -> list[dict]:
    """Add method identities after collection is complete."""
    results: list[dict] = []
    for record in masked_records:
        method_a, method_b, block_index = mappings[record["Source_SHA256"]]
        unblinded = dict(record)
        unblinded["Randomization_Block_Index"] = block_index
        unblinded["Candidate_A_Method"] = method_a
        unblinded["Candidate_B_Method"] = method_b
        unblinded["Preferred_Method"] = preferred_method(
            record["Reviewer_Choice"],
            (method_a, method_b),
        )
        results.append(unblinded)
    return results


def summarize_records(records: list[dict]) -> dict:
    """Return descriptive image-level counts from unblinded records."""
    preferences = Counter(record["Preferred_Method"] for record in records)
    adaptive_wins = preferences[METHOD_ADAPTIVE]
    fixed_wins = preferences[METHOD_FIXED]
    directional = adaptive_wins + fixed_wins
    return {
        "n_total_records": len(records),
        "n_reviewed_nonexcluded": len(records) - preferences["excluded"],
        "n_adaptive_preferred": adaptive_wins,
        "n_whole_image_preferred": fixed_wins,
        "n_equivalent": preferences["tie"],
        "n_neither": preferences["neither"],
        "n_excluded": preferences["excluded"],
        "n_directional_preferences": directional,
        "adaptive_fraction_among_directional_preferences": (
            adaptive_wins / directional if directional else None
        ),
    }


def git_metadata() -> dict[str, str | bool | None]:
    """Return informational Git state without making it a resume key."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}
    return {"commit": commit or None, "worktree_dirty": bool(status.strip())}


def implementation_fingerprint() -> dict[str, str]:
    """Hash code files that determine comparison behavior."""
    repository_root = Path(__file__).resolve().parents[1]
    paths = {
        "comparison_script": Path(__file__).resolve(),
        "production_pipeline": repository_root / "run_sholl_pipeline.py",
        "pipeline_config": repository_root / "pipeline_config.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def environment_fingerprint() -> dict[str, str]:
    """Record numerical-library versions that can affect results."""
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_image": skimage.__version__,
    }


def comparison_settings(args: argparse.Namespace) -> dict:
    """Build settings that must remain fixed within one output directory."""
    return {
        "tool_version": TOOL_VERSION,
        "reviewer_id": args.reviewer_id,
        "locked_fixed_clip_limit": args.fixed_clip_limit,
        "default_fixed_clip_selection": (
            "single setting per image from whole-image entropy thresholds"
        ),
        "tile_grid_size": list(TILE_GRID_SIZE),
        "common_nlm_h_override": args.nlm_h,
        "default_common_nlm_h_source": (
            "production baseline estimated from adaptive CLAHE output"
        ),
        "random_seed": args.random_seed,
        "method_labels_visible_during_review": bool(args.show_labels),
        "roi_policy": "selected on raw image and immutable after candidate exposure",
        "comparison_unit": "one source image with one reviewer-selected ROI",
    }


def initialize_config(
    path: Path,
    records_path: Path,
    settings: dict,
    input_manifest: list[dict[str, str]],
    implementation: dict[str, str],
    environment: dict[str, str],
) -> None:
    """Create a config or reject an inhomogeneous resume."""
    if records_path.exists() and not path.exists():
        raise RuntimeError(
            "Masked records exist without their configuration file. "
            "Use a new output directory or restore the matching config."
        )

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparisons = {
            "settings": settings,
            "input_manifest": input_manifest,
            "implementation": implementation,
            "environment": environment,
        }
        for key, current_value in comparisons.items():
            if existing.get(key) != current_value:
                raise RuntimeError(
                    f"Cannot resume because {key} changed. Use a new output "
                    "directory or restore the original comparison environment."
                )
        return

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": (
            "unblinded debug comparison"
            if settings.get("method_labels_visible_during_review", False)
            else "method-label-hidden internal qualitative CLAHE comparison"
        ),
        "settings": settings,
        "input_manifest": input_manifest,
        "implementation": implementation,
        "environment": environment,
        "git": git_metadata(),
        "interpretation_warning": (
            "Visual preference and downstream skeleton appearance do not "
            "establish biological accuracy. Independent manual reference "
            "traces are required for an accuracy claim."
        ),
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_summary(
    output_dir: Path,
    records: list[dict],
    show_labels: bool,
) -> None:
    """Write descriptive counts only after the batch is complete."""
    summary = summarize_records(records)
    summary.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_unit": "source image",
            "collection_mode": (
                "unblinded_debug" if show_labels else "method_label_hidden"
            ),
            "interpretation_warning": (
                "These are visual preferences, not independent accuracy "
                "measurements or biological replicates. The directional "
                "fraction excludes Equivalent, Neither, and Excluded records."
            ),
        }
    )
    (output_dir / "clahe_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    labels = ("Adaptive", "Whole image", "Equivalent", "Neither", "Excluded")
    values = (
        summary["n_adaptive_preferred"],
        summary["n_whole_image_preferred"],
        summary["n_equivalent"],
        summary["n_neither"],
        summary["n_excluded"],
    )
    colors = ("#0072B2", "#D55E00", "#999999", "#CC79A7", "#D9D9D9")
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.bar(labels, values, color=colors, width=0.68)
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Source images")
    title_prefix = "Unblinded debug" if show_labels else "Method-label-hidden"
    axis.set_title(
        f"{title_prefix} paired CLAHE review",
        loc="left",
        fontweight="bold",
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.text(
        0,
        -0.22,
        "Internal qualitative evidence; no manual-reference accuracy claim.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "clahe_comparison_summary.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def export_unblinded_qc_names(
    masked_qc_directory: Path,
    unblinded_qc_directory: Path,
    records: list[dict],
) -> None:
    """Copy completed masked panels to method-labeled filenames."""
    unblinded_qc_directory.mkdir(exist_ok=True)
    for record in records:
        source_hash = record["Source_SHA256"]
        source = masked_qc_directory / f"{source_hash[:12]}_masked.png"
        if not source.exists():
            if record["Status"] == "excluded":
                continue
            raise RuntimeError(f"Missing masked QC panel: {source}")
        method_a = record["Candidate_A_Method"].replace("_", "-")
        method_b = record["Candidate_B_Method"].replace("_", "-")
        preferred = record["Preferred_Method"].replace("_", "-")
        destination = unblinded_qc_directory / (
            f"{source_hash[:12]}_A-{method_a}_B-{method_b}"
            f"_preferred-{preferred}.png"
        )
        shutil.copy2(source, destination)


def finalize_complete_batch(
    output_dir: Path,
    masked_records: list[dict],
    mappings: dict[str, tuple[str, str, int]],
    show_labels: bool,
) -> None:
    """Unmask and summarize a complete configured batch."""
    unblinded = unblind_records(masked_records, mappings)
    write_records(
        output_dir / "clahe_results_unblinded.csv",
        unblinded,
        UNBLINDED_CSV_FIELDS,
    )
    write_summary(output_dir, unblinded, show_labels)
    export_unblinded_qc_names(
        output_dir / "masked_qc",
        output_dir / "unblinded_qc",
        unblinded,
    )


def ensure_no_prior_unblinding(output_dir: Path, batch_complete: bool) -> None:
    """Prevent collection from resuming after method identities were exposed."""
    if batch_complete:
        return
    artifacts = (
        output_dir / "clahe_results_unblinded.csv",
        output_dir / "clahe_comparison_summary.json",
        output_dir / "clahe_comparison_summary.png",
        output_dir / "unblinded_qc",
    )
    existing = [path for path in artifacts if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise RuntimeError(
            "Cannot resume an incomplete comparison after unblinded artifacts "
            f"exist ({names}). Use a new output directory."
        )


def ensure_interactive_backend() -> None:
    """Fail clearly when the script is launched without an interactive GUI."""
    backend = plt.get_backend().lower()
    noninteractive = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
    if backend in noninteractive or "inline" in backend:
        raise RuntimeError(
            "This comparison requires an interactive Matplotlib desktop "
            "backend. Run it from a local graphical session."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Prepared uint8 single-channel TIFF files or directories.",
    )
    parser.add_argument(
        "--reviewer-id",
        default="reviewer_1",
        help=(
            "Pseudonymous reviewer identifier recorded with each decision "
            "(default: reviewer_1)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("clahe_method_comparison"),
        help="Output directory (default: ./clahe_method_comparison).",
    )
    parser.add_argument(
        "--fixed-clip-limit",
        type=float,
        choices=FIXED_CLIP_LIMITS,
        help=(
            "Use one clip limit locked before evaluation. If omitted, one "
            "setting per image is selected from whole-image entropy."
        ),
    )
    parser.add_argument(
        "--nlm-h",
        type=int,
        choices=range(1, 21),
        metavar="1..20",
        help=(
            "Use one common NLM h for both methods. If omitted, the production "
            "baseline is estimated from adaptive CLAHE and then held fixed."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Counterbalanced A/B seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Reveal method labels for debugging; invalidates masking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.reviewer_id.strip():
        raise ValueError("--reviewer-id must not be empty.")
    ensure_interactive_backend()

    images, common_root = discover_images(args.inputs)
    input_manifest = build_input_manifest(images, common_root)
    expected_hashes = {item["source_sha256"] for item in input_manifest}
    mappings = build_candidate_mappings(
        [item["source_sha256"] for item in input_manifest],
        args.random_seed,
        args.reviewer_id,
    )
    source_hash_by_path = {
        image_path: item["source_sha256"]
        for image_path, item in zip(images, input_manifest)
    }
    review_images = sorted(
        images,
        key=lambda image_path: mappings[source_hash_by_path[image_path]][2],
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "clahe_masked_reviews.csv"
    initialize_config(
        output_dir / "clahe_comparison_config.json",
        records_path,
        comparison_settings(args),
        input_manifest,
        implementation_fingerprint(),
        environment_fingerprint(),
    )
    records = load_masked_records(
        records_path,
        args.reviewer_id,
        expected_hashes,
        args.random_seed,
        args.show_labels,
    )
    completed_hashes = {record["Source_SHA256"] for record in records}
    ensure_no_prior_unblinding(
        output_dir,
        batch_complete=completed_hashes == expected_hashes,
    )
    masked_qc_directory = output_dir / "masked_qc"
    masked_qc_directory.mkdir(exist_ok=True)
    roi_directory = output_dir / "roi_records"
    roi_directory.mkdir(exist_ok=True)
    candidate_cache_directory = output_dir / "candidate_cache"
    candidate_cache_directory.mkdir(exist_ok=True)
    validate_candidate_cache_files(records, candidate_cache_directory)

    print(f"Found {len(images)} unique TIFF image(s).")
    print(f"Existing completed reviews: {len(completed_hashes)}")
    print(
        "Interpretation: this is an internal qualitative comparison. "
        "It does not prove biological accuracy."
    )
    if args.show_labels:
        print("WARNING: method labels are visible; this is an unblinded debug run.")

    manifest_by_hash = {
        item["source_sha256"]: item for item in input_manifest
    }
    for index, image_path in enumerate(review_images, start=1):
        source_hash = source_hash_by_path[image_path]
        if source_hash in completed_hashes:
            print(
                f"[{index}/{len(review_images)}] "
                f"Skipping completed: {image_path.name}"
            )
            continue

        print(f"[{index}/{len(review_images)}] Processing: {image_path}")
        raw_image = load_uint8_grayscale(image_path)
        roi_path = roi_directory / f"{source_hash[:12]}_roi.json"
        roi_bounds = load_roi_record(roi_path, raw_image.shape)
        if roi_bounds is None:
            roi_bounds = select_review_region(raw_image, image_path.name)
            if roi_bounds is not None:
                write_roi_record(roi_path, roi_bounds, raw_image.shape)
        else:
            print("  Reusing the pre-exposure ROI saved before interruption.")
        metadata = manifest_by_hash[source_hash]
        if roi_bounds is None:
            record = {
                "Source_File": metadata["source_file"],
                "Relative_Path": metadata["relative_path"],
                "Source_Folder": metadata["source_folder"],
                "Source_SHA256": source_hash,
                "Reviewer_ID": args.reviewer_id,
                "Reviewed_UTC": datetime.now(timezone.utc).isoformat(),
                "Status": "excluded",
                "Exclusion_Stage": "pre_exposure",
                "ROI_X_Min_px": "",
                "ROI_X_Max_px": "",
                "ROI_Y_Min_px": "",
                "ROI_Y_Max_px": "",
                "Image_Width_px": raw_image.shape[1],
                "Image_Height_px": raw_image.shape[0],
                "Fixed_Clip_Limit": "",
                "Fixed_Clip_Selection": "not_run",
                "Tile_Grid": "8x8",
                "Adaptive_Sigma_Estimate": "",
                "Common_NLM_h": "",
                "Common_NLM_h_Source": "not_run",
                "Candidate_Cache_SHA256": "",
                "Random_Seed": args.random_seed,
                "Displayed_Method_Labels": bool(args.show_labels),
                "Reviewer_Choice": "excluded",
            }
            records.append(record)
            completed_hashes.add(source_hash)
            write_records(records_path, records, MASKED_CSV_FIELDS)
            print("  Pre-exposure exclusion recorded.")
            continue

        cache_path = candidate_cache_directory / f"{source_hash[:12]}.npz"
        fixed_selection = (
            "pre_locked"
            if args.fixed_clip_limit is not None
            else "single_setting_from_whole_image_entropy"
        )
        if cache_path.exists():
            print("  Reusing the exact candidates saved before interruption.")
            (
                cache_metadata,
                enhanced_by_method,
                skeleton_by_method,
            ) = load_candidate_cache(cache_path, source_hash, raw_image.shape)
            if cache_metadata.get("fixed_clip_selection") != fixed_selection:
                raise RuntimeError(
                    f"Candidate cache settings do not match: {cache_path}"
                )
            fixed_clip = float(cache_metadata["fixed_clip_limit"])
            adaptive_sigma = float(cache_metadata["adaptive_sigma_estimate"])
            common_h = int(cache_metadata["common_nlm_h"])
            h_source = str(cache_metadata["common_nlm_h_source"])
            expected_fixed_clip = (
                float(args.fixed_clip_limit)
                if args.fixed_clip_limit is not None
                else whole_image_clip_from_stats(raw_image)
            )
            if fixed_clip != expected_fixed_clip:
                raise RuntimeError(
                    f"Candidate cache fixed clip does not match: {cache_path}"
                )
            expected_sigma = float(
                estimate_sigma(
                    enhanced_by_method[METHOD_ADAPTIVE],
                    channel_axis=None,
                )
            )
            if not np.isclose(adaptive_sigma, expected_sigma):
                raise RuntimeError(
                    f"Candidate cache sigma does not match: {cache_path}"
                )
            expected_common_h = (
                int(args.nlm_h)
                if args.nlm_h is not None
                else nlm_baseline_from_sigma(expected_sigma)
            )
            expected_h_source = (
                "command_line_locked"
                if args.nlm_h is not None
                else "production_adaptive_noise_heuristic"
            )
            if common_h != expected_common_h or h_source != expected_h_source:
                raise RuntimeError(
                    f"Candidate cache NLM h does not match: {cache_path}"
                )
        else:
            print("  Computing matched preprocessing and skeleton candidates...")
            adaptive_image = apply_adaptive_patching(
                raw_image,
                float(np.var(raw_image)),
            )
            fixed_clip = (
                float(args.fixed_clip_limit)
                if args.fixed_clip_limit is not None
                else whole_image_clip_from_stats(raw_image)
            )
            fixed_image = apply_fixed_clahe(raw_image, fixed_clip)
            adaptive_sigma = float(
                estimate_sigma(adaptive_image, channel_axis=None)
            )
            common_h = args.nlm_h or nlm_baseline_from_sigma(adaptive_sigma)
            h_source = (
                "command_line_locked"
                if args.nlm_h
                else "production_adaptive_noise_heuristic"
            )
            enhanced_by_method = {
                METHOD_ADAPTIVE: adaptive_image,
                METHOD_FIXED: fixed_image,
            }
            skeleton_by_method = {
                method: skeleton_for_common_h(image, common_h)
                for method, image in enhanced_by_method.items()
            }
            cache_metadata = {
                "source_sha256": source_hash,
                "fixed_clip_limit": fixed_clip,
                "fixed_clip_selection": fixed_selection,
                "adaptive_sigma_estimate": adaptive_sigma,
                "common_nlm_h": common_h,
                "common_nlm_h_source": h_source,
            }
            write_candidate_cache(
                cache_path,
                cache_metadata,
                enhanced_by_method[METHOD_ADAPTIVE],
                enhanced_by_method[METHOD_FIXED],
                skeleton_by_method[METHOD_ADAPTIVE],
                skeleton_by_method[METHOD_FIXED],
            )
        candidate_cache_hash = sha256_file(cache_path)

        method_a, method_b, _block_index = mappings[source_hash]
        method_order = (method_a, method_b)

        reviewer_choice = review_pair(
            raw_image,
            enhanced_by_method,
            skeleton_by_method,
            image_path.name,
            method_order,
            roi_bounds,
            common_h,
            args.show_labels,
        )
        status = "excluded" if reviewer_choice == "excluded" else "reviewed"
        save_masked_qc(
            masked_qc_directory / f"{source_hash[:12]}_masked.png",
            raw_image,
            enhanced_by_method,
            skeleton_by_method,
            method_order,
            roi_bounds,
            image_path.name,
            common_h,
            reviewer_choice,
            args.show_labels,
        )

        x_min, x_max, y_min, y_max = roi_bounds
        record = {
            "Source_File": metadata["source_file"],
            "Relative_Path": metadata["relative_path"],
            "Source_Folder": metadata["source_folder"],
            "Source_SHA256": source_hash,
            "Reviewer_ID": args.reviewer_id,
            "Reviewed_UTC": datetime.now(timezone.utc).isoformat(),
            "Status": status,
            "Exclusion_Stage": (
                "post_exposure" if reviewer_choice == "excluded" else ""
            ),
            "ROI_X_Min_px": x_min,
            "ROI_X_Max_px": x_max,
            "ROI_Y_Min_px": y_min,
            "ROI_Y_Max_px": y_max,
            "Image_Width_px": raw_image.shape[1],
            "Image_Height_px": raw_image.shape[0],
            "Fixed_Clip_Limit": fixed_clip,
            "Fixed_Clip_Selection": fixed_selection,
            "Tile_Grid": "8x8",
            "Adaptive_Sigma_Estimate": f"{adaptive_sigma:.10g}",
            "Common_NLM_h": common_h,
            "Common_NLM_h_Source": h_source,
            "Candidate_Cache_SHA256": candidate_cache_hash,
            "Random_Seed": args.random_seed,
            "Displayed_Method_Labels": bool(args.show_labels),
            "Reviewer_Choice": reviewer_choice,
        }
        records.append(record)
        completed_hashes.add(source_hash)
        write_records(records_path, records, MASKED_CSV_FIELDS)
        print("  Masked decision recorded.")

    if completed_hashes == expected_hashes:
        validate_masked_records(
            records,
            args.reviewer_id,
            expected_hashes,
            args.random_seed,
            args.show_labels,
        )
        finalize_complete_batch(output_dir, records, mappings, args.show_labels)
        print("Configured batch complete; method identities have been released.")
        print(
            f"Unblinded results: "
            f"{output_dir / 'clahe_results_unblinded.csv'}"
        )
        print(
            f"Descriptive summary: "
            f"{output_dir / 'clahe_comparison_summary.json'}"
        )
    else:
        remaining = len(expected_hashes - completed_hashes)
        print(f"Masked records saved. {remaining} configured image(s) remain.")

    print(
        "Do not report visual preferences as accuracy. Independent manual "
        "reference traces are required for a superiority claim."
    )


if __name__ == "__main__":
    main()

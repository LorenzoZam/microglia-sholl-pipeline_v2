"""Collect blinded, image-level NLM ``h`` selections for later calibration.

For every prepared single-channel TIFF, this script reproduces the production
preprocessing used by ``run_sholl_pipeline.py``. It computes the current
baseline for comparison and displays the fixed second-test range h=5 through
h=17. Candidate positions and labels are randomized so the reviewer is not
anchored to the current baseline.

The script does not alter the production pipeline or automatically recommend a
new formula. It records observations that can later support that decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from skimage.restoration import estimate_sigma

from pipeline_config import NLM_BASELINE_SIGMA_FACTOR, nlm_baseline_from_sigma

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


MINIMUM_CANDIDATE_H = 5
MAXIMUM_CANDIDATE_H = 17
CSV_FIELDS = (
    "Source_File",
    "Relative_Path",
    "Source_SHA256",
    "Reviewed_UTC",
    "Status",
    "Sigma_Estimate",
    "Baseline_h",
    "Minimum_h_Shown",
    "Maximum_h_Shown",
    "Selected_h",
    "Selected_Minus_Baseline",
    "Selected_h_Per_Sigma",
    "ROI_X_Min_px",
    "ROI_X_Max_px",
    "ROI_Y_Min_px",
    "ROI_Y_Max_px",
    "Width_px",
    "Height_px",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_input_directory() -> Path:
    root = Tk()
    root.withdraw()
    selected = filedialog.askdirectory(title="Select prepared Iba1 TIFF directory")
    root.destroy()
    if not selected:
        raise RuntimeError("No input directory was selected.")
    return Path(selected)


def discover_images(inputs: list[Path]) -> tuple[list[Path], Path]:
    if not inputs:
        inputs = [select_input_directory()]

    images: list[Path] = []
    for item in inputs:
        resolved = item.resolve()
        if resolved.is_dir():
            images.extend(resolved.rglob("*.tif"))
            images.extend(resolved.rglob("*.tiff"))
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


def compute_baseline(processed_image: np.ndarray) -> tuple[float, int, list[int]]:
    """Reproduce the baseline formula currently used by the main pipeline."""
    sigma_estimate = float(estimate_sigma(processed_image, channel_axis=None))
    baseline_h = nlm_baseline_from_sigma(sigma_estimate)
    candidates = list(range(MINIMUM_CANDIDATE_H, MAXIMUM_CANDIDATE_H + 1))
    return sigma_estimate, baseline_h, candidates


def skeleton_for_h(processed_image: np.ndarray, h_value: int) -> np.ndarray:
    """Run the production post-CLAHE preprocessing for one NLM strength."""
    denoised = cv2.fastNlMeansDenoising(
        processed_image,
        None,
        h=h_value,
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


def overlay_skeleton(image: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    mask = skeleton.astype(bool)
    rgb[mask] = (0, 255, 70)
    return rgb


def select_review_region(
    processed_image: np.ndarray, image_name: str
) -> tuple[int, int, int, int]:
    """Let the reviewer set one field of view on the post-CLAHE image."""
    height, width = processed_image.shape
    fig, axis = plt.subplots(figsize=(11, 8))
    fig.canvas.manager.set_window_title("Select NLM calibration region")
    axis.imshow(processed_image, cmap="gray")
    axis.set_title(
        "Use the toolbar to zoom/pan to the benchmark cell(s), then confirm",
        fontweight="bold",
    )
    axis.set_xlabel(
        "The current field of view will be copied exactly to every h candidate."
    )

    state: dict[str, tuple[int, int, int, int] | bool | None] = {
        "bounds": None,
        "confirmed": False,
    }
    status = fig.text(
        0.02,
        0.025,
        "Tip: click the magnifying-glass tool, drag around the cells, then "
        "disable the tool before confirming.",
        fontsize=10,
    )

    def confirm_region(_event) -> None:
        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()
        x_min = max(0, int(np.floor(min(x_limits))))
        x_max = min(width, int(np.ceil(max(x_limits))))
        y_min = max(0, int(np.floor(min(y_limits))))
        y_max = min(height, int(np.ceil(max(y_limits))))
        if x_max - x_min < 4 or y_max - y_min < 4:
            status.set_text("The selected view is too small; zoom out and try again.")
            fig.canvas.draw_idle()
            return
        state["bounds"] = (x_min, x_max, y_min, y_max)
        state["confirmed"] = True
        plt.close(fig)

    button_axis = fig.add_axes([0.78, 0.015, 0.2, 0.05])
    confirm_button = Button(
        button_axis, "Use current view", color="#B8E0B8", hovercolor="#9DD49D"
    )
    confirm_button.on_clicked(confirm_region)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.1)
    fig.suptitle(f"Benchmark region: {image_name}", fontsize=13)
    try:
        fig.canvas.manager.window.state("zoomed")
    except Exception:
        pass
    plt.show(block=True)

    if not state["confirmed"] or state["bounds"] is None:
        plt.close(fig)
        raise RuntimeError(
            "Region-selection window was closed without confirming a field of view."
        )
    return state["bounds"]


def choose_candidate(
    processed_image: np.ndarray,
    skeletons: dict[int, np.ndarray],
    image_name: str,
    random_seed: int,
    show_h_labels: bool,
    roi_bounds: tuple[int, int, int, int],
) -> tuple[int | None, bool]:
    """Return the selected h and whether the reviewer requested another ROI."""
    candidate_values = list(skeletons)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(candidate_values)

    fig, axes = plt.subplots(4, 4, figsize=(17, 11))
    fig.canvas.manager.set_window_title("NLM h calibration")
    flat_axes = list(axes.flat)
    original_axis = flat_axes[0]
    original_axis.imshow(processed_image, cmap="gray")
    original_axis.set_title("CLAHE reference ROI", fontweight="bold")
    original_axis.set_xticks([])
    original_axis.set_yticks([])

    candidate_axes: dict[object, int] = {}
    labels = [chr(ord("A") + index) for index in range(len(candidate_values))]
    for axis, label, h_value in zip(flat_axes[1:], labels, candidate_values):
        axis.imshow(overlay_skeleton(processed_image, skeletons[h_value]))
        title = f"Candidate {label}"
        if show_h_labels:
            title += f" (h={h_value})"
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        candidate_axes[axis] = h_value

    used_axes = 1 + len(candidate_values)
    x_min, x_max, y_min, y_max = roi_bounds
    for axis in flat_axes[:used_axes]:
        axis.set_xlim(x_min - 0.5, x_max - 0.5)
        axis.set_ylim(y_max - 0.5, y_min - 0.5)

    for axis in flat_axes[used_axes:]:
        axis.axis("off")

    state: dict[str, int | bool | None] = {
        "selected": None,
        "confirmed": False,
        "change_region": False,
    }
    status = fig.text(
        0.02,
        0.025,
        "Click one candidate, then confirm. Green is the final production skeleton.",
        fontsize=10,
    )

    def select_axis(event) -> None:
        if event.inaxes not in candidate_axes:
            return
        state["selected"] = candidate_axes[event.inaxes]
        for axis in candidate_axes:
            selected = axis is event.inaxes
            for spine in axis.spines.values():
                spine.set_visible(selected)
                spine.set_color("#D55E00")
                spine.set_linewidth(3)
        status.set_text("Candidate selected. Confirm or choose a different panel.")
        fig.canvas.draw_idle()

    def confirm(_event) -> None:
        if state["selected"] is None:
            status.set_text("Select a candidate before confirming.")
            fig.canvas.draw_idle()
            return
        state["confirmed"] = True
        plt.close(fig)

    def exclude(_event) -> None:
        state["selected"] = None
        state["confirmed"] = True
        plt.close(fig)

    def change_region(_event) -> None:
        state["change_region"] = True
        state["confirmed"] = True
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", select_axis)
    confirm_axis = fig.add_axes([0.62, 0.012, 0.15, 0.045])
    region_axis = fig.add_axes([0.78, 0.012, 0.11, 0.045])
    exclude_axis = fig.add_axes([0.90, 0.012, 0.08, 0.045])
    confirm_button = Button(confirm_axis, "Confirm selection", color="#B8E0B8")
    region_button = Button(region_axis, "Change region", color="#F4D9A6")
    exclude_button = Button(exclude_axis, "Exclude", color="#E5E5E5")
    confirm_button.on_clicked(confirm)
    region_button.on_clicked(change_region)
    exclude_button.on_clicked(exclude)

    fig.suptitle(f"NLM calibration: {image_name}", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.08, wspace=0.04, hspace=0.16)
    try:
        fig.canvas.manager.window.state("zoomed")
    except Exception:
        pass
    plt.show(block=True)

    if not state["confirmed"]:
        plt.close(fig)
        raise RuntimeError("Calibration window was closed without confirming or excluding the image.")
    return state["selected"], bool(state["change_region"])


def write_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_selected_qc(
    output_path: Path,
    processed_image: np.ndarray,
    skeleton: np.ndarray,
    image_name: str,
    baseline_h: int,
    selected_h: int,
    roi_bounds: tuple[int, int, int, int],
) -> None:
    """Save a reviewable record of the candidate that was accepted."""
    x_min, x_max, y_min, y_max = roi_bounds
    image_crop = processed_image[y_min:y_max, x_min:x_max]
    skeleton_crop = skeleton[y_min:y_max, x_min:x_max]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_crop, cmap="gray")
    axes[0].set_title("CLAHE benchmark region")
    axes[1].imshow(overlay_skeleton(image_crop, skeleton_crop))
    axes[1].set_title(f"Accepted skeleton: h={selected_h}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{image_name} | current baseline={baseline_h}, selected={selected_h}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(output_dir: Path, records: list[dict]) -> None:
    selected = [record for record in records if record["Status"] == "selected"]
    selected_h = np.array([float(record["Selected_h"]) for record in selected])
    baseline_h = np.array([float(record["Baseline_h"]) for record in selected])
    sigma = np.array([float(record["Sigma_Estimate"]) for record in selected])
    deltas = selected_h - baseline_h
    ratios = selected_h[sigma > 0] / sigma[sigma > 0]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_design": "fixed candidate range with image-specific benchmark ROI",
        "minimum_candidate_h": MINIMUM_CANDIDATE_H,
        "maximum_candidate_h": MAXIMUM_CANDIDATE_H,
        "baseline_formula": (
            f"min(max(int({NLM_BASELINE_SIGMA_FACTOR:g} * sigma), 2), 20)"
        ),
        "n_reviewed": len(records),
        "n_selected": len(selected),
        "n_excluded": sum(record["Status"] == "excluded" for record in records),
        "selected_h_counts": dict(sorted(Counter(map(int, selected_h)).items())),
        "n_selected_at_minimum": (
            int(np.count_nonzero(selected_h == MINIMUM_CANDIDATE_H))
            if len(selected)
            else 0
        ),
        "n_selected_at_maximum": (
            int(np.count_nonzero(selected_h == MAXIMUM_CANDIDATE_H))
            if len(selected)
            else 0
        ),
        "median_baseline_h": float(np.median(baseline_h)) if len(selected) else None,
        "median_selected_h": float(np.median(selected_h)) if len(selected) else None,
        "mean_selected_minus_baseline": float(np.mean(deltas)) if len(selected) else None,
        "median_selected_minus_baseline": float(np.median(deltas)) if len(selected) else None,
        "diagnostic_median_selected_h_per_sigma": (
            float(np.median(ratios)) if len(ratios) else None
        ),
        "interpretation_warning": (
            "The diagnostic ratio is not a validated replacement formula. "
            "Selections require independent review before production changes."
        ),
    }
    (output_dir / "nlm_h_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not len(selected):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(baseline_h, selected_h, color="#0072B2", edgecolor="white", s=55)
    limits = [min(baseline_h.min(), selected_h.min()) - 1, max(baseline_h.max(), selected_h.max()) + 1]
    axes[0].plot(limits, limits, linestyle="--", color="#777777", linewidth=1)
    axes[0].set(xlabel="Current baseline h", ylabel="Selected h", xlim=limits, ylim=limits)
    axes[0].set_title("Selection relative to baseline", loc="left")
    axes[1].hist(deltas, bins=np.arange(deltas.min() - 0.5, deltas.max() + 1.5), color="#009E73", rwidth=0.8)
    axes[1].set(xlabel="Selected h − baseline h", ylabel="Images")
    axes[1].set_title("Direction of reviewer adjustment", loc="left")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "nlm_h_calibration_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Prepared Iba1 TIFF files or directories; omit to select a directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("nlm_h_second_test"),
        help="Calibration output directory (default: ./nlm_h_second_test).",
    )
    parser.add_argument(
        "--show-h-labels",
        action="store_true",
        help="Show h values during review; by default candidates are blinded and randomized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, common_root = discover_images(args.inputs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "nlm_h_selections.csv"
    records = load_records(records_path)
    completed_hashes = {record["Source_SHA256"] for record in records}

    print(f"Found {len(images)} TIFF image(s).")
    print(f"Existing completed reviews: {len(completed_hashes)}")
    for index, image_path in enumerate(images, start=1):
        source_hash = sha256_file(image_path)
        if source_hash in completed_hashes:
            print(f"[{index}/{len(images)}] Skipping completed image: {image_path.name}")
            continue

        print(f"[{index}/{len(images)}] Processing: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(
                f"Expected a single-channel 8-bit TIFF, got "
                f"{None if image is None else (image.shape, image.dtype)}: {image_path}"
            )

        processed = apply_adaptive_patching(image, float(np.var(image)))
        sigma, baseline_h, candidates = compute_baseline(processed)
        print(
            f"  sigma={sigma:.3f}; baseline={baseline_h}; "
            f"candidate range={candidates[0]}–{candidates[-1]}"
        )
        roi_bounds = select_review_region(processed, image_path.name)
        skeletons = {
            h_value: skeleton_for_h(processed, h_value) for h_value in candidates
        }
        while True:
            selected_h, change_region = choose_candidate(
                processed,
                skeletons,
                image_path.name,
                random_seed=int(source_hash[:16], 16),
                show_h_labels=args.show_h_labels,
                roi_bounds=roi_bounds,
            )
            if not change_region:
                break
            roi_bounds = select_review_region(processed, image_path.name)

        try:
            relative_path = str(image_path.relative_to(common_root))
        except ValueError:
            relative_path = str(image_path)
        selected = selected_h is not None
        if selected:
            qc_directory = output_dir / "selected_qc"
            qc_directory.mkdir(exist_ok=True)
            save_selected_qc(
                qc_directory / f"{source_hash[:12]}_h{selected_h}.png",
                processed,
                skeletons[selected_h],
                image_path.name,
                baseline_h,
                selected_h,
                roi_bounds,
            )
        x_min, x_max, y_min, y_max = roi_bounds
        record = {
            "Source_File": image_path.name,
            "Relative_Path": relative_path,
            "Source_SHA256": source_hash,
            "Reviewed_UTC": datetime.now(timezone.utc).isoformat(),
            "Status": "selected" if selected else "excluded",
            "Sigma_Estimate": f"{sigma:.10g}",
            "Baseline_h": baseline_h,
            "Minimum_h_Shown": candidates[0],
            "Maximum_h_Shown": candidates[-1],
            "Selected_h": selected_h if selected else "",
            "Selected_Minus_Baseline": selected_h - baseline_h if selected else "",
            "Selected_h_Per_Sigma": f"{selected_h / sigma:.10g}" if selected and sigma > 0 else "",
            "ROI_X_Min_px": x_min,
            "ROI_X_Max_px": x_max,
            "ROI_Y_Min_px": y_min,
            "ROI_Y_Max_px": y_max,
            "Width_px": image.shape[1],
            "Height_px": image.shape[0],
        }
        records.append(record)
        completed_hashes.add(source_hash)
        write_records(records_path, records)
        write_summary(output_dir, records)
        print(f"  Recorded: {'h=' + str(selected_h) if selected else 'excluded'}")

    write_summary(output_dir, records)
    print(f"Calibration records: {records_path}")
    print(f"Calibration summary: {output_dir / 'nlm_h_calibration_summary.json'}")


if __name__ == "__main__":
    main()

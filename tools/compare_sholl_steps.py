"""Compare 2, 5, and 10 µm Sholl intervals on one fixed cell geometry.

The image is preprocessed exactly once with functions from
``run_sholl_pipeline.py``. The user selects one soma once, and the resulting
skeleton, connected component, and corrected soma anchor are reused for every
profile. The starting radius (5 µm) and maximum radius are also held constant.

Run without an image argument to choose a prepared single-channel Iba1 TIFF
with a file dialog::

    python -m tools.compare_sholl_steps

Or provide the image explicitly::

    python -m tools.compare_sholl_steps path/to/image_Iba1.tif
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

from run_sholl_pipeline import (
    apply_adaptive_patching,
    apply_dilate,
    apply_morph_close,
    apply_tophat,
    binarize_image,
    bridge_nearby_fragments,
    bridge_nearby_fragments_refine,
    compute_sholl_intersections,
    get_connected_component,
    preview_denoising,
    remove_isolated_fibers,
    remove_isolated_fibers_refine,
    remove_small_fragments,
    skeletonize_image,
)


START_RADIUS_UM = 5.0
INTERVALS_UM = (2.0, 5.0, 10.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _resolution_value(tag) -> float:
    value = tag.value
    if isinstance(value, tuple):
        numerator, denominator = value
        if denominator == 0:
            raise ValueError("TIFF resolution denominator is zero.")
        return float(numerator) / float(denominator)
    return float(value)


def read_calibration(path: Path) -> tuple[float, float]:
    """Return verified TIFF pixel width and height in µm/px."""
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1 or tif.series[0].axes != "YX":
            raise ValueError(
                "Expected a prepared single-channel YX TIFF. Run "
                "prepare_iba1_dataset.py before this comparison."
            )
        page = tif.pages[0]
        x_tag = page.tags.get("XResolution")
        y_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        if x_tag is None or y_tag is None or unit_tag is None:
            raise ValueError("The TIFF does not contain complete spatial calibration.")
        x_resolution = _resolution_value(x_tag)
        y_resolution = _resolution_value(y_tag)
        unit = int(unit_tag.value)
        if unit == 3:
            micrometres_per_unit = 10_000.0
        elif unit == 2:
            micrometres_per_unit = 25_400.0
        else:
            raise ValueError(f"Unsupported TIFF ResolutionUnit={unit}.")
        if min(x_resolution, y_resolution) <= 0:
            raise ValueError("TIFF spatial resolution must be positive.")
        return (
            micrometres_per_unit / x_resolution,
            micrometres_per_unit / y_resolution,
        )


def select_image() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Select a prepared single-channel Iba1 TIFF",
        filetypes=[("TIFF images", "*.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    if not selected:
        raise RuntimeError("No image was selected.")
    return Path(selected)


def preprocess_once(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply the production preprocessing chain exactly once."""
    processed = apply_adaptive_patching(image, float(np.var(image)))
    denoised, h_used = preview_denoising(processed)
    top_hat = apply_tophat(denoised)
    binary = binarize_image(top_hat)
    fragments_removed = remove_small_fragments(binary)
    closed = apply_morph_close(fragments_removed)
    dilated = apply_dilate(closed)
    skeleton = skeletonize_image(dilated)
    skeleton = bridge_nearby_fragments(skeleton)
    skeleton = remove_isolated_fibers_refine(skeleton)
    skeleton = bridge_nearby_fragments_refine(skeleton)
    skeleton = remove_isolated_fibers(skeleton)
    return skeleton, int(h_used)


def select_soma_once(
    image: np.ndarray, skeleton: np.ndarray
) -> tuple[int, int, int, int, np.ndarray]:
    """Collect one click and return raw/corrected coordinates and component."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Prepared Iba1 image")
    axes[1].imshow(skeleton, cmap="gray")
    axes[1].set_title("Click the soma once on either panel")
    for axis in axes:
        axis.axis("off")
    fig.suptitle("Select one soma; close the window to cancel")
    plt.tight_layout()
    points = plt.ginput(1, timeout=-1, show_clicks=True)
    plt.close(fig)
    if not points:
        raise RuntimeError("Soma selection was cancelled.")

    raw_x = int(round(points[0][0]))
    raw_y = int(round(points[0][1]))
    raw_x = int(np.clip(raw_x, 0, skeleton.shape[1] - 1))
    raw_y = int(np.clip(raw_y, 0, skeleton.shape[0] - 1))
    component, (corrected_x, corrected_y) = get_connected_component(
        skeleton, (raw_y, raw_x)
    )
    if component is None or not np.any(component):
        raise RuntimeError("No skeleton component was found near the selected soma.")
    return raw_x, raw_y, int(corrected_x), int(corrected_y), component


def common_maximum_radius_um(
    component: np.ndarray,
    soma_x: int,
    soma_y: int,
    pixel_size_um: float,
) -> tuple[float, float]:
    """Return arbor extent and one common outer radius beyond that extent."""
    yy, xx = np.where(component)
    extent_px = float(np.max(np.hypot(xx - soma_x, yy - soma_y)))
    extent_um = extent_px * pixel_size_um

    # Values 5 + 10*n occur in all three radius grids. Select at least one
    # 10-µm interval beyond the detected component so every profile ends at
    # the same radius and includes an outer zero-intersection observation.
    target_um = extent_um + max(INTERVALS_UM)
    common_max_um = START_RADIUS_UM + math.ceil(
        max(0.0, target_um - START_RADIUS_UM) / max(INTERVALS_UM)
    ) * max(INTERVALS_UM)
    return extent_um, common_max_um


def calculate_profiles(
    component: np.ndarray,
    soma_x: int,
    soma_y: int,
    pixel_size_um: float,
    maximum_radius_um: float,
) -> tuple[list[dict], list[dict]]:
    profile_rows: list[dict] = []
    summary_rows: list[dict] = []
    for interval_um in INTERVALS_UM:
        radii_um = np.arange(
            START_RADIUS_UM,
            maximum_radius_um + interval_um * 0.25,
            interval_um,
            dtype=float,
        )
        radii_px = radii_um / pixel_size_um
        intersections = compute_sholl_intersections(
            component, soma_x, soma_y, radii_px
        )
        for radius_um, radius_px, count in zip(
            radii_um, radii_px, intersections
        ):
            profile_rows.append(
                {
                    "Interval_um": interval_um,
                    "Start_Radius_um": START_RADIUS_UM,
                    "Maximum_Radius_um": maximum_radius_um,
                    "Radius_um": float(radius_um),
                    "Radius_px": float(radius_px),
                    "Intersections": int(count),
                }
            )

        maximum_intersections = int(np.max(intersections))
        first_count = int(intersections[0])
        critical_index = int(np.argmax(intersections))
        summary_rows.append(
            {
                "Interval_um": interval_um,
                "N_Radii": len(radii_um),
                "Maximum_Intersections": maximum_intersections,
                "Critical_Radius_um": float(radii_um[critical_index]),
                "First_Radius_Intersections": first_count,
                "Sampled_Ramification_Index": (
                    maximum_intersections / first_count if first_count else np.nan
                ),
                "AUC_Intersections_x_um": float(
                    np.trapezoid(intersections, radii_um)
                ),
            }
        )
    return profile_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(profile_rows: list[dict], output_path: Path) -> None:
    profiles = {
        interval_um: [
            row for row in profile_rows if row["Interval_um"] == interval_um
        ]
        for interval_um in INTERVALS_UM
    }
    shared_radii = sorted(
        set.intersection(
            *(
                {round(row["Radius_um"], 9) for row in rows}
                for rows in profiles.values()
            )
        )
    )

    colors = ("#0072B2", "#D55E00", "#009E73")
    markers = ("o", "s", "^")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.8))

        for radius_um in shared_radii:
            axis.axvline(
                radius_um,
                color="#B8B8B8",
                linewidth=0.7,
                linestyle=(0, (2, 3)),
                zorder=0,
            )

        for interval_um, color, marker in zip(INTERVALS_UM, colors, markers):
            rows = profiles[interval_um]
            axis.plot(
                [row["Radius_um"] for row in rows],
                [row["Intersections"] for row in rows],
                color=color,
                linewidth=1.6,
                marker=marker,
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1.1,
                label=f"{interval_um:g} µm interval",
                zorder=2,
            )

        reference_rows = {
            round(row["Radius_um"], 9): row for row in profiles[INTERVALS_UM[0]]
        }
        axis.scatter(
            shared_radii,
            [reference_rows[radius]["Intersections"] for radius in shared_radii],
            s=60,
            facecolors="none",
            edgecolors="#222222",
            linewidths=1.2,
            label="Radius sampled by all intervals",
            zorder=4,
        )

        axis.set_xlabel("Radius (µm)")
        axis.set_ylabel("Sholl intersections")
        axis.set_title("Effect of radial sampling interval", loc="left", pad=10)
        axis.text(
            0,
            1.01,
            "Same cell, soma centre, and maximum radius",
            transform=axis.transAxes,
            color="#555555",
            fontsize=9,
            va="bottom",
        )
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7, zorder=0)
        axis.legend(frameon=False, fontsize=9, handlelength=2.4)
        axis.margins(x=0.02)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def show_and_save_circle_qc(
    image: np.ndarray,
    component: np.ndarray,
    soma_x: int,
    soma_y: int,
    profile_rows: list[dict],
    output_path: Path,
) -> None:
    """Show main-QC-style circle overlays for all three interval grids."""
    maximum_radius_px = max(row["Radius_px"] for row in profile_rows)
    crop_radius = int(math.ceil(maximum_radius_px * 1.15))
    y0 = max(0, soma_y - crop_radius)
    y1 = min(image.shape[0], soma_y + crop_radius + 1)
    x0 = max(0, soma_x - crop_radius)
    x1 = min(image.shape[1], soma_x + crop_radius + 1)

    background = image[y0:y1, x0:x1]
    cell_crop = component[y0:y1, x0:x1]
    local_x = soma_x - x0
    local_y = soma_y - y0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for axis, interval_um in zip(axes, INTERVALS_UM):
        rgb = np.stack([background] * 3, axis=-1).copy()
        rgb[cell_crop] = [0, 255, 0]
        axis.imshow(rgb)

        rows = [row for row in profile_rows if row["Interval_um"] == interval_um]
        for index, row in enumerate(rows):
            radius_px = row["Radius_px"]
            axis.add_patch(
                Circle(
                    (local_x, local_y),
                    radius_px,
                    fill=False,
                    edgecolor="red",
                    linewidth=0.8,
                    alpha=0.75,
                )
            )
            angle = (index % 12) * (2 * np.pi / 12)
            text_x = local_x + radius_px * np.cos(angle)
            text_y = local_y - radius_px * np.sin(angle)
            axis.text(
                text_x,
                text_y,
                str(row["Intersections"]),
                color="yellow",
                fontsize=6,
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.4},
            )

        axis.scatter(
            [local_x], [local_y], s=45, c="yellow", edgecolors="black", zorder=5
        )
        axis.set_title(
            f"{interval_um:g} µm interval — {len(rows)} circles\n"
            f"red: radii, green: fixed component, yellow: counts"
        )
        axis.axis("off")

    fig.suptitle(
        "Fixed-cell Sholl circle QC — same soma, component, start, and maximum",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    try:
        fig.canvas.manager.window.state("zoomed")
    except Exception:
        pass
    plt.show(block=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Prepared single-channel Iba1 TIFF; omit to use a file dialog.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="New or empty output directory (default: beside the input image).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = (args.image or select_image()).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else image_path.parent / f"{image_path.stem}_step_comparison"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")

    pixel_width_um, pixel_height_um = read_calibration(image_path)
    if not math.isclose(pixel_width_um, pixel_height_um, rel_tol=1e-6):
        raise ValueError(
            "Sholl circles require square pixels; X and Y calibration differ."
        )
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")

    print(f"Image: {image_path}")
    print(f"Calibration: {pixel_width_um:.9g} µm/px")
    print("Preprocessing the image once...")
    skeleton, h_used = preprocess_once(image)
    raw_x, raw_y, soma_x, soma_y, component = select_soma_once(image, skeleton)
    extent_um, maximum_radius_um = common_maximum_radius_um(
        component, soma_x, soma_y, pixel_width_um
    )
    profile_rows, summary_rows = calculate_profiles(
        component, soma_x, soma_y, pixel_width_um, maximum_radius_um
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sholl_step_profiles.csv", profile_rows)
    write_csv(output_dir / "sholl_step_summary.csv", summary_rows)
    save_plot(profile_rows, output_dir / "sholl_step_comparison.png")
    show_and_save_circle_qc(
        image,
        component,
        soma_x,
        soma_y,
        profile_rows,
        output_dir / "sholl_circle_qc.png",
    )
    cv2.imwrite(str(output_dir / "fixed_skeleton.png"), skeleton.astype(np.uint8) * 255)
    cv2.imwrite(
        str(output_dir / "fixed_cell_component.png"),
        component.astype(np.uint8) * 255,
    )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": image_path.name,
        "source_sha256": sha256_file(image_path),
        "skeleton_sha256": sha256_array(skeleton),
        "component_sha256": sha256_array(component),
        "pixel_size_um_per_px": pixel_width_um,
        "nlm_h": h_used,
        "raw_soma_xy_px": [raw_x, raw_y],
        "corrected_soma_xy_px": [soma_x, soma_y],
        "start_radius_um": START_RADIUS_UM,
        "intervals_um": list(INTERVALS_UM),
        "detected_component_extent_um": extent_um,
        "common_maximum_radius_um": maximum_radius_um,
    }
    with (output_dir / "comparison_provenance.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"Raw soma:       ({raw_x}, {raw_y}) px")
    print(f"Corrected soma: ({soma_x}, {soma_y}) px")
    print(f"Component extent: {extent_um:.3f} µm")
    print(f"Common maximum:   {maximum_radius_um:.3f} µm")
    print(f"Results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

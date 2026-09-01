"""Create auditable single-channel Iba1 TIFFs from the source hyperstacks.

This utility is specific to the staining protocol used for BioImage Archive
studies S-BIAD1280 and S-BIAD1581. In those studies, Iba1 was detected with an
Alexa Fluor 488 secondary antibody. The archived ImageJ hyperstacks use Fiji
channel 2 (one-based) with a green LUT for that signal.

The source TIFFs are never modified. Each output is a single-channel TIFF with
the original pixel values and spatial resolution, accompanied by CSV and JSON
manifests containing input/output SHA-256 hashes and extraction metadata.

Examples
--------
Extract one image::

    python -m tools.prepare_iba1_dataset "M image 02.tif" -o derived_iba1

Recursively extract every TIFF below one or more directories::

    python -m tools.prepare_iba1_dataset "Dataset Rat/S-BIAD1280" \
        "Dataset Rat/S-BIAD1581" -o derived_iba1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


TOOL_VERSION = "1.1.0"
DEFAULT_IBA1_CHANNEL = 2  # Fiji/ImageJ numbering is one-based.
PROTOCOL_DOI = "10.17504/protocols.io.kqdg3xbbeg25/v1"


@dataclass(frozen=True)
class DiscoveredTiff:
    source: Path
    relative_parent: Path


@dataclass(frozen=True)
class ExtractionPlan:
    source: Path
    source_relative_path: Path
    source_sha256: str
    output_relative_path: Path
    data: np.ndarray
    channel_count: int
    channel_one_based: int
    pixel_width_um: float
    pixel_height_um: float
    calibration_source: str
    lut_verification: str


@dataclass(frozen=True)
class ManifestRecord:
    source_file: str
    source_sha256: str
    output_file: str
    output_sha256: str
    width_px: int
    height_px: int
    dtype: str
    source_channel_count: int
    iba1_channel_one_based: int
    marker: str
    fluorophore: str
    pixel_width_um: float
    pixel_height_um: float
    calibration_source: str
    channel_verification: str
    staining_protocol_doi: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rational_to_float(value) -> float:
    if isinstance(value, tuple):
        numerator, denominator = value
        if denominator == 0:
            raise ValueError("TIFF resolution denominator is zero.")
        return float(numerator) / float(denominator)
    return float(value)


def _pixel_size_um(page: tifffile.TiffPage) -> tuple[float, float, str]:
    x_tag = page.tags.get("XResolution")
    y_tag = page.tags.get("YResolution")
    unit_tag = page.tags.get("ResolutionUnit")
    if x_tag is None or y_tag is None or unit_tag is None:
        raise ValueError(
            "Missing TIFF XResolution, YResolution, or ResolutionUnit; "
            "spatial calibration cannot be verified."
        )

    x_resolution = _rational_to_float(x_tag.value)
    y_resolution = _rational_to_float(y_tag.value)
    unit = int(unit_tag.value)
    if not math.isfinite(x_resolution) or x_resolution <= 0:
        raise ValueError("TIFF XResolution must be finite and positive.")
    if not math.isfinite(y_resolution) or y_resolution <= 0:
        raise ValueError("TIFF YResolution must be finite and positive.")

    if unit == 3:  # TIFF ResolutionUnit.CENTIMETER
        micrometres_per_unit = 10_000.0
        source = "TIFF resolution tags (centimetre)"
    elif unit == 2:  # TIFF ResolutionUnit.INCH
        micrometres_per_unit = 25_400.0
        source = "TIFF resolution tags (inch)"
    else:
        raise ValueError(
            f"Unsupported or unitless TIFF ResolutionUnit={unit}; "
            "spatial calibration cannot be verified."
        )

    return (
        micrometres_per_unit / x_resolution,
        micrometres_per_unit / y_resolution,
        source,
    )


def _is_green_lut(page: tifffile.TiffPage) -> bool:
    colormap = page.colormap
    if colormap is None or colormap.shape[0] != 3:
        return False
    endpoint = np.asarray(colormap[:, -1], dtype=float)
    return endpoint[1] > 0 and endpoint[1] > endpoint[0] and endpoint[1] > endpoint[2]


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem or "image"


def discover_tiffs(inputs: Iterable[Path], output_dir: Path) -> list[DiscoveredTiff]:
    files: dict[Path, DiscoveredTiff] = {}
    output_resolved = output_dir.resolve()
    for item in inputs:
        if item.is_file():
            if item.suffix.lower() not in {".tif", ".tiff"}:
                raise ValueError(f"Input is not a TIFF file: {item}")
            source = item.resolve()
            files.setdefault(source, DiscoveredTiff(source, Path()))
        elif item.is_dir():
            input_root = item.resolve()
            for candidate in item.rglob("*"):
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in {".tif", ".tiff"}
                    and output_resolved not in candidate.resolve().parents
                ):
                    source = candidate.resolve()
                    relative_parent = source.relative_to(input_root).parent
                    files.setdefault(
                        source, DiscoveredTiff(source, relative_parent)
                    )
        else:
            raise FileNotFoundError(f"Input does not exist: {item}")

    discovered = sorted(files.values(), key=lambda item: str(item.source).lower())
    if not discovered:
        raise ValueError("No TIFF files were found in the supplied inputs.")
    return discovered


def inspect_source(
    source: Path,
    relative_parent: Path,
    channel_one_based: int,
    allow_unverified_lut: bool,
) -> ExtractionPlan:
    with tifffile.TiffFile(source) as tif:
        if len(tif.series) != 1:
            raise ValueError(f"Expected one TIFF series, found {len(tif.series)}.")
        series = tif.series[0]
        if series.axes != "CYX":
            raise ValueError(
                f"Expected a CYX channel stack, found axes={series.axes!r} "
                f"and shape={series.shape}."
            )
        channel_count = int(series.shape[0])
        if not 1 <= channel_one_based <= channel_count:
            raise ValueError(
                f"Requested channel {channel_one_based}, but the TIFF has "
                f"{channel_count} channels."
            )

        channel_index = channel_one_based - 1
        if len(tif.pages) <= channel_index:
            raise ValueError("TIFF pages do not match the declared channel stack.")
        channel_page = tif.pages[channel_index]
        green_lut = _is_green_lut(channel_page)
        if not green_lut and not allow_unverified_lut:
            raise ValueError(
                f"Fiji channel {channel_one_based} does not have the expected "
                "green LUT. Refusing extraction without --allow-unverified-lut."
            )

        pixel_width_um, pixel_height_um, calibration_source = _pixel_size_um(
            channel_page
        )
        data = np.asarray(series.asarray()[channel_index])
        if data.ndim != 2:
            raise ValueError(f"Selected channel is not 2-D: shape={data.shape}.")
        if data.dtype.kind not in "ui":
            raise ValueError(f"Expected integer image data, found dtype={data.dtype}.")

    source_hash = sha256_file(source)
    output_name = f"{_safe_stem(source)}_Iba1_{source_hash[:10]}.tif"
    return ExtractionPlan(
        source=source,
        source_relative_path=relative_parent / source.name,
        source_sha256=source_hash,
        output_relative_path=relative_parent / output_name,
        data=data,
        channel_count=channel_count,
        channel_one_based=channel_one_based,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
        calibration_source=calibration_source,
        lut_verification=(
            "green LUT verified" if green_lut else "LUT verification overridden"
        ),
    )


def write_extraction(plan: ExtractionPlan, output_dir: Path) -> ManifestRecord:
    output_path = output_dir / plan.output_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixels_per_centimetre_x = 10_000.0 / plan.pixel_width_um
    pixels_per_centimetre_y = 10_000.0 / plan.pixel_height_um
    tifffile.imwrite(
        output_path,
        plan.data,
        photometric="minisblack",
        resolution=(pixels_per_centimetre_x, pixels_per_centimetre_y),
        resolutionunit="CENTIMETER",
        metadata={
            "axes": "YX",
            "unit": "um",
            "marker": "Iba1",
            "fluorophore": "Alexa Fluor 488",
            "source_channel_one_based": plan.channel_one_based,
            "source_sha256": plan.source_sha256,
            "staining_protocol_doi": PROTOCOL_DOI,
        },
    )

    # Verify that writing did not alter pixels or calibration.
    with tifffile.TiffFile(output_path) as tif:
        written = tif.asarray()
        width_um, height_um, _ = _pixel_size_um(tif.pages[0])
    if not np.array_equal(written, plan.data):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Pixel verification failed for {output_path}.")
    if not math.isclose(width_um, plan.pixel_width_um, rel_tol=1e-6):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"X calibration verification failed for {output_path}.")
    if not math.isclose(height_um, plan.pixel_height_um, rel_tol=1e-6):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Y calibration verification failed for {output_path}.")

    height_px, width_px = plan.data.shape
    return ManifestRecord(
        source_file=plan.source_relative_path.as_posix(),
        source_sha256=plan.source_sha256,
        output_file=plan.output_relative_path.as_posix(),
        output_sha256=sha256_file(output_path),
        width_px=width_px,
        height_px=height_px,
        dtype=str(plan.data.dtype),
        source_channel_count=plan.channel_count,
        iba1_channel_one_based=plan.channel_one_based,
        marker="Iba1",
        fluorophore="Alexa Fluor 488",
        pixel_width_um=plan.pixel_width_um,
        pixel_height_um=plan.pixel_height_um,
        calibration_source=plan.calibration_source,
        channel_verification=plan.lut_verification,
        staining_protocol_doi=PROTOCOL_DOI,
    )


def write_manifests(records: list[ManifestRecord], output_dir: Path) -> None:
    fieldnames = list(asdict(records[0]).keys())
    csv_path = output_dir / "iba1_dataset_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    calibration_summary: dict[str, int] = {}
    for record in records:
        key = f"{record.pixel_width_um:.9g} x {record.pixel_height_um:.9g} um/px"
        calibration_summary[key] = calibration_summary.get(key, 0) + 1

    document = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": Path(__file__).name,
        "tool_version": TOOL_VERSION,
        "marker": "Iba1",
        "fluorophore": "Alexa Fluor 488",
        "default_source_channel_one_based": DEFAULT_IBA1_CHANNEL,
        "staining_protocol_doi": PROTOCOL_DOI,
        "calibration_summary": calibration_summary,
        "records": [asdict(record) for record in records],
    }
    json_path = output_dir / "iba1_dataset_manifest.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="TIFF files or directories searched recursively for TIFF files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty directory for derived TIFFs and manifests.",
    )
    parser.add_argument(
        "--iba1-channel",
        type=int,
        default=DEFAULT_IBA1_CHANNEL,
        help="One-based Fiji channel number (default: 2).",
    )
    parser.add_argument(
        "--allow-unverified-lut",
        action="store_true",
        help="Allow extraction when the selected channel lacks a green LUT.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be new or empty: {output_dir}"
        )

    sources = discover_tiffs(args.inputs, output_dir)
    plans: list[ExtractionPlan] = []
    for discovered in sources:
        try:
            plans.append(
                inspect_source(
                    discovered.source,
                    relative_parent=discovered.relative_parent,
                    channel_one_based=args.iba1_channel,
                    allow_unverified_lut=args.allow_unverified_lut,
                )
            )
        except Exception as error:
            raise ValueError(
                f"Validation failed for {discovered.source}: {error}"
            ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    records = [write_extraction(plan, output_dir) for plan in plans]
    write_manifests(records, output_dir)

    print(f"Prepared {len(records)} Iba1 image(s) in: {output_dir}")
    print(f"Manifest: {output_dir / 'iba1_dataset_manifest.csv'}")
    for record in records:
        print(
            f"  {record.output_file}: {record.pixel_width_um:.6f} x "
            f"{record.pixel_height_um:.6f} um/px"
        )
    calibration_values = {
        (round(record.pixel_width_um, 9), round(record.pixel_height_um, 9))
        for record in records
    }
    if len(calibration_values) > 1:
        print(
            "WARNING: The batch contains multiple spatial calibrations. "
            "Verify these acquisition differences before pooled analysis."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

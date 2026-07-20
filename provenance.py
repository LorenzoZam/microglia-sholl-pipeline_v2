"""Utilities for machine-readable analysis provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "numpy", "pandas", "scipy", "opencv-python-headless", "matplotlib",
    "seaborn", "streamlit", "scikit-image", "networkx", "statsmodels",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repository):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_manifest(input_path, parameters, decisions=None):
    path = Path(input_path)
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": "MicroSholl",
        "software_version": "2.0.0",
        "git_commit": _git_commit(Path(__file__).parent),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "input": {
            "name": path.name,
            "sha256": sha256_file(path),
        },
        "parameters": parameters,
        "qc_decisions": decisions or {},
    }


def write_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)

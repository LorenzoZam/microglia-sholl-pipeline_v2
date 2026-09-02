"""Shared, explicitly documented configuration for MicroSholl preprocessing."""

from __future__ import annotations

import math


# Dataset-specific heuristic derived from the July 2026 manual review of six
# accepted Iba1 benchmark regions. This is a starting value for user review,
# not a universally validated optimum.
NLM_BASELINE_SIGMA_FACTOR = 0.7
NLM_MIN_H = 2
NLM_MAX_H = 20
NLM_REVIEW_HALF_WIDTH = 6

# Calibration of the prepared S-BIAD1280 Iba1 example images. Uploaded images
# must still be checked individually by the user.
EXAMPLE_PIXEL_SIZE_UM_PER_PX = 0.454546
UPLOAD_FALLBACK_PIXEL_SIZE_UM_PER_PX = 0.56
DEFAULT_SHOLL_STEP_PX = 4


def nlm_baseline_from_sigma(sigma_estimate: float) -> int:
    """Return the dataset-calibrated NLM starting value for a finite sigma."""
    sigma = float(sigma_estimate)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Sigma estimate must be finite and non-negative.")
    baseline = int(NLM_BASELINE_SIGMA_FACTOR * sigma)
    return min(max(baseline, NLM_MIN_H), NLM_MAX_H)


def nlm_review_range(baseline_h: int) -> list[int]:
    """Return the inclusive manual-review range around a baseline value."""
    baseline = int(baseline_h)
    minimum = max(1, baseline - NLM_REVIEW_HALF_WIDTH)
    maximum = baseline + NLM_REVIEW_HALF_WIDTH
    return list(range(minimum, maximum + 1))

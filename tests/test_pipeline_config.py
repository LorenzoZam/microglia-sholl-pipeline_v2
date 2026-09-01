import math

import pytest

from pipeline_config import nlm_baseline_from_sigma, nlm_review_range


@pytest.mark.parametrize(
    ("sigma", "expected"),
    [(0.0, 2), (10.0, 7), (20.0, 14), (100.0, 20)],
)
def test_dataset_calibrated_nlm_baseline(sigma, expected):
    assert nlm_baseline_from_sigma(sigma) == expected


@pytest.mark.parametrize("sigma", [math.nan, math.inf, -math.inf, -1.0])
def test_nlm_baseline_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="finite and non-negative"):
        nlm_baseline_from_sigma(sigma)


def test_nlm_review_range_is_inclusive_and_lower_bounded():
    assert nlm_review_range(7) == list(range(1, 14))

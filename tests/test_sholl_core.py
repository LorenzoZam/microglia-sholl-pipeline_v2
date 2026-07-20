import numpy as np
import pytest

from run_sholl_pipeline import (
    apply_adaptive_patching,
    compute_sholl_intersections,
    find_endpoints,
)


def test_empty_skeleton_has_no_intersections():
    skeleton = np.zeros((21, 21), dtype=np.uint8)
    assert compute_sholl_intersections(skeleton, 10, 10, [2, 5]).tolist() == [0, 0]


def test_single_radial_branch_crosses_each_circle_once():
    skeleton = np.zeros((21, 21), dtype=np.uint8)
    skeleton[10, 10:19] = 255
    assert compute_sholl_intersections(skeleton, 10, 10, [2, 4, 7]).tolist() == [1, 1, 1]


def test_two_opposed_branches_cross_twice():
    skeleton = np.zeros((21, 21), dtype=np.uint8)
    skeleton[10, 2:19] = 255
    assert compute_sholl_intersections(skeleton, 10, 10, [2, 5]).tolist() == [2, 2]


def test_tangent_does_not_count_as_crossing():
    skeleton = np.zeros((21, 21), dtype=np.uint8)
    skeleton[5, 8:13] = 255
    assert compute_sholl_intersections(skeleton, 10, 10, [5]).tolist() == [0]


def test_border_endpoints_are_detected():
    skeleton = np.zeros((5, 5), dtype=np.uint8)
    skeleton[0, :3] = 255
    assert set(find_endpoints(skeleton)) == {(0, 0), (2, 0)}


def test_small_image_is_supported_by_adaptive_patching():
    image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    result = apply_adaptive_patching(image, np.var(image))
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_too_small_image_has_clear_error():
    with pytest.raises(ValueError, match="at least 8"):
        apply_adaptive_patching(np.zeros((7, 8), dtype=np.uint8), 0)

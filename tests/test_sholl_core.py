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


def test_single_branch_is_rotationally_consistent():
    center = (10, 10)
    radii = [2, 4, 7]

    horizontal = np.zeros((25, 25), dtype=np.uint8)
    horizontal[center[1], center[0]:center[0] + 9] = 255

    vertical = np.zeros((25, 25), dtype=np.uint8)
    vertical[center[1]:center[1] + 9, center[0]] = 255

    diagonal = np.zeros((25, 25), dtype=np.uint8)
    for offset in range(9):
        diagonal[center[1] + offset, center[0] + offset] = 255

    expected = [1, 1, 1]
    assert compute_sholl_intersections(horizontal, *center, radii).tolist() == expected
    assert compute_sholl_intersections(vertical, *center, radii).tolist() == expected
    assert compute_sholl_intersections(diagonal, *center, radii).tolist() == expected


def test_four_unequal_arms_have_manual_sholl_profile():
    center_x = center_y = 15
    skeleton = np.zeros((31, 31), dtype=np.uint8)

    # Arm lengths from the center: right=8, left=6, up=4, down=2 pixels.
    skeleton[center_y, center_x:center_x + 9] = 255
    skeleton[center_y, center_x - 6:center_x + 1] = 255
    skeleton[center_y - 4:center_y + 1, center_x] = 255
    skeleton[center_y:center_y + 3, center_x] = 255

    # At radii 2, 4, 6, and 8 respectively, 4, 3, 2, and 1 arms reach
    # the circle. These counts are obtained directly from the arm lengths.
    observed = compute_sholl_intersections(
        skeleton, center_x, center_y, [2, 4, 6, 8]
    )
    assert observed.tolist() == [4, 3, 2, 1]


def test_u_shaped_branch_crosses_one_circle_twice():
    center_x = center_y = 15
    skeleton = np.zeros((31, 31), dtype=np.uint8)

    # One continuous path starts at the soma, exits radius 5 along the top
    # segment, loops downward, and re-enters radius 5 along the bottom segment.
    skeleton[center_y, center_x:center_x + 8] = 255
    skeleton[center_y:center_y + 4, center_x + 7] = 255
    skeleton[center_y + 3, center_x:center_x + 8] = 255

    assert compute_sholl_intersections(
        skeleton, center_x, center_y, [5]
    ).tolist() == [2]


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

import numpy as np

from morphology_features import (
    box_counting_fractal_dimension,
    box_counting_lacunarity,
    compute_graph_centralities,
    schoenen_ramification_index,
    skeleton_to_graph,
    soma_shape_metrics,
)


def test_line_has_fractal_dimension_near_one():
    mask = np.zeros((128, 128), dtype=bool)
    mask[64, 8:120] = True
    assert 0.85 <= box_counting_fractal_dimension(mask) <= 1.15


def test_empty_features_return_nan():
    empty = np.zeros((16, 16), dtype=bool)
    assert np.isnan(box_counting_fractal_dimension(empty))
    assert np.isnan(box_counting_lacunarity(empty))


def test_simple_graph_has_finite_centralities():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 1:8] = True
    graph = skeleton_to_graph(mask)
    betweenness, closeness = compute_graph_centralities(graph)
    assert np.isfinite(betweenness)
    assert np.isfinite(closeness)


def test_sampled_ramification_index_uses_first_radius():
    result = schoenen_ramification_index(
        intersections=[2, 6, 3], radii=[4, 8, 12], soma_point=(5, 5),
        skeleton_mask=np.zeros((11, 11), dtype=bool),
    )
    assert result == 3.0


def test_soma_circle_metrics_are_bounded():
    yy, xx = np.indices((64, 64))
    binary = (((xx - 32) ** 2 + (yy - 32) ** 2) <= 10 ** 2).astype(np.uint8) * 255
    area, circularity = soma_shape_metrics(binary, (32, 32))
    assert area > 0
    assert 0 <= circularity <= 1

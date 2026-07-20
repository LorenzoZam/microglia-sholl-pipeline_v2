import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from plot_sholl_profiles import (
    _mixed_model_variance_summary,
    _validate_mixed_model_input,
    add_cell_id,
    load_and_process_data,
)


def _valid_mixed_model_frame():
    rows = []
    for group, animal, cell in (
        ("Control", "A1", "A1/C1"),
        ("Treatment", "A2", "A2/C1"),
    ):
        for radius in np.arange(1.0, 6.0):
            rows.append({
                "Group": group,
                "Animal_ID": animal,
                "Cell_ID": cell,
                "Radius_um": radius,
                "Intersections": 1.0,
            })
    return pd.DataFrame(rows)


def test_cell_id_distinguishes_images():
    frame = pd.DataFrame({
        "Image_Name": ["image_a", "image_b"],
        "Soma_ID": [1, 1],
    })
    assert add_cell_id(frame)["Cell_ID"].nunique() == 2


def test_cell_id_is_stable_across_radii_and_distinct_across_cells():
    frame = pd.DataFrame({
        "Image_Name": ["image_a", "image_a", "image_a", "image_b"],
        "Soma_ID": [1, 1, 2, 1],
        "Radius": [4, 8, 4, 4],
    })

    identified = add_cell_id(frame)
    assert identified.loc[0, "Cell_ID"] == identified.loc[1, "Cell_ID"]
    assert identified.loc[0, "Cell_ID"] != identified.loc[2, "Cell_ID"]
    assert identified.loc[0, "Cell_ID"] != identified.loc[3, "Cell_ID"]
    assert identified["Cell_ID"].nunique() == 3


def test_summary_retains_zero_intersections(tmp_path):
    path = tmp_path / "profiles.csv"
    pd.DataFrame({
        "Soma_ID": [1, 2, 1, 2],
        "Radius": [4, 4, 8, 8],
        "Intersections": [2, 2, 0, 2],
    }).to_csv(path, index=False)

    radii, mean, sem = load_and_process_data(path)
    assert radii.tolist() == [4 * 0.56, 8 * 0.56]
    assert mean.tolist() == [2.0, 1.0]
    assert np.isclose(sem.iloc[1], 1.0)


def test_complete_three_cell_profiles_retain_zeroes_in_mean_and_sem(tmp_path):
    path = tmp_path / "complete_profiles.csv"
    pd.DataFrame({
        "Soma_ID": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "Radius": [4, 4, 4, 8, 8, 8, 12, 12, 12],
        "Intersections": [4, 2, 0, 2, 0, 0, 0, 0, 0],
    }).to_csv(path, index=False)

    radii, mean, sem = load_and_process_data(path)

    assert np.allclose(radii, np.array([4, 8, 12]) * 0.56)
    assert np.allclose(mean, [2.0, 2.0 / 3.0, 0.0])
    assert np.allclose(sem, [2.0 / np.sqrt(3.0), 2.0 / 3.0, 0.0])


def test_mixed_model_variance_summary_includes_all_components():
    result = SimpleNamespace(
        cov_re=pd.DataFrame([[4.0]]),
        vcomp=np.array([9.0]),
        scale=16.0,
    )

    summary = _mixed_model_variance_summary(result)

    assert summary["animal_variance"] == 4.0
    assert summary["cell_variance"] == 9.0
    assert summary["residual_variance"] == 16.0
    assert summary["total_variance"] == 29.0
    assert np.isclose(summary["animal_level_icc"], 4.0 / 29.0)
    assert np.isclose(summary["within_cell_correlation"], 13.0 / 29.0)


def test_mixed_model_rejects_animal_in_multiple_groups():
    frame = _valid_mixed_model_frame()
    frame.loc[frame.index[-1], "Animal_ID"] = "A1"
    with pytest.raises(ValueError, match="Animal_ID must belong to exactly one Group"):
        _validate_mixed_model_input(frame, spline_df=5)


def test_mixed_model_rejects_cell_in_multiple_animals():
    frame = _valid_mixed_model_frame()
    frame.loc[frame["Animal_ID"] == "A2", "Cell_ID"] = "A1/C1"
    with pytest.raises(ValueError, match="Cell_ID must belong to exactly one Animal_ID"):
        _validate_mixed_model_input(frame, spline_df=5)


def test_mixed_model_rejects_negative_intersections():
    frame = _valid_mixed_model_frame()
    frame.loc[0, "Intersections"] = -1
    with pytest.raises(
        ValueError, match="Intersections must be finite and non-negative"
    ):
        _validate_mixed_model_input(frame, spline_df=5)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_mixed_model_rejects_non_finite_intersections(invalid_value):
    frame = _valid_mixed_model_frame()
    frame.loc[0, "Intersections"] = invalid_value
    with pytest.raises(
        ValueError, match="Intersections must be finite and non-negative"
    ):
        _validate_mixed_model_input(frame, spline_df=5)


def test_mixed_model_rejects_duplicate_cell_radius_rows():
    frame = _valid_mixed_model_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate Cell_ID x Radius_um"):
        _validate_mixed_model_input(frame, spline_df=5)

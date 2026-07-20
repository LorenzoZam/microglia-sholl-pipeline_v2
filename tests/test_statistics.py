import numpy as np
import pandas as pd

from plot_sholl_profiles import add_cell_id, load_and_process_data


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

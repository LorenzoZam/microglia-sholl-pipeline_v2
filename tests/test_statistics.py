import numpy as np
import pandas as pd

from plot_sholl_profiles import add_cell_id, load_and_process_data


def test_cell_id_distinguishes_images():
    frame = pd.DataFrame({
        "Image_Name": ["image_a", "image_b"],
        "Soma_ID": [1, 1],
    })
    assert add_cell_id(frame)["Cell_ID"].nunique() == 2


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

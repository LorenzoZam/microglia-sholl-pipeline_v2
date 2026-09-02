# Research utilities

These scripts support dataset preparation and exploratory parameter review.
They are separate from the production Streamlit and desktop entry points.

Run them from the repository root as modules:

```powershell
python -m tools.prepare_iba1_dataset --help
python -m tools.compare_sholl_steps --help
python -m tools.calibrate_nlm_h --help
python -m tools.compare_clahe_methods --help
```

- `prepare_iba1_dataset.py` extracts the documented Iba1 channel and records
  calibration and hashes.
- `compare_sholl_steps.py` compares 2, 5 and 10 µm Sholl sampling intervals on
  one fixed skeleton and soma.
- `calibrate_nlm_h.py` provides blinded, ROI-based visual review of candidate
  NLM strengths. Its results are exploratory and dataset-specific.
- `compare_clahe_methods.py` runs a method-label-hidden, side-balanced
  comparison between single-setting whole-image CLAHE and the production
  adaptive patch-wise method. It records qualitative preferences and matched
  downstream skeleton views; it does not establish accuracy without
  independent reference traces.

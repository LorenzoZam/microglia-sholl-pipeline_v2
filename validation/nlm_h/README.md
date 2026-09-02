# Dataset-specific NLM starting-value calibration

This directory records the exploratory manual review used to choose the NLM
starting-value heuristic in MicroSholl. It does not establish external or
biological validation.

## Source and design

The reviewed inputs were single-channel Iba1 images prepared from BioImage
Archive study
[S-BIAD1280](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1280).
Noise was estimated after adaptive CLAHE, matching the production order.

The first review used `current baseline ± 6`. Seventeen images were reviewed:
16 received a selection and one was excluded. Twelve selections reached the
lower boundary, so that design did not locate the preferred value reliably.

The second review used a fixed `h=5–17` range and an image-specific benchmark
ROI shared by every candidate. Seven images were reviewed: six received a
selection and one was excluded. None of the six selections reached a range
boundary. All second-review images were from `Exposed CRBLM`.

## Implemented starting value

The second review produced a median selected-`h`/sigma ratio of approximately
0.677. The implemented starting value deliberately rounds this to one decimal
place rather than implying unsupported precision:

```text
starting h = clamp(int(0.7 × sigma), 2, 20)
manual review range = starting h ± 6 (lower-bounded at h=1)
```

This is a **dataset-specific heuristic starting value**. Users must inspect the
resulting skeleton and may adjust `h`; it should not be described as optimal or
generalized beyond comparable images without additional validation.

## Files

- `first_test_selections.csv`, `first_test_summary.json` and
  `first_test_summary.png`: initial boundary-limited review.
- `second_test_selections.csv`, `second_test_summary.json` and
  `second_test_summary.png`: fixed-range ROI review used for the implemented
  heuristic.

Full-resolution QC overlays and source microscopy files are retained locally
outside version control.

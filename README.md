# 🧠 MicroSholl — Advanced Microglia Morphology Analysis Pipeline

> **From raw confocal image to publication-ready morphometric data in minutes.**  
> A Python pipeline for robust, reproducible Sholl analysis of microglial cells, with interactive quality control and batch processing.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why MicroSholl](#why-microsholl)
3. [What's New in v2.0](#whats-new-in-v20)
4. [Pipeline Overview](#pipeline-overview)
5. [Feature Deep-Dives](#feature-deep-dives)
   - [Patch-wise Adaptive Preprocessing](#a-patch-wise-adaptive-preprocessing)
   - [Automated Denoising with Interactive Validation](#b-automated-denoising-with-interactive-validation)
   - [Soma Selection](#c-soma-selection)
   - [Sholl Analysis & Morphometrics](#d-sholl-analysis--morphometrics)
   - [Interactive QC Dashboard](#e-interactive-qc-dashboard)
   - [Batch Processing & Master CSV](#f-batch-processing--master-csv)
   - [Statistical Post-processing](#g-statistical-post-processing)
6. [Output Data Format](#output-data-format)
7. [Repository Structure](#repository-structure)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Replicability & Scientific Rationale](#replicability--scientific-rationale)
11. [Technologies Used](#technologies-used)
12. [Citation](#citation)

---

## Introduction

Microglial morphology is a direct, quantifiable readout of neuroinflammatory state. Ramified microglia with long, branched processes are characteristic of a surveillant, homeostatic phenotype, while amoeboid, process-retracted cells signal activation. Sholl analysis — counting intersections between concentric circles and a skeletonized cell arbor — is the gold-standard method for quantifying this complexity.

However, standard implementations (e.g., ImageJ/Fiji plugins) suffer from:
- **Manual, time-consuming workflows** that bottleneck large studies
- **Sensitivity to image quality**, causing fragmented skeletons and erroneous counts
- **No built-in quality control**, leading to undetected analysis errors
- **No traceability**, making it difficult to reproduce or audit results

**MicroSholl** is a semi-automated Python pipeline that solves all of these problems. It combines rigorous, publication-grade image processing with an interactive GUI, enabling neuroscientists to analyze entire experiments in a single session with full traceability.

---

## Why MicroSholl

| Feature | ImageJ Plugin | MicroSholl |
|---|:---:|:---:|
| Automated skeleton extraction | ✅ | ✅ |
| Adaptive noise reduction | ❌ | ✅ |
| Automated denoising parameter estimation | ❌ | ✅ |
| Multi-cell batch analysis | ❌ | ✅ |
| Interactive QC per cell | ❌ | ✅ |
| Fractal Dimension & graph metrics | ❌ | ✅ |
| Master CSV across image files | ❌ | ✅ |
| Full traceability (per-cell images) | ❌ | ✅ |

---

## What's New in v2.0

This release represents a major upgrade over the original pipeline, adding several key features:

### 🎛️ Automated Denoising UI (Zero-Lag Slider)
The original pipeline required manually selecting an `h` denoising value from a static grid. v2.0 replaces this with:
- **Mathematical noise estimation** using `skimage.restoration.estimate_sigma` (wavelet-based) to automatically compute an optimal starting `h` value
- **Pre-computed skeleton previews** for a range of ±4 values around the optimum, so the slider updates instantaneously with zero computational lag
- The user validates or adjusts the value with full visual feedback before committing

### 📊 Interactive QC Dashboard
Each cell now receives a full 6-panel quality control dashboard before its data is committed to the CSV:
- Original image with Sholl circles overlay
- Binary segmentation
- Skeleton graph (colored by betweenness centrality)
- Fractal Dimension log-log regression
- Per-cell Sholl profile curve
- Metric summary card with biological reference ranges
- GUI buttons: **[Accept]**, **[Reject]**, **[Accept All]** — replaces error-prone terminal input

### 🔬 Extended Morphometric Feature Extraction
Beyond raw Sholl intersections, v2.0 now computes and exports:
- **Fractal Dimension** (box-counting method)
- **Lacunarity** (morphological complexity)
- **Betweenness & Closeness Centrality** (graph-theory metrics on skeleton graph)
- **Schoenen Ramification Index** (ratio of maximum to primary branches)
- **Soma Area & Circularity** (shape descriptors)

### 📁 Batch Processing & Master CSV
The pipeline now accepts **multiple image files in a single session**:
- Select 1 to N images via a multi-file dialog (hold Ctrl/Shift)
- Each image is processed sequentially with its own denoising UI, soma selection, and QC review
- A global `MASTER_Sholl_Metrics_Batch.csv` is automatically generated at the end, with an `Image_Name` column for full traceability

---

## Pipeline Overview

```
  ┌─────────────────────────────────────────────────────────────┐
  │  USER selects 1–N image files via multi-file dialog         │
  └───────────────────────┬─────────────────────────────────────┘
                          │  for each image:
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  1. Patch-wise Adaptive CLAHE                               │
  │     → Divides image into local patches                      │
  │     → Applies CLAHE independently per patch                 │
  │     → Preserves fine structures in heterogeneous backgrounds│
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  2. Automated Denoising (NL-Means) + Interactive Slider UI  │
  │     → estimate_sigma() for mathematically optimal h value   │
  │     → Pre-computed skeleton range shown to user in real-time│
  │     → User confirms or fine-tunes before proceeding         │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  3. Morphological Binarization & Skeletonization            │
  │     → Top-hat transform → Otsu binarization                 │
  │     → Fragment removal → Morphological closing/dilation     │
  │     → Zhang-Suen skeletonization                            │
  │     → Gap bridging → Isolated fiber removal (2-pass)        │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  4. Interactive Soma Selection                              │
  │     → User clicks soma positions on the processed image     │
  │     → Skeleton displayed as reference on the right          │
  │     → Coordinates snapped to nearest skeleton point         │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  5. Per-Cell Connected Component Isolation                  │
  │     → 8-connectivity labeling isolates each cell's arbor    │
  │     → Prevents cross-cell contamination in dense images     │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  6. Sholl Analysis + Full Morphometrics Extraction          │
  │     → Concentric circles at step_size = 4 px               │
  │     → Intersection counting at each radius                  │
  │     → Fractal Dimension, Lacunarity, Graph centralities,    │
  │        Ramification Index, Soma shape                       │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  7. Interactive QC Dashboard (per cell)                     │
  │     → 6-panel visual review                                 │
  │     → [Accept] / [Reject] / [Accept All] GUI buttons        │
  │     → Rejected cells flagged in output folder               │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  8. CSV Export                                              │
  │     → Per-image CSV (sholl_intersections_<name>.csv)        │
  │     → Global MASTER_Sholl_Metrics_Batch.csv                 │
  │        (with Image_Name column for cross-image traceability)│
  └─────────────────────────────────────────────────────────────┘
```

---

## Feature Deep-Dives

### (A) Patch-wise Adaptive Preprocessing

Classical global CLAHE amplifies both signal and background uniformly, corrupting skeletonization in heterogeneous tissue. The pipeline divides the input image into overlapping patches and applies CLAHE locally, preserving fine terminal dendrites while suppressing background.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="data/M_02_global.png" width="480"><br>
      <em><strong>Global CLAHE.</strong> Background structures are amplified, causing fragmented skeletons and spurious bifurcations.</em>
    </td>
    <td align="center" width="50%">
      <img src="data/M_02_patchwise.png" width="480"><br>
      <em><strong>Patch-wise CLAHE.</strong> Local enhancement preserves fine terminal branches while suppressing background noise.</em>
    </td>
  </tr>
</table>

---

### (B) Automated Denoising with Interactive Validation


<p align="center">
  <img src="data/UIgif.gif" width="1100"><br>
  <em><strong>Denoising UI.</strong> The optimal <code>h</code> value is estimated mathematically (wavelet sigma estimation) and pre-computed for a ±4 range. The slider updates the skeleton preview instantaneously — no computation lag. The user validates the starting estimate or fine-tunes it before analysis begins.</em>
</p>

**Scientific rationale:** Non-local Means denoising strength (`h`) critically determines whether fine dendritic processes are preserved or eliminated. A value that is too low leaves noise fragments that corrupt the skeleton; a value that is too high removes real biology. The automated estimation provides a principled, reproducible starting point (Buades et al., 2005).

---

### (C) Soma Selection

<p align="center">
  <img src="data/soma_sel_UI.png" width="1100"><br>
  <em><strong>Soma selection.</strong> The processed image (left) is displayed alongside the final skeleton (right) as a reference. Users click directly on soma cell bodies to define analysis anchors. Coordinates are automatically snapped to the nearest skeleton point to ensure accurate centering.</em>
</p>

---

### (D) Sholl Analysis & Morphometrics

<p align="center">
  <img src="data/sholl_UI.png" width="520"><br>
  <em><strong>Sholl circle placement.</strong> Concentric circles are generated around each selected soma, extending to the farthest detected endpoint of the connected component.</em>
</p>

For each cell, the following metrics are extracted:

| Metric | Description | Reference |
|---|---|---|
| `Radius` | Distance from the soma center (px) | Sholl, 1953 |
| `Intersections` | Number of skeleton crossings at each radius | Sholl, 1953 |
| `Fractal_Dimension` | Box-counting fractal dimension of the arbor | Mandelbrot, 1982 |
| `Lacunarity` | Morphological heterogeneity of the arbor | Plotnick et al., 1996 |
| `Betweenness_Centrality` | Fraction of shortest paths through each node | Freeman, 1977 |
| `Closeness_Centrality` | Mean inverse distance to all other nodes | Bavelas, 1950 |
| `Ramification_Index` | Ratio max/primary branches (Schoenen RI) | Schoenen, 1982 |
| `Soma_Area` | Area of the detected soma region (px²) | — |
| `Soma_Circularity` | Shape circularity index [0–1] | — |

---

### (E) Interactive QC Dashboard

> **UI Screenshot Slot — insert `data/qc_dashboard_example.png`**

```
[ QC Dashboard Screenshot here ]
```
<p align="center">
  <img src="data/Screen_QC1.png" width="1100"><br>
</p>

Each cell presents a 6-panel dashboard before its data is committed:

- **Panel A** — Original image with Sholl circles and soma marker
- **Panel B** — Binary segmentation mask
- **Panel C** — Skeleton graph colored by betweenness centrality
- **Panel D** — Fractal Dimension log-log regression (box-counting)
- **Panel E** — Per-cell Sholl intersection profile
- **Panel F** — Metric summary card with biological reference ranges

The user responds via GUI buttons:
- **[Accept]** — include this cell in the output CSV
- **[Reject]** — exclude this cell; backtrace image is renamed `REJECTED_*` for audit
- **[Accept All]** — accept this and all remaining cells without further review

Rejected cells are **never silently dropped**: they are flagged and saved separately, enabling post-hoc audit.

---

### (F) Batch Processing & Master CSV

> **UI Screenshot Slot — insert `data/batch_dialog.png`**

```
[ Batch file selection dialog Screenshot here ]
```

The pipeline accepts **one or more image files** simultaneously:

```
# One image
file_paths = ("path/to/M05_Iba1.tif",)

# Multiple images (hold Ctrl/Shift in the dialog)
file_paths = ("path/to/M05_Iba1.tif", "path/to/F02_Iba1.tif", ...)
```

Each image is processed sequentially. At the end of the session, a **global master CSV** is written:

```
MASTER_Sholl_Metrics_Batch.csv
  Image_Name | Soma_ID | Radius | Intersections | Fractal_Dimension | ...
  ────────────────────────────────────────────────────────────────────────
  M05_Iba1   |    1    |    4   |       8       |      1.42         | ...
  M05_Iba1   |    1    |    8   |       5       |      1.42         | ...
  F02_Iba1   |    1    |    4   |      11       |      1.38         | ...
  F02_Iba1   |    2    |    4   |       7       |      1.51         | ...
```

The `Image_Name` column provides full traceability: every row can be linked back to its source image and soma.

---

### (G) Statistical Post-processing

Population-level Sholl curves and statistics are computed in `plot_sholl_profiles.py`, which reads master CSVs and produces:
- Mean Sholl intersection curves with SEM across cells
- Outlier inspection and removal
- Condition-level comparisons

<p align="center">
  <img src="data/Sham CB.png" width="800"><br>
  <em><strong>Population-level Sholl curves.</strong> Mean ± SEM intersection profiles across all accepted cells in the Sham condition.</em>
</p>

---

## Output Data Format

Each analysis run produces the following outputs in a dedicated folder (`<image_name>_sholl_output/`):

```
F02_sholl_output/
  ├── F02_gray_scale_image.png         ← Original image copy
  ├── F02_processed_image.png          ← After patch-wise CLAHE
  ├── F02_den_th_image.png             ← After denoising + top-hat
  ├── F02_binary.png                   ← Binary mask
  ├── F02_skeleton.png                 ← Final skeleton
  ├── F02_soma_and_radii.png           ← Sholl circles overlay
  ├── F02_backtrace_soma_1.png         ← Per-cell QC backtrace
  ├── F02_REJECTED_backtrace_soma_2.png   ← Flagged rejected cell
  ├── F02_qc_dashboard_soma_1.png      ← Per-cell QC dashboard
  └── sholl_intersections_F02.csv      ← Per-image morphometric data

MASTER_Sholl_Metrics_Batch.csv         ← Aggregated data from all images
```

---

## Repository Structure

```text
microglia-sholl-pipeline/
│
├── data/                       # Example input images and demo outputs
│
├── run_sholl_pipeline.py       # ★ Main pipeline:
│                               #   preprocessing → skeleton → soma selection
│                               #   → Sholl analysis → QC → CSV export
│                               #   Supports single and batch processing
│
├── morphology_features.py      # Morphometric feature extraction:
│                               #   fractal dimension, lacunarity,
│                               #   graph centralities, ramification index,
│                               #   soma shape, QC dashboard generation
│
├── plot_sholl_profiles.py      # Statistical post-processing and visualization:
│                               #   population-level curves, SEM, outlier removal
│
├── merge_sholl_results.py      # Utility to merge per-image CSVs
│
├── test_morphology_features.py # Unit tests for morphometric functions
│
├── README.md                   # This file
├── LICENSE                     # MIT License
└── requirements.txt            # Python dependencies
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/LorenzoZam/microglia-sholl-pipeline.git
cd microglia-sholl-pipeline

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Python ≥ 3.9 recommended.**

---

## Usage

### Single or Batch Analysis

```bash
python run_sholl_pipeline.py
```

A file dialog will open. Select **one or more** image files (`.tif`, `.tiff`, `.png`, `.jpg`, `.bmp`).

**Step by step:**
1. **Denoising UI** — review the pre-computed skeleton previews, adjust the slider if needed, confirm
2. **Soma selection** — click on each cell body in the processed image, then **close the window**
3. **QC Dashboard** — review each cell and click **[Accept]**, **[Reject]**, or **[Accept All]**
4. Repeat for each selected image
5. Find `MASTER_Sholl_Metrics_Batch.csv` in the same folder as your images

### Statistical visualization

```bash
python plot_sholl_profiles.py
```

Reads the master CSV and produces population-level curves.

---

## Replicability & Scientific Rationale

MicroSholl is designed with reproducibility as a core constraint — not an afterthought.

| Principle | Implementation |
|---|---|
| **No undocumented parameters** | All critical parameters (`h`, step size, minimum fragment size) are saved in the terminal log |
| **Automated, standardized denoising** | `estimate_sigma` provides a noise-level-informed starting point independent of user intuition |
| **Full audit trail** | Every intermediate image is saved; rejected cells are flagged rather than silently dropped |
| **Per-cell traceability** | Each CSV row can be traced to a specific soma in a specific image via `Image_Name` + `Soma_ID` |
| **Open dependencies** | All dependencies are open-source, versioned, and well-maintained scientific Python libraries |
| **Unit-tested morphometrics** | `test_morphology_features.py` provides automated tests for fractal dimension, lacunarity, and graph metrics |

This design means that any result produced by MicroSholl can be **fully reproduced** given the original images and parameter choices.

---

## Technologies Used

| Library | Role |
|---|---|
| [Python ≥ 3.9](https://python.org) | Core language |
| [NumPy](https://numpy.org) | Numerical computing |
| [Pandas](https://pandas.pydata.org) | Data handling and CSV export |
| [OpenCV](https://opencv.org) | Image I/O, CLAHE, morphological operations |
| [scikit-image](https://scikit-image.org) | Skeletonization, connected components, denoising estimation |
| [SciPy](https://scipy.org) | Spatial distance computations |
| [NetworkX](https://networkx.org) | Skeleton-to-graph conversion and centrality metrics |
| [Matplotlib](https://matplotlib.org) | Interactive UI, QC dashboards, Sholl plots |
| [tkinter](https://docs.python.org/3/library/tkinter.html) | File selection dialogs (stdlib) |

All dependencies and tested versions are listed in `requirements.txt`.

---

## Example image attribution

Example images are derived from the following publicly available dataset:

**Effects of PCB52 (2,2',5,5'-Tetrachlorobiphenyl) on the Rat Brain After Subacute Nose-Only Inhalation Exposure**  
BioImage Archive accession: **S-BIAD1280**  
DOI: **10.6019/S-BIAD1280**

Images are used exclusively for methodological demonstration and were not modified in a way that alters the original biological content.

---

## Citation

If you use this pipeline in your research, presentations, or publications, please cite:

```
Zammariello, L. (2025). MicroSholl: Advanced Microglia Morphology Analysis Pipeline.
GitHub: https://github.com/LorenzoZam/microglia-sholl-pipeline
```

> This software is provided "as is", without warranty of any kind. Contributions and forks are welcome under the MIT License.

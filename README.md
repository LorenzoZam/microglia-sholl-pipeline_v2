# 🧠 MicroSholl — Microglia Morphometrics and Batch Sholl Analysis

> A Python research pipeline for exploratory 2-D Sholl analysis of microglial
> images, with interactive quality control and batch processing.

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

Microglial morphology is a useful quantitative correlate of cellular state, but
it is not a direct measurement of inflammatory function. Ramification and soma
shape should be interpreted in the context of species, brain region, staining,
imaging, and complementary molecular or functional measurements. Sholl analysis
counts crossings between concentric circles and a reconstructed cell arbor and
is one established way to describe spatial complexity.

Common challenges in Sholl-analysis workflows include:
- **Manual, time-consuming workflows** that bottleneck large studies
- **Sensitivity to image quality**, causing fragmented skeletons and erroneous counts
- **No built-in quality control**, leading to undetected analysis errors
- **No traceability**, making it difficult to reproduce or audit results

**MicroSholl** is a semi-automated Python pipeline intended to make this workflow
inspectable and easier to repeat. Its outputs require visual QC and experimental
validation before use in inferential or publication workflows.

---

## Why MicroSholl

| Feature | MicroSholl |
|---|:---:|
| Automated skeleton extraction | ✅ |
| Adaptive noise reduction | ✅ |
| User-reviewed denoising | ✅ |
| Multi-cell batch analysis | ✅ |
| Interactive QC per cell | ✅ |
| Fractal dimension and graph descriptors | ✅ |
| Master CSV across image files | ✅ |
| Per-cell images and parameter metadata | ✅ |

---

## What's New in v2.0

This release represents a major upgrade over the original pipeline, adding several key features:

### 🎛️ Automated Denoising UI (Zero-Lag Slider)
The original pipeline required manually selecting an `h` denoising value from a static grid. v2.0 replaces this with:
- **Wavelet-based noise estimation** using `skimage.restoration.estimate_sigma` to compute a heuristic starting `h` value
- **Pre-computed skeleton previews** for a range of ±4 values around that starting value
- The user validates or adjusts the value with full visual feedback before committing

### 📊 Interactive QC Dashboard
Each cell now receives a full 6-panel quality control dashboard before its data is committed to the CSV:
- Original image with Sholl circles overlay
- Binary segmentation
- Skeleton graph (colored by betweenness centrality)
- Fractal Dimension log-log regression
- Per-cell Sholl profile curve
- Metric summary card with configurable QC ranges (not biological norms)
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
- A global `MASTER_Sholl_Metrics_Batch.csv` is automatically generated at the end, with an `Image_Name` column for source-image identification

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
  │     → Applies local enhancement in heterogeneous backgrounds │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  2. Automated Denoising (NL-Means) + Interactive Slider UI  │
  │     → estimate_sigma() for a heuristic starting h value     │
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
  │     → Intended to limit cross-cell mixing in dense images    │
  └───────────────────────┬─────────────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  6. Sholl Analysis + Morphometric Descriptor Extraction     │
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

Global CLAHE can amplify both signal and background in heterogeneous tissue. The pipeline instead divides the input image into overlapping patches and applies CLAHE locally; its effect on segmentation and skeletonization must be validated for each dataset.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="data/M_02_global.png" width="480"><br>
      <em><strong>Global CLAHE.</strong> Example output from whole-image enhancement.</em>
    </td>
    <td align="center" width="50%">
      <img src="data/M_02_patchwise.png" width="480"><br>
      <em><strong>Patch-wise CLAHE.</strong> Example output from local enhancement.</em>
    </td>
  </tr>
</table>

---

### (B) Automated Denoising with Interactive Validation


<p align="center">
  <img src="data/UIgif.gif" width="1100"><br>
  <em><strong>Denoising UI.</strong> A heuristic starting <code>h</code> value is derived from wavelet noise estimation, and previews are pre-computed for a ±4 range. The user reviews or adjusts the estimate before analysis begins.</em>
</p>

**Methodological note:** Non-local Means denoising strength (`h`) affects whether
fine processes and noise are retained. The desktop interface uses wavelet noise
estimation as a starting point; this mapping is heuristic and must be visually
validated. The Streamlit interface starts from a configurable default.

---

### (C) Soma Selection

<p align="center">
  <img src="data/soma_sel_UI.png" width="1100"><br>
  <em><strong>Soma selection.</strong> The processed image (left) is displayed alongside the final skeleton (right) as a reference. Users click directly on soma cell bodies to define analysis anchors. Coordinates are automatically snapped to the nearest skeleton point to provide a repeatable computational anchor; users must verify the selected location.</em>
</p>

---

### (D) Sholl Analysis & Morphometrics

<p align="center">
  <img src="data/sholl_UI.png" width="520"><br>
  <em><strong>Sholl circle placement.</strong> Concentric circles are generated around each selected soma, extending to the farthest detected endpoint of the connected component.</em>
</p>

For each cell, the following computationally defined metrics are extracted.
The Sholl intersection counts and ramification index are preliminary software
measurements, not validated biological ground truth. Their accuracy and
repeatability must be evaluated against manual annotations or an established
reference workflow on representative images before scientific use.

| Metric | Description | Reference |
|---|---|---|
| `Radius` | Distance from the soma center (px) | Sholl, 1953 |
| `Intersections` | Software-counted skeleton intersections at each radius | Sholl, 1953 |
| `Fractal_Dimension` | Box-counting fractal dimension of the arbor | Mandelbrot, 1982 |
| `Lacunarity` | Morphological heterogeneity of the arbor | Plotnick et al., 1996 |
| `Betweenness_Centrality` | Fraction of shortest paths through each node | Freeman, 1977 |
| `Closeness_Centrality` | Mean inverse distance to all other nodes | Bavelas, 1950 |
| `Ramification_Index` | Software-defined ratio of maximum intersections to the estimated primary-branch count | Schoenen, 1982 |
| `Soma_Area` | Area of the detected soma region (px²) | — |
| `Soma_Circularity` | Shape circularity index [0–1] | — |

---

### (E) Interactive QC Dashboard

<p align="center">
  <img src="data/Screen_QC1.png" width="1100"><br>
</p>

Each cell presents a 6-panel dashboard before its data is committed:

- **Panel A** — Original image with Sholl circles and soma marker
- **Panel B** — Binary segmentation mask
- **Panel C** — Skeleton graph colored by betweenness centrality
- **Panel D** — Fractal Dimension log-log regression (box-counting)
- **Panel E** — Per-cell Sholl intersection profile
- **Panel F** — Metric summary card with configurable QC ranges

The user responds via GUI buttons:
- **[Accept]** — include this cell in the output CSV
- **[Reject]** — exclude this cell; backtrace image is renamed `REJECTED_*` for audit
- **[Accept All]** — accept this and all remaining cells without further review

Rejected cells are **never silently dropped**: they are flagged and saved separately, enabling post-hoc audit.

---

### (F) Batch Processing & Master CSV


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

The `Image_Name` and `Soma_ID` columns identify the source image and selected soma for each row.

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
microglia-sholl-pipeline_v2/
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
├── tests/                      # Unit and regression tests
├── app.py                      # Streamlit interface
│
├── README.md                   # This file
├── LICENSE                     # MIT License
└── requirements.txt            # Python dependencies
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/LorenzoZam/microglia-sholl-pipeline_v2.git
cd microglia-sholl-pipeline_v2

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**CI is configured to test Python 3.10–3.12.**

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

### Streamlit app

```bash
streamlit run app.py
```

The web interface accepts TIFF, PNG, and JPEG uploads. Cloud storage is ephemeral;
download both CSV outputs before ending the session. The configured upload limit
is 200 MB per file. Do not upload sensitive human data to a public deployment
without an approved data-handling process.

---

## Replicability & Scientific Rationale

The repository includes the following mechanisms intended to support repeatable analysis:

| Principle | Implementation |
|---|---|
| **Reviewed denoising** | The desktop workflow asks the user to review the selected NLM setting |
| **Review artifacts** | The desktop workflow saves intermediate images and flags rejected cells |
| **Per-cell traceability** | Each CSV row can be traced to a specific soma in a specific image via `Image_Name` + `Soma_ID` |
| **Automated checks** | CI runs baseline scientific-function tests and a Streamlit startup check |

Exact reproduction can depend on the operating system and numerical-library
versions. Archived analyses should therefore preserve the inputs, outputs,
parameter records, and an exact environment snapshot.

### Scientific limitations

- The pipeline analyzes 2-D images; it does not reconstruct 3-D arbors.
- Segmentation and gap bridging are heuristic and require dataset-specific validation.
- Morphology is not synonymous with microglial inflammatory or functional state.
- The displayed QC ranges are software defaults, not validated biological norms.
- Cells are nested within images and animals. Treating cells as independent
  biological replicates is pseudoreplication; define the experimental unit first.
- Validate agreement and repeatability against manual annotations or an established
  reference workflow before using measurements as study endpoints.

---

## Technologies Used

| Library | Role |
|---|---|
| [Python 3.10–3.12](https://python.org) | CI-supported core language versions |
| [NumPy](https://numpy.org) | Numerical computing |
| [Pandas](https://pandas.pydata.org) | Data handling and CSV export |
| [OpenCV](https://opencv.org) | Image I/O, CLAHE, morphological operations |
| [scikit-image](https://scikit-image.org) | Skeletonization, connected components, denoising estimation |
| [SciPy](https://scipy.org) | Spatial distance computations |
| [NetworkX](https://networkx.org) | Skeleton-to-graph conversion and centrality metrics |
| [Matplotlib](https://matplotlib.org) | Interactive UI, QC dashboards, Sholl plots |
| [tkinter](https://docs.python.org/3/library/tkinter.html) | File selection dialogs (stdlib) |

Runtime dependencies are listed in `requirements.txt`. Development and test
dependencies are listed separately in `requirements-dev.txt`. Record exact
installed versions when archiving an analysis.

---

## Example image attribution

Example images are derived from the following publicly available dataset:

**Effects of PCB52 (2,2',5,5'-Tetrachlorobiphenyl) on the Rat Brain After Subacute Nose-Only Inhalation Exposure**  
BioImage Archive accession: **S-BIAD1280**  
DOI: **10.6019/S-BIAD1280**

Images are used exclusively for methodological demonstration and were not modified in a way that alters the original biological content.

---

## Citation

If you use this pipeline in your research, presentations, or publications, please cite the software as follows:

**APA:**
> Zammariello, L. (2026). MicroSholl: Microglia Morphometrics and Batch Sholl Analysis (v2.0.0). GitHub. https://github.com/LorenzoZam/microglia-sholl-pipeline_v2

**BibTeX:**
```bibtex
@software{Zammariello_MicroSholl_2026,
  author = {Zammariello, Lorenzo},
  title = {{MicroSholl: Microglia Morphometrics and Batch Sholl Analysis}},
  url = {https://github.com/LorenzoZam/microglia-sholl-pipeline_v2},
  version = {2.0.0},
  year = {2026}
}
```

> [!NOTE]
> You can also use the **"Cite this repository"** button on the GitHub sidebar to export the citation in your preferred format.

> This software is provided "as is", without warranty of any kind. Contributions and forks are welcome under the MIT License.

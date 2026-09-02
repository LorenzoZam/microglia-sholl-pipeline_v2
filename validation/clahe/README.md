# CLAHE comparison protocol

## Status

This directory documents an internal comparison of:

1. single-setting whole-image CLAHE; and
2. MicroSholl's adaptive patch-wise CLAHE strategy.

Two exploratory review passes have been completed with the four public Iba1
example images. The utility is
<a href="../../tools/compare_clahe_methods.py">tools/compare_clahe_methods.py</a>.

The comparison is a **method-label-hidden qualitative ablation**, not proof of
biological accuracy. The current Iba1 images contributed to development and
parameter review, so they are not an independent validation dataset.

## Why “whole-image CLAHE” rather than “global CLAHE”

OpenCV CLAHE already operates on an internal tile grid. Here, “single-setting
whole-image CLAHE” means that one <code>clipLimit</code> and one 8 × 8 tile grid
are applied to the complete image. The adaptive arm changes patch scale and
CLAHE strength from local statistics, overlaps patches, and combines them with
Gaussian weights.

This comparison evaluates that **composite adaptive strategy**. It cannot
identify whether a difference comes from patch scale, clip selection, overlap,
or blending without further ablations.

## Paired review design

For every configured source image, the script:

1. verifies a single-channel, 8-bit TIFF and records its SHA-256;
2. asks the reviewer to select one ROI on the raw image before seeing either
   candidate;
3. permits exclusion before A/B exposure and otherwise fixes the ROI;
4. uses the same ROI and 0–255 display limits for A and B;
5. applies one common NLM <code>h</code> and identical downstream processing;
6. presents images in a deterministic randomized order with alternating,
   reviewer-specific A/B allocation;
7. records Prefer A, Prefer B, Equivalent, Neither, or Exclude; and
8. saves only masked decisions and A/B panels until the configured batch is
   complete.

The default common <code>h</code> is the current production baseline estimated
after adaptive CLAHE and then held fixed for both arms. Thus, skeleton panels
are a secondary **integrated pipeline outcome**, not a measurement of CLAHE in
isolation. A preselected common value can instead be locked with
<code>--nlm-h</code>.

Completed masked rows are written after every image, so an interrupted session
can resume. The source manifest, code hashes, package versions, and settings
must match at resume.

The exact ROI and generated candidates are cached before review. This matters
because parallel adaptive-patch accumulation can otherwise differ by one
intensity level at a small number of pixels between executions. A resumed
image therefore reuses the exact arrays first displayed.

“Method-label-hidden” describes the interface, not cryptographic allocation
concealment. A reviewer who deliberately inspects the open-source allocation
code, seed, hashes, and configuration could reconstruct A/B identities.

## Exploratory results

Both passes used the same four public Iba1 source images and one reviewer. The
whole-image comparator used the image-level entropy rule implemented by the
tool, while the common downstream NLM strength was estimated from the adaptive
candidate and then held fixed within each pair.

| Review pass | Configured images | Non-excluded | Adaptive preferred | Whole-image preferred | Equivalent or neither |
|---|---:|---:|---:|---:|---:|
| Initial general-ROI pass | 4 | 3 | 0 | 3 | 0 |
| Targeted heterogeneous-background pass | 4 | 4 | 3 | 1 | 0 |

The targeted ROIs were selected on raw images and contained visible cellular
signal as well as spatially heterogeneous background. However, this second
pass was designed after the initial results were known, and background
heterogeneity was not defined by a prespecified quantitative threshold. The
two rows are not independent experiments, the images are not biological
replicates, and the review recorded a preference but not its reason. Without
manual reference traces, neither result measures branch-preservation or
segmentation accuracy.

The combined observation supports only a conditional interpretation: visual
preference differed between the initial and targeted ROI selections, so the
relative behavior of the methods may depend on background and image region.
It does not support universal superiority of either method.

## Running the internal comparison

From the repository root:

~~~powershell
python -m tools.compare_clahe_methods "C:\path\to\prepared Iba1 images" --reviewer-id reviewer_1 --output-dir clahe_method_comparison
~~~

If no input path is supplied, a directory chooser opens.
When launched directly with the VS Code Run button and no arguments, the
reviewer identifier defaults to <code>reviewer_1</code>. Supply
<code>--reviewer-id</code> when separate reviewers or review sessions must be
distinguished.

By default, the whole-image comparator receives one image-level setting chosen
from whole-image entropy: 4.8 for low entropy, 3.6 for high entropy, and 2.8
otherwise. The reviewer does not see or choose this value before A/B review.
This default is convenient for an internal technical comparison, but it is not
a tuned universal baseline.

For a stronger held-out design, choose one fixed setting on a separate tuning
set and lock it before evaluating different images:

~~~powershell
python -m tools.compare_clahe_methods "C:\path\to\held-out Iba1 images" --reviewer-id reviewer_1 --fixed-clip-limit 2.8 --nlm-h 8 --output-dir clahe_heldout_reviewer_1
~~~

Use a separate output directory for each reviewer. The
<code>--show-labels</code> option is for debugging only: it is recorded as
unblinded and must not be pooled with masked reviews.

## Outputs

During incomplete collection:

- <code>clahe_masked_reviews.csv</code> contains A/B decisions without method
  identities;
- <code>masked_qc/</code> contains matched raw, enhanced, and skeleton panels
  labeled only A/B;
- <code>roi_records/</code> and <code>candidate_cache/</code> preserve the
  exact pre-exposure crop and displayed candidate arrays for safe resume; and
- <code>clahe_comparison_config.json</code> records the input manifest,
  settings, code hashes, Git state, and numerical-library versions.

Only after every configured source image has a record:

- <code>clahe_results_unblinded.csv</code> maps A/B to methods;
- <code>unblinded_qc/</code> copies the panels to method-labeled filenames;
- <code>clahe_comparison_summary.json</code> reports descriptive image-level
  counts and the exact denominator; and
- <code>clahe_comparison_summary.png</code> displays preference counts.

The source-folder name is recorded but must not be treated as an animal
identifier. No inferential p-value or confidence interval is produced because
the current files do not establish independent animals or cluster structure.

## What the result can support

A completed review may support wording such as:

> In an internal method-label-hidden review of N source images, adaptive
> patch-wise CLAHE was visually preferred in X cases, the single-setting
> whole-image comparator in Y, with Z equivalent, neither, or excluded cases.

It does **not** support:

- “adaptive CLAHE is mathematically superior”;
- “adaptive CLAHE preserves true branches more accurately”;
- “adaptive CLAHE improves Sholl accuracy”; or
- treating images, cells, pixels, or radii as independent biological
  replicates.

Greater entropy, contrast, skeleton length, endpoint count, or foreground
fraction can increase when background artifacts are amplified. None is
independent ground truth.

## Required next step for an accuracy claim

For quantitative validation, create reference traces from raw images without
showing either pipeline result, preferably with two masked raters. Predefine a
physical-distance tolerance and compare:

- skeleton precision, recall, and F1;
- false-positive skeleton length;
- fraction of reference arbor retained in the soma-connected component;
- endpoint and branch-point detection/localization; and
- Sholl-profile error at identical soma coordinates and physical radii.

Choose a primary metric before inspecting results. Separate tuning and
evaluation by verified <code>Animal_ID</code>, not by cell or filename, and
report paired effects with uncertainty that respects image/animal clustering.
Inter-rater agreement provides context for attainable reference consistency.

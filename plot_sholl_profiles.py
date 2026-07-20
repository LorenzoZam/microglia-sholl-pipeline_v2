"""
plot_sholl_profiles.py — Statistical analysis and visualization of Sholl data.

v2 Update:
  • Hierarchical Mixed-Effects Model (statsmodels MixedLM) with Natural Cubic
    Spline basis on radius and random intercepts for Animal_ID.
  • Original simple mean±SEM curves preserved as fallback.
  • Outlier detection via IQR (max_radius or AUC).
  • Extended morphometric summary barplots (if columns present).

Dependencies: pandas, matplotlib, numpy, scipy, statsmodels, patsy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import scipy.integrate

# Apply Seaborn theme globally for scientific aesthetics
sns.set_theme(style="ticks", context="talk", palette="colorblind")

# ─── Configuration ──────────────────────────────────────────────────────────
conversion_factor = 0.56  # 1 pixel = 0.56 µm
SPLINE_DF = 5             # degrees of freedom for natural cubic spline


def add_cell_id(df):
    """Return a copy with a stable cell identifier across images/animals."""
    df = df.copy()
    parts = []
    for column in ("Animal_ID", "Image_Name", "Image"):
        if column in df.columns:
            parts.append(df[column].astype(str))
    parts.append(df["Soma_ID"].astype(str))
    cell_id = parts[0]
    for part in parts[1:]:
        cell_id = cell_id + "/" + part
    df["Cell_ID"] = cell_id
    return df


def radius_um(df):
    """Use row-level calibration when available, otherwise the CLI default."""
    if "Radius (µm)" in df.columns:
        return pd.to_numeric(df["Radius (µm)"], errors="coerce")
    if "Pixel size (µm/px)" in df.columns:
        return df["Radius"] * pd.to_numeric(
            df["Pixel size (µm/px)"], errors="coerce"
        )
    return df["Radius"] * conversion_factor
# ────────────────────────────────────────────────────────────────────────────


# ===========================================================================
#  Data loading helpers
# ===========================================================================

def load_dataframe(file_path):
    """Load CSV with auto-detected separator, validating required columns."""
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(file_path, sep=sep)
            df = df.rename(columns={"Radius (px)": "Radius", "Cell": "Soma_ID"})
            if {"Intersections", "Soma_ID", "Radius"}.issubset(df.columns):
                return df
        except Exception:
            continue
    raise ValueError(
        f"Cannot read file: {file_path}. "
        "Expected columns: Intersections, Soma_ID, Radius."
    )


def load_and_process_data(file_path):
    """Load data and compute mean ± SEM by radius, retaining zeroes."""
    df = load_dataframe(file_path)
    df = add_cell_id(df)
    df["Radius_um"] = radius_um(df)
    stats = df.groupby("Radius_um")["Intersections"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    stats["std"] = stats["std"].fillna(0)
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    radii = stats["Radius_um"]
    return radii, stats["mean"], stats["sem"]


# ===========================================================================
#  Classic plots (backward-compatible)
# ===========================================================================

def plot_comparison_curve_simple(file_paths, labels, colors):
    """Original mean ± SEM Sholl curve (no mixed model)."""
    plt.figure(figsize=(10, 6))

    for file, lbl, color in zip(file_paths, labels, colors):
        radii, mean, sem = load_and_process_data(file)
        df = load_dataframe(file)
        df = add_cell_id(df)
        n = df["Cell_ID"].nunique()
        label_with_n = f"{lbl} (n={n})"

        plt.plot(radii, mean, marker='o', markersize=5, linestyle='-', linewidth=2.5,
                 label=label_with_n, color=color)
        plt.fill_between(radii, mean - sem, mean + sem,
                         color=color, alpha=0.25, linewidth=0)

    plt.xlabel("Distance from Soma (µm)", fontweight='bold')
    plt.ylabel("Number of Intersections", fontweight='bold')
    plt.title("Sholl Curve (Mean ± SEM)", fontweight='bold', pad=15)
    plt.legend(frameon=False, fontsize=10)
    sns.despine()
    plt.tight_layout()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except Exception:
        pass
    plt.show()


# ===========================================================================
#  Mixed-effects Sholl model
# ===========================================================================

def fit_mixed_effects_sholl(df, spline_df=SPLINE_DF):
    """
    Fit a Hierarchical Mixed-Effects Model to Sholl intersection data.

    Model:
        Intersections ~ cr(Radius_um, df) * C(Group)
        Random effect:  (1 | Animal_ID)

    Uses Natural Cubic Splines (patsy ``cr()``) to model the non-linear
    relationship between radius and intersections, with Group as a
    categorical fixed effect and Animal_ID as random intercepts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Intersections, Radius, Animal_ID, Group.
    spline_df : int
        Degrees of freedom for the natural cubic spline.

    Returns
    -------
    statsmodels MixedLMResults
        Fitted model object.
    """
    try:
        import statsmodels.formula.api as smf
        from patsy import dmatrix  # noqa: F401  — validate patsy is available
    except ImportError as e:
        raise ImportError(
            "Mixed-effects model requires 'statsmodels' and 'patsy'. "
            "Install with:  pip install statsmodels"
        ) from e

    # Prepare data
    df = df.copy()
    df = add_cell_id(df)
    df["Radius_um"] = radius_um(df)
    df["Intersections"] = df["Intersections"].astype(float)

    # Ensure required columns
    for col in ("Animal_ID", "Group"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found.  Run merge_sholl_results.py first "
                "to generate Animal_ID and Group columns."
            )

    # Build formula with natural cubic spline × Group interaction
    formula = (
        f"Intersections ~ cr(Radius_um, df={spline_df}) * C(Group)"
    )

    print("\n" + "=" * 70)
    print("Fitting Hierarchical Mixed-Effects Model")
    print(f"  Formula:  {formula}")
    print(f"  Random effect:  (1 | Animal_ID)")
    print(f"  N observations: {len(df)}")
    print(f"  N animals:      {df['Animal_ID'].nunique()}")
    print(f"  N groups:       {df['Group'].nunique()}")
    print("=" * 70)

    model = smf.mixedlm(
        formula, data=df, groups=df["Animal_ID"],
        vc_formula={"Cell": "0 + C(Cell_ID)"},
    )
    result = model.fit(reml=True)

    print(result.summary())

    # Intra-class correlation (ICC)
    var_animal = float(result.cov_re.iloc[0, 0])
    var_resid = float(result.scale)
    icc = var_animal / (var_animal + var_resid)
    print(f"\nIntra-class Correlation (ICC):  {icc:.4f}")
    print(f"  → {icc*100:.1f}% of residual variance is between animals")

    return result


def plot_mixed_effects_curves(df, result, colors=None):
    """
    Plot fixed-effect model predictions with raw mean ± SEM overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Same DataFrame used for fitting (needs Radius, Group, Animal_ID).
    result : MixedLMResults
        Fitted model.
    colors : list or None
        Colours per group.
    """
    df = df.copy()
    df = add_cell_id(df)
    df["Radius_um"] = radius_um(df)
    groups = sorted(df["Group"].unique())

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(groups))]

    # Create a fine grid of radii for smooth predictions
    r_min = df["Radius_um"].min()
    r_max = df["Radius_um"].max()
    r_grid = np.linspace(r_min, r_max, 200)

    fig, ax = plt.subplots(figsize=(10, 6))

    for grp, color in zip(groups, colors):
        # Build prediction DataFrame — use the first Animal_ID as reference
        # (fixed-effects only prediction)
        ref_animal = df.loc[df["Group"] == grp, "Animal_ID"].iloc[0]
        pred_df = pd.DataFrame({
            "Radius_um": r_grid,
            "Group": grp,
            "Animal_ID": ref_animal,
            "Radius": r_grid / conversion_factor,
        })

        pred = result.predict(pred_df)

        # Also compute the raw group mean ± SEM for scatter overlay
        grp_data = df[df["Group"] == grp]
        raw_stats = grp_data.groupby("Radius_um")["Intersections"].agg(
            ["mean", "sem"]
        ).reset_index()
        # Use std / sqrt(n) for sem if pandas < 1.0
        if "sem" not in raw_stats.columns or raw_stats["sem"].isna().all():
            raw_grp = grp_data.groupby("Radius_um")["Intersections"]
            raw_stats = raw_grp.agg(["mean", "std", "count"]).reset_index()
            raw_stats["sem"] = raw_stats["std"] / np.sqrt(raw_stats["count"])

        n_cells = grp_data["Cell_ID"].nunique()
        n_animals = grp_data["Animal_ID"].nunique()

        # Plot predicted curve
        ax.plot(r_grid, pred, color=color, linewidth=3,
                label=f"{grp} (cells={n_cells}, animals={n_animals})")

        # Scatter raw means
        ax.scatter(raw_stats["Radius_um"], raw_stats["mean"],
                   color=color, s=30, alpha=0.6, zorder=3, edgecolor='white', linewidth=0.5)
        ax.fill_between(raw_stats["Radius_um"],
                        raw_stats["mean"] - raw_stats["sem"],
                        raw_stats["mean"] + raw_stats["sem"],
                        color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Distance from Soma (µm)", fontweight='bold')
    ax.set_ylabel("Number of Intersections", fontweight='bold')
    ax.set_title("Mixed-Effects Sholl Model Analysis", fontweight='bold', pad=15)
    ax.legend(frameon=False, fontsize=10)
    sns.despine()
    fig.tight_layout()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except Exception:
        pass
    plt.show()


# ===========================================================================
#  Bar plots
# ===========================================================================

def plot_max_radius_per_cell(file_path, label=""):
    df = load_dataframe(file_path)
    df = df[df["Intersections"] >= 0]

    max_radii = []
    for _, group in df.groupby("Soma_ID"):
        last_non_zero = group[group["Intersections"] > 0]["Radius"].max()
        max_radii.append(
            last_non_zero if pd.notna(last_non_zero) else group["Radius"].max()
        )

    max_radii.sort()
    max_radii_um = [r * conversion_factor for r in max_radii]

    mean_r = np.mean(max_radii_um)
    std_r = np.std(max_radii_um, ddof=1)

    plt.figure(figsize=(10, 6))
    
    # Use stem/lollipop plot instead of heavy bars
    x_pos = range(1, len(max_radii_um) + 1)
    plt.vlines(x=x_pos, ymin=0, ymax=max_radii_um, color='teal', alpha=0.6, linewidth=2)
    plt.scatter(x_pos, max_radii_um, color='darkcyan', s=40, zorder=3)

    stat_text = f"Mean ± SD: {mean_r:.1f} ± {std_r:.1f} µm"
    plt.text(0.05, 0.95, stat_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    plt.xlabel("Cells (Sorted Distribution)", fontweight='bold')
    plt.ylabel("Max Radius (µm)", fontweight='bold')
    plt.title(f"Cellular Max Radius Distribution — {label}", fontweight='bold', pad=15)
    plt.xticks([]) # Hide individual sequential integers for clean look
    sns.despine(bottom=True)
    
    plt.tight_layout()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except Exception:
        pass
    plt.show()


def plot_grouped_max_radius_barplot(file_paths, labels, colors):
    all_cells_radius = []
    
    for fp, lbl in zip(file_paths, labels):
        df = load_dataframe(fp)
        df = df[df["Intersections"] >= 0]
        
        for soma_id, group in df.groupby("Soma_ID"):
            last_nz = group[group["Intersections"] > 0]["Radius"].max()
            max_r = last_nz if pd.notna(last_nz) else group["Radius"].max()
            all_cells_radius.append({"Group": lbl, "Max_Radius_um": max_r * conversion_factor})
            
    plot_df = pd.DataFrame(all_cells_radius)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Beautiful seaborn violin + swarm visualization
    sns.violinplot(
        data=plot_df, x="Group", y="Max_Radius_um", 
        ax=ax, color='white', inner=None, linewidth=1.5
    )
    sns.stripplot(
        data=plot_df, x="Group", y="Max_Radius_um", 
        ax=ax, hue="Group", palette=colors[:len(plot_df["Group"].unique())], size=6, alpha=0.7, jitter=True, zorder=1, legend=False
    )
    
    # Overlay robust mean + error representation
    sns.pointplot(
        data=plot_df, x="Group", y="Max_Radius_um", 
        ax=ax, color='black', errorbar='sd', linestyle='none', 
        capsize=0.1, markers="D", markersize=8, zorder=3
    )

    ax.set_ylabel("Max Radius (µm)", fontweight='bold')
    ax.set_xlabel("")
    ax.set_title("Distribution of Cell Sizes per Group", fontweight='bold', pad=15)
    sns.despine(trim=True)
    fig.tight_layout()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except Exception:
        pass
    plt.show()


# ===========================================================================
#  Morphometric summary barplot (new columns)
# ===========================================================================

def plot_morphometric_summary(file_paths, labels, colors):
    """
    Generate grouped barplots for per-cell morphometric features
    (Fractal_Dimension, Lacunarity, Betweenness_Centrality, etc.)
    if the corresponding columns exist in the CSV.
    """
    morph_cols = [
        'Fractal_Dimension', 'Lacunarity',
        'Betweenness_Centrality', 'Closeness_Centrality',
        'Ramification_Index', 'Soma_Area', 'Soma_Circularity'
    ]

    # Collect per-cell means (metrics are constant per soma, so just
    # take the first value per Soma_ID)
    all_data = []
    for fp, lbl in zip(file_paths, labels):
        df = load_dataframe(fp)
        present_cols = [c for c in morph_cols if c in df.columns]
        if not present_cols:
            print(f"  No morphometric columns found in {fp}, skipping.")
            continue
        df = add_cell_id(df)
        cell_df = df.groupby("Cell_ID")[present_cols].first().reset_index()
        cell_df["Group"] = lbl
        all_data.append(cell_df)

    if not all_data:
        print("No morphometric data found in any file.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    present_cols = [c for c in morph_cols if c in combined.columns]

    groups = combined["Group"].unique()
    color_dict = dict(zip(groups, colors))

    chunk_size = 4
    chunks = [present_cols[i:i + chunk_size] for i in range(0, len(present_cols), chunk_size)]
    
    for chunk_idx, cols_chunk in enumerate(chunks, 1):
        n_metrics = len(cols_chunk)
        # Provide plenty of horizontal space per subplot
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics + 2, 6))
        if n_metrics == 1:
            axes = [axes]
        
        fig.canvas.manager.set_window_title(f"Morphometrics - Part {chunk_idx}")

        for ax, col in zip(axes, cols_chunk):
            sns.boxplot(
                data=combined, x="Group", y=col,
                ax=ax, color="white", showfliers=False, width=0.5, linewidth=1.5
            )
            # Increased point size (size=7) and opacity (alpha=0.9) to make single cells pop out
            sns.stripplot(
                data=combined, x="Group", y=col,
                ax=ax, hue="Group", palette=color_dict, size=7, alpha=0.9, jitter=0.2, zorder=1, legend=False
            )
            
            ax.set_xlabel("")
            ax.set_ylabel("")
            clean_name = col.replace('_', ' ')
            ax.set_title(clean_name, fontweight='bold', fontsize=13, pad=12)
            
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=11)

        sns.despine()
        fig.suptitle(f"Morphometric Distributions per Group (Part {chunk_idx}/{len(chunks)})", fontsize=16, fontweight='bold', y=1.05)
        # Added w_pad for extra horizontal breathing room between plots
        fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=2.0)
        
        try:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.state('zoomed')
        except Exception:
            pass

    # Call show once at the end so all windows pop up together
    plt.show()


# ===========================================================================
#  Outlier detection (unchanged logic)
# ===========================================================================

def calculate_auc_per_soma(df):
    aucs = {}
    df = add_cell_id(df)
    for soma_id, group in df.groupby("Cell_ID"):
        radii = radius_um(group).values
        intersections = group["Intersections"].values
        auc = scipy.integrate.trapezoid(intersections, radii) if len(radii) > 1 else 0
        aucs[soma_id] = auc
    return aucs


def detect_outliers(df, method='iqr', threshold=1.5, metric='max_radius'):
    if metric == 'max_radius':
        df = add_cell_id(df)
        values = [(sid, radius_um(g).max())
                  for sid, g in df.groupby("Cell_ID")]
    elif metric == 'auc':
        values = list(calculate_auc_per_soma(df).items())
    else:
        raise ValueError("Metric must be 'max_radius' or 'auc'")

    data_vals = [v for _, v in values]
    q1, q3 = np.percentile(data_vals, 25), np.percentile(data_vals, 75)
    iqr = q3 - q1
    lo, hi = q1 - threshold * iqr, q3 + threshold * iqr

    outliers = [sid for sid, val in values if val < lo or val > hi]
    print(f"Detected {len(outliers)} outliers ({method}, {metric}): {outliers}")
    return outliers


def filter_outliers(df, outliers):
    identified = add_cell_id(df)
    return identified[~identified["Cell_ID"].isin(outliers)]


# ===========================================================================
#  File selection helpers
# ===========================================================================

def select_files_and_labels():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select one or more Sholl CSV files",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    file_paths = list(file_paths)
    labels = []
    for fp in file_paths:
        default_label = os.path.splitext(os.path.basename(fp))[0]
        label = input(
            f"Enter label for '{fp}' (default: {default_label}): "
        ).strip()
        labels.append(label if label else default_label)
    return file_paths, labels


def get_default_colors(n):
    return list(plt.get_cmap('tab10').colors[:n])


# ===========================================================================
#  Main execution
# ===========================================================================

if __name__ == "__main__":
    file_paths, labels = select_files_and_labels()
    if not file_paths:
        print("No files selected. Exiting.")
        exit()

    # Prompt for custom conversion factor
    user_cf = input(f"\nEnter pixel-to-µm conversion factor (default {conversion_factor}): ").strip()
    if user_cf:
        try:
            conversion_factor = float(user_cf)
            print(f"  > Using custom conversion factor: {conversion_factor} µm/px")
        except ValueError:
            print(f"  > Invalid input. Falling back to default: {conversion_factor} µm/px")
    else:
        print(f"  > Using default conversion factor: {conversion_factor} µm/px")

    colors = get_default_colors(len(file_paths))

    # ── Outlier detection (optional) ────────────────────────────────────
    detect_out = (
        input("Detect outliers? (y/n, default n): ").strip().lower() == 'y'
    )
    if detect_out:
        metric = (
            input("Metric ('max_radius' or 'auc', default 'max_radius'): "
                  ).strip() or 'max_radius'
        )
        threshold = float(
            input("IQR threshold (default 1.5): ").strip() or 1.5
        )
        exclude = (
            input("Exclude outliers? (y/n, default n): ").strip().lower()
            == 'y'
        )

        for i, file in enumerate(file_paths):
            df = load_dataframe(file)
            outliers = detect_outliers(df, metric=metric, threshold=threshold)
            if exclude and outliers:
                df = filter_outliers(df, outliers)
                filtered_path = file.replace('.csv', '_filtered.csv')
                df.to_csv(filtered_path, index=False)
                file_paths[i] = filtered_path
                print(f"Filtered file saved: {filtered_path}")

    # ── Choose analysis mode ────────────────────────────────────────────
    use_mixed = (
        input("\nUse mixed-effects model? (y/n, default n): "
              ).strip().lower() == 'y'
    )

    if use_mixed:
        # Load all files into a single DataFrame
        all_dfs = []
        for fp, lbl in zip(file_paths, labels):
            df = load_dataframe(fp)
            # If the file already has Group/Animal_ID, use them
            if "Group" not in df.columns:
                df["Group"] = lbl
            if "Animal_ID" not in df.columns:
                # Derive from filename
                base = os.path.splitext(os.path.basename(fp))[0]
                df["Animal_ID"] = base
                print(f"  Warning: no 'Animal_ID' column in {fp}, "
                      f"using filename '{base}' as Animal_ID")
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        # Fit and plot
        result = fit_mixed_effects_sholl(combined, spline_df=SPLINE_DF)
        plot_mixed_effects_curves(combined, result, colors)
    else:
        # Classic mean ± SEM
        plot_comparison_curve_simple(file_paths, labels, colors)

    # ── Common plots ────────────────────────────────────────────────────
    plot_grouped_max_radius_barplot(file_paths, labels, colors)

    # ── Morphometric summary (if columns available) ─────────────────────
    plot_morphometric_summary(file_paths, labels, colors)

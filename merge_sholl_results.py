import pandas as pd
import os
import csv
import re
import tkinter as tk
from tkinter import filedialog

# ============================================================================
#  Merge multiple Sholl CSV files into a unified dataset.
#
#  Updates (v2):
#    - Passes through ALL columns (supports extended morphometric CSVs)
#    - Derives Animal_ID from source filename  (configurable regex)
#    - Adds a Group column from user-provided labels
#    - Reassigns Soma_ID to avoid conflicts across files
# ============================================================================

# --- Configuration -----------------------------------------------------------
# The regex extracts the Animal_ID from the source filename.
# Default pattern: take everything before the first underscore.
#   e.g.  "Sham1_image02.csv"  ->  Animal_ID = "Sham1"
#         "TBI3_slice4.csv"    ->  Animal_ID = "TBI3"
# Adjust the pattern below if your naming convention is different.
ANIMAL_ID_PATTERN = r"^(.+?)_"  # captures text before the first underscore
# -----------------------------------------------------------------------------


def detect_separator(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
    return dialect.delimiter


def extract_animal_id(filename, pattern=ANIMAL_ID_PATTERN):
    """
    Extract an animal identifier from the filename using a regex pattern.

    Falls back to the full filename (without extension) if no match is found.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(pattern, base)
    if match:
        return match.group(1)
    return base


# --- Select CSV files --------------------------------------------------------
root = tk.Tk()
root.withdraw()
csv_files = filedialog.askopenfilenames(
    title="Select CSV files to merge",
    filetypes=[("CSV files", "*.csv")]
)
csv_files = list(csv_files)
if not csv_files:
    print("No files selected. Exiting.")
    exit()

# --- Collect group labels per file -------------------------------------------
group_labels = []
for fp in csv_files:
    default_label = os.path.splitext(os.path.basename(fp))[0]
    label = input(f"Enter GROUP label for '{os.path.basename(fp)}' "
                  f"(default: {default_label}): ").strip()
    group_labels.append(label if label else default_label)

# --- Select output directory -------------------------------------------------
output_dir = filedialog.askdirectory(title="Select the output directory for the merged file")
if not output_dir:
    print("No directory selected. Exiting.")
    exit()

dataframes = []
last_soma_id = 0

for file, group in zip(csv_files, group_labels):
    print(f"Reading file: {file}")
    separator = detect_separator(file)
    print(f"  Detected separator: '{separator}'")
    df = pd.read_csv(file, sep=separator)
    df.columns = df.columns.str.strip()
    print(f"  Columns found: {df.columns.tolist()}")

    if 'Soma_ID' not in df.columns:
        raise ValueError(
            f"'Soma_ID' not found in {file}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Preserve the source ID and assign a stable cross-file Cell_ID.
    df['Source_Soma_ID'] = df['Soma_ID']
    old_ids = df['Soma_ID'].unique()
    id_map = {old: new for old, new in
              zip(old_ids, range(last_soma_id + 1,
                                last_soma_id + 1 + len(old_ids)))}
    df['Soma_ID'] = df['Soma_ID'].map(id_map)
    last_soma_id = df['Soma_ID'].max()

    # Round intersections if present
    if 'Intersections' in df.columns:
        df['Intersections'] = df['Intersections'].round().astype('Int64')

    # Add Animal_ID (from filename) and Group (from user input)
    df['Animal_ID'] = extract_animal_id(file)
    df['Group'] = group
    source_name = os.path.splitext(os.path.basename(file))[0]
    df['Cell_ID'] = source_name + "/" + df['Source_Soma_ID'].astype(str)

    print(f"  Animal_ID = '{df['Animal_ID'].iloc[0]}',  Group = '{group}'")
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)

output_path = os.path.join(output_dir, "merged_data.csv")
merged_df.to_csv(output_path, sep="\t", index=False)

print(f"\nMerging completed!  {len(dataframes)} files merged.")
print(f"Total cells (unique Soma_ID): {merged_df['Soma_ID'].nunique()}")
print(f"Columns: {merged_df.columns.tolist()}")
print(f"File saved as '{output_path}'")

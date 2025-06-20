import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm


def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = df[~df["automapper"]]
    df = df[~df["missing_levels"]]
    df = df[~df["missing_song"]]
    df = df[~df["default_skip"]]

    df = df.drop(
        ["missing_levels", "missing_song", "automapper", "default_skip"], axis=1
    )

    return df


meta_df = pd.read_csv("dataset/beatmaps/color_notes/metadata.csv")
meta_df = clean_data(meta_df)

# Directory containing .npz files
npz_dir = Path("dataset/beatmaps/color_notes/npz")  # Change as needed

# Dictionary to store keys per file
data = []

# Set to collect all unique keys
all_keys = set()

# Extract keys from each file
for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing files"):
    try:
        with np.load(
            os.path.join(npz_dir, f"{row['song']}.npz"), allow_pickle=True
        ) as npz:
            keys = list(npz.files)
            all_keys.update(keys)
            data.append({"filename": row["song"], "keys": keys})
    except Exception as e:
        print(f"Error reading {row['song']}: {e}")

# Create DataFrame with one row per file
rows = []
for entry in data:
    row = {"filename": entry["filename"]}
    for key in all_keys:
        row[key] = "✓" if key in entry["keys"] else ""
    rows.append(row)

df = pd.DataFrame(rows)

# Save to CSV
output_path = "npz_keys_summary.csv"
df.to_csv(output_path, index=False)
print(f"CSV written to {output_path}")

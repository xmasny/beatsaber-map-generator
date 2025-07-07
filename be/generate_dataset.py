import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

type = "Easy"
batch_size = 100

base_path = "dataset/beatmaps/color_notes"
filename = f"{base_path}/notes_dataset/notes_{type.lower()}.parquet"
npz_dir = f"{base_path}/npz"

keys_to_drop = [
    "Easy",
    "Normal",
    "Hard",
    "Expert",
    "ExpertPlus",
    "notes",
    "stacked_mel_3",
    "note_labels",
]


def word_to_one_hot(combined_word: str):
    one_hot = np.zeros((3, 4, 19))
    one_hot[:, :, 0] = 1
    words = combined_word.split("_")
    for word in words:
        c = 1 if word[0] == "L" else 10
        d = int(word[1])
        x = int(word[2])
        y = int(word[3])
        one_hot[y, x, 0] = 0
        one_hot[y, x, c + d] = 1
    return one_hot


def extract_window(mel: np.ndarray, index: int, window_size: int):
    half = window_size // 2
    start = index - half
    end = index + half + 1
    pad_left = max(0, -start)
    pad_right = max(0, end - mel.shape[1])
    start = max(0, start)
    end = min(mel.shape[1], end)
    window = mel[:, start:end]
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")
    return window


# --- LOAD PARQUET AND DROP DUPLICATES ---
df = pd.read_parquet(filename)
df_files = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
print(f"Loaded {filename} with {len(df)} unique rows.")

# --- PREPARE OUTPUT DIRS ---
for sub in ["onsets", "classification"]:
    for ctype in ["compressed", "uncompressed"]:
        os.makedirs(f"dataset/batch/{sub}/train/{ctype}", exist_ok=True)

# --- PROCESS .NPZ FILES IN BATCHES ---
onsets_combined_data = {}
classification_combined_data = {}
file_counter = 0
batch_counter = 0

for row in tqdm(df_files.itertuples(), total=len(df_files), desc="Processing rows"):
    npz_path = os.path.join(npz_dir, f"{row.name}.npz")

    if not os.path.exists(npz_path):
        print(f"Missing file: {npz_path}")
        continue

    try:
        data = np.load(npz_path, allow_pickle=True)

        for key in data.files:
            if key not in keys_to_drop:
                unique_key = f"{row.name}_{key}"
                onsets_combined_data[unique_key] = data[key]

        # Onsets array
        onsets_key = f"{row.name}_onsets_{type.lower()}"
        onsets_combined_data[onsets_key] = data["onsets"].item()[type]["onsets_array"]

        song_steps = df[df["name"] == f"{row.name}"]

        for step in tqdm(
            song_steps.itertuples(),
            total=len(song_steps),
            desc=f"Processing steps for {row.name}",
        ):
            classification_combined_data[f"{row.name}_{step.stack}_classes"] = (
                word_to_one_hot(str(step.combined_word))
            )
            classification_combined_data[f"{row.name}_{step.stack}_mel"] = (
                extract_window(data["song"], int(str(step.stack)), 45)
            )
            del data
            gc.collect()

    except Exception as e:
        print(f"Error with {npz_path}: {e}")
        continue

    file_counter += 1

    # Save batch
    if file_counter % batch_size == 0:
        # --- Save Onsets ---
        np.savez_compressed(
            f"dataset/batch/onsets/train/compressed/batch_{batch_counter:03}.npz",
            **onsets_combined_data,
        )
        np.savez(
            f"dataset/batch/onsets/train/uncompressed/batch_{batch_counter:03}.npz",
            **onsets_combined_data,
        )
        onsets_combined_data = {}

        # --- Save Classification ---
        np.savez_compressed(
            f"dataset/batch/classification/train/compressed/batch_{batch_counter:03}.npz",
            **classification_combined_data,
        )
        np.savez(
            f"dataset/batch/classification/train/uncompressed/batch_{batch_counter:03}.npz",
            **classification_combined_data,
        )
        classification_combined_data = {}

        batch_counter += 1

# --- FINAL BATCH ---
if onsets_combined_data:
    np.savez_compressed(
        f"dataset/batch/onsets/train/compressed/batch_{batch_counter:03}.npz",
        **onsets_combined_data,
    )
    np.savez(
        f"dataset/batch/onsets/train/uncompressed/batch_{batch_counter:03}.npz",
        **onsets_combined_data,
    )

if classification_combined_data:
    np.savez_compressed(
        f"dataset/batch/classification/train/compressed/batch_{batch_counter:03}.npz",
        **classification_combined_data,
    )
    np.savez(
        f"dataset/batch/classification/train/uncompressed/batch_{batch_counter:03}.npz",
        **classification_combined_data,
    )

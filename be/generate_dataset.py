from io import BytesIO
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# === CONFIGURATION ===
type = "Easy"
base_path = "local_npz_data"
filename = f"dataset/beatmaps/color_notes/notes_{type.lower()}.parquet"
batch_size = 10

onsets_out_dir = "dataset/batch/onsets/train"
class_out_dir = "dataset/batch/classification/train"
os.makedirs(f"{onsets_out_dir}/compressed", exist_ok=True)
os.makedirs(f"{onsets_out_dir}/uncompressed", exist_ok=True)
os.makedirs(f"{class_out_dir}/compressed", exist_ok=True)
os.makedirs(f"{class_out_dir}/uncompressed", exist_ok=True)

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


def process_row(row, df_all):
    try:
        npz_path = os.path.join(base_path, f"{row.name}.npz")
        data = np.load(npz_path, allow_pickle=True)

        onsets_data = {}
        for key in data.files:
            if key not in keys_to_drop:
                unique_key = f"{row.name}_{key}"
                onsets_data[unique_key] = data[key]

        onsets_data[f"{row.name}_onsets_{type.lower()}"] = data["onsets"].item()[type][
            "onsets_array"
        ]

        classification_data = {}
        song_steps = df_all[df_all["name"] == row.name]
        for step in song_steps.itertuples():
            classification_data[f"{row.name}_{step.stack}_classes"] = word_to_one_hot(
                str(step.combined_word)
            )
            classification_data[f"{row.name}_{step.stack}_mel"] = extract_window(
                data["song"], int(str(step.stack)), 45
            )

        return onsets_data, classification_data
    except Exception as e:
        print(f"Error processing {row.name}: {e}")
        return {}, {}


# --- LOAD PARQUET ---
df = pd.read_parquet(filename)
df_files = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)

# --- PARALLEL PROCESSING ---
onsets_combined_data = {}
classification_combined_data = {}
file_counter = 0
batch_counter = 0

with ProcessPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_row, row, df) for row in df_files.itertuples()]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        onsets_data, class_data = future.result()

        onsets_combined_data.update(onsets_data)
        classification_combined_data.update(class_data)
        file_counter += 1

        if file_counter % batch_size == 0:
            # Save onsets
            compressed_path = (
                f"{onsets_out_dir}/compressed/batch_{batch_counter:03}.npz"
            )
            uncompressed_path = (
                f"{onsets_out_dir}/uncompressed/batch_{batch_counter:03}.npz"
            )
            np.savez_compressed(compressed_path, **onsets_combined_data)
            np.savez(uncompressed_path, **onsets_combined_data)
            onsets_combined_data = {}

            # Save classification
            compressed_path = f"{class_out_dir}/compressed/batch_{batch_counter:03}.npz"
            uncompressed_path = (
                f"{class_out_dir}/uncompressed/batch_{batch_counter:03}.npz"
            )
            np.savez_compressed(compressed_path, **classification_combined_data)
            np.savez(uncompressed_path, **classification_combined_data)
            classification_combined_data = {}

            batch_counter += 1

# --- FINAL BATCH ---
if onsets_combined_data:
    compressed_path = f"{onsets_out_dir}/compressed/batch_{batch_counter:03}.npz"
    uncompressed_path = f"{onsets_out_dir}/uncompressed/batch_{batch_counter:03}.npz"
    np.savez_compressed(compressed_path, **onsets_combined_data)
    np.savez(uncompressed_path, **onsets_combined_data)

if classification_combined_data:
    compressed_path = f"{class_out_dir}/compressed/batch_{batch_counter:03}.npz"
    uncompressed_path = f"{class_out_dir}/uncompressed/batch_{batch_counter:03}.npz"
    np.savez_compressed(compressed_path, **classification_combined_data)
    np.savez(uncompressed_path, **classification_combined_data)

from io import BytesIO
import numpy as np
import requests
from tqdm import tqdm
import pandas as pd
import os

type = "Easy"

base_url = "http://kaistore.dcs.fmph.uniba.sk/beatsaber-map-generator/dataset/beatmaps/color_notes"
filename = f"notes_{type.lower()}.parquet"
url = rf"{base_url}/notes_dataset/{filename}"
batch_size = 10


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

    # Pad if needed
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")

    return window


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
# Stream the request so we can download chunk-by-chunk
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

# --- LOAD PARQUET AND DROP DUPLICATES ---
df = pd.read_parquet(filename)
df_files = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
print(f"Downloaded {filename} with {len(df)} unique rows.")

# --- PROCESS .NPZ FILES IN BATCHES ---
onsets_combined_data = {}
classification_combined_data = {}
file_counter = 0
batch_counter = 0

os.makedirs("dataset/batch/onsets/train", exist_ok=True)
os.makedirs("dataset/batch/classification/train", exist_ok=True)

for row in tqdm(df_files.itertuples(), total=len(df_files), desc="Processing rows"):
    npz_url = f"{base_url}/npz/{row.name}.npz"

    try:
        with requests.get(npz_url, stream=True) as r:
            r.raise_for_status()
            file_in_memory = BytesIO(r.content)
            data = np.load(file_in_memory, allow_pickle=True)

            # Filter unwanted keys
            for key in data.files:
                if key not in keys_to_drop:
                    unique_key = f"{row.name}_{key}"
                    onsets_combined_data[unique_key] = data[key]

            onsets_combined_data[f"{row.name}_onsets_{type.lower()}"] = data[
                "onsets"
            ].item()[type]["onsets_array"]

            del onsets_combined_data[f"{row.name}_onsets"]

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
    except Exception as e:
        print(f"Error with {npz_url}: {e}")
        continue

    file_counter += 1

    # Save a batch every N files
    if file_counter % batch_size == 0:
        # === ONSETS ===
        out_filename_onsets = (
            f"dataset/batch/onsets/train/compressed/batch_{batch_counter:03}.npz"
        )
        np.savez_compressed(out_filename_onsets, **onsets_combined_data)
        print(
            f"Saved compressed {out_filename_onsets} with {len(onsets_combined_data)} arrays."
        )

        out_filename_onsets_uncompressed = (
            f"dataset/batch/onsets/train/uncompressed/batch_{batch_counter:03}.npz"
        )
        np.savez(out_filename_onsets_uncompressed, **onsets_combined_data)
        print(
            f"Saved uncompressed {out_filename_onsets_uncompressed} with {len(onsets_combined_data)} arrays."
        )

        onsets_combined_data = {}

        # === CLASSIFICATION ===
        out_filename_class = f"dataset/batch/classification/train/compressed/batch_{batch_counter:03}.npz"
        np.savez_compressed(out_filename_class, **classification_combined_data)
        print(
            f"Saved compressed {out_filename_class} with {len(classification_combined_data)} arrays."
        )

        out_filename_class_uncompressed = f"dataset/batch/classification/train/uncompressed/batch_{batch_counter:03}.npz"
        np.savez(out_filename_class_uncompressed, **classification_combined_data)
        print(
            f"Saved uncompressed {out_filename_class_uncompressed} with {len(classification_combined_data)} arrays."
        )

        classification_combined_data = {}

        batch_counter += 1

# --- SAVE FINAL BATCH ---
if onsets_combined_data:
    out_filename_onsets = (
        f"dataset/batch/onsets/train/compressed/batch_{batch_counter:03}.npz"
    )
    np.savez_compressed(out_filename_onsets, **onsets_combined_data)
    print(
        f"Saved final compressed {out_filename_onsets} with {len(onsets_combined_data)} arrays."
    )

    out_filename_onsets_uncompressed = (
        f"dataset/batch/onsets/train/uncompressed/batch_{batch_counter:03}.npz"
    )
    np.savez(out_filename_onsets_uncompressed, **onsets_combined_data)
    print(
        f"Saved final uncompressed {out_filename_onsets_uncompressed} with {len(onsets_combined_data)} arrays."
    )

if classification_combined_data:
    out_filename_class = (
        f"dataset/batch/classification/train/compressed/batch_{batch_counter:03}.npz"
    )
    np.savez_compressed(out_filename_class, **classification_combined_data)
    print(
        f"Saved final compressed {out_filename_class} with {len(classification_combined_data)} arrays."
    )

    out_filename_class_uncompressed = (
        f"dataset/batch/classification/train/uncompressed/batch_{batch_counter:03}.npz"
    )
    np.savez(out_filename_class_uncompressed, **classification_combined_data)
    print(
        f"Saved final uncompressed {out_filename_class_uncompressed} with {len(classification_combined_data)} arrays."
    )

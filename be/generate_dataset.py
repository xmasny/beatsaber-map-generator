import os
import argparse
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--type", type=str, default="Easy", help="Difficulty type (e.g. Easy)"
)
parser.add_argument("--start", type=int, default=0, help="Start index")
parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
parser.add_argument(
    "--checkpoint_file", type=str, default=None, help="Optional checkpoint file"
)
args = parser.parse_args()

# --- Config ---
type = args.type
intermediate_batch_size = 25
final_batch_size = 100
combine_factor = final_batch_size // intermediate_batch_size

base_path = "dataset/beatmaps/color_notes"
filename = f"{base_path}/notes_dataset/notes_{type.lower()}.parquet"
npz_dir = f"{base_path}/npz"

intermediate_path = "dataset/batch/intermediate"
final_path = "dataset/batch/final"
os.makedirs(intermediate_path, exist_ok=True)
os.makedirs(final_path, exist_ok=True)

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

# --- Optional Checkpointing ---
processed_names = set()
if args.checkpoint_file and os.path.exists(args.checkpoint_file):
    with open(args.checkpoint_file, "r") as f:
        processed_names = set(line.strip() for line in f)


# --- Helpers ---
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


# --- Load Dataset ---
df = pd.read_parquet(filename)
df_files = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
df_files = (
    df_files[args.start : args.end] if args.end is not None else df_files[args.start :]
)

# --- Main Processing ---
intermediate_files = []
file_counter = 0
intermediate_counter = 0

onsets_combined_data = {}
classification_combined_data = {}

for row in tqdm(df_files.itertuples(), total=len(df_files), desc="Processing rows"):
    if row.name in processed_names:
        continue

    npz_path = os.path.join(npz_dir, f"{row.name}.npz")
    if not os.path.exists(npz_path):
        continue

    try:
        data = np.load(npz_path, allow_pickle=True)
        for key in data.files:
            if key not in keys_to_drop:
                unique_key = f"{row.name}_{key}"
                onsets_combined_data[unique_key] = data[key]

        onsets_key = f"{row.name}_onsets_{type.lower()}"
        onsets_combined_data[onsets_key] = data["onsets"].item()[type]["onsets_array"]

        song_steps = df[df["name"] == f"{row.name}"]
        for step in song_steps.itertuples():
            classification_combined_data[f"{row.name}_{step.stack}_classes"] = (
                word_to_one_hot(str(step.combined_word))
            )
            classification_combined_data[f"{row.name}_{step.stack}_mel"] = (
                extract_window(data["song"], int(str(step.stack)), 45)
            )

        del data
        gc.collect()

        if args.checkpoint_file:
            with open(args.checkpoint_file, "a") as f:
                f.write(f"{row.name}\n")

    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        continue

    file_counter += 1

    if file_counter % intermediate_batch_size == 0:
        onset_file = os.path.join(
            intermediate_path, f"onsets_{intermediate_counter:03}.npz"
        )
        class_file = os.path.join(
            intermediate_path, f"class_{intermediate_counter:03}.npz"
        )
        np.savez_compressed(onset_file, **onsets_combined_data)
        np.savez_compressed(class_file, **classification_combined_data)
        intermediate_files.append((onset_file, class_file))
        onsets_combined_data = {}
        classification_combined_data = {}
        intermediate_counter += 1

# Save any remaining intermediate data
if onsets_combined_data:
    onset_file = os.path.join(
        intermediate_path, f"onsets_{intermediate_counter:03}.npz"
    )
    class_file = os.path.join(intermediate_path, f"class_{intermediate_counter:03}.npz")
    np.savez_compressed(onset_file, **onsets_combined_data)
    np.savez_compressed(class_file, **classification_combined_data)
    intermediate_files.append((onset_file, class_file))
    intermediate_counter += 1

# --- Final merge of ALL intermediate files ---
all_intermediate_onsets = sorted(
    [f for f in os.listdir(intermediate_path) if f.startswith("onsets_")]
)
all_intermediate_classes = sorted(
    [f for f in os.listdir(intermediate_path) if f.startswith("class_")]
)

intermediate_files = list(zip(all_intermediate_onsets, all_intermediate_classes))

final_batch_counter = 0
for i in range(0, len(intermediate_files), combine_factor):
    group = intermediate_files[i : i + combine_factor]
    final_onsets = {}
    final_classes = {}

    for onset_file, class_file in group:
        onset_fp = os.path.join(intermediate_path, onset_file)
        class_fp = os.path.join(intermediate_path, class_file)

        with np.load(onset_fp, allow_pickle=True) as d:
            final_onsets.update(d)
        with np.load(class_fp, allow_pickle=True) as d:
            final_classes.update(d)

        # Delete intermediate files after loading
        os.remove(onset_fp)
        os.remove(class_fp)

    np.savez_compressed(
        os.path.join(final_path, f"onsets_batch_{final_batch_counter:03}.npz"),
        **final_onsets,
    )
    np.savez_compressed(
        os.path.join(final_path, f"class_batch_{final_batch_counter:03}.npz"),
        **final_classes,
    )

    final_batch_counter += 1
    del final_onsets, final_classes
    gc.collect()

print(
    f"✅ Final merge complete: {final_batch_counter} final batches saved and intermediate files deleted."
)

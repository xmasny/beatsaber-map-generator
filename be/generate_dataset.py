import os
import argparse
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Optional
from dataclasses import dataclass


@dataclass
class ArgparseType:
    type: str
    start: int
    end: Optional[int]
    checkpoint_file: Optional[str]
    gen_class_only: bool
    gen_onset_only: bool
    final_only: bool
    intermediate_only: bool


# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--type", type=str, default="Easy", help="Difficulty type (e.g. Easy)"
)
parser.add_argument(
    "--start", type=int, default=0, help="Start index for training subset"
)
parser.add_argument(
    "--end", type=int, default=None, help="End index (exclusive) for training subset"
)
parser.add_argument(
    "--checkpoint_file", type=str, default=None, help="Optional checkpoint file"
)
parser.add_argument(
    "--gen_class_only", action="store_true", help="Generate classification dataset only"
)
parser.add_argument(
    "--gen_onset_only", action="store_true", help="Generate onset dataset only"
)
parser.add_argument(
    "--final_only", action="store_true", help="Final only, no generation"
)
parser.add_argument(
    "--intermediate_only",
    action="store_true",
    help="Only generate intermediate files, no final merge",
)
args = ArgparseType(**vars(parser.parse_args()))

# Derived logic: if neither class nor onset flags set, do both
gen_class = args.gen_class_only or (not args.gen_class_only and not args.gen_onset_only)
gen_onset = args.gen_onset_only or (not args.gen_class_only and not args.gen_onset_only)

# --- Config ---
type = args.type
intermediate_batch_size = 25
final_batch_size = 100
combine_factor = final_batch_size // intermediate_batch_size

base_path = "dataset/beatmaps/color_notes"
filename = f"{base_path}/notes_dataset/notes_{type.lower()}.parquet"
npz_dir = f"{base_path}/npz"

intermediate_path = "dataset/batch/intermediate"
final_base_path = "dataset/batch"
splits = ["valid", "test", "train"]
final_paths = {s: os.path.join(final_base_path, s) for s in splits}
os.makedirs(intermediate_path, exist_ok=True)
for p in final_paths.values():
    os.makedirs(p, exist_ok=True)

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


# --- Load Dataset and Split ---
df = pd.read_parquet(filename)
df_files = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)

full_train_df, test_df = train_test_split(df_files, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

train_df = (
    full_train_df[args.start : args.end].copy()
    if args.end is not None
    else full_train_df[args.start :].copy()
)

valid_df["split"] = "valid"
test_df["split"] = "test"
train_df["split"] = "train"

print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

df_files = pd.concat([valid_df, test_df, train_df], ignore_index=True)

# --- Main Processing ---
split_counters = {s: 0 for s in splits}
intermediate_files_by_split = {s: [] for s in splits}

onsets_combined_data = {}
classification_combined_data = {}
file_counter = 0

if not args.final_only:
    for split in splits:
        onsets_combined_data = {}
        classification_combined_data = {}
        file_counter = 0
        df_split = df_files[df_files["split"] == split]
        for row in tqdm(
            df_split.itertuples(),
            total=len(df_split),
            desc="Processing rows for split " + split,
        ):
            if row.name in processed_names:
                continue
            npz_path = os.path.join(npz_dir, f"{row.name}.npz")
            if not os.path.exists(npz_path):
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)

                if gen_onset:
                    for key in data.files:
                        if key not in keys_to_drop:
                            unique_key = f"{row.name}_{key}"
                            onsets_combined_data[unique_key] = data[key]
                    onsets_key = f"{row.name}_onsets_{type.lower()}"
                    onsets_combined_data[onsets_key] = data["onsets"].item()[type][
                        "onsets_array"
                    ]

                if gen_class:
                    song_steps = df[df["name"] == f"{row.name}"]
                    for step in song_steps.itertuples():
                        classification_combined_data[
                            f"{row.name}_{step.stack}_classes"
                        ] = word_to_one_hot(str(step.combined_word))
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
                onset_file, class_file = None, None
                if gen_onset and onsets_combined_data:
                    onset_file = os.path.join(
                        intermediate_path,
                        f"onsets_{split}_{split_counters[split]:03}.npz",
                    )
                    np.savez_compressed(onset_file, **onsets_combined_data)
                    print(f"Saved: {onset_file}")
                    onsets_combined_data = {}
                if gen_class and classification_combined_data:
                    class_file = os.path.join(
                        intermediate_path,
                        f"class_{split}_{split_counters[split]:03}.npz",
                    )
                    np.savez_compressed(class_file, **classification_combined_data)
                    print(f"Saved: {class_file}")
                    classification_combined_data = {}
                intermediate_files_by_split[split].append((onset_file, class_file))
                split_counters[split] += 1

        # Save leftovers
        if gen_onset and onsets_combined_data:
            onset_file = os.path.join(
                intermediate_path, f"onsets_{split}_{split_counters[split]:03}.npz"
            )
            np.savez_compressed(onset_file, **onsets_combined_data)
            print(f"Saved: {onset_file} (remaining data) {len(onsets_combined_data)}")
            onsets_combined_data = {}
        if gen_class and classification_combined_data:
            class_file = os.path.join(
                intermediate_path, f"class_{split}_{split_counters[split]:03}.npz"
            )
            np.savez_compressed(class_file, **classification_combined_data)
            print(
                f"Saved: {class_file} (remaining data) {len(classification_combined_data)}"
            )
            classification_combined_data = {}
        if gen_class or gen_onset:
            intermediate_files_by_split[split].append(
                (onset_file if gen_onset else None, class_file if gen_class else None)
            )
            split_counters[split] += 1

# --- Collect generated intermediate files ---
if args.final_only:
    intermediate_files_by_split = {s: [] for s in splits}
    for split in splits:
        onset_pattern = f"onsets_{split}_"
        class_pattern = f"class_{split}_"
        onset_files = sorted(
            [
                f
                for f in os.listdir(intermediate_path)
                if f.startswith(onset_pattern) and f.endswith(".npz")
            ]
        )
        class_files = sorted(
            [
                f
                for f in os.listdir(intermediate_path)
                if f.startswith(class_pattern) and f.endswith(".npz")
            ]
        )

        # Match onset and class files by batch index
        batch_indexes = sorted(
            set([f.split("_")[-1].split(".")[0] for f in onset_files + class_files])
        )
        for batch_index in batch_indexes:
            onset_file = os.path.join(
                intermediate_path, f"onsets_{split}_{batch_index}.npz"
            )
            class_file = os.path.join(
                intermediate_path, f"class_{split}_{batch_index}.npz"
            )

            onset_file = onset_file if os.path.exists(onset_file) else None
            class_file = class_file if os.path.exists(class_file) else None

            intermediate_files_by_split[split].append((onset_file, class_file))

# --- Final merge ---
if not args.intermediate_only:
    for split in splits:
        final_counter = 0
        for i in range(0, len(intermediate_files_by_split[split]), combine_factor):
            group = intermediate_files_by_split[split][i : i + combine_factor]
            final_onsets, final_classes = {}, {}
            merged_onset_files = []
            merged_class_files = []

            for onset_file, class_file in group:
                if gen_onset and onset_file and os.path.exists(onset_file):
                    with np.load(onset_file, allow_pickle=True) as d:
                        final_onsets.update(d)
                    merged_onset_files.append(onset_file)
                if gen_class and class_file and os.path.exists(class_file):
                    with np.load(class_file, allow_pickle=True) as d:
                        final_classes.update(d)
                    merged_class_files.append(class_file)

            if gen_onset and final_onsets:
                final_onset_path = os.path.join(
                    final_paths[split], f"onsets_batch_{final_counter:03}.npz"
                )
                np.savez_compressed(final_onset_path, **final_onsets)
                print(f"Saved: {final_onset_path}")
                for f in merged_onset_files:
                    os.remove(f)
                    print(f"Deleted: {f}")

            if gen_class and final_classes:
                final_class_path = os.path.join(
                    final_paths[split], f"class_batch_{final_counter:03}.npz"
                )
                np.savez_compressed(final_class_path, **final_classes)
                print(f"Saved: {final_class_path}")
                for f in merged_class_files:
                    os.remove(f)
                    print(f"Deleted: {f}")

            final_counter += 1
            gc.collect()

    # Remove intermediate folder if empty
if os.path.isdir(intermediate_path) and not os.listdir(intermediate_path):
    os.rmdir(intermediate_path)
    print(f"Removed empty intermediate folder: {intermediate_path}")

print(
    "\u2705 Final merge complete: train/valid/test datasets saved in separate folders."
)

from collections import defaultdict
import os
import argparse
import random
import re
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Optional
from dataclasses import dataclass
from random_word import RandomWords
import multiprocessing as mp
import time


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
    intermediate_batch_size: int
    final_batch_size: int
    num_runs: int


def parse_args() -> ArgparseType:
    parser = argparse.ArgumentParser(
        description="Generate Beat Saber dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--type", type=str, default="Easy", help="Difficulty type (e.g. Easy)"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for training subset"
    )
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument(
        "--checkpoint_file", type=str, default=None, help="Optional checkpoint file"
    )
    parser.add_argument(
        "--gen_class_only",
        action="store_true",
        help="Generate classification dataset only",
    )
    parser.add_argument(
        "--gen_onset_only", action="store_true", help="Generate onset dataset only"
    )
    parser.add_argument(
        "--final_only", action="store_true", help="Only perform final merge"
    )
    parser.add_argument(
        "--intermediate_only",
        action="store_true",
        help="Only generate intermediate files",
    )
    parser.add_argument(
        "--intermediate_batch_size",
        type=int,
        default=25,
        help="Intermediate batch size",
    )
    parser.add_argument(
        "--final_batch_size", type=int, default=100, help="Final batch size"
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of times to run the script"
    )

    return ArgparseType(**vars(parser.parse_args()))


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


def process_row(args):
    name, song_steps, npz_path, type, gen_onset, gen_class = args
    if not os.path.exists(npz_path):
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        result = {"name": name, "onsets": {}, "classification": {}}
        grouped_onsets = defaultdict(dict)

        if gen_onset:
            for key in data.files:
                if key not in keys_to_drop:
                    result["onsets"][f"{name}_{key}"] = data[key]
            for full_key, value in result["onsets"].items():
                match = re.match(r"(song[0-9]+_[0-9a-f]+)_(.+)", full_key)
                if match:
                    song_name, sub_key = match.groups()
                    grouped_onsets[song_name][sub_key] = value
                    result["onsets"] = dict(grouped_onsets)
        if gen_class:
            for step in song_steps:
                result["classification"][f"{name}_{step['stack']}_classes"] = (
                    word_to_one_hot(step["combined_word"])
                )
                result["classification"][f"{name}_{step['stack']}_mel"] = (
                    extract_window(data["song"], int(step["stack"]), 45)
                )

        return result
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        return None


def main(args: ArgparseType):
    # Derived logic: if neither class nor onset flags set, do both
    r = RandomWords()
    gen_class = args.gen_class_only or (
        not args.gen_class_only and not args.gen_onset_only
    )
    gen_onset = args.gen_onset_only or (
        not args.gen_class_only and not args.gen_onset_only
    )

    # --- Config ---
    type = args.type
    combine_factor = args.final_batch_size // args.intermediate_batch_size

    base_path = "dataset/beatmaps/color_notes"
    filename = f"{base_path}/notes_dataset/notes_{type.lower()}.parquet"
    npz_dir = f"{base_path}/npz"

    shuffle_seed = random.randint(0, 2**32 - 1)

    base_batch_path = (
        f"dataset/batch/{type.lower()}/{r.get_random_word()}_{shuffle_seed}"
    )

    intermediate_path = f"{base_batch_path}/intermediate"
    final_base_path = base_batch_path

    splits = ["train", "valid", "test"]
    final_paths = {s: os.path.join(final_base_path, s) for s in splits}
    os.makedirs(intermediate_path, exist_ok=True)
    for p in final_paths.values():
        os.makedirs(p, exist_ok=True)
        os.makedirs(os.path.join(p, "onsets"), exist_ok=True)
        os.makedirs(os.path.join(p, "class"), exist_ok=True)

    # --- Optional Checkpointing ---
    processed_names = set()
    if args.checkpoint_file and os.path.exists(args.checkpoint_file):
        with open(args.checkpoint_file, "r") as f:
            processed_names = set(line.strip() for line in f)

    # --- Load Dataset and Split ---
    df = pd.read_parquet(filename)
    df_files = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    full_train_df, test_df = train_test_split(
        df_files, test_size=0.3, random_state=shuffle_seed
    )
    valid_df, test_df = train_test_split(
        test_df, test_size=0.5, random_state=shuffle_seed
    )
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

    if not args.final_only:
        cpu_count = mp.cpu_count()
        for split in splits:
            df_split = df_files[df_files["split"] == split]
            file_counter = 0

            song_steps_by_name = (
                df[["name", "stack", "combined_word"]]
                .groupby("name")
                .apply(
                    lambda group: group[["stack", "combined_word"]].to_dict("records")
                )
                .to_dict()
            )

            args_list = [
                (
                    row.name,
                    song_steps_by_name.get(row.name, []),
                    os.path.join(npz_dir, f"{row.name}.npz"),
                    type,
                    gen_onset,
                    gen_class,
                )
                for row in df_split.itertuples()
                if row.name not in processed_names
            ]

            onsets_combined_data = {}
            classification_combined_data = {}

            with mp.Pool(cpu_count) as pool:
                for result in tqdm(
                    pool.imap_unordered(process_row, args_list),
                    total=len(args_list),
                    desc=f"Processing {split}",
                ):
                    if not result:
                        continue
                    if gen_onset:
                        onsets_combined_data.update(result["onsets"])
                    if gen_class:
                        classification_combined_data.update(result["classification"])
                    if args.checkpoint_file:
                        with open(args.checkpoint_file, "a") as f:
                            f.write(f"{result['name']}\n")
                    file_counter += 1
                    if file_counter % args.intermediate_batch_size == 0:
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
                            np.savez_compressed(
                                class_file, **classification_combined_data
                            )
                            print(f"Saved: {class_file}")
                            classification_combined_data = {}
                        intermediate_files_by_split[split].append(
                            (onset_file, class_file)
                        )
                        split_counters[split] += 1

            # Save leftovers
            onset_file, class_file = None, None
            if gen_onset and onsets_combined_data:
                onset_file = os.path.join(
                    intermediate_path, f"onsets_{split}_{split_counters[split]:03}.npz"
                )
                np.savez_compressed(onset_file, **onsets_combined_data)
                print(f"Saved: {onset_file} (remaining data)")
            if gen_class and classification_combined_data:
                class_file = os.path.join(
                    intermediate_path, f"class_{split}_{split_counters[split]:03}.npz"
                )
                np.savez_compressed(class_file, **classification_combined_data)
                print(f"Saved: {class_file} (remaining data)")
            if gen_class or gen_onset:
                intermediate_files_by_split[split].append((onset_file, class_file))
                split_counters[split] += 1

    # --- Final merge ---
    if not args.intermediate_only:
        for split in splits:
            final_counter = 0
            for i in tqdm(
                range(0, len(intermediate_files_by_split[split]), combine_factor),
                desc=f"Merging final {split}",
                total=len(intermediate_files_by_split[split]) // combine_factor + 1,
            ):
                group = intermediate_files_by_split[split][i : i + combine_factor]
                final_onsets, final_classes = {}, {}
                merged_onset_files, merged_class_files = [], []

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
                        final_paths[split], "onsets", f"batch_{final_counter:03}.npz"
                    )
                    np.savez_compressed(final_onset_path, **final_onsets)
                    print(f"Saved: {final_onset_path}")
                    for f in merged_onset_files:
                        os.remove(f)

                if gen_class and final_classes:
                    final_class_path = os.path.join(
                        final_paths[split], "class", f"batch_{final_counter:03}.npz"
                    )
                    np.savez_compressed(final_class_path, **final_classes)
                    print(f"Saved: {final_class_path}")
                    for f in merged_class_files:
                        os.remove(f)

                final_counter += 1
                gc.collect()

    if os.path.isdir(intermediate_path) and not os.listdir(intermediate_path):
        os.rmdir(intermediate_path)
        print(f"Removed empty intermediate folder: {intermediate_path}")

    print(
        "✅ Final merge complete: train/valid/test datasets saved in separate folders."
    )


if __name__ == "__main__":
    args = parse_args()

    start_time = time.time()

    for i in range(args.num_runs):
        print(
            f"\n🔄 Running dataset generation script (Run {i + 1}/{args.num_runs})..."
        )
        main(args)

    elapsed = (time.time() - start_time) / 60
    print(f"\n⏱️ Finished in {elapsed:.2f} minutes.")
    print("✅ All runs completed successfully.")

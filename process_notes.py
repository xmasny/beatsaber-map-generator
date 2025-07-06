import glob
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
import gc
import psutil
import os
import time
import sys

# === CLI Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing")
parser.add_argument(
    "--skip-existing", action="store_true", help="Skip already processed songs"
)
args = parser.parse_args()

USE_MULTIPROCESSING = args.parallel
SKIP_EXISTING = args.skip_existing

# === Config ===
base_path = Path("dataset/beatmaps/color_notes")
OUTPUT_DIR = base_path / "notes_chunks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 🚨 Log files
SKIPPED_RAM_LOG = "skipped_due_to_ram.txt"
SKIPPED_USER_LOG = "skipped_by_user.txt"


# === Memory Helper ===
def current_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


# === Helpers ===
def add_combined_word_column(df):
    df["word"] = (
        df["c"].replace({0.0: "L", 1.0: "R"}).astype(str)
        + df["d"].astype(int).astype(str)
        + df["x"].astype(int).astype(str)
        + df["y"].astype(int).astype(str)
    )
    df_combined = (
        df.groupby("b")["word"]
        .apply(lambda x: "_".join(sorted(x)))
        .reset_index()
        .rename(columns={"word": "combined_word"})
    )
    return df.merge(df_combined, on="b")


def process_song(row_tuple):
    song_name, upvotes, downvotes, score, bpm = row_tuple
    out_path = OUTPUT_DIR / f"{song_name}.parquet"

    if SKIP_EXISTING and out_path.exists():
        return str(out_path)

    try:
        ram_before = current_ram_mb()

        npz_path = base_path / "npz" / f"{song_name}.npz"
        if not npz_path.exists():
            print(f"⚠️ Missing file: {npz_path}")
            return None

        data = np.load(npz_path, allow_pickle=True)
        notes_dict = data["notes"].item()
        mel = data["song"]
        timestamps = librosa.times_like(mel, sr=22050)

        rows = []
        for level in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
            notes = notes_dict.get(level)
            if notes is None:
                continue

            df_level = pd.DataFrame(notes)
            df_level = add_combined_word_column(df_level)

            beat_time_to_sec = df_level["b"] / float(bpm) * 60
            stack_indices = (
                np.searchsorted(timestamps, beat_time_to_sec, side="right") - 1
            )
            stack_indices = np.clip(stack_indices, 0, len(timestamps) - 1)
            df_level["stack"] = stack_indices

            df_level["name"] = song_name
            df_level["upvotes"] = upvotes
            df_level["downvotes"] = downvotes
            df_level["score"] = score
            df_level["bpm"] = bpm
            df_level["difficulty"] = level

            rows.append(df_level)

            # 🚨 RAM check after each difficulty
            ram_now = current_ram_mb()
            if ram_now - ram_before > 500:
                print(
                    f"🚨 Skipping {song_name} during {level}: RAM grew by {ram_now - ram_before:.2f} MB"
                )
                with open(SKIPPED_RAM_LOG, "a") as f:
                    f.write(f"{song_name}\n")
                return None

        if rows:
            df_out = pd.concat(rows, ignore_index=True)
            df_out.to_parquet(out_path, index=False)
            del df_out, rows
            gc.collect()
            return str(out_path)

    except Exception as e:
        print(f"❌ Failed: {song_name} — {type(e).__name__}: {e}")
        return None


# === Safe wrapper for KeyboardInterrupt skipping ===
last_interrupt_time = [0]


def safe_process_song(row):
    try:
        return process_song(row)
    except KeyboardInterrupt:
        song_name = row[0]
        now = time.time()
        if now - last_interrupt_time[0] < 2:
            print("\n⏹️  Ctrl+C again — exiting.")
            sys.exit(0)
        else:
            print(f"\n⏭️  Skipping {song_name}. Press Ctrl+C again quickly to quit.")
            last_interrupt_time[0] = now  # type: ignore
            with open(SKIPPED_USER_LOG, "a") as f:
                f.write(f"{song_name}\n")
            return None


# === Main script ===
if __name__ == "__main__":
    print("🚀 Starting memory-safe note processing...")

    meta_df = pd.read_parquet(base_path / "metadata.parquet")
    meta_df = (
        meta_df[
            ~meta_df["automapper"]
            & ~meta_df["missing_levels"]
            & ~meta_df["missing_song"]
            & ~meta_df["default_skip"]
            & ~meta_df["incorrect_word"]
        ]
        .drop(["missing_levels", "missing_song", "automapper", "default_skip"], axis=1)
        .reset_index(drop=True)
    )

    print(f"🎧 Processing {len(meta_df)} songs...")

    rows = [
        (row.song, row.upvotes, row.downvotes, row.score, row.bpm)
        for row in meta_df.itertuples(index=False)
    ]

    if USE_MULTIPROCESSING:
        print(
            f"⚙️ Using multiprocessing with up to {multiprocessing.cpu_count()} workers"
        )
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(process_song, rows), total=len(rows), desc="Songs"))
    else:
        print("⚙️ Running single-threaded with Ctrl+C skip support.")
        for row in tqdm(rows, desc="Songs"):
            safe_process_song(row)

    # Combine all per-song parquet files
    print("🧩 Combining .parquet files...")
    parquet_files = glob.glob(str(OUTPUT_DIR / "*.parquet"))
    df_all = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df_all.to_parquet(base_path / "notes.parquet", index=False)
    print("✅ Saved: notes.parquet")

    df_all.columns
    print("✅ Done processing.")

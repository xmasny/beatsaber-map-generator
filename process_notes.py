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


# === Optional: Memory Logger ===
def log_mem(note=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[{note}] RAM: {mem:.2f} MB")


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


def process_song(row):
    try:
        song_name = row.song
        out_path = OUTPUT_DIR / f"{song_name}.parquet"

        if SKIP_EXISTING and out_path.exists():
            return str(out_path)

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

            beat_time_to_sec = df_level["b"] / float(data["bpm"]) * 60
            stack_indices = (
                np.searchsorted(timestamps, beat_time_to_sec, side="right") - 1
            )
            stack_indices = np.clip(stack_indices, 0, len(timestamps) - 1)
            df_level["stack"] = stack_indices

            df_level["name"] = song_name
            df_level["upvotes"] = row.upvotes
            df_level["downvotes"] = row.downvotes
            df_level["score"] = row.score
            df_level["bpm"] = row.bpm
            df_level["difficulty"] = level

            rows.append(df_level)

        if rows:
            df_out = pd.concat(rows, ignore_index=True)
            df_out.to_parquet(out_path, index=False)
            del df_out, rows
            gc.collect()
            return str(out_path)

    except Exception as e:
        print(f"❌ Failed: {row.song} — {type(e).__name__}: {e}")
        return None


# === Main script ===
if __name__ == "__main__":
    print("🚀 Starting memory-safe note processing...")
    log_mem("Start")

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

    log_mem("After filtering metadata")
    print(f"🎧 Processing {len(meta_df)} songs...")

    if USE_MULTIPROCESSING:
        print(f"⚙️ Using multiprocessing with up to 4 workers")
        with ProcessPoolExecutor(max_workers=4) as executor:
            list(
                tqdm(
                    executor.map(process_song, meta_df.itertuples(index=False)),
                    total=len(meta_df),
                    desc="Songs",
                )
            )
    else:
        print("⚙️ Running single-threaded processing.")
        for row in tqdm(meta_df.itertuples(index=False), desc="Songs"):
            process_song(row)

    print("✅ Done processing.")

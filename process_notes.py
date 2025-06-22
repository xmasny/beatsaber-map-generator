import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
import gc

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


def process_song(row_dict):
    row = pd.Series(row_dict)
    song_name = row["song"]
    out_path = OUTPUT_DIR / f"{song_name}.parquet"

    if SKIP_EXISTING and out_path.exists():
        return str(out_path)

    try:
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
            df_level["stack"] = [
                np.abs(timestamps - t).argmin() for t in beat_time_to_sec
            ]

            df_level["name"] = song_name
            df_level["upvotes"] = row["upvotes"]
            df_level["downvotes"] = row["downvotes"]
            df_level["score"] = row["score"]
            df_level["bpm"] = row["bpm"]
            df_level["difficulty"] = level

            rows.append(df_level)

        if rows:
            df_out = pd.concat(rows, ignore_index=True)
            df_out.to_parquet(out_path, index=False)
            del df_out, rows
            gc.collect()
            return str(out_path)

    except Exception as e:
        print(f"❌ Failed: {song_name} — {type(e).__name__}: {e}")
        return None


# === Main script ===
if __name__ == "__main__":
    print("🚀 Starting memory-safe parallel note processing...")

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
    rows = meta_df.to_dict(orient="records")

    if USE_MULTIPROCESSING:
        print(f"⚙️ Using multiprocessing with {multiprocessing.cpu_count()} workers.")
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(process_song, rows), total=len(rows), desc="Songs"))
    else:
        print("⚙️ Running single-threaded processing.")
        for row in tqdm(rows, desc="Songs"):
            process_song(row)

    # Combine all per-song parquet files
    print("🧩 Combining .parquet files...")
    parquet_files = glob.glob(str(OUTPUT_DIR / "*.parquet"))
    df_all = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df_all.to_parquet(base_path / "notes.parquet", index=False)
    print("✅ Saved: notes.parquet")

import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing")
args = parser.parse_args()

USE_MULTIPROCESSING = args.parallel

# === Config ===
CHUNK_SIZE = 100
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


def process_chunk(chunk_df, base_path, chunk_id):
    rows = []

    for _, row in chunk_df.iterrows():
        try:
            data = np.load(base_path / "npz" / f"{row['song']}.npz", allow_pickle=True)
            for level in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
                notes = data["notes"].item().get(level)
                if notes is None:
                    continue

                df_level = pd.DataFrame(notes)
                df_level = add_combined_word_column(df_level)

                mel = data["song"]
                timestamps = librosa.times_like(mel, sr=22050)
                beat_time_to_sec = df_level["b"] / float(data["bpm"]) * 60
                df_level["stack"] = [
                    np.abs(timestamps - t).argmin() for t in beat_time_to_sec
                ]

                df_level["name"] = row["song"]
                df_level["upvotes"] = row["upvotes"]
                df_level["downvotes"] = row["downvotes"]
                df_level["score"] = row["score"]
                df_level["bpm"] = row["bpm"]
                df_level["difficulty"] = level

                rows.append(df_level)
        except FileNotFoundError:
            continue

    if rows:
        df_out = pd.concat(rows, ignore_index=True)
        out_path = OUTPUT_DIR / f"chunk_{chunk_id}.parquet"
        df_out.to_parquet(out_path, index=False)
        return out_path
    return None


def process_chunk_wrapper(args):
    return process_chunk(*args)


def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = df[
        ~df["automapper"]
        & ~df["missing_levels"]
        & ~df["missing_song"]
        & ~df["default_skip"]
        & ~df["incorrect_word"]  # include this filter
    ]
    df = df.drop(
        ["missing_levels", "missing_song", "automapper", "default_skip"], axis=1
    )
    return df


# === Main script ===
if __name__ == "__main__":
    print("🚀 Starting note generation...")

    # Load metadata
    meta_df = pd.read_parquet(base_path / "metadata.parquet")
    meta_df = clean_data(meta_df).reset_index(drop=True)
    # Split into chunks
    song_chunks = [
        meta_df.iloc[i : i + CHUNK_SIZE] for i in range(0, len(meta_df), CHUNK_SIZE)
    ]
    chunk_inputs = [(chunk_df, base_path, i) for i, chunk_df in enumerate(song_chunks)]

    print(f"📦 Processing {len(chunk_inputs)} chunks...")
    if USE_MULTIPROCESSING:
        print(f"⚙️ Using multiprocessing with {multiprocessing.cpu_count()} workers.")
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(
                tqdm(
                    executor.map(process_chunk_wrapper, chunk_inputs),
                    total=len(chunk_inputs),
                    desc="Processing valid chunks",
                )
            )
    else:
        print(
            "⚙️ Running single-threaded chunk processing (set USE_MULTIPROCESSING = True for Linux)."
        )
        results = []
        for args in tqdm(
            chunk_inputs, desc="Processing valid chunks (single-threaded)"
        ):
            results.append(process_chunk_wrapper(args))

    # Combine final output
    files = glob.glob(str(OUTPUT_DIR / "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(base_path / "notes.parquet", index=False)
    print("✅ Combined notes.parquet saved.")

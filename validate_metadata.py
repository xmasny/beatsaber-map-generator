import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing")
args = parser.parse_args()

USE_MULTIPROCESSING = args.parallel

# === Config ===
base_path = Path("dataset/beatmaps/color_notes")
pattern = r"^(?:[LR][0-8][0-3][0-2])(?:_(?:[LR][0-8][0-3][0-2]))*$"


# === Helper functions ===
def is_valid_combined_word(series: pd.Series) -> bool:
    if not series.str.match(pattern).all(bool_only=True):
        return False
    split_words = series.str.split("_")
    if (split_words.apply(len) > 12).any(bool_only=True):
        return False
    if split_words.apply(lambda x: len(x) != len(set(x))).any(bool_only=True):
        return False
    return True


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


def validate_song(index, song_id):
    try:
        data = np.load(base_path / "npz" / f"{song_id}.npz", allow_pickle=True)
        for level in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
            notes = data["notes"].item().get(level)
            if notes is None:
                continue
            df_level = pd.DataFrame(notes)
            df_level = add_combined_word_column(df_level)
            if not is_valid_combined_word(df_level["combined_word"]):
                return index
    except FileNotFoundError:
        return None
    return None


def validate_song_wrapper(args):
    return validate_song(*args)


# === Main script ===
if __name__ == "__main__":
    print("🚀 Starting metadata validation...")

    # Load metadata
    meta_df = pd.read_csv(base_path / "metadata.csv")
    full_meta_df = meta_df.copy()
    full_meta_df["incorrect_word"] = False

    # Filter out songs to skip
    df = meta_df[
        ~meta_df["automapper"]
        & ~meta_df["missing_levels"]
        & ~meta_df["missing_song"]
        & ~meta_df["default_skip"]
    ].copy()

    song_inputs = [(i, row["song"]) for i, row in df.iterrows()]

    # Run validation
    print(f"🔍 Validating {len(song_inputs)} songs...")
    if USE_MULTIPROCESSING:
        print(f"⚙️ Using multiprocessing with {multiprocessing.cpu_count()} workers.")
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(
                tqdm(
                    executor.map(validate_song_wrapper, song_inputs),
                    total=len(song_inputs),
                    desc="Validating beatmaps",
                )
            )
    else:
        print(
            "⚙️ Running single-threaded validation (set USE_MULTIPROCESSING = True for faster runs on Linux)."
        )
        results = []
        for args in tqdm(song_inputs, desc="Validating beatmaps (single-threaded)"):
            results.append(validate_song_wrapper(args))

    # Mark invalid songs
    invalid_indices = [i for i in results if i is not None]
    full_meta_df.loc[invalid_indices, "incorrect_word"] = True

    # Save metadata
    out_path = base_path / "metadata.parquet"
    full_meta_df.to_parquet(out_path, index=False)
    print(f"✅ Validation complete. Invalid songs: {len(invalid_indices)}")
    print(f"✅ Metadata written to {out_path}")

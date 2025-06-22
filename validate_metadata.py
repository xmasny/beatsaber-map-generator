import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import argparse

# === CLI Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing")
args = parser.parse_args()

USE_MULTIPROCESSING = args.parallel
SAVE_EVERY = 1000  # Save progress every N songs
TIMEOUT = 30  # seconds

# === Config ===
base_path = Path("dataset/beatmaps/color_notes")
pattern = r"^(?:[LR][0-8][0-3][0-2])(?:_(?:[LR][0-8][0-3][0-2]))*$"
metadata_path = base_path / "metadata.parquet"


# === Helper Functions ===
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


# === Main Script ===
if __name__ == "__main__":
    print("🚀 Starting metadata validation...")

    if metadata_path.exists():
        meta_df = pd.read_parquet(metadata_path)
    else:
        meta_df = pd.read_csv(base_path / "metadata.csv")

    if "incorrect_word" not in meta_df.columns:
        meta_df["incorrect_word"] = False
    if "validated" not in meta_df.columns:
        meta_df["validated"] = False
    if "timed_out" not in meta_df.columns:
        meta_df["timed_out"] = False

    full_meta_df = meta_df.copy()

    df = meta_df[
        ~meta_df["automapper"]
        & ~meta_df["missing_levels"]
        & ~meta_df["missing_song"]
        & ~meta_df["default_skip"]
        & ~meta_df["validated"]
    ].copy()

    song_inputs = [(i, row["song"]) for i, row in df.iterrows()]

    results = []
    validated_indices = []
    invalid_indices = []
    timed_out_indices = []

    def save_progress():
        if validated_indices:
            full_meta_df.loc[validated_indices, "validated"] = True
        if invalid_indices:
            full_meta_df.loc[invalid_indices, "incorrect_word"] = True
        if timed_out_indices:
            full_meta_df.loc[timed_out_indices, "timed_out"] = True
        full_meta_df.to_parquet(metadata_path, index=False)
        print(f"📅 Saved progress.")
        validated_indices.clear()
        invalid_indices.clear()
        timed_out_indices.clear()

    if USE_MULTIPROCESSING:
        print(f"⚙️ Using multiprocessing with {multiprocessing.cpu_count()} workers.")
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {
                executor.submit(validate_song_wrapper, args): args[0]
                for args in song_inputs
            }
            for i, (future, index) in enumerate(
                tqdm(futures.items(), total=len(futures), desc="Validating beatmaps")
            ):
                try:
                    result = future.result(timeout=TIMEOUT)
                    validated_indices.append(index)
                    if result is not None:
                        invalid_indices.append(result)
                except TimeoutError:
                    print(f"⏱️ Timeout on song index {index}")
                    timed_out_indices.append(index)
                except Exception as e:
                    print(f"⚠️ Error on song index {index}: {e}")
                    validated_indices.append(index)
                if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(futures):
                    save_progress()
    else:
        print("⚙️ Running single-threaded validation.")
        for i, args in enumerate(
            tqdm(song_inputs, desc="Validating beatmaps (single-threaded)")
        ):
            index = args[0]
            try:
                result = validate_song_wrapper(args)
                validated_indices.append(index)
                if result is not None:
                    invalid_indices.append(result)
            except Exception as e:
                print(f"⚠️ Error on song index {index}: {e}")
                validated_indices.append(index)
            if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(song_inputs):
                save_progress()

    print("✅ Validation finished.")

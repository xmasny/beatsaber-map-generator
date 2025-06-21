import glob
from pathlib import Path
import numpy as np
import os
import pandas as pd
import librosa
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

CHUNK_SIZE = 100  # Number of songs per chunk
OUTPUT_DIR = Path("dataset/beatmaps/color_notes/notes_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = df[~df["automapper"]]
    df = df[~df["missing_levels"]]
    df = df[~df["missing_song"]]
    df = df[~df["default_skip"]]

    df = df.drop(
        ["missing_levels", "missing_song", "automapper", "default_skip"], axis=1
    )

    return df


def clean_data_notes(df):
    # Create the 'word' column
    df["word"] = (
        df["c"].replace({0.0: "L", 1.0: "R"}).astype(str)
        + df["d"].astype(int).astype(str)
        + df["x"].astype(int).astype(str)
        + df["y"].astype(int).astype(str)
    )

    # Create a new DataFrame with combined words per beat, without dropping original columns
    df_combined = (
        df.groupby("b")["word"]
        .apply(lambda x: "_".join(sorted(x)))
        .reset_index()
        .rename(columns={"word": "combined_word"})
    )

    # Merge back to original df (optional, if you want both word and combined_word)
    df = df.merge(df_combined, on="b")

    return df


pattern = r"^(?:[LR][0-8][0-3][0-2])(?:_(?:[LR][0-8][0-3][0-2]))*$"

full_meta_df = pd.read_csv("dataset/beatmaps/color_notes/metadata.csv")
full_meta_df["incorrect_word"] = False


def process_chunk(chunk_df, base_path, chunk_id):
    rows = []

    for index, row in chunk_df.iterrows():
        try:
            data = np.load(
                os.path.join(base_path, "npz", f"{row['song']}.npz"), allow_pickle=True
            )

            for level in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
                try:
                    notes = data["notes"].item().get(level)
                    if notes is None:
                        continue

                    df_level = pd.DataFrame(notes)
                    df_level = clean_data_notes(df_level)
                    # If incorrect word pattern
                    if (
                        not df_level["combined_word"]
                        .str.match(pattern)
                        .all(bool_only=True)
                    ):
                        full_meta_df.loc[index, "incorrect_word"] = True
                        continue
                    # If combined_word has more than 12 parts
                    word_split = df_level["combined_word"].str.split("_")
                    if (word_split.apply(len) > 12).any(bool_only=True):
                        full_meta_df.loc[index, "incorrect_word"] = True
                        continue

                    # If combined_word has duplicate parts
                    if word_split.apply(lambda x: len(x) != len(set(x))).any(
                        bool_only=True
                    ):
                        full_meta_df.loc[index, "incorrect_word"] = True
                        continue

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

                    rows.append(
                        df_level[
                            [
                                "name",
                                "upvotes",
                                "downvotes",
                                "bpm",
                                "score",
                                "difficulty",
                                "b",
                                "c",
                                "d",
                                "x",
                                "y",
                                "combined_word",
                                "stack",
                            ]
                        ]
                    )
                except Exception:
                    continue
        except FileNotFoundError:
            continue

    if rows:
        df_out = pd.concat(rows, ignore_index=True)
        out_path = OUTPUT_DIR / f"chunk_{chunk_id}.parquet"
        df_out.to_parquet(out_path, index=False)
        return out_path
    return None


# === Main script ===
if __name__ == "__main__":
    base_path = Path("dataset/beatmaps/color_notes")
    meta_df = pd.read_csv(base_path / "metadata.csv")
    meta_df = clean_data(meta_df)

    song_chunks = [
        meta_df.iloc[i : i + CHUNK_SIZE] for i in range(0, len(meta_df), CHUNK_SIZE)
    ]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(process_chunk, chunk_df, base_path, i)
            for i, chunk_df in enumerate(song_chunks)
        ]

        for _ in tqdm(futures, total=len(futures), desc="Processing song chunks"):
            _.result()

    print("✅ All chunks saved to:", OUTPUT_DIR)

    files = glob.glob("dataset/beatmaps/color_notes/notes_chunks/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet("dataset/beatmaps/color_notes/notes.parquet", index=False)
    print("✅ Combined Parquet written.")

    full_meta_df.to_parquet(
        "dataset/beatmaps/color_notes/metadata.parquet", index=False
    )
    print("✅ Metadata Parquet written.")

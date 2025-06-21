import glob
from pathlib import Path
import numpy as np
import os
import pandas as pd
import librosa
from tqdm import tqdm

CHUNK_SIZE = 100
OUTPUT_DIR = Path("dataset/beatmaps/color_notes/notes_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pattern = r"^(?:[LR][0-8][0-3][0-2])(?:_(?:[LR][0-8][0-3][0-2]))*$"

base_path = Path("dataset/beatmaps/color_notes")
other_meta_df = pd.read_csv(base_path / "metadata.csv")
other_meta_df["incorrect_word"] = False


def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = df[
        ~df["automapper"]
        & ~df["missing_levels"]
        & ~df["missing_song"]
        & ~df["default_skip"]
    ]
    df = df.drop(
        ["missing_levels", "missing_song", "automapper", "default_skip"], axis=1
    )
    return df


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

    df = df.merge(df_combined, on="b")
    return df


def is_valid_combined_word(series: pd.Series) -> bool:
    if not series.str.match(pattern).all(bool_only=True):
        return False
    split_words = series.str.split("_")
    if (split_words.apply(len) > 12).any(bool_only=True):
        return False
    if split_words.apply(lambda x: len(x) != len(set(x))).any(bool_only=True):
        return False
    return True


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
                    df_level = add_combined_word_column(df_level)

                    if not is_valid_combined_word(df_level["combined_word"]):
                        other_meta_df.loc[index, "incorrect_word"] = True
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
    meta_df = pd.read_csv(base_path / "metadata.csv")
    meta_df = clean_data(meta_df)

    song_chunks = [
        meta_df.iloc[i : i + CHUNK_SIZE] for i in range(0, len(meta_df), CHUNK_SIZE)
    ]

    for i, chunk_df in tqdm(
        list(enumerate(song_chunks)),
        total=len(song_chunks),
        desc="Processing song chunks",
    ):
        process_chunk(chunk_df, base_path, i)

    print("✅ All chunks saved to:", OUTPUT_DIR)

    files = glob.glob(str(OUTPUT_DIR / "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet("dataset/beatmaps/color_notes/notes.parquet", index=False)
    print("✅ Combined Parquet written.")

    other_meta_df.to_parquet(
        "dataset/beatmaps/color_notes/metadata.parquet", index=False
    )
    print("✅ Metadata Parquet written.")

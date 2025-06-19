# %%
import numpy as np
import os
import pandas as pd
import librosa


# %%
def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    min_bpm = kwargs.get("min_bpm", 60.0)
    max_bpm = kwargs.get("max_bpm", 300.0)
    min_votes = kwargs.get("min_votes", 100)
    min_score = kwargs.get("min_score", 0.95)

    df = df[~df["automapper"]]
    df = df[~df["missing_levels"]]
    df = df[~df["missing_song"]]
    df = df[~df["default_skip"]]

    df = df.drop(
        ["missing_levels", "missing_song", "automapper", "default_skip"], axis=1
    )

    df = df[(df["bpm"] >= min_bpm) & (df["bpm"] <= max_bpm)]
    df = df[(df["upvotes"] + df["downvotes"]) > min_votes]
    df = df[df["score"] > min_score]

    return df


# %%
meta_df = pd.read_csv("../dataset/beatmaps/color_notes/metadata.csv")
meta_df = clean_data(meta_df)

# %%
base_path = "../dataset/beatmaps/color_notes"


# %%
def clean_data_notes(df):
    # Add column 'word' combining c, d, x, y
    df["word"] = (
        df["c"].replace({0.0: "L", 1.0: "R"}).astype(str)
        + df["d"].astype(int).astype(str)
        + df["x"].astype(int).astype(str)
        + df["y"].astype(int).astype(str)
    )
    df = df.drop(columns=["c", "d", "x", "y"])
    # Combine rows with same 'b' and connect 'word' lexically
    df = df.groupby("b")["word"].apply(lambda x: "_".join(sorted(x))).reset_index()
    return df


# %%
df = pd.DataFrame(columns=["name", "difficulty", "b", "word", "stack"])

# %%
for _, row in meta_df.iterrows():
    try:
        data = np.load(
            os.path.join(base_path, "npz", f"{row['song']}.npz"), allow_pickle=True
        )
        for level in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
            try:
                new_df = pd.DataFrame(data["notes"].item()[level])
                new_df = clean_data_notes(new_df)
                mel = data["song"]
                timestamps = librosa.times_like(mel, sr=22050)
                beat_time_to_sec = new_df["b"] / float(data["bpm"]) * 60
                new_df["stack"] = [
                    np.abs(timestamps - t).argmin() for t in beat_time_to_sec
                ]
                new_df["name"] = row["song"]
                new_df["difficulty"] = level
                df = pd.concat([df, new_df], ignore_index=True)
            except KeyError:
                print(f"Level {level} not found for song {row['song']}")
                continue
            except Exception as e:
                print(f"Error processing song {row['song']} at level {level}: {e}")
                continue
    except FileNotFoundError:
        continue

# %%
df

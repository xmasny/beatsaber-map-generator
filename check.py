import numpy as np
import os
import pandas as pd

from tqdm import tqdm

base_path = "dataset/beatmaps/color_notes"

df = pd.read_csv(os.path.join(base_path, "combined_songs.csv"))
df["missing_song"] = False
df["missing_levels"] = False

path = os.path.join(base_path, "npz")

for index, song in tqdm(df.iterrows(), total=len(df)):
    try:
        data = np.load(os.path.join(path, song["song"]), allow_pickle=True)
        if "song" not in data:
            song["missing_song"] = True

        difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
        if not any(diff in data for diff in difficulties):
            song["missing_levels"] = True

    except FileNotFoundError:
        song["missing_song"] = True
        song["missing_levels"] = True
        continue
    except Exception as e:
        print(f"Error processing {song['song']}: {e}")
        song["missing_song"] = True
        song["missing_levels"] = True
        continue

df.to_csv(os.path.join(base_path, "combined_songs.csv"), index=False)

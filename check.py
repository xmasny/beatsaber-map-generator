import numpy as np
import os
import pandas as pd

from tqdm import tqdm

df = pd.read_csv("dataset/beatmaps/color_notes/combined_songs.csv")
df["missing_song"] = False
df["missing_levels"] = False

path = "dataset/beatmaps/color_notes/npz"
for index, song in tqdm(df.iterrows(), total=len(df)):
    data = np.load(os.path.join(path, song["song"]), allow_pickle=True)
    if "song" not in data:
        song["missing_song"] = True

    difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
    if not any(diff in data for diff in difficulties):
        song["missing_levels"] = True

df.to_csv("dataset/beatmaps/color_notes/combined_songs.csv", index=False)

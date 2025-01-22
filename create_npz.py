import os

import numpy as np
import pandas as pd
from tqdm import tqdm

os.makedirs("dataset/beatmaps/color_notes/npz", exist_ok=True)
difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]

df = pd.read_csv("dataset/beatmaps/color_notes/combined_songs.csv", header=0)

progress_bar = tqdm(total=len(df))

def load_npy(song):
    key = song["song"].split(".")[0]

    data_dict = {}

    for difficulty in difficulties:
        try:
            if song[difficulty]:
                data_dict[difficulty] = np.load(f"dataset/beatmaps/color_notes/{difficulty}/{key}.npy")
        except FileNotFoundError:
            pass
        try:
            data_dict["song"] = np.load(f"dataset/songs/mel229/{key}.npy", allow_pickle=True)
        except FileNotFoundError:
            pass

    if data_dict != {}:
        np.savez(f"dataset/beatmaps/color_notes/npz/{key}.npz", **data_dict)
    progress_bar.update(1)

df.apply(load_npy, axis=1)
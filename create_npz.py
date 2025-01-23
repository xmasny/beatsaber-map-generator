import logging
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", filename="create_npz.log"
)

difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
types = ["color_notes", "obstacles", "bomb_notes"]

for object_type in types:

    os.makedirs(f"dataset/beatmaps/{object_type}/npz", exist_ok=True)

    df = pd.read_csv(f"dataset/beatmaps/{object_type}/combined_songs.csv", header=0)

    progress_bar = tqdm(total=len(df), desc=f"Creating {object_type} npz files")


    def load_npy(song):
        if os.path.exists(f"dataset/beatmaps/{object_type}/npz/{song['song'].split('.')[0]}.npz"):
            progress_bar.update(1)
            return

        key = song["song"].split(".")[0]

        data_dict = {}

        for difficulty in difficulties:
            try:
                if song[difficulty]:
                    data_dict[difficulty] = np.load(f"dataset/beatmaps/{object_type}/{difficulty}/{key}.npy")
            except FileNotFoundError:
                logging.error(f"File not found: {key}")
            try:
                data_dict["song"] = np.load(f"dataset/songs/mel229/{key}.npy", allow_pickle=True)
            except FileNotFoundError:
                logging.error(f"File not found: {key}")
            except pickle.UnpicklingError:
                try:
                    data_dict["song"] = np.load(f"dataset/songs/mel229/{key}.npy")
                except Exception as e:
                    logging.error(f"Error loading {key}: {e}")
                    print(f"Error loading {key}: {e}")
                    continue
        if data_dict != {}:
            np.savez(f"dataset/beatmaps/{object_type}/npz/{key}.npz", **data_dict)
        progress_bar.update(1)


    df.apply(load_npy, axis=1)

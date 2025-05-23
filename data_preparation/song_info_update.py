import json
import logging
import os
import re
from csv import QUOTE_NONNUMERIC

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(filename="song_info_update.log", level=logging.ERROR)

for type in ("color_notes", ):

    path = f"dataset/beatmaps/{type}"

    for file in os.listdir(f"dataset/beatmaps/{type}"):

        if file.endswith(".csv"):
            diff_name = file.split(".")[0]
            with open(os.path.join(path, file)) as csvfile:
                df = pd.read_csv(csvfile, header=None, names=["song", "npz file"])

                df["upvotes"] = 0
                df["downvotes"] = 0
                df["score"] = 0.0
                df["bpm"] = 0.0
                df["duration"] = 0
                df["automapper"] = False
                df[diff_name] = True

            for index, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Updating {file}"
            ):
                split = re.split(r"_|\.", df.at[index, "song"])
                response = requests.get(f"https://api.beatsaver.com/maps/id/{split[1]}")
                if response.status_code == 200:
                    data = response.json()

                    df.at[index, "upvotes"] = data["stats"]["upvotes"]
                    df.at[index, "downvotes"] = data["stats"]["downvotes"]
                    df.at[index, "score"] = data["stats"]["score"]
                    df.at[index, "bpm"] = data["metadata"]["bpm"]
                    df.at[index, "duration"] = data["metadata"]["duration"]
                    df.at[index, "automapper"] = data["automapper"]
                else:
                    logging.error(
                        f"Error: {response.status_code} - {response.text} - {df.at[index, 'song']}"
                    )
                    try:
                        with open(f"data/{split[0]}/generated/SongInfo.dat") as f:
                            data = json.load(f)
                        df.at[index, "upvotes"] = data["stats"]["upvotes"]
                        df.at[index, "downvotes"] = data["stats"]["downvotes"]
                        df.at[index, "score"] = data["stats"]["score"]
                        df.at[index, "bpm"] = data["metadata"]["bpm"]
                        df.at[index, "duration"] = data["metadata"]["duration"]
                        df.at[index, "automapper"] = data["automapper"]
                    except Exception as e:
                        logging.error(f"Error: {e} - {df.at[index, 'song']}")
                        continue
            df.drop('npz file', axis=1, inplace=True)
            df.to_csv(
                os.path.join(path, file),
                index=False,
                header=True,
                quoting=QUOTE_NONNUMERIC,
            )

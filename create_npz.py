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
types = ["color_notes"]

for object_type in types:

    df_metadata = pd.read_csv(f"dataset/beatmaps/{object_type}/metadata.csv", header=0)
    df_filtered = df_metadata[~df_metadata["automapper"]].copy()
    df_issues = pd.read_csv(
        f"dataset/beatmaps/{object_type}/onset_validation_issues.csv", header=0
    )

    df_issues = pd.DataFrame(df_issues["song"].drop_duplicates())

    for index, issue in tqdm(df_issues.iterrows(), total=len(df_issues)):

        if os.path.exists(f"dataset/beatmaps/{object_type}/npz/{issue['song']}.npz"):
            os.remove(f"dataset/beatmaps/{object_type}/npz/{issue['song']}.npz")

        song = df_filtered[df_filtered["song"] == issue["song"]].iloc[0]

        key = song["song"]

        data_dict = {}

        for difficulty in difficulties:
            try:
                if song[difficulty]:
                    data_dict[difficulty] = np.load(
                        f"dataset/beatmaps/{object_type}/{difficulty}/{key}.npy"
                    )
            except FileNotFoundError:
                logging.error(f"Map File not found: {key}, {difficulty}")
            except ValueError:
                try:
                    data_dict[difficulty] = np.load(
                        f"dataset/beatmaps/{object_type}/{difficulty}/{key}.npy",
                        allow_pickle=True,
                    )
                except pickle.UnpicklingError as e:
                    logging.error(f"Error loading map {key}, {difficulty}: {e}")
                    print(f"Error loading map {key}, {difficulty}: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Error loading map {key}, {difficulty}: {e}")
                    print(f"Error loading map {key}, {difficulty}: {e}")
                    continue
            except Exception as e:
                logging.error(f"Error loading map {key}, {difficulty}: {e}")
                print(f"Error loading map {key}, {difficulty}: {e}")
                continue
        try:
            data_dict["song"] = np.load(
                f"dataset/songs/mel229/{key}.npy", allow_pickle=True
            )
        except FileNotFoundError:
            logging.error(f"Mel File not found: {key}")
        except ValueError as e:
            logging.error(f"Error loading mel file {key}: {e}")
            print(f"Error loading mel file {key}: {e}")

        except pickle.UnpicklingError:
            try:
                data_dict["song"] = np.load(f"dataset/songs/mel229/{key}.npy")
            except Exception as e:
                logging.error(f"Error loading mel file {key}: {e}")
                print(f"Error loading mel file {key}: {e}")
        except Exception as e:
            logging.error(f"Error loading mel file {key}: {e}")
            print(f"Error loading mel file {key}: {e}")

        if data_dict != {}:
            np.savez(f"dataset/beatmaps/{object_type}/npz/{key}.npz", **data_dict)

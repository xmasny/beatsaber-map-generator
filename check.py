import numpy as np
import os
import pandas as pd
from tqdm import tqdm

for type in ["color_notes"]:

    base_path = f"dataset/beatmaps/{type}/"
    df = pd.read_csv(os.path.join(base_path, "metadata.csv"))
    df["missing_song"] = False
    df["missing_levels"] = False
    df["frames"] = 0
    df["npz_size_mb"] = 0.0

    path = os.path.join(base_path, "npz")

    for index, song in tqdm(df.iterrows(), total=len(df)):
        try:
            # Load the .npz file
            data = np.load(os.path.join(path, song["song"]), allow_pickle=True)

            # Check if the song file is missing
            if "song" not in data:
                df.at[index, "missing_song"] = True
            else:
                df.at[index, "frames"] = np.shape(data["song"])[1]
                file_size = os.path.getsize(
                    os.path.join(path, song["song"])
                )  # in bytes
                size_mb = file_size / (1024 * 1024)
                df.at[index, "npz_size_mb"] = round(size_mb, 3)

            # Check if all difficulty levels are missing
            difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
            if not any(diff in data for diff in difficulties):
                df.at[index, "missing_levels"] = True

        except FileNotFoundError:
            # If the .npz file doesn't exist, mark as missing
            df.at[index, "missing_song"] = True
            df.at[index, "missing_levels"] = True
        except Exception as e:
            # Catch any other exceptions and mark as missing
            print(f"Error processing {song['song']}: {e}")
            df.at[index, "missing_song"] = True
            df.at[index, "missing_levels"] = True

        # Optionally save progress every N rows (cast index to int)
        if int(index) % 100 == 0:  # type: ignore
            df.to_csv(os.path.join(base_path, "metadata.csv"), index=False)

    # Final save
    df.to_csv(os.path.join(base_path, "metadata.csv"), index=False)

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_song(row, path):
    index, song = row
    result = {
        "index": index,
        "missing_song": False,
        "missing_levels": False,
        "npz_size_mb": 0.0,
    }
    try:
        npz_path = os.path.join(path, song["song"] + ".npz")
        data = np.load(npz_path, allow_pickle=True)

        if "song" not in data:
            result["missing_song"] = True
        else:
            file_size = os.path.getsize(npz_path)
            result["npz_size_mb"] = round(file_size / (1024 * 1024), 3)

        difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
        if not any(diff in data for diff in difficulties):
            result["missing_levels"] = True

    except FileNotFoundError:
        result["missing_song"] = True
        result["missing_levels"] = True
    except Exception as e:
        print(f"Error processing {song['song']}: {e}")
        result["missing_song"] = True
        result["missing_levels"] = True

    return result


for type in ["color_notes"]:
    base_path = f"dataset/beatmaps/{type}/"
    df = pd.read_csv(os.path.join(base_path, "metadata.csv"))
    path = os.path.join(base_path, "npz")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_song, row, path) for row in df.iterrows()]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            i = result["index"]
            df.at[i, "missing_song"] = result["missing_song"]
            df.at[i, "missing_levels"] = result["missing_levels"]
            df.at[i, "npz_size_mb"] = result["npz_size_mb"]

    df.to_csv(os.path.join(base_path, "metadata.csv"), index=False)

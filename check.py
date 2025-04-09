import numpy as np
import os

path = "dataset/beatmaps/color_notes/npz"
for filename in os.listdir(path):
    if filename.endswith(".npz"):
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        if "song" not in data or len(data) < 2:
            print(f"Invalid file: {filename}, keys found: {list(data.keys())}")

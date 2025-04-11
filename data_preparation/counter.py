import os
import numpy as np
from tqdm import tqdm

pbar = tqdm(os.listdir("dataset/beatmaps/color_notes"))
axes = []

for file in pbar:
    if file.endswith(".npz"):
        # Load mel spectrogram and object data
        object_array = np.load("dataset/beatmaps/color_notes/" + file)

        for obj in object_array:
            axes.append(obj[0])

print(np.unique(axes))

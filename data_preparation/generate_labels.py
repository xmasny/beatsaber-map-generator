import os
import numpy as np

labels = {
    "color_notes": [],
    "bomb_notes": [],
    "obstacles": [],
}

for c in range(2):
    for d in range(9):
        for x in range(4):
            for y in range(3):
                labels["color_notes"].append([c, d, x, y])

for x in range(4):
    for y in range(3):
        labels["bomb_notes"].append([x, y])

np.save("dataset/color_notes.npy", labels["color_notes"])
np.save("dataset/bomb_notes.npy", labels["bomb_notes"])
np.save("dataset/obstacles.npy", labels["obstacles"])

color_notes = np.load("dataset/color_notes.npy")
bomb_notes = np.load("dataset/bomb_notes.npy")
obstacles = np.load("dataset/obstacles.npy")

data = {
    "color_notes": color_notes,
    "bomb_notes": bomb_notes,
    "obstacles": obstacles,
}

np.savez("dataset/labels.npz", **data)

os.remove("dataset/color_notes.npy")
os.remove("dataset/bomb_notes.npy")
os.remove("dataset/obstacles.npy")

import numpy as np

labels = []

for c in range(2):
    for d in range(9):
        for x in range(4):
            for y in range(3):
                labels.append([c, d, x, y])

labels = np.array(labels)

np.savez("dataset/beatmaps/color_notes/labels.npz", labels)

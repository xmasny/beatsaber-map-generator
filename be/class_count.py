import numpy as np
import os
from tqdm import tqdm

count = {
    "train_class_count": np.zeros((3, 4, 19), dtype=np.int32),
    "train_iterations": 0,
    "validation_class_count": np.zeros((3, 4, 19), dtype=np.int32),
    "validation_iterations": 0,
    "test_class_count": np.zeros((3, 4, 19), dtype=np.int32),
    "test_iterations": 0,
}

sweep = input("Enter the sweep: ")

shuffle_folder = f"dataset/batch/easy/{sweep}"

for type in ["train", "validation", "test"]:
    folder = f"{shuffle_folder}/{type}/class"
    files = os.listdir(folder)
    for npz_file in tqdm(files, position=0, desc=f"Processing {type} files"):
        data = np.load(f"{folder}/{npz_file}")
        matching_keys = [k for k in data if "_classes" in k]
        count[f"{type}_iterations"] += len(matching_keys)
        for file in tqdm(matching_keys, position=1, desc=f"Processing {npz_file}"):
            count[f"{type}_class_count"] += np.int32(data[file])

np.save(f"{shuffle_folder}/class_count.npy", np.asarray(count))

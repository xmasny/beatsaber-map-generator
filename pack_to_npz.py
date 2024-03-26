import os
import numpy as np
from tqdm import tqdm
import wandb
import csv

wandb.init(project="beat-saber-map-generator", tags=["pack"])


# Function to get files from folders
def get_files_from_folders(folders):
    files = []
    for folder in folders:
        for root, _, filenames in os.walk(folder):
            files.extend([os.path.join(root, filename) for filename in filenames])
    return files

# Function to load NumPy arrays from files
def load_arrays_from_files(files, npz_filename):
    songs_folder = "dataset/songs/"
    arrays = {
        "beatmaps": {},
        "songs": {},
    }
    folder = os.path.dirname(npz_filename)
    with open(f"{folder}.csv", "a", newline='') as csvf:
        with tqdm(total=len(files), desc=f"Packing {npz_filename}") as pbar:
            for file in files:
                array_name = os.path.basename(file)
                arrays["songs"][array_name] = np.load(songs_folder + array_name)
                arrays["beatmaps"][array_name] = np.load(file)
                writer = csv.writer(csvf, dialect='unix')
                writer.writerow([array_name, npz_filename])
                pbar.update(1)
    return arrays

# Function to save arrays into an NPZ file
def save_arrays_to_npz(arrays, npz_filename):
    np.savez(npz_filename, **arrays)

# List of folders containing files to pack
folders = [
    # "dataset/beatmaps/color_notes/Easy",
    # "dataset/beatmaps/color_notes/Normal",
    # "dataset/beatmaps/color_notes/Hard",
    # "dataset/beatmaps/color_notes/Expert",
    # "dataset/beatmaps/color_notes/ExpertPlus",
    "dataset/beatmaps/bomb_notes/Easy",
    "dataset/beatmaps/bomb_notes/Normal",
    "dataset/beatmaps/bomb_notes/Hard",
    "dataset/beatmaps/bomb_notes/Expert",
    "dataset/beatmaps/bomb_notes/ExpertPlus",
    # "dataset/beatmaps/obstacles/Easy",
    # "dataset/beatmaps/obstacles/Normal",
    # "dataset/beatmaps/obstacles/Hard",
    # "dataset/beatmaps/obstacles/Expert",
    # "dataset/beatmaps/obstacles/ExpertPlus",
]

# Number of files to pack into each npz file
files_per_npz = 1000

# Iterate over folders, get files, and pack them into npz files
for folder in folders:
    # Get files from the folder
    files = get_files_from_folders([folder])
    # Split files into chunks of size files_per_tar
    
    with tqdm(total=len(files), desc=f"Packing folder {folder}") as pbar:
        for i in range(0, len(files), files_per_npz):
            # Generate npz filename
            subfolder = folder.split("/")[-1]
            npz_filename = f"{folder}/{subfolder}_{i // files_per_npz}.npz"
            # Get files for this iteration
            files_for_this_npz = files[i : i + files_per_npz]
            # Load NumPy arrays from files
            arrays = load_arrays_from_files(files_for_this_npz, npz_filename)
            # Save arrays into NPZ file
            save_arrays_to_npz(arrays, npz_filename)
            pbar.update(len(files_for_this_npz))

wandb.finish()
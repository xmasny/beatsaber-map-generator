import tarfile
import os
from tqdm import tqdm


# Function to pack files into tar files
def pack_files_into_tar(files, tar_filename):
    with tarfile.open(tar_filename, "w") as tar:
        with tqdm(total=len(files), desc=f"Packing {tar_filename}") as pbar:
            for file in files:
                if os.path.exists(
                    f"dataset/songs/{os.path.basename(file)}"
                ) and os.path.exists(file):
                    tar.add(
                        file, arcname=os.path.join("beatmaps", os.path.basename(file))
                    )
                    tar.add(
                        f"dataset/songs/{os.path.basename(file)}",
                        arcname=os.path.join("songs", os.path.basename(file)),
                    )
                pbar.update(1)


# Function to get files from folders
def get_files_from_folders(folders):
    files = []
    for folder in folders:
        for root, _, filenames in os.walk(folder):
            files.extend([os.path.join(root, filename) for filename in filenames])
    return files


# List of folders containing files to pack
folders = [
    "dataset/beatmaps/color_notes/Easy",
    "dataset/beatmaps/color_notes/Normal",
    "dataset/beatmaps/color_notes/Hard",
    "dataset/beatmaps/color_notes/Expert",
    "dataset/beatmaps/color_notes/ExpertPlus",
    "dataset/beatmaps/bomb_notes/Easy",
    "dataset/beatmaps/bomb_notes/Normal",
    "dataset/beatmaps/bomb_notes/Hard",
    "dataset/beatmaps/bomb_notes/Expert",
    "dataset/beatmaps/bomb_notes/ExpertPlus",
    "dataset/beatmaps/obstacles/Easy",
    "dataset/beatmaps/obstacles/Normal",
    "dataset/beatmaps/obstacles/Hard",
    "dataset/beatmaps/obstacles/Expert",
    "dataset/beatmaps/obstacles/ExpertPlus",
]

# Number of files to pack into each tar file
files_per_tar = 1000

# Iterate over folders, get files, and pack them into tar files
for folder in folders:
    # Get files from the folder
    files = get_files_from_folders([folder])
    # Split files into chunks of size files_per_tar
    for i in range(0, len(files), files_per_tar):
        # Generate tar filename
        subfolder = folder.split("/")[-1]
        tar_filename = f"{folder}/{subfolder}_{i // files_per_tar}.tar"
        # Get files for this iteration
        files_for_this_tar = files[i : i + files_per_tar]
        # Pack files into tar file
        pack_files_into_tar(files_for_this_tar, tar_filename)

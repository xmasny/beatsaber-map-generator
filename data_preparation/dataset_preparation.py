import os
import shutil
import zipfile
import pandas as pd
from tqdm import tqdm
import wandb

all_difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]


def move_and_zip_selected_type_difficulty():
    while True:
        try:
            object_type = input("Choose object type: ")
            difficulty = "color_notes"

            df = pd.read_csv(
                f"dataset/beatmaps/{object_type}/{difficulty}.csv",
                header=None,
            ).values.tolist()

            if not os.path.exists(f"dataset/beatmaps/{object_type}/{difficulty}"):
                raise Exception("Path does not exist, please try again.")

            if not os.path.exists(f"dataset/beatmaps/{object_type}/{difficulty}/mels"):
                os.makedirs(f"dataset/beatmaps/{object_type}/{difficulty}/mels")

            for filename in tqdm(df, desc=f"Copying {object_type} {difficulty}"):
                shutil.copy(
                    f"dataset/songs/mel229/{filename[0]}",
                    f"dataset/beatmaps/{object_type}/{difficulty}/mels/{filename[0]}",
                )

        except Exception as e:
            print(e)
            continue

        if input(f"Do you want to zip {object_type} {difficulty}? (y/n): ") == "y":
            zip_folder(object_type, difficulty)

        if input("Do you want to zip different files? (y/n): ") == "n":
            break


def zip_folder(object_type, difficulty):
    source_folder = f"dataset/beatmaps/{object_type}/{difficulty}"
    total_files = sum(
        [len(files) for _, _, files in os.walk(source_folder)]
    )  # Count total number of files

    with zipfile.ZipFile(
        f"dataset/beatmaps/{object_type}_{difficulty}.zip", "w"
    ) as zipf:
        # Wrap os.walk with tqdm to track progress
        with tqdm(
            total=total_files, desc=f"Zipping {object_type} {difficulty}"
        ) as pbar:
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    if not file.endswith(".zip"):
                        zipf.write(
                            os.path.join(root, file),
                            os.path.relpath(os.path.join(root, file), source_folder),
                        )
                    pbar.update(1)

    # Remove the specified directory
    shutil.rmtree(f"{source_folder}/mels")


def move_and_zip_all_difficulties():
    object_type = input("Choose object type: ")
    zip_all = input("Do you want to zip all difficulties? (y/n): ")

    for difficulty in all_difficulties:
        if not os.path.exists(f"dataset/beatmaps/{object_type}/{difficulty}/mels"):
            os.makedirs(f"dataset/beatmaps/{object_type}/{difficulty}/mels")
        for filename in tqdm(
            os.listdir(f"dataset/beatmaps/{object_type}/{difficulty}"),
            desc=f"Copying {object_type} {difficulty}",
        ):
            if filename.endswith(".npy"):
                shutil.copy(
                    f"dataset/songs/{filename}",
                    f"dataset/beatmaps/{object_type}/{difficulty}/mels/{filename}",
                )
        if zip_all == "y":
            if input(f"Do you want to zip {object_type} {difficulty}? (y/n): ") == "y":
                zip_folder(object_type, difficulty)


if __name__ == "__main__":
    wandb.init(project="beat-saber-map-generator")

    select = input("Do you want to zip selected difficulties and objects? (y/n): ")
    if select == "y":
        move_and_zip_selected_type_difficulty()
    else:
        move_and_zip_all_difficulties()

    print("Done zipping!")
    wandb.finish()

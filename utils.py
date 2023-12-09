import json
import os
import collections


def print_scripts():
    print("All scripts: ")
    print("--------------------")
    print("Get all users: 1")
    print("Get all maps: 2")
    print("Get all maps urls: 3")
    print("Get all maps from user: 4")
    print("Zips work: 5")
    print("Save song info: 6")
    print("Get song versions: 7")
    print("Get song info versions: 8")
    print("Generate v3 beatmap: 9")
    print("Generate & save mel spectrogram: 10")
    print("Remove pics: 11")
    print("Folders to zip: 12")
    print("Save filenames to json: 13")


def create_all_data_dirs_json(filename):
    print("Creating all_data_dirs.json...")
    files = os.listdir("data")

    sorted_songs = sorted(files, key=extract_number)

    with open(f"saved_data/{filename}.json", "w") as file:
        file.write(json.dumps(sorted_songs))
        print(f"{filename} saved")


def extract_number(song):
    return int("".join(filter(str.isdigit, song)))


def remove_pics():
    print("Removing pngs...")
    folders = os.listdir("data")
    for folder in folders:
        for file in os.listdir(f"data/{folder}"):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(f"data/{folder}/{file}")

        if extract_number(folder) % 100 == 0:
            print(f"Removed pics from {folder}")


def get_all_filenames(directory):
    all_filenames = []

    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            full_path = os.path.join(foldername, filename)
            # Extract only the filename from the full path
            file_only = os.path.basename(full_path)
            all_filenames.append(file_only)

    return all_filenames


def save_filenames_to_json():
    directory_path = "data"
    filenames = get_all_filenames(directory_path)
    filenames = collections.Counter(filenames)

    # Save the filenames to a JSON file
    json_filename = "saved_data/filenames.json"
    with open(json_filename, "w") as json_file:
        json.dump(filenames, json_file)

    print(f"Filenames saved to {json_filename}")

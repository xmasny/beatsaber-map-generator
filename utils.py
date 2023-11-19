import json
import os


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

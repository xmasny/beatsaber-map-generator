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
    print("Save all full filenames to json: 14")
    print("Get maps by characteristic and difficuly: 15")


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


def get_all_filenames(directory="data"):
    all_filenames = []

    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.endswith(
                (".dat", ".json", ".txt")
            ) or filename.lower().endswith(
                ("info.json", "info.dat", "info - copy.dat")
            ):
                continue
            full_path = os.path.join(foldername, filename)
            # Extract only the filename from the full path
            file_only = os.path.basename(full_path)
            all_filenames.append(file_only)

    filenames = collections.Counter(all_filenames)

    # Save the filenames to a JSON file
    json_filename = "saved_data/filenames.json"
    with open(json_filename, "w") as json_file:
        json.dump(all_filenames, json_file)

    print(f"Filenames saved to {json_filename}")


def get_all_filenames_full_route(directory="data"):
    all_filenames = []

    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.endswith((".dat", ".json", ".txt")):
                continue
            full_path = os.path.join(foldername, filename)
            all_filenames.append(full_path)

    # Save the filenames to a JSON file as an array
    json_filename = "saved_data/filenamesfullroute.json"
    with open(json_filename, "w") as json_file:
        json.dump(all_filenames, json_file, indent=2)

    print(f"Filenames saved to {json_filename}")

def get_maps_by_characteristic_and_difficulty(directory="data"):
    song_levels = {}    

    for foldername, _, filenames in os.walk(directory):
        for filename in filenames:
            if "info" in filename.lower() and not filename.endswith(("BPMInfo.dat", "SongInfo.dat")):
                with open(os.path.join(foldername, filename), "r") as file:
                    song_info = json.load(file)
                    
                    with open(os.path.join(foldername,"generated", "SongInfo.dat"), "r") as file:
                        song_info_extra = json.load(file)
                    
                    if song_info.get("_difficultyBeatmapSets"):
                        song_levels[foldername[5:]] = {}
                        song_levels[foldername[5:]]["songFilename"] = song_info["_songFilename"]
                        song_levels[foldername[5:]]["songId"] = song_info_extra["id"]
                        song_levels[foldername[5:]]["difficultySet"] = {}
                        
                        for beatmap_set in song_info["_difficultyBeatmapSets"]:
                            characteristic_name = beatmap_set["_beatmapCharacteristicName"]
                            song_levels[foldername[5:]]["difficultySet"][characteristic_name] = {}

                            for difficulty in beatmap_set["_difficultyBeatmaps"]:
                                difficulty_name = difficulty["_difficulty"]
                                
                                song_levels[foldername[5:]]["difficultySet"][characteristic_name][difficulty_name] = difficulty["_beatmapFilename"]
                                print(f"Song {foldername[5:]} added to song_levels")
    
    sorted_song_levels = dict(sorted(song_levels.items(), key=lambda x: int(x[0][4:])))
    
    output_path = "saved_data/song_levels.json"
    with open(output_path, "w") as file:
        json.dump(sorted_song_levels, file, indent=2)
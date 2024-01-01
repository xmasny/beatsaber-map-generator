import json
import os
from tqdm import tqdm
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
    print("Create beatmap arrays and save: 9")
    print("Generate & save mel spectrogram: 10")
    print("Folders to zip: 11")
    print("Get maps by characteristic and difficulty: 12")
    print("Get all song files: 13")


def create_all_data_dirs_json(filename):
    print("Creating all_data_dirs.json...")
    files = os.listdir("data")

    sorted_songs = sorted(files, key=extract_number)

    with open(f"saved_data/{filename}.json", "w") as file:
        file.write(json.dumps(sorted_songs))
        print(f"{filename} saved")

def extract_number(song):
    return int(song[0][4:])


def get_maps_by_characteristic_and_difficulty(directory="data"):
    song_levels = {}  
    
    progress_bar = tqdm(os.listdir(directory))

    for foldername in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, foldername)):
            if "info" in filename.lower() and not filename.endswith(("BPMInfo.dat", "SongInfo.dat")):
                try:
                    with open(os.path.join(directory, foldername, filename), "r") as file:
                        song_info = json.load(file)
                        
                    with open(os.path.join(directory, foldername,"generated", "SongInfo.dat"), "r") as file:
                        song_info_extra = json.load(file)
                        
                        if song_info.get("_difficultyBeatmapSets"):
                            song_levels[foldername] = {}
                            song_levels[foldername]["songFilename"] = song_info["_songFilename"]
                            song_levels[foldername]["songId"] = song_info_extra["id"]
                            song_levels[foldername]["difficultySet"] = {}
                            
                            for beatmap_set in song_info["_difficultyBeatmapSets"]:
                                characteristic_name = beatmap_set["_beatmapCharacteristicName"]
                                song_levels[foldername]["difficultySet"][characteristic_name] = {}

                                for difficulty in beatmap_set["_difficultyBeatmaps"]:
                                    difficulty_name = difficulty["_difficulty"]
                                    
                                    song_levels[foldername]["difficultySet"][characteristic_name][difficulty_name] = difficulty["_beatmapFilename"]
                            progress_bar.update(1)
                            break
                except Exception as e:
                    print(e)
                    print(foldername + filename)
                    progress_bar.update(1)
                    continue
    sorted_song_levels = dict(sorted(song_levels.items(), key=extract_number))
    
    output_path = "saved_data/song_levels.json"
    with open(output_path, "w") as file:
        json.dump(sorted_song_levels, file, indent=2)
        
def get_all_song_files(directory="data"):
    song_files = []
    with open("saved_data/song_levels.json", "r") as file:
        song_levels = json.load(file)
    
    for song in song_levels:
        song_files.append((song, song_levels[song]["songFilename"], song_levels[song]["songId"]))
    
    with open("saved_data/song_files.json", "w") as file:
        json.dump(song_files, file)
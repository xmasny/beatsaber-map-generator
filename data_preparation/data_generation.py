import csv
from io import TextIOWrapper
import json
import os
import traceback
from collections import Counter
import zipfile
import pandas as pd
from tqdm import tqdm

import librosa
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    filename="outputs/data_generation.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


from utils import create_all_data_dirs_json


class DataGeneration:
    def __init__(self, file: TextIOWrapper):
        self.terminal_file = file

        with open("saved_data/map_info.json", "r", encoding="utf-8") as f:
            self.map_info = json.load(f)

        with open("saved_data/map_zips.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            self.map_csv = list(reader)

    def save_song_info(self):
        for song in self.map_csv:
            if song[0] in os.listdir("data"):
                song_info = [id for id in self.map_info if id["id"] == song[1]]

                if "generated" not in os.listdir(f"data/{song[0]}"):
                    os.mkdir(f"data/{song[0]}/generated")

                with open(f"data/{song[0]}/generated/SongInfo.dat", "w") as f:
                    json.dump(song_info[0], f)
                    message = f"{song[0]} SongInfo.dat saved"
                    print(message)
                    self.terminal_file.write(f"{message}/n")
                    self.terminal_file.flush()
            else:
                message = f"{song[0]} not found"
                print(message)
                self.terminal_file.write(f"{message}\n")
                self.terminal_file.flush()

    def get_song_versions(self):
        versions = []
        for song in self.map_csv:
            try:
                if song[0] in os.listdir("data"):
                    directory_path = f"data/{song[0]}"
                    info_dat_file = next(
                        (
                            filename
                            for filename in os.listdir(directory_path)
                            if filename.lower() == "info.dat"
                        ),
                        None,
                    )
                    print(song[0], "versions")
                    self.terminal_file.write(f"{song[0]} versions\n")
                    self.terminal_file.flush()

                    if info_dat_file is not None:
                        info_dat_file_path = os.path.join(directory_path, info_dat_file)
                        difficulty_beatmaps_filenames = []
                        try:
                            with open(info_dat_file_path, "r") as f:
                                song_info = json.load(f)

                            for difficultyBeatmapSet in song_info[
                                "_difficultyBeatmapSets"
                            ]:
                                for difficultyBeatmap in difficultyBeatmapSet[
                                    "_difficultyBeatmaps"
                                ]:
                                    difficulty_beatmaps_filenames.append(
                                        difficultyBeatmap["_beatmapFilename"]
                                    )

                            for difficultyFilename in difficulty_beatmaps_filenames:
                                try:
                                    with open(
                                        f"{directory_path}/{difficultyFilename}", "r"
                                    ) as f:
                                        difficulty_data = json.load(f)

                                        if "version" in difficulty_data:
                                            versions.append(difficulty_data["version"])
                                        elif "_version" in difficulty_data:
                                            versions.append(difficulty_data["_version"])
                                        else:
                                            message = f"'version' or '_version' not found in {difficultyFilename}"
                                            print(song[0], message)
                                            self.terminal_file.write(
                                                f"{song[0]} {message}\n"
                                            )
                                            self.terminal_file.flush()
                                except OSError as e:
                                    message = f"Error opening difficulty file: {e}"
                                    print(message)
                                    self.terminal_file.write(f"{message}\n")
                                    self.terminal_file.flush()
                                    continue
                                except json.JSONDecodeError as e:
                                    message = f"Error loading JSON data from difficulty file: {e}"
                                    print(message)
                                    self.terminal_file.write(f"{message}\n")
                                    self.terminal_file.flush()
                                    continue
                        except OSError as e:
                            message = f"Error opening 'Info.dat' file: {e}"
                            print(message)
                            self.terminal_file.write(f"{message}\n")
                            self.terminal_file.flush()
                            continue
                        except json.JSONDecodeError as e:
                            message = (
                                f"Error loading JSON data from 'Info.dat' file: {e}"
                            )
                            print(message)
                            self.terminal_file.write(f"{message}\n")
                            self.terminal_file.flush()
                            continue
                    else:
                        message = "'Info.dat' file not found in directory."
                        print(song[0], message)
                        print(os.listdir(directory_path))
                        print("-------------------")
                        self.terminal_file.write(f"{song[0]} {message}\n")
                        self.terminal_file.write(f"{os.listdir(directory_path)}\n")
                        self.terminal_file.write("-------------------\n")
                        self.terminal_file.flush()
                else:
                    message = "not found"
                    print(song[0], message)
                    self.terminal_file.write(f"{song[0]} {message}\n")
                    self.terminal_file.flush()
            except Exception as e:
                message = f"Unexpected error occurred: {e}\n{traceback.format_exc()}"
                print(message)
                self.terminal_file.write(f"{message}\n")
                self.terminal_file.flush()
                continue

        message = f"Versions: {Counter(versions)}"
        print(message)
        self.terminal_file.write(f"{message}\n")
        self.terminal_file.flush()

    def get_song_info_versions(self):
        versions = []
        for song in self.map_csv:
            try:
                if song[0] in os.listdir("data"):
                    directory_path = f"data/{song[0]}"
                    info_dat_file = next(
                        (
                            filename
                            for filename in os.listdir(directory_path)
                            if filename.lower() == "info.dat"
                        ),
                        None,
                    )
                    if info_dat_file is not None:
                        info_dat_file_path = os.path.join(directory_path, info_dat_file)
                        try:
                            with open(info_dat_file_path, "r") as f:
                                song_info = json.load(f)

                            if "version" in song_info:
                                versions.append(song_info["version"])

                                print(f"{song[0]} info version {song_info['version']}")
                                self.terminal_file.write(
                                    f"{song[0]} info version {song_info['version']}\n"
                                )
                                self.terminal_file.flush()
                            elif "_version" in song_info:
                                versions.append(song_info["_version"])
                                print(f"{song[0]} info version {song_info['_version']}")
                                self.terminal_file.write(
                                    f"{song[0]} info version {song_info['_version']}\n"
                                )
                                self.terminal_file.flush()
                            else:
                                message = f"'version' or '_version' not found in {info_dat_file}"
                                print(song[0], message)
                                self.terminal_file.write(f"{song[0]} {message}\n")
                                self.terminal_file.flush()
                        except OSError as e:
                            message = f"Error opening 'Info.dat' file: {e}"
                            print(message)
                            self.terminal_file.write(f"{message}\n")
                            self.terminal_file.flush()
                            continue
                        except json.JSONDecodeError as e:
                            message = (
                                f"Error loading JSON data from 'Info.dat' file: {e}"
                            )
                            print(message)
                            self.terminal_file.write(f"{message}\n")
                            self.terminal_file.flush()
                            continue
                    else:
                        message = "'Info.dat' file not found in directory."
                        print(song[0], message)
                        print(os.listdir(directory_path))
                        print("-------------------")
                        self.terminal_file.write(f"{song[0]} {message}\n")
                        self.terminal_file.write(f"{os.listdir(directory_path)}\n")
                        self.terminal_file.write("-------------------\n")
                        self.terminal_file.flush()
                else:
                    message = "not found"
                    print(song[0], message)
                    self.terminal_file.write(f"{song[0]} {message}\n")
                    self.terminal_file.flush()
            except Exception as e:
                message = f"Unexpected error occurred: {e}\n{traceback.format_exc()}"
                print(message)
                self.terminal_file.write(f"{message}\n")
                self.terminal_file.flush()
                continue

        message = f"Versions: {Counter(versions)}"
        print(message)
        self.terminal_file.write(f"{message}\n")
        self.terminal_file.flush()

    def mel_gen_and_save(self):
        errored = []
        difficulty = input("Choose a difficulty: ")
        with open(f"dataset/beatmaps/color_notes/{difficulty}.csv", "r") as f:
            df = pd.read_csv(f)
            songs = df.iloc[:, 0].tolist()
            progressbar = tqdm(songs, desc="Generating mel spectrograms")

        with open(f"dataset/song_levels.json", "r") as f:
            song_levels = json.load(f)

        for index, song in enumerate(progressbar):
            try:
                if os.path.exists(f"dataset/songs/mel229/{song}"):
                    continue
                else:
                    song_split = song.split("_")
                    audio_data, sample_rate = librosa.load(
                        f"data/{song_split[0]}/{song_levels[song_split[0]]['songFilename']}"
                    )

                    if sample_rate != 22050:
                        print(
                            f"Sample rate for {song_split[0]} is {sample_rate}, resampling to 22050",
                        )
                        logging.info(
                            f"Sample rate for {song_split[0]} is {sample_rate}, resampling to 22050"
                        )
                        audio_data = librosa.resample(
                            audio_data, orig_sr=sample_rate, target_sr=22050
                        )
                        sample_rate = 22050

                    # Compute the Mel spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(
                        y=audio_data, sr=sample_rate, n_mels=229
                    )
                    np.save(
                        f"dataset/songs/mel229/{song}",
                        mel_spectrogram,
                    )

            except Exception as e:
                errored.append(song)
                print(f"Error generating mel spectrogram for {song_split[0]}")
                logging.error(f"Error generating mel spectrogram for {song_split[0]}")
                print(e)
                continue

        if errored and errored != []:
            with open(f"saved_data/errored_songs.json", "w") as file:
                file.write(json.dumps(errored))

    def zip_to_download(self):
        prefix = "data/"
        zip_filename = "songs.zip"
        create_all_data_dirs_json("zip_to_download")

        with open(f"saved_data/zip_to_download.json", "r") as f:
            song_folders: list = json.load(f)

        song_folders_to_zip = song_folders[:10]

        modified_list = [prefix + x for x in song_folders_to_zip]
        song_folders_to_zip = modified_list

        print("Zipping", len(song_folders_to_zip), "songs")

        zip_folders(zip_filename, *song_folders_to_zip)

    def create_beatmap_arrays_and_save(self):
        with open(f"dataset/song_levels.json", "r") as f:
            song_levels = json.load(f)

        progress_bar = tqdm(song_levels, desc="Parsing songs")

        count_all = 0
        count_missing = 0

        allowed_keys = ["b", "x", "y", "c", "d", "a", "w", "h"]

        for song in progress_bar:
            if "Standard" in song_levels[song]["difficultySet"]:
                for level in song_levels[song]["difficultySet"]["Standard"]:
                    try:
                        count_all += 1

                        color_notes = None
                        bomb_notes = None
                        obstacles = None

                        diffic = song_levels[song]["difficultySet"]["Standard"][level]
                        with open(f"data/{song}/{diffic}", "r") as f:
                            data = json.load(f)
                        if all(
                            key in data
                            for key in ("colorNotes", "bombNotes", "obstacles")
                        ):
                            color_notes = data["colorNotes"]
                            bomb_notes = data["bombNotes"]
                            obstacles = data["obstacles"]

                            for index, note in enumerate(color_notes):
                                if len(note) != 6:
                                    keys_to_remove = [
                                        key
                                        for key in note.keys()
                                        if key not in allowed_keys
                                    ]
                                    for key in keys_to_remove:
                                        color_notes[index].pop(key)

                            for index, note in enumerate(bomb_notes):
                                if len(note) != 3:
                                    keys_to_remove = [
                                        key
                                        for key in note.keys()
                                        if key not in allowed_keys
                                    ]
                                    for key in keys_to_remove:
                                        bomb_notes[index].pop(key)

                            for index, obstacle in enumerate(obstacles):
                                if len(obstacle) != 6:
                                    keys_to_remove = [
                                        key
                                        for key in obstacle.keys()
                                        if key not in allowed_keys
                                    ]
                                    for key in keys_to_remove:
                                        obstacles[index].pop(key)

                        elif all(key in data for key in ("_notes", "_obstacles")):
                            color_notes = []
                            bomb_notes = []
                            obstacles = []
                            for obstacle in data["_obstacles"]:
                                obstacles.append(
                                    {
                                        "b": obstacle["_time"],
                                        "d": obstacle["_duration"],
                                        "x": obstacle["_lineIndex"],
                                        "w": obstacle["_width"],
                                        "y": 0 if obstacle["_type"] == 0 else 2,
                                        "h": 5 if obstacle["_type"] == 0 else 3,
                                    }
                                )

                            for note in data["_notes"]:
                                if note["_type"] == 0 or note["_type"] == 1:
                                    color_notes.append(
                                        {
                                            "b": note["_time"],
                                            "x": note["_lineIndex"],
                                            "y": note["_lineLayer"],
                                            "c": note["_type"],
                                            "d": note["_cutDirection"],
                                            "a": 0,
                                        }
                                    )

                                elif note["_type"] == 3:
                                    bomb_notes.append(
                                        {
                                            "b": note["_time"],
                                            "x": note["_lineIndex"],
                                            "y": note["_lineLayer"],
                                        }
                                    )
                        else:
                            print(
                                f"Skipping {song} - {level} due to missing valid keys"
                            )
                            count_missing += 1
                            continue

                        if color_notes == [] and bomb_notes == [] and obstacles == []:
                            count_missing += 1
                            continue

                        if color_notes:
                            ordered_list = [
                                [
                                    sorted_pair[1]
                                    for sorted_pair in sorted(dictionary.items())
                                ]
                                for dictionary in color_notes
                            ]
                            np.save(
                                f"dataset/beatmaps/color_notes/{level}/{song}_{song_levels[song]['songId']}",
                                ordered_list,
                            )

                        if bomb_notes:
                            ordered_list = [
                                [
                                    sorted_pair[1]
                                    for sorted_pair in sorted(dictionary.items())
                                ]
                                for dictionary in bomb_notes
                            ]
                            np.save(
                                f"dataset/beatmaps/bomb_notes/{level}/{song}_{song_levels[song]['songId']}",
                                ordered_list,
                            )
                        if obstacles:
                            ordered_list = [
                                [
                                    sorted_pair[1]
                                    for sorted_pair in sorted(dictionary.items())
                                ]
                                for dictionary in obstacles
                            ]
                            np.save(
                                f"dataset/beatmaps/obstacles/{level}/{song}_{song_levels[song]['songId']}",
                                ordered_list,
                            )
                    except Exception as e:
                        print(f"Error parsing {song} - {level}: {e}")
                        continue

        print(f"Total number of beatmaps: {count_all}")
        print(f"Number of beatmaps with missing data: {count_missing}")

        count_validation = count_all - count_missing

        print(f"Number of beatmaps successfully parsed: {count_validation}")
        dir_len = len(os.listdir("dataset/beatmaps/color_notes"))

        if count_validation == dir_len:
            print("All beatmaps have been successfully parsed")


def zip_folders(zip_filename, *folders):
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for folder in folders:
            # Walk through the folder and add all files to the zip file
            for foldername, _, filenames in os.walk(folder):
                for filename in filenames:
                    # Create the full filepath by using os module.
                    file_path = os.path.join(foldername, filename)
                    # Add file to zip
                    zip_file.write(file_path)

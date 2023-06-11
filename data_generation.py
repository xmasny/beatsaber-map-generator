import json
import csv
import os
from collections import Counter
from io import TextIOWrapper


class DataGeneration:
    def __init__(self, file: TextIOWrapper):
        self.terminal_file = file

        with open('saved_data/map_info.json', 'r') as f:
            self.map_info = json.load(f)

        with open('saved_data/map_zips.csv', 'r') as f:
            reader = csv.reader(f)
            self.map_csv = list(reader)

    def save_song_info(self):
        for song in self.map_csv:
            if song[0] in os.listdir('data'):
                song_info = [id for id in self.map_info if id['id'] == song[1]]

                if 'generated' not in os.listdir(f'data/{song[0]}'):
                    os.mkdir(f'data/{song[0]}/generated')

                with open(f'data/{song[0]}/generated/SongInfo.dat', 'w') as f:
                    json.dump(song_info[0], f)
                    message = f"{song[0]} SongInfo.dat saved"
                    print(message)
                    self.terminal_file.write(f"{message}\n")
                    self.terminal_file.flush()
            else:
                message = f"{song[0]} not found"
                print(message)
                self.terminal_file.write(f"{message}\n")
                self.terminal_file.flush()

    def get_song_versions(self):
        versions = []
        for song in self.map_csv:
            if song[0] in os.listdir('data'):
                directory_path = f"data/{song[0]}"

                info_dat_file = next((filename for filename in os.listdir(
                    directory_path) if filename.lower() == 'info.dat'), None)

                print(song[0], "versions")
                self.terminal_file.write(f"{song[0]} versions\n")
                self.terminal_file.flush()

                if info_dat_file is not None:
                    info_dat_file_path = os.path.join(
                        directory_path, info_dat_file)
                    difficultyBeatmapsFilenames = []
                    try:
                        with open(info_dat_file_path, 'r') as f:
                            song_info = json.load(f)

                        for difficultyBeatmapSet in song_info['_difficultyBeatmapSets']:
                            for difficultyBeatmap in difficultyBeatmapSet['_difficultyBeatmaps']:
                                difficultyBeatmapsFilenames.append(
                                    difficultyBeatmap['_beatmapFilename'])

                        for difficultyFilename in difficultyBeatmapsFilenames:
                            with open(f'{directory_path}/{difficultyFilename}', 'r') as f:
                                difficulty_data = json.load(f)

                                if 'version' in difficulty_data:
                                    versions.append(difficulty_data['version'])
                                elif '_version' in difficulty_data:
                                    versions.append(
                                        difficulty_data['_version'])
                                else:
                                    message = f"'version' or '_version' not found in {difficultyFilename}"
                                    print(song[0], message)
                                    self.terminal_file.write(
                                        f"{song[0]} {message}\n")
                                    self.terminal_file.flush()

                    except OSError as e:
                        message = f"Error opening 'Info.dat' file: {e}"
                        print(message)
                        self.terminal_file.write(f"{message}\n")
                        continue
                    except json.JSONDecodeError as e:
                        message = f"Error loading JSON data from 'Info.dat' file: {e}"
                        print(message)
                        self.terminal_file.write(f"{message}\n")
                        continue

                else:
                    message = f"'Info.dat' file not found in directory."
                    print(song[0], message)
                    print(os.listdir(directory_path))
                    print('-------------------')
                    self.terminal_file.write(f"{song[0]} {message}\n")
                    self.terminal_file.write(f"{os.listdir(directory_path)}\n")
                    self.terminal_file.write("-------------------\n")
                    self.terminal_file.flush()

            else:
                message = "not found"
                print(song[0], message)
                self.terminal_file.write(f"{song[0]} {message}\n")
                self.terminal_file.flush()

        message = f"Versions: {Counter(versions)}"
        print(message)
        self.terminal_file.write(f"{message}\n")
        self.terminal_file.flush()

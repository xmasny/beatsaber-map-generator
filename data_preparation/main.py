import datetime
import time
import wandb

from data_generation import *
from download_script import DownloadScript
from utils import *

wandb.init(project="test-beat-saber-map-generator", mode="disabled")

folders = [
    "data",
    "terminal",
    "saved_data",
    "dataset/songs",
    "dataset/songs/mel128",
    "dataset/songs/mel229",
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

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print_scripts()
script_no = input("Choose script: ")

start_time = time.time()


def start_date():
    date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return date


print("Start time:", start_date)

try:
    with open(f"terminal/{start_time}.txt", "a") as file:
        download = DownloadScript(file)
        generation = DataGeneration(file)

        if script_no == "1":
            file.write(f"get_all_users Start time: {start_date()}\n")
            download.get_all_users()

        if script_no == "2":
            file.write(f"get_all_maps Start time: {start_date()}\n")
            download.get_all_maps()

        if script_no == "3":
            file.write(f"get_all_maps_urls Start time: {start_date()}\n")
            download.get_all_maps_urls()

        if script_no == "4":
            file.write(f"get_all_maps_from_user Start time: {start_date()}\n")
            download.get_all_maps_from_user()

        if script_no == "5":
            file.write(f"get_all_zips Start time: {start_date()}\n")
            download.get_all_zips()

            file.write(f"unzip_all_zips Start time: {start_date()}\n")
            download.unzip_all_zips()

        if script_no == "6":
            file.write(f"save_song_info Start time: {start_date()}\n")
            generation.save_song_info()

        if script_no == "7":
            file.write(f"get_song_versions Start time: {start_date()}\n")
            generation.get_song_versions()

        if script_no == "8":
            file.write(f"get_song_info_versions Start time: {start_date()}\n")
            generation.get_song_info_versions()

        if script_no == "9":
            file.write(f"create_beatmap_arrays_and_save Start time: {start_date()}\n")
            generation.create_beatmap_arrays_and_save()

        if script_no == "10":
            file.write(f"mel_gen_and_save Start time: {start_date()}\n")
            generation.mel_gen_and_save()

        if script_no == "11":
            file.write(f"create_all_data_dirs_json Start time: {start_date()}\n")
            generation.zip_to_download()

        if script_no == "12":
            file.write(
                f"get_maps_by_characteristic_and_difficuly Start time: {start_date()}\n"
            )
            get_maps_by_characteristic_and_difficulty()

        if script_no == "13":
            file.write(f"get_all_song_files Start time: {start_date()}\n")
            get_all_song_files()

    end_time = time.time()
    runtime = end_time - start_time

    print("Runtime:", (runtime / 60), "minutes")

    wandb.finish()

except Exception as e:
    runtime_exception = time.time() - start_time
    print(e)
    print("Runtime:", runtime_exception, "seconds")
    with open(f"terminal/{start_time}.txt", "a") as file:
        file.write(f"Error time: {start_date()}\n")
        file.write(f"Error: {e}\n")
        file.write(f"Runtime: {runtime_exception} seconds\n")
    wandb.finish()
    exit()

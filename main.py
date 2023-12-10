import datetime
import time

from data_generation import *
from download_script import DownloadScript
from utils import *

folders = ["data", "terminal", "saved_data"]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

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
            file.write(f"generate_v3_beatmap Start time: {start_date()}\n")
            generation.generate_v3_beatmap()

        if script_no == "10":
            file.write(f"mel_gen_and_save Start time: {start_date()}\n")
            generation.mel_gen_and_save()

        if script_no == "11":
            file.write(f"remove_pngs Start time: {start_date()}\n")
            remove_pics()

        if script_no == "12":
            file.write(f"create_all_data_dirs_json Start time: {start_date()}\n")
            generation.zip_to_download()

        if script_no == "13":
            file.write(f"save_filenames_to_json Start time: {start_date()}\n")
            get_all_filenames()

        if script_no == "14":
            file.write(f"save_all_full_filenames_to_json Start time: {start_date()}\n")
            get_all_filenames_full_route()

    end_time = time.time()
    runtime = end_time - start_time

    print("Runtime:", runtime, "seconds")

except Exception as e:
    runtime_exception = time.time() - start_time
    print(e)
    print("Runtime:", runtime_exception, "seconds")
    with open(f"terminal/{start_time}.txt", "a") as file:
        file.write(f"Error time: {start_date()}\n")
        file.write(f"Error: {e}\n")
        file.write(f"Runtime: {runtime_exception} seconds\n")
    exit()

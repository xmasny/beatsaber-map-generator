import time
import datetime
from download_script import DownloadScript
from utils import *
import os

folders = ["data", "terminal", "saved_data"]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


script_no = input("Choose script: ")

start_time = time.time()


def start_date():
    start_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return start_date


print("Start time:", start_date)

print_scripts()


try:
    with open(f"terminal/{start_time}.txt", 'a') as file:

        download = DownloadScript(file)

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

    end_time = time.time()
    runtime = end_time - start_time

    print("Runtime:", runtime, "seconds")

except Exception as e:
    runtime_exeption = time.time() - start_time
    print(e)
    print("Runtime:", runtime_exeption, "seconds")
    with open(f"terminal/{start_time}.txt", 'a') as file:
        file.write(f"Error time: {start_date()}\n")
        file.write(f"Error: {e}\n")
        file.write(f"Runtime: {runtime_exeption} seconds\n")
    exit()

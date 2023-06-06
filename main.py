import time
import datetime
from download_script import DownloadScript
import os

folders = ["data", "terminal", "saved_data"]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


start_time = time.time()


def start_date():
    start_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return start_date


print("Start time:", start_date)


try:
    with open(f"terminal/{start_time}.txt", 'a') as file:

        download = DownloadScript(file)

        # file.write(f"get_all_users Start time: {start_date()}\n")
        # download.get_all_users()

        # file.write(f"get_all_maps Start time: {start_date()}\n")
        # download.get_all_maps()

        # file.write(f"get_all_maps_from_user Start time: {start_date()}\n")
        # download.get_all_maps_from_user()

        file.write(f"get_all_zips Start time: {start_date()}\n")
        download.get_all_zips()

        file.write(f"unzip_all_zips Start time: {start_date()}\n")
        download.unzip_all_zips()

    end_time = time.time()
    runtime = end_time - start_time

    print("Runtime:", runtime, "seconds")

except Exception as e:
    with open(f"terminal/{start_time}.txt", 'a') as file:
        file.write(f"Error time: {start_date()}\n")
        file.write(f"Error: {e}\n")
    exit()

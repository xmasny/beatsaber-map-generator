import requests
import math
import zipfile
import os
import json
from io import TextIOWrapper
import csv
import time

time.time()


class DownloadScript:
    def __init__(self, file: TextIOWrapper):
        self.terminal_file = file
        self.songFolder = "data"
        self.zipFiles = []
        self.allUsersJson = []
        self.map_info = []
        self.maps_ids = []

    def get_all_users(self, start=0, end=20000):
        # get all users in range of pages
        for page in range(start, end):
            while True:
                try:
                    response = requests.get(
                        f"https://api.beatsaver.com/users/list/{page}")
                    break
                except Exception as e:
                    print(e)
                    self.terminal_file.write(f"{e}\n")
                    time.sleep(10)
            jsonUsers = response.json()
            for user in jsonUsers:
                self.allUsersJson.append([
                    user["id"],
                    user['stats']["totalMaps"],
                ])
            print(f"Users page {page} done")
            self.terminal_file.write(f"Users page {page} done\n")
            self.terminal_file.flush()
        with open(f'saved_data/users{time.time()}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.allUsersJson)

    def get_all_maps(self):
        # get all zip maps from users
        song_index = 1
        allMapsUrls = []
        for user in self.allUsersJson:
            for page in range(0, math.ceil(int(user[1] / 20))):
                while True:
                    try:
                        response = requests.get(
                            f"https://api.beatsaver.com/maps/uploader/{user[0]}/{page}")
                        break
                    except Exception as e:
                        print(e)
                        self.terminal_file.write(f"{e}\n")
                        time.sleep(10)
                jsonUserMaps = response.json()
                print(
                    f"User {user[0]} maps page {page} done")
                self.terminal_file.write(
                    f"User {user[0]} maps page {page} done\n")
                self.terminal_file.flush()
                for map in jsonUserMaps['docs']:
                    self.map_info.append(map)
                    self.maps_ids.append(map['id'])
                    for version in map['versions']:
                        allMapsUrls.append([
                            f'song{song_index}', map['id'], version['downloadURL']])
                        song_index += 1
        with open('saved_data/map_zips.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(allMapsUrls)

        with open('saved_data/map_info.json', 'w') as file:
            json.dump(self.map_info, file)

    def get_all_maps_from_user(self, start_id=None, end_id=None):
        '''
        Download all maps from user
        start_id: start index of allMapsUrls
        end_id: end index of allMapsUrls

        default: download all maps
        '''
        with open('saved_data/map_zips.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            allMapsUrls = list(reader)
            response = None

        for map in allMapsUrls[start_id:end_id]:
            while True:
                try:
                    response = requests.get(map[2])
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
            with open(f"data/{map[0]}.zip", 'wb') as f:
                f.write(response.content)
            print(f"{map[0]}/{len(allMapsUrls)} downloaded")
            self.terminal_file.write(
                f"{map[0]}/{len(allMapsUrls)} downloaded\n")
            self.terminal_file.flush()

    def get_all_zips(self):
        # get all zip files
        for file_name in os.listdir(self.songFolder):
            if file_name.endswith(".zip"):
                self.zipFiles.append(file_name)

    def unzip_all_zips(self):
        # unzip file
        for zip_file in self.zipFiles:
            with zipfile.ZipFile(f"data/{zip_file}", "r") as zip_ref:
                zip_ref.extractall(f"data/{zip_file[:-4]}")
                print(f"{zip_file} unzipped")
                self.terminal_file.write(f"{zip_file} unzipped\n")
                # remove song zip file
            os.remove(f"data/{zip_file}")
            print(f"{zip_file} removed")
            self.terminal_file.write(f"{zip_file} removed\n")

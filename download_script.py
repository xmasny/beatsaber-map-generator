import requests
import math
import zipfile
import os
import json
import numpy as np
from io import TextIOWrapper


class DownloadScript:
    def __init__(self, file: TextIOWrapper):
        self.terminal_file = file
        self.songFolder = "data"
        self.zipFiles = []
        self.allUsersJson = []
        self.allMapsUrls = []
        self.map_info = []
        self.maps_ids = []

    def get_all_users(self, start=0, end=20000):
        # get all users in range of pages
        for page in range(start, end):
            response = requests.get(
                f"https://api.beatsaver.com/users/list/{page}")
            jsonUsers = response.json()
            for user in jsonUsers:
                self.allUsersJson.append({
                    "id": user["id"],
                    "totalMaps": user['stats']["totalMaps"],
                })
            print(f"Users page {page} done")
            self.terminal_file.write(f"Users page {page} done\n")
            self.terminal_file.flush()

    def get_all_maps(self):
        # get all zip maps from users
        for user in self.allUsersJson:
            for page in range(0, math.ceil(int(user['totalMaps'] / 20))):
                response = requests.get(
                    f"https://api.beatsaver.com/maps/uploader/{user['id']}/{page}")
                jsonUserMaps = response.json()
                print(
                    f"User {user['id']} maps page {page} done")
                self.terminal_file.write(
                    f"User {user['id']} maps page {page} done\n")
                self.terminal_file.flush()
                for map in jsonUserMaps['docs']:
                    self.map_info.append(map)
                    self.maps_ids.append(map['id'])
                    for version in map['versions']:
                        self.allMapsUrls.append(version['downloadURL'])
        with open('map_ids.txt', 'w') as file:
            for item in self.maps_ids:
                file.write(str(item) + '\n')

    def get_all_maps_from_user(self, start_id=None, end_id=None):
        '''
        Download all maps from user
        start_id: start index of allMapsUrls
        end_id: end index of allMapsUrls

        default: download all maps
        '''
        for index, url in enumerate(self.allMapsUrls[start_id:end_id]):
            response = requests.get(url)
            with open(f"data/song{index}.zip", 'wb') as f:
                f.write(response.content)
            print(f"song{index}/{len(self.allMapsUrls)-1} downloaded")
            self.terminal_file.write(
                f"song{index}/{len(self.allMapsUrls)-1} downloaded\n")
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
                # remove song zip file
            os.remove(f"data/{zip_file}")

    def get_all_map_info(self):
        # get all map info
        for index, map in enumerate(self.map_info):
            os.makedirs(f"data/song{index}/generated")

            print(f"song{index}/{len(self.map_info)-1} detail info created")
            self.terminal_file.write(
                f"song{index}/{len(self.map_info)-1} detail info created\n")
            with open(f"data/song{index}/generated/DetailInfo.dat", 'w') as f:
                json.dump(map, f)

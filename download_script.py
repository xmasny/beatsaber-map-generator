import requests
import math
import zipfile
import os


class DownloadScript:
    def __init__(self):
        self.songFolder = "data"
        self.zipFiles = []
        self.allUsersJson = []
        self.allMapsUrls = []

    def get_all_users(self, start=0, end=10):
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

    def get_all_maps(self):
        # get all zip maps from users
        for user in self.allUsersJson:
            for page in range(0, math.ceil(int(user['totalMaps'] / 20))):
                response = requests.get(
                    f"https://api.beatsaver.com/maps/uploader/{user['id']}/{page}")
                jsonUserMaps = response.json()
                print(f"User {user['id']} maps page {page} done")
                for map in jsonUserMaps['docs']:
                    for version in map['versions']:
                        self.allMapsUrls.append(version['downloadURL'])

    def get_all_maps_from_user(self, star_id=None, end_id=None):
        '''
        Download all maps from user
        star_id: start index of allMapsUrls
        end_id: end index of allMapsUrls

        default: download all maps
        '''
        for index, url in enumerate(self.allMapsUrls[star_id:end_id]):
            response = requests.get(url)
            with open(f"data/song{index}.zip", 'wb') as f:
                f.write(response.content)
            print(f"song{index}.zip downloaded")

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

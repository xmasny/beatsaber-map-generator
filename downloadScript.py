import requests
import math
import time
import zipfile
import os

start_time = time.time()


songFolder = "data"
zip_files = []
allUsersJson = []
allMapsUrls = []

# get all users in range
for page in range(0, 1):
    response = requests.get(f"https://api.beatsaver.com/users/list/{page}")
    jsonUsers = response.json()
    for user in jsonUsers:
        allUsersJson.append({
            "id": user["id"],
            "totalMaps": user['stats']["totalMaps"],
        })
    print(f"Users page {page} done")

# get all zip maps from users
for user in allUsersJson:
    for page in range(0, math.ceil(int(user['totalMaps'] / 20))):
        response = requests.get(
            f"https://api.beatsaver.com/maps/uploader/{user['id']}/{page}")
        jsonUserMaps = response.json()
        print(f"User {user['id']} maps page {page} done")
        for map in jsonUserMaps['docs']:
            for version in map['versions']:
                allMapsUrls.append(version['downloadURL'])

# download maps
for index, url in enumerate(allMapsUrls[:3]):
    response = requests.get(url)
    with open(f"data/song{index}.zip", 'wb') as f:
        f.write(response.content)
    print(f"song{index}.zip downloaded")


# get all zip files
for file_name in os.listdir(songFolder):
    if file_name.endswith(".zip"):
        zip_files.append(file_name)


# unzip file
for zip_file in zip_files:
    with zipfile.ZipFile(f"data/{zip_file}", "r") as zip_ref:
        zip_ref.extractall(f"data/{zip_file[:-4]}")
        # remove song zip file
    os.remove(f"data/{zip_file}")

end_time = time.time()
runtime = end_time - start_time

print("Runtime:", runtime, "seconds")

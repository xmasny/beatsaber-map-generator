import json
import shutil
with open("saved_data/errored_songs.json", "r") as file:
    errored_songs = json.load(file)

for song in errored_songs:
    shutil.rmtree(f"data/{song[0]}")
    print(f"Removed {song[0]}")
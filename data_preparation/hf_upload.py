import time
from huggingface_hub import HfApi
from requests import HTTPError
import wandb

wandb.login()


wandb.init(project="beat-saber-map-generator", tags=["upload"])

api = HfApi()

while True:
    try:
        api.upload_folder(
            folder_path=f"./dataset/beatmaps",
            path_in_repo=f".",
            repo_id="masny5/beatsaber_songs_and_metadata",
            repo_type="dataset",
            commit_message=f"Add new songs and metadata",
            token="",
            ignore_patterns=["*.npy", "songs/*"],
            multi_commits=True,
            multi_commits_verbose=True,
        )

        print("Upload successful")
        break

    except HTTPError as e:
        if e.response.status_code == 429:
            print(e.response.status_code)
            print("You have exceeded our hourly quotas for action: commit.")

            for i in range(15):
                print(f"Waiting {15 - i} minutes", end="\r", flush=True)
                time.sleep(60)

            continue

        elif e.response.status_code == 413:
            print(e.response.status_code)
            print("Request Entity Too Large")
            break

        elif e.response.status_code == 500:
            print(e.response.status_code)
            print("Internal Server Error")
            continue
    except Exception as e:
        print(e)
        print("Upload failed")
        break

wandb.finish()

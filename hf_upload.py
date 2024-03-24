import time
from huggingface_hub import HfApi
from requests import HTTPError
import wandb

wandb.login()


wandb.init(project="beat-saber-map-generator", tags=["upload"])

api = HfApi()

type = [
    "beatmaps/color_notes/Easy",
    "beatmaps/color_notes/Normal",
    "beatmaps/color_notes/Hard",
    "beatmaps/color_notes/Expert",
    "beatmaps/color_notes/ExpertPlus",
    # "beatmaps/bomb_notes/Easy",
    # "beatmaps/bomb_notes/Normal",
    # "bomb_notes/Hard",
    # "bomb_notes/Expert",
    # "bomb_notes/ExpertPlus",
    # "obstacles/Easy",
    # "obstacles/Normal",
    # "obstacles/Hard",
    # "obstacles/Expert",
    # "obstacles/ExpertPlus",
]

while True:
    try:
        api.upload_folder(
            folder_path=f"./dataset",
            path_in_repo=f".",
            repo_id="masny5/beatsaber_songs_and_metadata",
            repo_type="dataset",
            commit_message=f"Add new songs and metadata",
            token='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',
            ignore_patterns=["*.npy", 'songs/*'],
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
    except Exception as e:
        print(e)
        print("Upload failed")
        break

wandb.finish()

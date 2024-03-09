import time
from huggingface_hub import HfApi, login
from requests import HTTPError
import wandb

wandb.login()


login()

wandb.init(project="beat-saber-map-generator")

api = HfApi()

while True:
    try:
        api.upload_folder(
            folder_path="./dataset",
            path_in_repo=".",
            repo_id="masny5/beatsaber_songs_and_metadata",
            repo_type="dataset",
            commit_message="Add new songs and metadata",
            multi_commits=True,
            multi_commits_verbose=True,
        )
        
        print("Upload successful")
        break
    
    except HTTPError as e:
        print(e.response.status_code)
        print("You have exceeded our hourly quotas for action: commit.")
        
        for i in range(15):
            print(f"Waiting {15 - i} minutes", end="\r", flush=True)
            time.sleep(60)
        
        continue
    
    except Exception as e:
        print(e)
        print("Upload failed")
        

wandb.finish()
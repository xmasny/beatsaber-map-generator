import os
import shutil
import zipfile
from tqdm import tqdm
import wandb

all_difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]

def zip_selected_type_difficulty():
    while True:
        try:
            object_type = input("Choose object type: ")
            difficulty = input("Choose difficulty: ")
            
            if not os.path.exists(f"dataset/beatmaps/{object_type}/{difficulty}"):
                raise Exception("Path does not exist, please try again.")
            
            if not os.path.exists(f"dataset/beatmaps/{object_type}/{difficulty}/mels"):
                os.makedirs(f"dataset/beatmaps/{object_type}/{difficulty}/mels")
            
            for filename in os.listdir(f"dataset/beatmaps/{object_type}/{difficulty}"):
                if filename.endswith(".npy"):
                    shutil.copy(f"dataset/songs/{filename}", f"dataset/beatmaps/{object_type}/{difficulty}/mels/{filename}")
                
        except Exception as e:
            print(e)
            continue
        
        zip_folder(object_type, difficulty)
        
        if input("Do you want to zip different files? (y/n): ") == "n":
            break

def zip_folder(object_type, difficulty):
    source_folder = f"dataset/beatmaps/{object_type}/{difficulty}"
    total_files = sum([len(files) for _, _, files in os.walk(source_folder)])  # Count total number of files
    
    with zipfile.ZipFile(f"dataset/beatmaps/{object_type}_{difficulty}.zip", 'w') as zipf:
        # Wrap os.walk with tqdm to track progress
        with tqdm(total=total_files, desc=f"Zipping {object_type} {difficulty}") as pbar:
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    if not file.endswith(".zip"):
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), source_folder))
                    pbar.update(1)
            
    
    # Remove the specified directory
    shutil.rmtree(f"{source_folder}/mels")

def zip_all_difficulties():
    object_type = input("Choose object type: ")
    
    for difficulty in all_difficulties:
        if not os.path.exists(f"dataset/beatmaps/{object_type}/{difficulty}/mels"):
            os.makedirs(f"dataset/beatmaps/{object_type}/{difficulty}/mels")
        for filename in tqdm(os.listdir(f"dataset/beatmaps/{object_type}/{difficulty}"), desc=f"Copying {object_type} {difficulty}"):
            if filename.endswith(".npy"):
                shutil.copy(f"dataset/songs/{filename}", f"dataset/beatmaps/{object_type}/{difficulty}/mels/{filename}")
        zip_folder(object_type, difficulty)
    
if __name__ == "__main__":
    wandb.init(project="beat-saber-map-generator")

    select = input("Do you want to zip selected difficulties and objects? (y/n): ")
    if select == "y":
        zip_selected_type_difficulty()
    else:
        zip_all_difficulties()  
    
    print("Done zipping!")
    wandb.finish()
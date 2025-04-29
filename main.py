import shutil
import sys
import hydra
import os
import torch.multiprocessing as mp

if sys.platform.startswith("linux"):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Start method was already set; that's OK

from config import RunConfig
from training.onset_train import main as train


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: RunConfig):
    try:
        train(cfg.params)
    except Exception as e:
        if os.path.exists("dataset/valid_dataset"):
            shutil.rmtree("dataset/valid_dataset")
            print("Removed dataset/valid_dataset")
        print(e)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")

    repeats = int(input("Enter the number of times to repeat the training: "))

    for i in range(repeats):
        print(f"Training iteration {i + 1}/{repeats}")
        try:
            main()
        except Exception as e:
            print(f"Error during training iteration {i + 1}: {e}")
            if os.path.exists("dataset/valid_dataset"):
                shutil.rmtree("dataset/valid_dataset")
                print("Removed dataset/valid_dataset")
            continue

import shutil
import sys
import hydra
import os

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

    main()

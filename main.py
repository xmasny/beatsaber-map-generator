import sys
import hydra
from omegaconf import OmegaConf
import torch

from config import RunConfig
from training.onset_train import main as train


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: RunConfig):
    train(cfg.color_notes)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")
    main()

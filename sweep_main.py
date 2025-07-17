from omegaconf import OmegaConf
from training.onset_train import main as onset_train
from training.class_train import main as class_train
from config import *
import sys
import hydra


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def sweep_main(cfg: RunConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.model_type == "onsets":
        onset_train(cfg.model)
    elif cfg.model_type == "class":
        class_train(cfg.model)
    else:
        raise ValueError("Invalid model type")


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")

    sweep_main()

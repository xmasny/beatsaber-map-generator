import random
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from config import *
from dl.models.onsets import OnsetFeatureExtractor
from training.loader import BaseLoader
from training.onset_ignite import ignite_train
from utils import MyDataParallel


class AttributeDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __setattr__(self, attr, value):
        self[attr] = value


def non_collate(batch):
    return batch


dataset = BaseLoader()

sweep_id = input("Enter the sweep id: ")
batch_size = int(input("Enter the batch size: "))

is_parallel = input("Is parallel? (True/False): ").lower() == "true"

if not is_parallel:
    gpu_index = int(input("Enter the GPU index: "))
else:
    gpu_index = -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu", gpu_index)


def sweep_train(config=None):
    SEED = random.randint(0, 2**32 - 1)  # or fix it for true reproducibility

    common_dataset_args = {
        "difficulty": DifficultyName.ALL,
        "object_type": ObjectType.COLOR_NOTES,
        "enable_condition": True,
        "seq_length": 16384,
        "skip_step": 2000,
        "with_beats": True,
        "batch_size": batch_size,
        "min_sum_votes": 100,
        "min_score": 0.92,
        "min_bpm": 60.0,
        "max_bpm": 300.0,
        "split_seed": SEED,
    }

    train_dataset = BaseLoader(split=Split.TRAIN, **common_dataset_args)
    valid_dataset = BaseLoader(split=Split.VALIDATION, **common_dataset_args)

    train_dataset_len = len(train_dataset)
    valid_dataset_len = len(valid_dataset)

    train_loader = train_dataset.get_dataloader()
    valid_loader = valid_dataset.get_dataloader()

    with wandb.init(config=config):  # type: ignore
        config = wandb.config

        warmup_steps = config.epoch_length * config.epochs * config.warmup_steps

        run_parameters = {
            **common_dataset_args,
            **config,
            "model_type": "onsets",
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "num_layers": 2,
            "patience": 3,
            "eta_min": 1e-6,
        }

        wandb.config.update(
            {
                "train_dataset_len": train_dataset_len,
                "valid_dataset_len": valid_dataset_len,
                "wandb_mode": "online",
            },
        )

        model = OnsetFeatureExtractor(
            input_features=n_mels,
            output_features=1,
            dropout=config.dropout,
            num_layers=run_parameters["num_layers"],
            rnn_dropout=config.rnn_dropout,
            inference_chunk_length=round(run_parameters["seq_length"] / FRAME),
        ).to(device)

        if is_parallel:
            model = MyDataParallel(model)

        optimizer = Adam(
            model.parameters(),
            config.start_lr,
            weight_decay=config.weight_decay,
        )

        if config.lr_scheduler_name == "CyclicLR":
            lr_scheduler = CyclicLR(
                optimizer,
                config.start_lr,
                config.end_lr,
                1000,
                cycle_momentum=False,
            )
        elif config.lr_scheduler_name == "CosineAnnealingLR":
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                config.epochs * config.epoch_length,
                eta_min=run_parameters["eta_min"],
            )
        else:
            raise ValueError

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        wandb.define_metric("epoch")
        wandb.define_metric("metrics/*", step_metric="epoch")

        wandb.define_metric("weight_sum/*", step_metric="epoch")

        wandb.define_metric("validation/step")
        wandb.define_metric("validation/*", step_metric="validation/step")

        ignite_train(
            train_dataset,
            valid_dataset,
            model,  # type: ignore
            train_loader,
            valid_loader,
            optimizer,
            train_dataset_len,
            valid_dataset_len,
            device,
            lr_scheduler,
            wandb_logger=None,
            **run_parameters,
        )


wandb.agent(
    sweep_id,
    sweep_train,
    project="beat-saber-map-generator",
)

wandb.finish()

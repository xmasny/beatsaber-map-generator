import torch
import wandb
from training.onset_ignite import ignite_train
from utils import MyDataParallel
from training.loader import BaseLoader, SavedValidDataloader
from config import *
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torch.optim import Adam
from dl.models.onsets import SimpleOnsets


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = BaseLoader()

dataset.load()

train_dataset = dataset[Split.TRAIN]
valid_dataset = dataset[Split.VALIDATION]

train_dataset_len = train_dataset.n_shards
valid_dataset_len = valid_dataset.n_shards

run_parameters = AttributeDict()


run_parameters.difficulty = "Easy"
run_parameters.object_type = "color_notes"

songs_batch_size = 60
num_workers = 4

valid_loader = DataLoader(valid_dataset, batch_size=songs_batch_size, collate_fn=non_collate, num_workers=num_workers)  # type: ignore

dataset.save_valid_data(valid_loader, valid_dataset_len, run_parameters)  # type: ignore

valid_dataset = SavedValidDataloader(run_parameters)  # type: ignore
valid_dataset_len = len(valid_dataset)
valid_loader = DataLoader(valid_dataset, batch_size=songs_batch_size, collate_fn=non_collate)  # type: ignore

sweep_id = "c734g143"
train_batch_size = 60


def sweep_train(config=None):
    train_loader = DataLoader(train_dataset, batch_size=songs_batch_size, collate_fn=non_collate, num_workers=num_workers)  # type: ignore
    with wandb.init(config=config):  # type: ignore
        config = wandb.config

        run_parameters = {
            **config,
            "songs_batch_size": songs_batch_size,
            "num_workers": num_workers,
            "train_batch_size": train_batch_size,
        }

        wandb.config.update(
            {
                "train_dataset_len": train_dataset_len,
                "valid_dataset_len": valid_dataset_len,
                "wandb_mode": "online",
                "songs_batch_size": songs_batch_size,
                "num_workers": num_workers,
            },
        )

        model = SimpleOnsets(
            input_features=n_mels,
            output_features=1,
            dropout=config.dropout,
            num_layers=config.num_layers,
            rnn_dropout=config.rnn_dropout,
            inference_chunk_length=round(config.seq_length / FRAME),
        ).to(device)

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
                eta_min=config.eta_min,
            )
        else:
            raise ValueError

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        wandb.define_metric("epoch")
        wandb.define_metric("metrics/*", step_metric="epoch")

        wandb.define_metric("validation/step")
        wandb.define_metric("validation/*", step_metric="validation/step")

        ignite_train(
            dataset,
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
    project="test-beat-saber-map-generator",
)

wandb.finish()

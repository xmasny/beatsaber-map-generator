import os
import shutil

import torch
import wandb
from ignite.contrib.handlers.wandb_logger import WandBLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from config import *
from dl.models.onsets import SimpleOnsets
from training.loader import BaseLoader
from training.onset_ignite import ignite_train
from utils import MyDataParallel
import random


def main(run_parameters: RunConfig):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        SEED = random.randint(0, 2**32 - 1)  # or fix it for true reproducibility

        common_dataset_args = {
            "difficulty": DifficultyName[run_parameters.difficulty.upper()],
            "object_type": ObjectType[run_parameters.object_type.upper()],
            "enable_condition": run_parameters.enable_condition,
            "seq_length": run_parameters.seq_length,
            "skip_step": run_parameters.skip_step,
            "with_beats": run_parameters.with_beats,
            "batch_size": run_parameters.songs_batch_size,
            "num_workers": run_parameters.num_workers,
            # "split_seed": SEED,
        }

        train_dataset = BaseLoader(split=Split.TRAIN, **common_dataset_args)
        valid_dataset = BaseLoader(split=Split.VALIDATION, **common_dataset_args)

        model = SimpleOnsets(
            input_features=n_mels,
            output_features=1,
            dropout=run_parameters.dropout,
            rnn_dropout=run_parameters.rnn_dropout,
            enable_condition=run_parameters.enable_condition,
            num_layers=run_parameters.num_layers,
            enable_beats=run_parameters.with_beats,
            inference_chunk_length=round(run_parameters.seq_length / FRAME),
        ).to(device)

        if run_parameters.is_parallel:
            model = MyDataParallel(model)

        wandb_logger = WandBLogger(
            project="test-beat-saber-map-generator",
            config={**run_parameters},
            mode=run_parameters.wandb_mode,
        )

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("metrics/*", step_metric="epoch")
        wandb.define_metric("weight_sum/*", step_metric="epoch")
        wandb.define_metric("validation/step")
        wandb.define_metric("validation/*", step_metric="validation/step")

        train_dataset_len = len(train_dataset)
        valid_dataset_len = len(valid_dataset)

        train_loader = train_dataset.get_dataloader()
        valid_loader = valid_dataset.get_dataloader()

        # Define your optimizer
        optimizer = Adam(
            model.parameters(),
            run_parameters.start_lr,
            weight_decay=run_parameters.weight_decay,
        )

        if run_parameters.lr_scheduler_name == "CyclicLR":
            lr_scheduler = CyclicLR(
                optimizer,
                run_parameters.start_lr,
                run_parameters.end_lr,
                1000,
                cycle_momentum=False,
            )
        elif run_parameters.lr_scheduler_name == "CosineAnnealingLR":
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                run_parameters.epochs * 100,
                eta_min=run_parameters.eta_min,
            )
        else:
            raise ValueError("Unsupported scheduler")

        wandb.config.update(
            {
                "train_dataset_len": train_dataset_len,
                "valid_dataset_len": valid_dataset_len,
                "seed": SEED,
            }
        )

    except KeyboardInterrupt as e:
        print(e)
        if os.path.exists("dataset/valid_dataset"):
            shutil.rmtree("dataset/valid_dataset")
            print("Removed dataset/valid_dataset")

    try:
        ignite_train(
            train_dataset,
            valid_dataset,
            model,
            train_loader,
            valid_loader,
            optimizer,
            train_dataset_len,
            valid_dataset_len,
            device,
            lr_scheduler,
            wandb_logger,
            **run_parameters,  # type: ignore
        )
    except KeyboardInterrupt as e:
        print(e)
        if os.path.exists("dataset/valid_dataset"):
            shutil.rmtree("dataset/valid_dataset")
            print("Removed dataset/valid_dataset")

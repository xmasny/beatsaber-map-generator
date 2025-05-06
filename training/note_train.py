import os
import shutil
import torch
import wandb
import random

from ignite.contrib.handlers.wandb_logger import WandBLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

from config import *
from training.loader import BaseLoader
from training.note_ignite import ignite_train
from utils import MyDataParallel
from dl.models.layers import SparseNoteClassifier


def main(run_parameters: RunParams):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(
            run_parameters.gpu_index if run_parameters.gpu_index >= 0 else -1
        )
        SEED = random.randint(0, 2**32 - 1)

        common_dataset_args = {
            "difficulty": DifficultyName[run_parameters.difficulty.upper()],
            "object_type": ObjectType.COLOR_NOTES,
            "enable_condition": run_parameters.enable_condition,
            "seq_length": run_parameters.seq_length,
            "skip_step": run_parameters.skip_step,
            "with_beats": run_parameters.with_beats,
            "batch_size": run_parameters.batch_size,
            "num_workers": run_parameters.num_workers,
            "min_sum_votes": run_parameters.min_sum_votes,
            "min_score": run_parameters.min_score,
            "min_bpm": run_parameters.min_bpm,
            "max_bpm": run_parameters.max_bpm,
            "model_type": run_parameters.model_type,
            "mel_window": run_parameters.mel_window,
            "split_seed": SEED,
        }

        train_dataset = BaseLoader(split=Split.TRAIN, **common_dataset_args)
        valid_dataset = BaseLoader(split=Split.VALIDATION, **common_dataset_args)

        model = SparseNoteClassifier(
            window=run_parameters.mel_window,
            n_mels=n_mels,
            symbolic_dim=1,
            hidden_dim=128,
        ).to(device)

        if run_parameters.is_parallel:
            model = MyDataParallel(model)

        if wandb.run is not None:
            wandb.finish()

        wandb_logger = WandBLogger(
            project=run_parameters.wandb_project,
            config={**run_parameters},
            mode=run_parameters.wandb_mode,
            resume=run_parameters.wandb_resume,
            id=run_parameters.wandb_resume_id,
        )

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("metrics/*", step_metric="epoch")
        wandb.define_metric("weight_sum/*", step_metric="epoch")
        wandb.define_metric("validation/step")
        wandb.define_metric("validation/*", step_metric="epoch")

        train_dataset_len = len(train_dataset)
        valid_dataset_len = len(valid_dataset)

        train_loader = train_dataset.get_dataloader()
        valid_loader = valid_dataset.get_dataloader()

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
                run_parameters.epochs * run_parameters.epoch_length,
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

    except KeyboardInterrupt:
        if os.path.exists("dataset/valid_dataset"):
            shutil.rmtree("dataset/valid_dataset")

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
    except KeyboardInterrupt:
        if os.path.exists("dataset/valid_dataset"):
            shutil.rmtree("dataset/valid_dataset")

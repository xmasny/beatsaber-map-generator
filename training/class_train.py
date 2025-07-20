import os
import shutil
import requests
import torch
import wandb
import numpy as np

from ignite.contrib.handlers.wandb_logger import WandBLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

from config import *
from dl.models.onsets import MultiClassOnsetClassifier
from dl.models.util import pick_least_used_gpu
from training.class_loader import ClassBaseLoader
from training.class_ignite import ignite_train
from utils import ClassDataParallel

from dotenv import load_dotenv

load_dotenv()

base_dataset_api = os.getenv("BASE_DATASET_API", "http://localhost:8000/dataset/batch/")


def main(run_parameters: RunParams):
    try:
        payload = {
            "difficulty": run_parameters.difficulty.lower(),
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not run_parameters.is_parallel and run_parameters.gpu_index < 0:
            run_parameters.gpu_index = pick_least_used_gpu()

        torch.cuda.set_device(run_parameters.gpu_index)
        print(
            f"Assigned GPU: {run_parameters.gpu_index} - {torch.cuda.get_device_name(run_parameters.gpu_index)}"
        )

        response = requests.post(base_dataset_api + "claim-seed", json=payload)
        SEED = response.json().get("batch")

        if "split_seed" in run_parameters:
            SEED = run_parameters.split_seed

        common_dataset_args = {
            "difficulty": DifficultyName[run_parameters.difficulty.upper()],
            "object_type": ObjectType.COLOR_NOTES,
            "enable_condition": run_parameters.enable_condition,
            "batch_size": run_parameters.batch_size,
            "model_type": run_parameters.model_type,
            "split_seed": SEED,
        }

        train_dataset = ClassBaseLoader(split=Split.TRAIN, **common_dataset_args)
        valid_dataset = ClassBaseLoader(split=Split.VALIDATION, **common_dataset_args)

        class_count: ClassCount = np.load(
            f"dataset/batch/{run_parameters.difficulty.lower()}/{SEED}/class_count.npy",
            allow_pickle=True,
        ).item()

        model = MultiClassOnsetClassifier(
            class_count["train_class_count"],
            focal_loss_gamma=run_parameters.focal_loss_gamma,
            loss_fn=run_parameters.loss_fn,
        ).to(device)

        if run_parameters.is_parallel:
            model = ClassDataParallel(model)

        if wandb.run is not None:
            wandb.finish()

        wandb_logger = WandBLogger(
            project=run_parameters.wandb_project,
            config={**run_parameters},
            mode=run_parameters.wandb_mode,
        )

        print(wandb.config)

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("metrics/*", step_metric="epoch")
        wandb.define_metric("weight_sum/*", step_metric="epoch")
        wandb.define_metric("validation/step")
        wandb.define_metric("validation/*", step_metric="epoch")

        train_dataset_len = class_count["train_iterations"]
        valid_dataset_len = class_count["validation_iterations"]

        train_loader = train_dataset.get_dataloader(
            batch_size=run_parameters.batch_size
        )
        valid_loader = valid_dataset.get_dataloader(
            batch_size=run_parameters.batch_size
        )

        optimizer = AdamW(
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
                run_parameters.epochs * train_dataset_len,  # type: ignore
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

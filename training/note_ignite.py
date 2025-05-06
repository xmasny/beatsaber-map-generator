# training/ignite_note.py

import os
import shutil
from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from pathlib import Path
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Average
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from tqdm import tqdm
import wandb

from dl.models.layers import SparseNoteClassifier
from training.loader import BaseLoader
from utils import setup_checkpoint_upload


def score_function(engine: Engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def ignite_train(
    train_dataset: BaseLoader,
    valid_dataset: BaseLoader,
    model: SparseNoteClassifier,
    train_loader,
    valid_loader,
    optimizer: Optimizer,
    train_dataset_len,
    valid_dataset_len,
    device,
    lr_scheduler: CyclicLR | CosineAnnealingLR,
    wandb_logger: WandBLogger,
    **run_parameters,
):
    epochs = run_parameters.get("epochs", 100)
    epoch_length = run_parameters.get("epoch_length", 100)
    checkpoint_interval = run_parameters.get("checkpoint_interval", 100)
    validation_interval = run_parameters.get("validation_interval", 100)
    warmup_steps = run_parameters.get("warmup_steps", 0)
    wandb_mode = run_parameters.get("wandb_mode", "online")
    n_saved_model = run_parameters.get("n_saved_model", 10)
    n_saved_checkpoint = run_parameters.get("n_saved_checkpoint", 10)
    resume_checkpoint = run_parameters.get("resume_checkpoint", None)

    target_lr = optimizer.param_groups[0]["lr"]

    def cycle(iteration, num_songs_pbar: Optional[tqdm] = None):
        if num_songs_pbar:
            num_songs_pbar.reset()
        for index, songs in enumerate(iteration):
            for song in songs:
                if num_songs_pbar:
                    num_songs_pbar.update(1)
                if "not_working" in song:
                    continue
                for segment in train_dataset.process(song_meta=song):
                    yield segment

    def train_step(engine: Engine, batch):
        model.train()
        if warmup_steps > 0 and engine.state.iteration < warmup_steps:
            lr_scale = min(1.0, float(engine.state.iteration + 1) / float(warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * target_lr

        optimizer.zero_grad()
        preds, losses = model.run_on_batch(batch)
        loss = losses["loss"]
        loss.backward()
        optimizer.step()
        if lr_scheduler and engine.state.iteration >= warmup_steps:
            lr_scheduler.step()

        return preds, {k: v.item() for k, v in losses.items()}

    def eval_step(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            preds, losses = model.run_on_batch(batch)
            return preds, {k: v.item() for k, v in losses.items()}

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    # Attach metrics
    avg_loss = Average(output_transform=lambda output: output[1]["loss"])
    avg_loss.attach(trainer, "loss")
    avg_loss.attach(evaluator, "loss")
    # Average(output_transform=lambda x: x[1]["loss-color"]).attach(evaluator, "loss-color")
    # Average(output_transform=lambda x: x[1]["loss-direction"]).attach(evaluator, "loss-direction")
    # Average(output_transform=lambda x: x[1]["loss-x"]).attach(evaluator, "loss-x")
    # Average(output_transform=lambda x: x[1]["loss-y"]).attach(evaluator, "loss-y")

    epoch_length_valid = [None]
    epoch_length_train = [None]

    # Logging
    @trainer.on(Events.EPOCH_COMPLETED(every=validation_interval))
    def log_validation(engine: Engine):
        evaluator.run(
            cycle(valid_loader, valid_num_songs_pbar),
            epoch_length=epoch_length_valid[0] or None,
        )

        if epoch_length_valid[0] is None:
            epoch_length_valid[0] = evaluator.state.epoch_length  # type: ignore
            print(f"Discovered real epoch_length_valid: {epoch_length_valid[0]}")

        metrics = evaluator.state.metrics
        epoch = engine.state.epoch

        print(f"[Validation] Epoch {epoch} - Loss: {metrics['loss']:.4f}")
        if wandb_mode != "disabled":
            for k, v in metrics.items():
                wandb_logger.log({f"validation/{k}": v, "epoch": epoch})

    # Checkpointing
    to_save = {"model": model, "optimizer": optimizer}
    if wandb_mode != "disabled":

        if resume_checkpoint:
            checkpoint_path = os.path.join(wandb.run.dir, "checkpoints", f"checkpoint_<X>.pth")  # type: ignore
            Checkpoint.load_objects(
                to_load={"model": model, "optimizer": optimizer},
                checkpoint=torch.load(checkpoint_path),
            )

        handler = Checkpoint(
            to_save,
            DiskSaver(os.path.join(wandb.run.dir, "checkpoints"), create_dir=True, require_empty=False),  # type: ignore
            n_saved=n_saved_checkpoint,
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=(validation_interval * epoch_length)),
            handler,
        )

        best_checkpoint = Checkpoint(
            to_save,
            DiskSaver(os.path.join(wandb.run.dir), create_dir=False, require_empty=False),  # type: ignore
            n_saved=2,
            score_function=score_function,
            score_name="validation_loss",
            greater_or_equal=True,
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=validation_interval), best_checkpoint
        )

        best_model = ModelCheckpoint(
            dirname=os.path.join(wandb.run.dir),  # type: ignore
            filename_prefix="model",
            n_saved=2,
            create_dir=True,
            require_empty=False,
            score_function=score_function,
            score_name="validation_loss",
            greater_or_equal=True,
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=validation_interval),
            best_model,
            {"mymodel": model},
        )

        model_handler = ModelCheckpoint(
            dirname=os.path.join(wandb.run.dir, "model_checkpoints"),  # type: ignore
            filename_prefix="model",
            n_saved=n_saved_model,
            create_dir=True,
            require_empty=False,
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=validation_interval * epoch_length),
            model_handler,
            {"mymodel": model},
        )

        wandb.watch(model, log="all", criterion=avg_loss)

        setup_checkpoint_upload(trainer, {"model": model, "optimizer": optimizer}, wandb.run.dir, validation_interval=validation_interval)  # type: ignore

    # Progress bars
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: x[1]["loss"])
    ProgressBar(persist=True).attach(evaluator, output_transform=lambda x: x[1]["loss"])

    train_num_songs_pbar = tqdm(total=train_dataset_len, desc="Train songs")
    valid_num_songs_pbar = tqdm(total=valid_dataset_len, desc="Valid songs")

    for epoch in range(epochs):
        trainer.run(
            cycle(train_loader, train_num_songs_pbar),
            max_epochs=1,
            epoch_length=epoch_length_train[0] or None,
        )

        if epoch_length_train[0] is None:
            epoch_length_train[0] = trainer.state.epoch_length  # type: ignore
            print(f"Discovered real epoch_length_train: {epoch_length_train[0]}")

    if wandb_mode != "disabled":
        wandb_logger.close()

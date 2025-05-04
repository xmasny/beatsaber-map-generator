# training/ignite_note.py

import os
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from pathlib import Path
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Average
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger

from dl.models.layers import (
    AudioSymbolicNoteSelectorMultiHead,
    AudioSymbolicNoteSelector,
)


def ignite_note_train(
    model: AudioSymbolicNoteSelectorMultiHead | AudioSymbolicNoteSelector,
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
    checkpoint_interval = run_parameters.get("checkpoint_interval", 200)
    validation_interval = run_parameters.get("validation_interval", 100)
    warmup_steps = run_parameters.get("warmup_steps", 0)
    wandb_mode = run_parameters.get("wandb_mode", "online")

    target_lr = optimizer.param_groups[0]["lr"]

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
    Average(output_transform=lambda x: x[1]["loss"]).attach(trainer, "loss")
    Average(output_transform=lambda x: x[1]["loss"]).attach(evaluator, "loss")
    Average(output_transform=lambda x: x[1]["loss-color"]).attach(
        evaluator, "loss-color"
    )
    Average(output_transform=lambda x: x[1]["loss-direction"]).attach(
        evaluator, "loss-direction"
    )
    Average(output_transform=lambda x: x[1]["loss-x"]).attach(evaluator, "loss-x")
    Average(output_transform=lambda x: x[1]["loss-y"]).attach(evaluator, "loss-y")

    # Logging
    @trainer.on(Events.EPOCH_COMPLETED(every=validation_interval))
    def log_validation(engine: Engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        epoch = engine.state.epoch

        print(f"[Validation] Epoch {epoch} - Loss: {metrics['loss']:.4f}")
        if wandb_mode != "disabled":
            for k, v in metrics.items():
                wandb_logger.log({f"validation/{k}": v, "epoch": epoch})

    # Checkpointing
    checkpoint_dir = Path("logs") / "checkpoints"
    to_save = {"model": model, "optimizer": optimizer}
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=checkpoint_interval),
        Checkpoint(to_save, DiskSaver(checkpoint_dir, create_dir=True), n_saved=5),
    )

    # Progress bars
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: x[1]["loss"])
    ProgressBar(persist=True).attach(evaluator, output_transform=lambda x: x[1]["loss"])

    trainer.run(train_loader, max_epochs=epochs, epoch_length=epoch_length)

    if wandb_mode != "disabled":
        wandb_logger.close()

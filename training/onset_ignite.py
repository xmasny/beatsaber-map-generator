import os
import shutil
from logging import getLogger
from math import ceil
from pathlib import Path

import numpy as np
import torch
import wandb

from config import *
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torch.optim.optimizer import Optimizer

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger

from ignite.metrics import Average

from dl.models.onsets import OnsetsBase
from dl.models.util import generate_window_data
from notes_generator.training.evaluate import evaluate
from training.loader import BaseLoader

from tqdm import tqdm

from utils import setup_checkpoint_upload


logger = getLogger(__name__)


def get_number_of_difficulties(song):
    return sum(bool(song.get(diff)) for diff in DIFFICULTY_NAMES)


def generate_valid_length(
    valid_dataset: BaseLoader,
    valid_loader,
    batch_size: int,
):
    # Temporarily skip preprocessing

    valid_dataset.skip_processing = True

    pbar = tqdm(
        valid_loader,
        total=ceil(len(valid_dataset) / batch_size),
        desc="Generate valid length",
    )
    valid_dataset_len = 0
    for songs in pbar:
        for song in songs:
            try:
                song_iterations = generate_window_data(
                    data_length=song["frames"]
                ) * get_number_of_difficulties(song)
                valid_dataset_len += song_iterations
            except Exception as e:
                print(e)
                continue
    pbar.close()

    # Restore full preprocessing mode
    valid_dataset.skip_processing = False

    return ceil(valid_dataset_len / batch_size)


def write_metrics(metrics, mode: str, epoch: int, wandb_mode: str):
    loss = metrics["loss"]
    logger.info(f"{mode} Results - Epoch: {epoch}  " f"Avg loss: {loss:.4f}")
    # wandb
    if wandb_mode == "disabled":
        return
    wandb.log(
        {
            f"metrics/{mode}-avg_loss": loss,
            f"metrics/{mode}-avg_loss_onset": metrics["loss-onset"],
            "epoch": epoch,
        }
    )
    if "loss-notes" in metrics:
        wandb.log(
            {f"metrics/{mode}-avg_loss_notes": metrics["loss-notes"], "epoch": epoch}
        )


def score_function(engine: Engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def ignite_train(
    train_dataset: BaseLoader,
    valid_dataset: BaseLoader,
    model: OnsetsBase,
    train_loader,
    valid_loader,
    optimizer: Optimizer,
    train_dataset_len,
    valid_dataset_len,
    device,
    lr_scheduler: CyclicLR | CosineAnnealingLR,
    wandb_logger: Optional[WandBLogger],
    **run_parameters,
) -> None:

    resume_checkpoint = run_parameters.get("resume_checkpoint", None)
    batch_size = run_parameters.get("batch_size", 20)
    fuzzy_width = run_parameters.get("fuzzy_width", 1)
    fuzzy_scale = run_parameters.get("fuzzy_scale", 1.0)
    enable_early_stop = run_parameters.get("enable_early_stop", True)
    patience = run_parameters.get("patience", 10)
    loss_interval = run_parameters.get("loss_interval", 100)
    validation_interval = run_parameters.get("validation_interval", 100)
    checkpoint_interval = run_parameters.get("checkpoint_interval", 200)
    n_saved_checkpoint = run_parameters.get("n_saved_checkpoint", 10)
    n_saved_model = run_parameters.get("n_saved_model", 40)
    disable_eval = run_parameters.get("disable_eval", False)
    eval_tolerance = run_parameters.get("eval_tolerance", 0.05)
    log_dir = run_parameters.get("log_dir", "logs")
    fuzzy_scale = run_parameters.get("fuzzy_scale", 1.0)
    fuzzy_width = run_parameters.get("fuzzy_width", 1)
    epoch_length = run_parameters.get("epoch_length", 100)
    warmup_steps = run_parameters.get("warmup_steps", 0)
    epochs = run_parameters.get("epochs", 100)
    wandb_mode = run_parameters.get("wandb_mode", "online")

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

    # Define a function to handle a single training iteration
    def train_step(engine: Engine, batch):
        if warmup_steps > 0 and engine.state.iteration < warmup_steps:
            lr_scale = min(1.0, float(engine.state.iteration + 1) / float(warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * target_lr
        optimizer.zero_grad()
        predictions, losses = model.run_on_batch(batch, fuzzy_width, fuzzy_scale)
        loss = sum(losses.values())
        loss.backward()  # type: ignore
        optimizer.step()
        if lr_scheduler:
            if warmup_steps > 0 and engine.state.iteration < warmup_steps:
                pass
            else:
                lr_scheduler.step()

        losses = {key: value.item() for key, value in {"loss": loss, **losses}.items()}

        i = engine.state.iteration

        lr = [pg["lr"] for pg in optimizer.param_groups][-1]

        if wandb_mode != "disabled":
            wandb.log({"train/lr": lr, "train/step": i})

        for key, value in losses.items():
            if wandb_mode != "disabled":
                wandb.log({f"train/{key}": value, "train/step": i})

        return predictions, losses

    # Define a function to handle a single evaluation iteration
    def eval_step(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            predictions, losses = model.run_on_batch(batch)
            loss = sum(losses.values())
            losses = {
                key: value.item() for key, value in {"loss": loss, **losses}.items()
            }
            model.train()
            return predictions, losses

    # Create Ignite engines
    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    target_lr = [pg["lr"] for pg in optimizer.param_groups][-1]

    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: x[1]["loss"])
    ProgressBar(persist=True).attach(evaluator, output_transform=lambda x: x[1]["loss"])

    checkpoint = Path(log_dir) / "checkpoint"

    @trainer.on(Events.STARTED)
    def resume_training(engine: Engine):
        if resume_checkpoint:
            engine.state.iteration = resume_checkpoint
            engine.state.epoch = int(resume_checkpoint / engine.state.epoch_length)

    @trainer.on(Events.ITERATION_COMPLETED(every=loss_interval))
    def log_training_loss(engine: Engine):
        if isinstance(engine.state.output, dict):
            loss = engine.state.output.get("loss")
            if loss is not None:
                iteration_max = (
                    engine.state.max_epochs * engine.state.epoch_length
                    if engine.state.max_epochs is not None
                    and engine.state.epoch_length is not None
                    else None
                )
                logger.info(
                    f"Iteration[{engine.state.iteration}/{iteration_max}] "
                    f"Loss: {loss:.4f}"
                )

    @trainer.on(Events.EPOCH_COMPLETED(every=validation_interval))
    def log_validation_results(engine: Engine):
        i = engine.state.iteration
        lr = [pg["lr"] for pg in optimizer.param_groups][-1]

        evaluator.run(
            cycle(valid_loader, valid_num_songs_pbar),
            # epoch_length=epoch_length_valid,
        )
        model.eval()
        with torch.no_grad():
            if disable_eval:
                pass
            else:
                for key, value in evaluate(
                    model,
                    cycle(valid_loader),
                    device,
                    epoch_length_valid,
                    eval_tolerance,
                ).items():
                    k = "validation/" + key.replace(" ", "_")
                    v = np.mean(value)
                    if wandb_mode != "disabled":
                        wandb.log({k: v, "validation/step": i})

            if wandb_mode != "disabled":
                wandb.log({"validation/lr": lr, "validation/step": i})

        metrics = evaluator.state.metrics
        write_metrics(metrics, "validation", engine.state.epoch, wandb_mode)
        model.train()

    @trainer.on(Events.EPOCH_COMPLETED(every=validation_interval))
    def weight_sum(engine: Engine):
        if wandb_mode == "disabled":
            return
        # Define weights for each metric
        f1_accuracy = wandb.summary["validation/metric/onset/f1"]
        train_loss = wandb.summary["train/loss"]
        val_loss = wandb.summary["metrics/validation-avg_loss"]

        f1_weight = 1.0
        train_loss_weight = -1.0
        val_loss_weight = -1.0

        # Compute the combined metric
        weight_sum = (
            f1_weight * f1_accuracy
            + train_loss_weight * train_loss
            + val_loss_weight * val_loss
        )

        wandb.log(
            {
                "weight_sum/f1/train_loss/val_loss": weight_sum,
                "epoch": engine.state.epoch,
            }
        )

    avg_loss = Average(output_transform=lambda output: output[1]["loss"])
    avg_loss_onset = Average(output_transform=lambda output: output[1]["loss-onset"])
    avg_loss.attach(trainer, "loss")
    avg_loss.attach(evaluator, "loss")
    avg_loss_onset.attach(evaluator, "loss-onset")
    if enable_early_stop:
        handler = EarlyStopping(
            patience=patience, score_function=score_function, trainer=trainer
        )
        evaluator.add_event_handler(Events.COMPLETED, handler)
    to_save = {"trainer": trainer, "optimizer": optimizer}
    if lr_scheduler:
        to_save["lr_scheduler"] = lr_scheduler
    if checkpoint.exists() and not resume_checkpoint:
        shutil.rmtree(str(checkpoint))
    if wandb_mode != "disabled":

        handler = Checkpoint(
            to_save,
            DiskSaver(os.path.join(wandb.run.dir, "checkpoints"), create_dir=True, require_empty=False),  # type: ignore
            n_saved=n_saved_checkpoint,
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=checkpoint_interval), handler
        )

        best_checkpoint = Checkpoint(
            to_save,
            DiskSaver(os.path.join(wandb.run.dir), create_dir=False, require_empty=False),  # type: ignore
            n_saved=1,
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
            n_saved=1,
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
            Events.ITERATION_COMPLETED(every=checkpoint_interval),
            model_handler,
            {"mymodel": model},
        )

        wandb.watch(model, log="all", criterion=avg_loss)

        setup_checkpoint_upload(trainer, {"model": model, "optimizer": optimizer}, wandb.run.dir, validation_interval=validation_interval)  # type: ignore

    train_num_songs_pbar = tqdm(total=train_dataset_len, desc="Train songs")
    valid_num_songs_pbar = tqdm(total=valid_dataset_len, desc="Valid songs")

    epoch_length_valid = generate_valid_length(
        valid_dataset=valid_dataset,
        valid_loader=valid_loader,
        batch_size=batch_size,
    )

    logger.info(
        f"epoch_length: {epoch_length} epoch_length_valid: {epoch_length_valid}"
    )
    # Run the training process
    trainer.run(
        cycle(train_loader, train_num_songs_pbar),
        max_epochs=epochs,
        epoch_length=epoch_length,
    )

    if wandb_mode != "disabled":
        wandb.join()

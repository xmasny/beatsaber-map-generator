import math
import shutil
from logging import getLogger
from math import ceil
from pathlib import Path

import numpy as np
import torch
import wandb

from config import *
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger

from ignite.metrics import Average

from dl.models.onsets import OnsetsBase, SimpleOnsets
from notes_generator.training.evaluate import concatenate_tensors_by_key, evaluate
from training.loader import BaseLoader

from tqdm import tqdm


# logger = getLogger(__name__)


def generate_valid_length(
    valid_loader,
    dataset: BaseLoader,
    song_batch_num: int,
    batch_size: int,
):
    pbar = tqdm(valid_loader, total=song_batch_num)
    valid_dataset_len = 0
    for songs in pbar:
        for song in DataLoader(songs, collate_fn=collate_fn):
            for segment in dataset.process_song(
                song=song,
                beats_array=song["data"]["beats"],
                condition=song["data"]["condition"],
                onsets=song["data"]["onset"],
            ):
                valid_dataset_len += 1
    return ceil(valid_dataset_len / batch_size)


def score_function(engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def collate_fn(batch):
    return batch[0]


def ignite_train(
    dataset: BaseLoader,
    model: OnsetsBase,
    train_loader,
    valid_loader,
    optimizer,
    train_dataset_len,
    valid_dataset_len,
    device,
    wandb_logger,
    **run_parameters,
) -> None:

    resume_checkpoint = run_parameters.get("resume_checkpoint", None)
    train_batch_size = run_parameters.get("train_batch_size", 20)
    songs_batch_size = run_parameters.get("songs_batch_size", 20)
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
    lr_find = run_parameters.get("lr_find", False)
    start_lr = run_parameters.get("start_lr", 1e-4)
    end_lr = run_parameters.get("end_lr", 1e-4)
    epoch_length = run_parameters.get("epoch_length", 100)
    warmup_steps = run_parameters.get("warmup_steps", 0)
    epochs = run_parameters.get("epochs", 100)
    wandb_mode = run_parameters.get("wandb_mode", "disabled")

    if lr_find:
        lr_find_loss = []
        lr_find_lr = []
        optimizer = torch.optim.Adam(model.parameters(), start_lr)
        lr_find_epochs = 2
        lr_lambda = lambda x: math.exp(
            x * math.log(end_lr / start_lr) / (lr_find_epochs * epoch_length)
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        smoothing = 0.05

    def cycle(iteration, num_songs_pbar: Optional[tqdm] = None):
        if num_songs_pbar:
            num_songs_pbar.reset()
        segment_batch = []
        # while True:
        for index, songs in enumerate(iteration):
            for song in DataLoader(songs, collate_fn=collate_fn):
                if num_songs_pbar:
                    num_songs_pbar.update(1)
                for segment in dataset.process_song(
                    song=song,
                    beats_array=song["data"]["beats"],
                    condition=song["data"]["condition"],
                    onsets=song["data"]["onset"],
                ):

                    segment_batch.append(segment)
                    if len(segment_batch) == train_batch_size:
                        segment_batch = concatenate_tensors_by_key(segment_batch)
                        yield segment_batch
                        segment_batch = []

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
        if lr_find:
            loss_v = loss.item()  # type: ignore
            i = engine.state.iteration
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)
            if i == 1:
                lr_find_loss.append(loss_v)
            else:
                loss_v = smoothing * loss_v + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss_v)

        losses = {key: value.item() for key, value in {"loss": loss, **losses}.items()}

        i = engine.state.iteration

        return predictions, losses

    # Define a function to handle a single evaluation iteration
    def eval_step(engine, batch):
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
    def resume_training(engine):
        if resume_checkpoint:
            engine.state.iteration = resume_checkpoint
            engine.state.epoch = int(resume_checkpoint / engine.state.epoch_length)

    @trainer.on(Events.ITERATION_COMPLETED(every=loss_interval))
    def log_training_loss(engine):
        loss = engine.state.output[1]["loss"]
        iteration_max = engine.state.max_epochs * engine.state.epoch_length
        """ logger.info(
            f"Iteration[{engine.state.iteration}/{iteration_max}] " f"Loss: {loss:.4f}"
        ) """

    @trainer.on(Events.ITERATION_COMPLETED(every=validation_interval))
    def log_validation_results(engine):
        i = engine.state.iteration
        lr = [pg["lr"] for pg in optimizer.param_groups][-1]

        evaluator.run(
            cycle(valid_loader, valid_num_songs_pbar),
            epoch_length=epoch_length_valid,
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
                    k = "validation-" + key.replace(" ", "_")
                    v = np.mean(value)
                    if wandb_mode != "disabled":
                        wandb.log({k: v}, step=i)

        metrics = evaluator.state.metrics
        model.train()

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
    if checkpoint.exists() and not resume_checkpoint:
        shutil.rmtree(str(checkpoint))
    handler = Checkpoint(
        to_save,
        DiskSaver(str(checkpoint), create_dir=True, require_empty=False),
        n_saved=n_saved_checkpoint,
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=checkpoint_interval), handler
    )
    model_handler = ModelCheckpoint(
        dirname=str(checkpoint),
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
    if wandb_mode != "disabled":
        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"loss": loss},
        )

        wandb_logger.watch(model)

    train_num_songs_pbar = tqdm(total=train_dataset_len)
    valid_num_songs_pbar = tqdm(total=valid_dataset_len)

    valid_song_batch_num = ceil(valid_dataset_len / songs_batch_size)
    epoch_length_valid = generate_valid_length(
        valid_loader,
        dataset,
        valid_song_batch_num,
        train_batch_size,
    )

    # Run the training process
    trainer.run(
        cycle(train_loader, train_num_songs_pbar),
        max_epochs=epochs,
        epoch_length=epoch_length,
    )

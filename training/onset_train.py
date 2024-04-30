import os
import numpy as np
import torch
import wandb
from dl.models.onsets import SimpleOnsets
from training.onset_ignite import ignite_train
from utils import MyDataParallel
from training.loader import BaseLoader
from config import *
from torch.utils.data import DataLoader
from ignite.contrib.handlers.wandb_logger import WandBLogger

import shutil

from tqdm import tqdm


def non_collate(batch):
    return batch


_valid_dataset_path = "dataset/valid_dataset/{object_type}/{difficulty}"


def save_valid_data(
    valid_loader: DataLoader, valid_dataset_len: int, run_parameters: RunConfig
):
    path = _valid_dataset_path.format(
        object_type=run_parameters.object_type, difficulty=run_parameters.difficulty
    )

    if os.path.exists(path):
        shutil.rmtree(
            "dataset/valid_dataset",
        )

    os.makedirs(path)
    pbar = tqdm(total=valid_dataset_len, desc="Saving valid dataset")

    for i, batch in enumerate(valid_loader):
        for song in batch:
            np.save(
                f'{path}/song{song["id"]}_{song["song_id"]}.npy',
                song,
                allow_pickle=True,
            )
            pbar.update(1)


def saved_valid_loader(run_parameters: RunConfig):
    path = _valid_dataset_path.format(
        object_type=run_parameters.object_type, difficulty=run_parameters.difficulty
    )
    for song in os.listdir(path):

        yield np.load(f"{path}/{song}", allow_pickle=True).item()


def main(run_parameters: RunConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    difficulty = getattr(DifficultyName, f"{run_parameters.difficulty.upper()}").name
    object_type = getattr(ObjectType, f"{run_parameters.object_type.upper()}").name
    dataset = BaseLoader(
        difficulty=DifficultyName[difficulty],
        object_type=ObjectType[object_type],
        enable_condition=run_parameters.enable_condition,
        seq_length=run_parameters.seq_length,
        skip_step=run_parameters.skip_step,
        with_beats=run_parameters.with_beats,
    )
    model = SimpleOnsets(
        input_features=n_mels,
        output_features=1,
        dropout=run_parameters.dropout,
        rnn_dropout=run_parameters.rnn_dropout,
        enable_condition=run_parameters.enable_condition,
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

    dataset.load()

    train_dataset = dataset[Split.TRAIN]
    valid_dataset = dataset[Split.VALIDATION]

    train_dataset_len = train_dataset.n_shards
    valid_dataset_len = valid_dataset.n_shards

    train_loader = DataLoader(train_dataset, batch_size=run_parameters.songs_batch_size, collate_fn=non_collate)  # type: ignore
    valid_loader = DataLoader(valid_dataset, batch_size=run_parameters.songs_batch_size, collate_fn=non_collate)  # type: ignore
    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=run_parameters.lr)
    wandb.config.update(
        {
            "train_dataset_len": train_dataset_len,
            "valid_dataset_len": valid_dataset_len,
        }
    )
    if run_parameters.save_valid_dataset:
        save_valid_data(valid_loader, valid_dataset_len, run_parameters)
        saved_loader = saved_valid_loader(run_parameters)
        valid_loader = DataLoader(saved_loader, batch_size=run_parameters.songs_batch_size, collate_fn=non_collate)  # type: ignore

    try:
        ignite_train(
            dataset,
            model,
            train_loader,
            valid_loader,
            optimizer,
            train_dataset_len,
            valid_dataset_len,
            device,
            wandb_logger,
            **run_parameters,  # type: ignore
        )
    except KeyboardInterrupt as e:
        print(e)
        wandb_logger.close()

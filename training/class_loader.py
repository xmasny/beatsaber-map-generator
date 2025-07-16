import logging
import os
import time

import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import *
from training.downloader import Downloader, download_fn


base_dataset_api = "http://kaistore.dcs.fmph.uniba.sk/api"

base_dataset_path = "http://kaistore.dcs.fmph.uniba.sk/beatsaber-map-generator/"

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s - %(asctime)s - %(message)s",
    filename="data_tester.log",
)


def group_npz_pairs(npz_data: dict) -> list[dict]:
    result = []
    keys = npz_data.keys()

    mel_keys = [k for k in keys if k.endswith("_mel")]

    for mel_key in tqdm(mel_keys, desc="Pairing mel/classes", unit="pair"):
        base = mel_key[:-4]  # Strip "_mel"
        class_key = base + "_classes"

        if class_key in npz_data:
            result.append(
                {
                    "id": base,
                    "mel": npz_data[mel_key],
                    "classes": npz_data[class_key],
                }
            )

    return result


def flatten_collate(batch: list[list[dict]]) -> list[dict]:
    # batch is a list of lists of dicts
    # flatten it
    return [item for sublist in batch for item in sublist]


def non_collate(batch):
    return batch[0]


class ClassBaseLoader(Dataset):
    def __init__(
        self,
        difficulty: DifficultyName = DifficultyName.ALL,
        object_type: ObjectType = ObjectType.COLOR_NOTES,
        enable_condition: bool = True,
        split: Split = Split.TRAIN,
        split_seed: str = "42",
        batch_size: int = 2,
        num_workers: int = 0,
        model_type: str = "class",
    ):
        self.difficulty = difficulty
        self.object_type = object_type
        self.enable_condition = enable_condition
        self.split = split
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type

        self.downloader = Downloader(
            download_fn, position_start=4 if split == Split.TRAIN else 5
        )
        self.files = []
        self.no_songs = 0

        self._load()

        if self.split == Split.VALIDATION and self.files:
            for f in self.files:
                self.downloader.enqueue(f, f)

    def _load(self):
        response = requests.post(
            f"{base_dataset_api}/get-list-of-batches",
            json={
                "difficulty": self.difficulty.value.lower(),
                "split_seed": self.split_seed,
                "split": self.split.value,
                "model_type": self.model_type,
            },
        ).json()

        class_count_path = f"dataset/batch/{self.difficulty.value.lower()}/{self.split_seed}/class_count.npy"
        self.downloader.enqueue(class_count_path, class_count_path)

        self.files = response.get("files", [])
        self.no_songs = response.get("no_songs", 0)

    def collate_batch(self, batch: list[dict]):
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0]
            if key != "id"
        }

    def process(self, file_path):
        batch = []
        with np.load(file_path, allow_pickle=True) as file:
            for row in self.iter_file(file):
                batch.append(row)
                if len(batch) == self.batch_size:
                    yield self.collate_batch(batch)
                    batch = []
            if batch:
                yield self.collate_batch(batch)

    def get_dataloader(self, batch_size=None):
        return DataLoader(
            self,
            collate_fn=non_collate,
        )

    def iter_file(self, file):
        file_iter = iter(file.files)
        while True:
            try:
                classes_key = next(file_iter)
                mel_key = next(file_iter)
            except StopIteration:
                break  # no more pairs

            id = mel_key[:-4]  # remove "_mel"
            yield {
                "id": id,
                "mel": torch.tensor(file[mel_key], dtype=torch.float32),
                "classes": torch.tensor(file[classes_key], dtype=torch.float32),
            }

    def __getitem__(self, idx):
        if self.split == Split.TRAIN:
            max_lookahead = 3
            for i in range(idx, min(idx + max_lookahead, len(self.files))):
                file_path = self.files[i]
                self.downloader.enqueue(file_path, file_path)

        current_path = self.files[idx]
        self.downloader.enqueue(current_path, current_path)

        while not os.path.exists(current_path):
            time.sleep(1)

        if idx > 0 and self.split == Split.TRAIN and self.model_type == "onset":
            prev_path = self.files[idx - 1]
            if os.path.exists(prev_path):
                try:
                    os.remove(prev_path)
                    print(f"[Cleanup] Deleted previous file: {prev_path}")
                except Exception as e:
                    print(f"[Cleanup error] Could not delete {prev_path}: {e}")

        return current_path

    def __len__(self):
        return len(self.files)

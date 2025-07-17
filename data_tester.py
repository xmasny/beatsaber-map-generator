import wandb
from config import DifficultyName, Split
from training.onset_loader import TestDataset

from torch.utils.data import DataLoader

from tqdm import tqdm

from math import ceil

import logging

from training.onset_train import non_collate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", filename="data_tester.log"
)

difficulty = "EASY"
batch_size = 2
num_workers = 0

dataset = TestDataset(
    difficulty=DifficultyName[difficulty],
)

wandb.init(
    project="test-beat-saber-map-generator",
    config={"difficulty": difficulty},
    mode="disabled",
)

dataset.load()

all_data = dataset[Split.ALL]

dataloader = DataLoader(all_data, batch_size=batch_size, num_workers=num_workers, collate_fn=non_collate)  # type: ignore

all_data_len = all_data.n_shards

pbar = tqdm(dataloader, total=ceil(all_data_len / batch_size))

pbar_songs = tqdm(total=all_data_len)

print(f"Number of shards: {all_data_len}")

for i, batch in enumerate(pbar):
    try:
        for song in batch:
            pbar_songs.update(1)
            pass
    except Exception as e:
        logging.error(e, song["song_id"], song["id"])
        print(e, song["song_id"], song["id"])
        pbar_songs.update(1)


wandb.finish()

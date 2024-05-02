from config import DifficultyName, ObjectType, Split
from training.loader import BaseLoader

from torch.utils.data import DataLoader

from tqdm import tqdm

from math import ceil

import logging

from training.onset_train import non_collate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", filename="data_tester.log"
)

difficulty = input("Enter difficulty: ").upper()
batch_size = int(input("Enter batch size: "))
num_workers = int(input("Enter number of workers: "))
object_type = "color_notes".upper()

dataset = BaseLoader(
    difficulty=DifficultyName[difficulty],
    object_type=ObjectType[object_type],
)

dataset.load()

all_data = dataset[Split.ALL]

dataloader = DataLoader(all_data, batch_size=batch_size, num_workers=num_workers, collate_fn=non_collate)  # type: ignore

all_data_len = all_data.n_shards

pbar = tqdm(dataloader, total=ceil(all_data_len / batch_size))

print(f"Number of shards: {all_data_len}")
try:
    for i, batch in enumerate(pbar):
        try:
            for song in batch:
                pass
        except Exception as e:
            logging.error(e, song["song_id"], song["id"])
            print(e, song["song_id"], song["id"])
except Exception as e:
    logging.error(e)
    print(e)

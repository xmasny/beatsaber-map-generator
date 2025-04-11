import typing

import torch
from torch import nn

import csv
import os


def round_decimal(x: torch.Tensor, n_dig: int) -> torch.Tensor:
    return torch.round(x * 10**n_dig) / (10**n_dig)


def batch_first(data):
    shapes = [-1] + list(data.shape[1:])
    return data.reshape(*shapes)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def convert1d(target):
    left = target[:, :, 0]
    right = target[:, :, 1]
    target = left * 8 + right
    return target


def log_window_to_csv(
    csv_path: str,
    start_index: int,
    end_index: int,
    audio_length: int,
    params: dict,
):
    # Check if file exists to write header only once
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["start_index", "end_index", "audio_length", "params"])

        writer.writerow([start_index, end_index, audio_length, str(params)])


def generate_window_data(data_length, seq_length=500, skip_step=62):
    count = 0
    for skip in range(0, seq_length, skip_step):  # this controls how the sliding starts
        for start in range(skip, data_length - seq_length + 1, seq_length):
            count += 1
    return count


class MyDataParallel(nn.DataParallel):
    def run_on_batch(
        self,
        batch,
        fuzzy_width=1,
        fuzzy_scale=1.0,
        merge_scale: typing.Optional[float] = None,
    ):
        return self.module.run_on_batch(
            batch,
            fuzzy_width=fuzzy_width,
            fuzzy_scale=fuzzy_scale,
            merge_scale=merge_scale,
            net=self,
        )

    def predict(self, batch):
        return self.module.predict(batch)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict)

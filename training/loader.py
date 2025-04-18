from io import BytesIO
import logging
import os
import re
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import requests
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from config import *
from utils import clean_data

base_dataset_path = (
    "http://kaistore.dcs.fmph.uniba.sk/beatsaber-map-generator/dataset/beatmaps"
)

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s - %(asctime)s - %(message)s",
    filename="data_tester.log",
)


def non_collate(batch):
    return batch


# Cache to store splits by (object_type, seed)
_SPLIT_CACHE = {}


class BaseLoader(Dataset):
    def __init__(
        self,
        difficulty: DifficultyName = DifficultyName.ALL,
        object_type: ObjectType = ObjectType.COLOR_NOTES,
        enable_condition: bool = True,
        with_beats: bool = True,
        seq_length: int = 16000,
        skip_step: int = 2000,
        valid_split_ratio: float = 0.2,
        skip_processing: bool = False,
        split: Split = Split.TRAIN,
        split_seed: int = 42,
        batch_size: int = 2,
        num_workers: int = 0,
    ):
        self.difficulty = difficulty
        self.object_type = object_type
        self.enable_condition = enable_condition
        self.with_beats = with_beats
        self.seq_length = seq_length
        self.skip_step = skip_step
        self.valid_split_ratio = valid_split_ratio
        self.skip_processing = skip_processing
        self.split = split
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = {}
        self.indices = {}
        self._load_and_split()

    def _load_and_split(self):
        dataset_path = f"{base_dataset_path}/{self.object_type.value}/metadata.csv"
        df = pd.read_csv(dataset_path)
        df = clean_data(df)
        self.df = df

        cache_key = (self.object_type.value, self.split_seed)

        if cache_key in _SPLIT_CACHE:
            self.indices = _SPLIT_CACHE[cache_key]
            return

        indices = np.arange(len(df))
        np.random.seed(self.split_seed)
        np.random.shuffle(indices)

        train_end = int(0.6 * len(indices))
        val_end = int(0.8 * len(indices))

        self.indices = {
            "train": indices[:train_end],
            "validation": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        _SPLIT_CACHE[cache_key] = self.indices

    def process(self, song_meta):
        song_path = (
            f"{base_dataset_path}/{self.object_type.value}/npz/{song_meta['song']}"
        )

        try:
            response = requests.get(song_path)
            response.raise_for_status()  # raises error if 404 or similar
            song = dict(np.load(BytesIO(response.content), allow_pickle=True))
        except AssertionError as e:
            logging.error(
                f'song{song_meta["id"]}_{song_meta["song_id"]} difficulty {self.difficulty}: {e}'
            )
            print(
                f'Error in song{song_meta["id"]}_{song_meta["song_id"]} difficulty {self.difficulty}: {e}'
            )
            song_meta["not_working"] = True

        song_meta["data"] = song

        if not self.with_beats and "not_working" not in song_meta:
            song_meta["data"].pop("beats_array")

        return song_meta

    def get_dataloader(self, shuffle=True):
        subset = Subset(self, self.indices[self.split.value])  # type: ignore
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=non_collate,
        )

    def get_split_df(self, split: Split):
        return self.df.iloc[self.indices[split.value]]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.skip_processing:
            return row
        return self.process(row.to_dict())

    def __len__(self):
        return len(self.indices[self.split.value])


def iter_array(array, length, skip, params):
    max_len = len(array)
    for i in range(skip, max_len, length):
        data = array[i : (i + length)]
        if len(data) == length:
            yield data, i, i + length, params


def get_bpm_info(song):
    bpm = song["meta"]["bpm"]

    energy = np.sum(song["data"]["mel"], axis=0)
    start_index = np.argmax(energy > 0)

    start_time = librosa.frames_to_time(start_index, sr=sample_rate)

    start = float(start_time * 1000)
    beats = int(4)
    return [(bpm, start, beats)]


def get_onset_array(song: dict, difficulty: DifficultyName = DifficultyName.ALL):
    def get_onsets_for_beatmap(beatmap, timestamps, bpm):
        onsets_array = np.zeros_like(timestamps, dtype=np.float32)
        for obj in beatmap:
            beat_time = obj[1]
            beat_time_to_sec = beat_time / bpm * 60
            closest_frame_idx = np.argmin(np.abs(timestamps - beat_time_to_sec))
            onsets_array[closest_frame_idx] = 1
        return onsets_array.reshape(-1, 1)

    if not isinstance(song["song"], np.ndarray):
        song["song"] = np.array(song["song"])

    timestamps = librosa.times_like(song["song"], sr=sample_rate)
    onsets = {}

    for key, beatmap in song.items():
        if key in DIFFICULTY_NAMES:
            onsets[key] = {}
            onsets[key]["onsets_array"] = get_onsets_for_beatmap(
                beatmap, timestamps, song["bpm"]
            )
            pascal_key = re.sub(r"([a-z])([A-Z])", r"\1_\2", key).upper()
            onsets[key]["condition"] = DifficultyNumber[pascal_key].value

    return onsets


def process_bpminfo(
    bpm_info: List, max_length_ms: float, units: TimeUnit = TimeUnit.milliseconds
) -> Tuple[List, List, List, List]:
    bpms, starts, ends, beats = [], [], [], []
    max_length = convert_units(max_length_ms, units)
    for i, (bpm, start, beat) in enumerate(bpm_info):
        start = convert_units(start, units)
        if start > max_length:
            continue
        if i < len(bpm_info) - 1:
            bpms.append(bpm)
            starts.append(start)
            next_start = convert_units(bpm_info[i + 1][1], units)
            ends.append(min(next_start, max_length))
            beats.append(beat)
        else:
            bpms.append(bpm)
            starts.append(start)
            ends.append(max_length)
            beats.append(beat)
    return bpms, starts, ends, beats


def gen_beats_array(
    length: int,
    bpm_info: List[Tuple[float, float, int]],
    mel_length: int,
    distinguish_downbeat: bool = False,
):
    """

    Parameters
    ----------
    length : int
        length of onset label sequence
    bpm_info : List[Tuple[float, float, int]]
        list of tuple (bpm, start(ms), beats)
        If there are tempo changes during the song, bpm_info contains
        more than one tuples.
    distinguish_downbeat : bool
        If `True`, distinguish downbeat of a measure from other beats.
    mel_length : int
        length of mel spectrogram sequence

    Returns
    -------

    """

    validate(bpm_info, mel_length)

    # The time range that nth(>= 0) frame represents is
    #   (n - 0.5) * FRAME < time <= (n + 0.5) * FRAME [ms]
    # , where `FRAME` is a constant for frame length.
    # Set an upper bound of the time for last frame (n = length - 1)
    max_length_ms = max(0, (length - 0.5) * FRAME)
    arr = np.zeros(length)
    bpms, starts, ends, beats = process_bpminfo(bpm_info, max_length_ms)
    for i, (bpm, start, end, beat) in enumerate(zip(bpms, starts, ends, beats)):
        # calc beat timing in milliseconds
        interval = 60 * 1000 / bpm  # ms
        beats_timing = np.arange(start, end, interval)

        # convert to array index
        # The conversion method (float -> int) is taking round, which is
        # the same as the preprocessing method for onset.
        # (in preprocessing/onset_converter.py)
        arg_beats = np.round(beats_timing / FRAME).astype("int64")

        # assign flag to beats array
        for b in range(beat):
            if b == 0:  # downbeat (beginning of a measure)
                if distinguish_downbeat:
                    arr[arg_beats[b::beat]] = 2
                else:
                    arr[arg_beats[b::beat]] = 1
            else:  # other beats
                arr[arg_beats[b::beat]] = 1

    return arr.reshape([-1, 1])


def convert_units(ms: float, units: TimeUnit) -> float:
    if units == TimeUnit.milliseconds:
        return float(ms)
    elif units == TimeUnit.seconds:
        return float(ms) / 1000.0
    elif units == TimeUnit.frames:
        return float(ms) / 32.0


def validate(bpm_info: List[Tuple[float, float, int]], mel_length: int):
    for x, (bpm, start, beat) in enumerate(bpm_info):
        assert bpm > 0
        assert start >= 0
        assert beat > 0
        assert (
            np.round(start / FRAME) <= mel_length
        ), f"The start position of bpm_info ({start}) is outside the song length ({mel_length})."


def assert_length(arr: np.ndarray, length: int):
    """Pad with zeros to ensure the array length
    Parameters
    ----------
    arr
    length

    Returns
    -------

    """
    if arr.shape[0] >= length:
        return arr[:length]
    length2 = length - arr.shape[0]
    arr2 = np.zeros((length2, arr.shape[1]))
    return np.concatenate([arr, arr2])

import logging
import os
import re
import shutil
from typing import Tuple
from sklearn.model_selection import train_test_split

import librosa
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset
from tqdm import tqdm

from config import *
from dl.models.util import log_window_to_csv
from notes_generator.preprocessing import mel
from utils import clean_data

base_dataset_path = (
    "http://kaistore.dcs.fmph.uniba.sk/beatsaber-map-generator/dataset/beatmaps"
)

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s - %(asctime)s - %(message)s",
    filename="data_tester.log",
)


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
    ):
        self.difficulty = difficulty
        self.object_type = object_type
        self.enable_condition = enable_condition
        self.with_beats = with_beats
        self.seq_length = seq_length
        self.skip_step = skip_step
        self.valid_split_ratio = valid_split_ratio
        self.skip_processing = skip_processing

        self.load()

    def load(self):
        dataset_path = f"{base_dataset_path}/{self.object_type.value}/metadata.csv"

        df = pd.read_csv(dataset_path)

        df = clean_data(df)

        # First split into train+valid and test
        train, test_valid = train_test_split(df, train_size=0.6, random_state=42)

        # Then split train_valid into train and valid
        test, valid = train_test_split(test_valid, test_size=0.5, random_state=42)

        self.dataset = {
            "train": train,
            "validation": valid,
            "test": test,
        }

    def process(self, song):
        if self.skip_processing:
            return song
        bpm_info = get_bpm_info(song)
        onsets = get_onset_array(song, self.difficulty)
        song_len = round(song["meta"]["duration"]) * 1000  # in ms
        mel_array_len = len(song["data"]["mel"][0])

        try:
            beats_array = gen_beats_array(mel_array_len, bpm_info, song_len)
        except AssertionError as e:
            logging.error(
                f'song{song["id"]}_{song["song_id"]} difficulty {self.difficulty}: {e}'
            )
            print(
                f'Error in song{song["id"]}_{song["song_id"]} difficulty {self.difficulty}: {e}'
            )
            song["not_working"] = True

        song["data"] = {
            "mel": song["data"]["mel"],
            "onsets": onsets,
        }

        if self.with_beats and "not_working" not in song:
            song["data"]["beats"] = beats_array

        return song

    def get_split(self, split: Split):
        return self.dataset[split.value]

    def __getitem__(self, idx):
        return self.process(self.dataset[self.split][idx])  # type: ignore

    def __len__(self) -> int:
        return len(self.dataset[self.split])  # type: ignore


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

    if not isinstance(song["data"]["mel"], np.ndarray):
        song["data"]["mel"] = np.array(song["data"]["mel"])

    timestamps = librosa.times_like(song["data"]["mel"], sr=sample_rate)
    onsets = {}

    for key, beatmap in song["beatmaps"].items():
        if beatmap:
            onsets[key] = {}
            onsets[key]["onsets_array"] = get_onsets_for_beatmap(
                beatmap, timestamps, song["meta"]["bpm"]
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

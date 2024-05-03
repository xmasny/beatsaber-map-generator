import logging
import os
import shutil
from typing import Tuple
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from datasets import load_dataset, IterableDataset

_valid_dataset_path = "dataset/valid_dataset/{object_type}/{difficulty}"

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(message)s", filename="data_tester.log"
)


class BaseLoader(IterableDataset):  # type: ignore
    def __init__(
        self,
        difficulty: DifficultyName = DifficultyName.EASY,
        object_type: ObjectType = ObjectType.COLOR_NOTES,
        enable_condition: bool = True,
        stream_dataset: bool = True,
        with_beats: bool = True,
        seq_length: int = 16000,
        skip_step: int = 2000,
    ):
        self.difficulty = difficulty
        self.object_type = object_type
        self.enable_condition = enable_condition
        self.stream_dataset = stream_dataset
        self.with_beats = with_beats
        self.seq_length = seq_length
        self.skip_step = skip_step

    def iter(self, song: SongIteration) -> SongIteration | dict:
        bpm_info = get_bpm_info(song)
        onsets = get_onset_array(song)
        song_len = round(song["meta"]["duration"]) * 1000  # in ms
        onsets_array_len = len(onsets)

        try:
            beats_array = gen_beats_array(onsets_array_len, bpm_info, song_len)
        except AssertionError as e:
            logging.error(f'Error in song{song["song_id"]} {song["id"]}: {e}')
            print(
                f'Error in song{song["song_id"]} {song["id"]} difficulty {self.difficulty}: {e}'
            )
        condition = DifficultyNumber[self.difficulty.name].value

        data = dict(
            condition=condition,  # difficulty
            onset=onsets,
            mel=song["data"]["mel"],  # type: ignore
        )

        if self.with_beats:
            # beat array(2 at downbeats, 1 at other beats)
            data["beats"] = beats_array  # type: ignore

        song["data"] = data  # type: ignore
        return song

    def process_song(
        self,
        song: SongIteration,
        beats_array: np.ndarray,
        condition: int,
        onsets: np.ndarray,
    ):
        for audio, start_index, end_index, params in self.iter_audio(song):
            score_segment = self.cut_segment(
                onsets, start_index, end_index, audio.shape[0]  # type: ignore
            )
            # Convert to the form of start, end, frame
            data = dict(
                condition=torch.Tensor([condition]),  # difficulty
                onset=torch.from_numpy(score_segment).float(),
                audio=torch.from_numpy(audio).float(),  # audio (mel-spectrogram)
            )
            if self.with_beats:
                # beat array(2 at downbeats, 1 at other beats)
                data["beats"] = torch.from_numpy(
                    self.cut_segment(
                        beats_array, start_index, end_index, audio.shape[0]
                    )
                ).float()

            yield data

    def iter_audio(self, song: SongIteration):
        """Iterate audio"""
        seq_length = int(round(self.seq_length / FRAME))
        song_mel = song["data"]["mel"].T  # type: ignore
        skip_step = int(round(self.skip_step / FRAME))
        for skip in range(0, seq_length, skip_step):
            params = dict(skip=skip, seq_length=seq_length)
            for data_tuple in iter_array(song_mel, seq_length, skip, params):
                # Add debugging statement to print data_tuple
                yield data_tuple

    def cut_segment(
        self, score_data: np.ndarray, start_index: int, end_index: int, length: int
    ):
        """Ensure the length of data array.

        Parameters
        ----------
        score_data : np.ndarray
            An array containing score data where the length will be adjusted.
        start_index : int
            The start index to cut the array.
        end_index : int
            The end index to cut the array.
        length : length
            The length of the array to be returned.

        Returns
        -------

        """
        # Cut the score data by specified length
        score_segment = score_data[start_index:end_index]
        # Pad if the length is insufficient
        score_segment = assert_length(score_segment, length)
        return score_segment

    def load(self):
        dataset = load_dataset(
            "masny5/beatsaber_songs_and_metadata",
            f"{self.object_type.value}-{self.difficulty.value}",
            streaming=self.stream_dataset,
            trust_remote_code=True,
        )
        # Map the dataset to the iterator - generate onset, beats
        self.dataset = dataset.map(self.iter)

    def __len__(self):
        return getattr(self.dataset, "n_shards", None)

    def __getitem__(self, split: Split) -> IterableDataset:
        return self.dataset.get(split.value)  # type: ignore

    def __iter__(self):
        return iter(self.dataset)

    def save_valid_data(
        self,
        valid_loader: DataLoader,
        valid_dataset_len: int,
        run_parameters: RunConfig,
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
                if "not_working" not in song:
                    np.save(
                        f'{path}/song{song["id"]}_{song["song_id"]}.npy',
                        song,
                        allow_pickle=True,
                    )
                pbar.update(1)


def iter_array(array, length, skip, params):
    max_len = len(array)
    for i in range(skip, max_len, length):
        data = array[i : (i + length)]
        if len(data) == length:
            yield data, i, i + length, params


def get_bpm_info(song: SongIteration):
    bpm = song["meta"]["bpm"]

    energy = np.sum(song["data"]["mel"], axis=0)  # type: ignore
    start_index = np.argmax(energy > 0)

    start_time = librosa.frames_to_time(start_index, sr=sample_rate)

    start = float(start_time * 1000)
    beats = int(4)
    return [(bpm, start, beats)]


def get_onset_array(song: SongIteration):
    if not isinstance(song["data"]["mel"], np.ndarray):  # type: ignore
        song["data"]["mel"] = np.array(song["data"]["mel"])  # type: ignore
    timestamps = librosa.times_like(song["data"]["mel"], sr=sample_rate)  # type: ignore

    onsets = np.zeros_like(timestamps, dtype=np.float32)

    for obj in song["beatmap"]:  # type: ignore
        beat_time = obj[1]

        beat_time_to_sec = beat_time / song["meta"]["bpm"] * 60
        closest_frame_idx = np.argmin(np.abs(timestamps - beat_time_to_sec))
        onsets[closest_frame_idx] = 1

    return onsets.reshape(-1, 1)


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


class SavedValidDataloader(IterableDataset):
    def __init__(self, run_parameters: RunConfig):
        self.path = _valid_dataset_path.format(
            object_type=run_parameters.object_type, difficulty=run_parameters.difficulty
        )
        self.songs = os.listdir(self.path)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        song = np.load(f"{self.path}/{self.songs[idx]}", allow_pickle=True)
        return song

    def __iter__(self):
        for song in self.songs:
            yield np.load(f"{self.path}/{song}", allow_pickle=True).item()


class TestDataset(BaseLoader):
    def __init__(
        self,
        difficulty: DifficultyName = DifficultyName.EASY,
        object_type: ObjectType = ObjectType.COLOR_NOTES,
        enable_condition: bool = True,
        stream_dataset: bool = True,
        with_beats: bool = True,
        seq_length: int = 16000,
        skip_step: int = 2000,
    ):
        super().__init__(
            difficulty=difficulty,
            object_type=object_type,
            enable_condition=enable_condition,
            stream_dataset=stream_dataset,
            with_beats=with_beats,
            seq_length=seq_length,
            skip_step=skip_step,
        )

    def load(self):
        dataset = load_dataset(
            "training/dataset.py",
            f"{self.object_type.value}-{self.difficulty.value}",
            streaming=self.stream_dataset,
        )
        # Map the dataset to the iterator - generate onset, beats
        self.dataset = dataset.map(self.iter)

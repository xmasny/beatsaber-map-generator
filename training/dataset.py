import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import datasets
import os
import re

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_HOMEPAGE = ""

_LICENSE = ""

_BASE_DATA_PAT_FORMAT_STR = "{type}/{difficulty}"

_DOWNLOAD_URL = "http://kaistore.dcs.fmph.uniba.sk/beatsaber-map-generator/dataset/"


def _types():
    return ["bomb_notes", "color_notes", "obstacles"]


def _difficulties():
    return ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]


def split_dataset(songs):
    train, valid = train_test_split(songs, train_size=0.8)
    valid, test = train_test_split(valid, test_size=0.5)

    return train, valid, test


class BeatSaberSongsAndMetadataConfig(datasets.BuilderConfig):
    """BuilderConfig for BeatSaberSongsAndMetadata"""

    def __init__(self, type: str, difficulty: str, **kwargs):
        """BuilderConfig for BeatSaberSongsAndMetadata.
        Args:
            type: str, type of beatmap
            difficulty: str, difficulty of beatmap
            **kwargs: keyword arguments forwarded to super.
        """

        if type not in _types():
            raise ValueError(f"type must be one of {_types()}")

        if difficulty not in _difficulties():
            raise ValueError(f"difficulty must be one of {_difficulties()}")

        name = f"{type}-{difficulty}"
        description = "Preprocessed BeatSaber songs and metadata for object type: {type} and difficulty: {difficulty}"

        super(BeatSaberSongsAndMetadataConfig, self).__init__(
            name=name, description=description, **kwargs
        )

        self.type = type
        self.difficulty = difficulty
        self.data_dir = _BASE_DATA_PAT_FORMAT_STR.format(
            type=type, difficulty=difficulty
        )


class BeatSaberSongsAndMetadata(datasets.GeneratorBasedBuilder):
    """BeatSaberSongsAndMetadata dataset."""

    BUILDER_CONFIGS = [
        BeatSaberSongsAndMetadataConfig(
            type=type,
            difficulty=difficulty,
            version=datasets.Version("1.0.0"),
        )
        for type in _types()
        for difficulty in _difficulties()
    ]

    def generate_urls(self, split):
        beatmaps_urls = [
            f"{_DOWNLOAD_URL}beatmaps/{self.config.data_dir}/{song[0]}"
            for song in split
            if song[0].endswith(".npy")
        ]
        songs_urls = [
            f"{_DOWNLOAD_URL}songs/{song[0]}"
            for song in split
            if song[0].endswith(".npy")
        ]

        return beatmaps_urls, songs_urls

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "song_id": datasets.Value("string"),
                    "beatmap": datasets.Array2D(shape=(None, 6), dtype="float32"),
                    "data": {
                        "mel": datasets.Array2D(shape=(None, 128), dtype="float32"),
                    },
                    "meta": {
                        "song": datasets.Value("string"),
                        "npzFile": datasets.Value("string"),
                        "upvotes": datasets.Value("int32"),
                        "downvotes": datasets.Value("int32"),
                        "score": datasets.Value("float32"),
                        "bpm": datasets.Value("float32"),
                        "duration": datasets.Value("int32"),
                        "automapper": datasets.Value("bool"),
                    },
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        all_songs_csv = dl_manager.download(
            f"{_DOWNLOAD_URL}beatmaps/{self.config.data_dir}.csv"
        )

        with open(all_songs_csv) as csvfile:  # type: ignore
            df = pd.read_csv(
                csvfile,
                header=None,
                names=[
                    "song",
                    "npzFile",
                    "upvotes",
                    "downvotes",
                    "score",
                    "bpm",
                    "duration",
                    "automapper",
                ],
            )

            songs = df.values.tolist()

        train, valid, test = split_dataset(songs)

        beatmaps_urls_train, songs_urls_train = self.generate_urls(train)
        beatmaps_urls_valid, songs_urls_valid = self.generate_urls(valid)
        beatmaps_urls_test, songs_urls_test = self.generate_urls(test)
        beatmap_files_all, song_files_all = self.generate_urls(songs)

        beatmap_files_train = dl_manager.download(beatmaps_urls_train)
        song_files_train = dl_manager.download(songs_urls_train)

        beatmap_files_valid = dl_manager.download(beatmaps_urls_valid)
        song_files_valid = dl_manager.download(songs_urls_valid)

        beatmap_files_test = dl_manager.download(beatmaps_urls_test)
        song_files_test = dl_manager.download(songs_urls_test)

        beatmap_files_all = dl_manager.download(beatmap_files_all)
        song_files_all = dl_manager.download(song_files_all)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_train,
                    "song_files": song_files_train,
                    "metadata": train,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_valid,
                    "song_files": song_files_valid,
                    "metadata": valid,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_test,
                    "song_files": song_files_test,
                    "metadata": test,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.ALL,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_all,
                    "song_files": song_files_all,
                    "metadata": songs,
                },
            ),
        ]

    def _generate_examples(self, beatmap_files, song_files, metadata):
        """Yields examples."""

        for beatmap_file, song_file, data in zip(beatmap_files, song_files, metadata):
            with open(beatmap_file, "rb") as npy_file:
                beatmap: np.ndarray = np.load(npy_file)

            with open(song_file, "rb") as npy_file:
                song: np.ndarray = np.load(npy_file)

            split = re.split(r"_|\.", os.path.basename(beatmap_file))
            id_ = int(split[0][4:])
            song_id = split[1]

            meta = dict(
                zip(
                    [
                        "song",
                        "npzFile",
                        "upvotes",
                        "downvotes",
                        "score",
                        "bpm",
                        "duration",
                        "automapper",
                    ],
                    data,
                )
            )

            yield f"{id_}_{song_id}", {
                "id": id_,
                "song_id": song_id,
                "beatmap": beatmap,
                "data": {
                    "mel": song,
                },
                "meta": meta,
            }

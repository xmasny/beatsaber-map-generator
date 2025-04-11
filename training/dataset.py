import os
import platform
import random
import re
import signal

import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

_BASE_DATA_PAT_FORMAT_STR = "{type}"

_DOWNLOAD_URL = (
    "http://kaistore.dcs.fmph.uniba.sk/beatsaber-map-generator/dataset/beatmaps/"
)


def _types():
    return ["bomb_notes", "color_notes", "obstacles"]


def _difficulties():
    return ["All", "Easy", "Normal", "Hard", "Expert", "ExpertPlus"]


def split_dataset(songs):
    shuffle_seed = random.randint(0, 2**32 - 1)
    print(f"Using shuffle seed: {shuffle_seed}")
    if platform.system() == "Linux":
        try:

            def handler(signum, frame):
                raise TimeoutError("Timed out")

            signal.signal(signal.SIGALRM, handler)  # type: ignore
            signal.alarm(30)  # type: ignore

            answer = input("Do you want to set a seed for the shuffle? (y/n): ")
            if answer:
                signal.alarm(0)  # type: ignore
            if answer == "y":
                shuffle_seed = int(input("Enter the seed: "))
        except TimeoutError:
            print("No input given, using random seed")
    else:
        answer = input("Do you want to set a seed for the shuffle? (y/n): ")
        if answer == "y":
            shuffle_seed = int(input("Enter the seed: "))

    try:
        import wandb  # type: ignore

        wandb.config.update({"shuffle_seed": shuffle_seed})
    except ImportError:
        pass

    train, valid = train_test_split(songs, train_size=0.6, random_state=shuffle_seed)
    valid, test = train_test_split(valid, test_size=0.5, random_state=shuffle_seed)

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
        self.data_dir = _BASE_DATA_PAT_FORMAT_STR.format(type=type)


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
            f"{_DOWNLOAD_URL}{self.config.data_dir}/npz/{beatmap[0]}"
            for beatmap in split
            if beatmap[0].endswith(".npz")
        ]

        return beatmaps_urls

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "song_id": datasets.Value("string"),
                    "beatmaps": {
                        "Normal": datasets.Sequence(
                            datasets.Array2D(shape=(None, 6), dtype="float32")
                        ),
                        "Easy": datasets.Sequence(
                            datasets.Array2D(shape=(None, 6), dtype="float32")
                        ),
                        "Hard": datasets.Sequence(
                            datasets.Array2D(shape=(None, 6), dtype="float32")
                        ),
                        "Expert": datasets.Sequence(
                            datasets.Array2D(shape=(None, 6), dtype="float32")
                        ),
                        "ExpertPlus": datasets.Sequence(
                            datasets.Array2D(shape=(None, 6), dtype="float32")
                        ),
                    },
                    "data": {
                        "mel": datasets.Array2D(shape=(None, 229), dtype="float32"),
                    },
                    "meta": {
                        "song": datasets.Value("string"),
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
            f"{_DOWNLOAD_URL}{self.config.data_dir}/combined_songs.csv"
        )

        with open(all_songs_csv) as csvfile:  # type: ignore
            df = pd.read_csv(csvfile)

            automapper = df["automapper"] == True
            missing_song = df["missing_song"] == True
            missing_levels = df["missing_levels"] == True

            df = df[~automapper]
            df = df[~missing_song]
            df = df[~missing_levels]

        songs = df.values.tolist()

        songs = [value for value in songs if 100.0 <= value[5] <= 500.0]

        train, valid, test = split_dataset(songs)

        beatmaps_urls_train = self.generate_urls(train)
        beatmaps_urls_valid = self.generate_urls(valid)
        beatmaps_urls_test = self.generate_urls(test)
        beatmaps_urls_all = self.generate_urls(songs)

        beatmap_files_train = dl_manager.download(beatmaps_urls_train)
        beatmap_files_valid = dl_manager.download(beatmaps_urls_valid)
        beatmap_files_test = dl_manager.download(beatmaps_urls_test)
        beatmap_files_all = dl_manager.download(beatmaps_urls_all)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_train,
                    "metadata": train,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_valid,
                    "metadata": valid,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_test,
                    "metadata": test,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.ALL,  # type: ignore
                gen_kwargs={
                    "beatmap_files": beatmap_files_all,
                    "metadata": songs,
                },
            ),
        ]

    def _generate_examples(self, beatmap_files, metadata):
        """Yields examples."""
        beatmap_data = None
        beatmap_song = None
        for beatmap_file, data in zip(beatmap_files, metadata):
            with open(beatmap_file, "rb") as npz_file:
                beatmap = np.load(npz_file, allow_pickle=True)
                beatmap_song = beatmap["song"]
                beatmap_data = {
                    "Easy": beatmap["Easy"] if "Easy" in beatmap else [],
                    "Normal": beatmap["Normal"] if "Normal" in beatmap else [],
                    "Hard": beatmap["Hard"] if "Hard" in beatmap else [],
                    "Expert": beatmap["Expert"] if "Expert" in beatmap else [],
                    "ExpertPlus": (
                        beatmap["ExpertPlus"] if "ExpertPlus" in beatmap else []
                    ),
                }

            split = re.split(r"[_.]", os.path.basename(beatmap_file))
            id_ = int(split[0][4:])
            song_id = split[1]

            meta = dict(
                zip(
                    [
                        "song",
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
                "beatmaps": beatmap_data,
                "data": {
                    "mel": beatmap_song,
                },
                "meta": meta,
            }

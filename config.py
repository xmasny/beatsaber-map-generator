import enum
from typing import List, Dict, Optional, TypedDict, Union

from numpy import ndarray
from omegaconf import DictConfig

# Constants

n_fft = 2048
hop_length = 512
sample_rate = 22050
n_mels = 128
FRAME = 32


# Enumerations
class ObjectType(enum.Enum):
    BOMB_NOTES = "bomb_notes"
    COLOR_NOTES = "color_notes"
    OBSTACLES = "obstacles"


class DifficultyName(enum.Enum):
    EASY = "Easy"
    NORMAL = "Normal"
    HARD = "Hard"
    EXPERT = "Expert"
    EXPERT_PLUS = "ExpertPlus"


class DifficultyNumber(enum.Enum):
    EASY = 1
    NORMAL = 3
    HARD = 5
    EXPERT = 7
    EXPERT_PLUS = 9


class Split(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


class TimeUnit(enum.Enum):
    milliseconds = "milliseconds"
    frames = "frames"
    seconds = "seconds"


# Type Definitions
class Meta(TypedDict):
    song: str
    npzFile: str
    upvotes: int
    downvotes: int
    score: float
    bpm: float
    duration: int
    automapper: bool


class Data(TypedDict):
    condition: Optional[List[int]]
    onset: ndarray
    beats: Optional[ndarray]
    mel: ndarray


class SongIteration(TypedDict):
    id: int
    beatmap: ndarray
    song_id: str
    data: Data | List[Data]
    meta: Meta
    not_working: Optional[bool]


class RunConfig(DictConfig):
    object_type: str
    difficulty: str
    model_dir: str
    songs_batch_size: int
    train_batch_size: int
    enable_condition: bool
    seq_length: int
    skip_step: int
    with_beats: bool
    num_layers: int
    dropout: float
    rnn_dropout: float
    enable_early_stop: bool
    patience: int
    checkpoint_interval: int
    validation_interval: int
    fuzzy_scale: float
    fuzzy_width: int
    end_lr: float
    start_lr: float
    epoch_length: int
    warmup_steps: int
    epochs: int
    wandb_mode: str
    is_parallel: bool
    save_valid_dataset: bool
    num_workers: int
    lr_scheduler_select: str
    weight_decay: float
    eta_min: float

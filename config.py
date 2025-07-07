import enum
from typing import List, Dict, Optional, TypedDict, Union

from numpy import ndarray
from omegaconf import DictConfig
import scipy as sp

# Constants

n_fft = 2048
hop_length = 512
sample_rate = 22050
n_mels = 229
FRAME = (hop_length / sample_rate) * 1000
FRAME_MS_ROUNDED = round(FRAME, 2)

DIFFICULTY_NAMES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]


# Enumerations
class ObjectType(enum.Enum):
    BOMB_NOTES = "bomb_notes"
    COLOR_NOTES = "color_notes"
    OBSTACLES = "obstacles"


class ModelType(enum.Enum):
    ONSETS = "onsets"
    NOTES = "notes"


class DifficultyName(enum.Enum):
    ALL = "All"
    EASY = "Easy"
    NORMAL = "Normal"
    HARD = "Hard"
    EXPERT = "Expert"
    EXPERT_PLUS = "ExpertPlus"


class DifficultyNumber(enum.Enum):
    ALL = 0
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
    song: ndarray


class Beatmaps(TypedDict):
    Easy: List[Data] | list
    Normal: List[Data] | list
    Hard: List[Data] | list
    Expert: List[Data] | list
    ExpertPlus: List[Data] | list


class SongIteration(TypedDict):
    id: int
    beatmaps: Beatmaps
    song_id: str
    data: Data | List[Data]
    meta: Meta
    not_working: Optional[bool]


class RunParams(DictConfig):
    wandb_resume: str | None
    wandb_resume_id: str | None
    song_reset_number: int
    split_seed: int
    object_type: str
    model_type: str
    difficulty: str
    model_dir: str
    batch_size: int
    notes_batch_size: int
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
    lr_scheduler_name: str
    weight_decay: float
    eta_min: float
    min_sum_votes: int
    min_score: float
    min_bpm: float
    max_bpm: float
    gpu_index: int
    wandb_project: str
    mel_window: int


class RunConfig(DictConfig):
    params: RunParams

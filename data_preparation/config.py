import enum
from typing import List, Dict, Optional, TypedDict

from numpy import ndarray

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

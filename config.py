import enum


n_fft = 2048
hop_length = 512
sample_rate = 22050
n_mels = 128


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

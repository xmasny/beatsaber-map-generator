import json
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

class OnsetTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        score_base_path: Path,
        audio_base_path: Path,
    ):
        super().__init__()
        self.score_base_path = score_base_path
        self.audio_base_path = audio_base_path
        self.score_dict = self.load_score_dict()

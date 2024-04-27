from torch.utils.data import DataLoader

from dl.dataset_old import SongAndBeatmapsDataset
from utils import collate_fn

def load_dataset():

    train_dataset = SongAndBeatmapsDataset()
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)
    labels = train_dataset.labels

    return train_dataset, train_loader, labels
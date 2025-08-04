import os
from dotenv import load_dotenv
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score
import pandas as pd
import wandb
from config import *
from dl.models.onsets import SimpleOnsets
from tqdm import tqdm

from training.loader import BaseLoader
from utils import MyDataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()

base_dataset_api = os.getenv("BASE_DATASET_API", "http://localhost:8000/dataset/batch/")


batch_size = 42  # Adjust as needed


def cycle(dataloader, num_songs_pbar: Optional[tqdm], test_dataset):
    if num_songs_pbar:
        num_songs_pbar.reset()
    for index, songs in enumerate(dataloader):
        for song in songs:
            if num_songs_pbar:
                num_songs_pbar.update(1)
            if "not_working" in song:
                continue
            for segment in test_dataset.process(song_meta=song):
                yield segment


def evaluate_full(
    model: MyDataParallel,
    test_loader,
    test_dataset,
    device,
    num_songs_pbar,
):
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    i = 0

    with torch.no_grad():
        for batch in cycle(test_loader, num_songs_pbar, test_dataset):
            preds, _ = model.run_on_batch(batch)
            targets = batch["onset"].argmax(-1).to(device)
            pred_classes = preds["onset"].argmax(-1).to(device)

            all_preds.append(pred_classes.cpu())
            all_targets.append(targets.cpu())
    # After collecting all_preds and all_targets
    y_pred = torch.cat(all_preds).cpu().numpy().reshape(-1)
    y_true = torch.cat(all_targets).cpu().numpy().reshape(-1)

    assert y_pred.ndim == 1 and y_true.ndim == 1
    assert np.issubdtype(y_pred.dtype, np.integer)
    assert np.issubdtype(y_true.dtype, np.integer)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(2), zero_division=0
    )

    class_labels = [f"class_{i}" for i in range(2)]

    df = pd.DataFrame(
        {
            "class": class_labels,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )

    # Log overall scores
    wandb.log(
        {
            "test/macro_f1": macro_f1,
            "test/micro_f1": micro_f1,
        }
    )

    # Log per-class scalars
    for i, cls in enumerate(class_labels):
        wandb.log(
            {
                f"test/class_{cls}/precision": precision[i],  # type: ignore
                f"test/class_{cls}/recall": recall[i],  # type: ignore
                f"test/class_{cls}/f1_score": f1[i],  # type: ignore
                f"test/class_{cls}/support": support[i],  # type: ignore
            }
        )

    # Log per-class metrics as a wandb table
    wandb_table = wandb.Table(dataframe=df)
    wandb.log({"test/per_class_metrics": wandb_table})

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(df)

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "per_class": df}


params = [
    {
        "difficulty": DifficultyName.ALL,
        "seed": 1631495479,
        "wandb_id": "2mcdjnve",
        "model_version": "model-epoch-35:v1",
    },
]

for param in params:

    # Resume existing run by ID (replace with your actual ID)
    wandb.init(project="beat-saber-map-generator", id=param["wandb_id"], resume="must")

    artifact = wandb.use_artifact(
        f"dp-dl-beatsaber/beat-saber-map-generator/{param['model_version']}",
        type="model",
    )
    MODEL_PATH = artifact.download()

    model = SimpleOnsets(
        input_features=n_mels,
        output_features=1,
        num_layers=2,
        rnn_dropout=0.1,
    )
    model = MyDataParallel(model)
    model.load_state_dict(
        torch.load(MODEL_PATH + "/model_epoch_35.pt", map_location=device)
    )

    common_dataset_args = {
        "difficulty": param["difficulty"],
        "split_seed": param["seed"],
        "batch_size": batch_size,
    }

    test_dataset = BaseLoader(split=Split.TEST, **common_dataset_args)

    test_loader = test_dataset.get_dataloader()
    test_dataset_len = len(test_dataset)

    test_num_songs_pbar = tqdm(total=test_dataset_len, desc="Test songs")

    # Then call evaluation
    results = evaluate_full(
        model, test_loader, test_dataset, device, test_num_songs_pbar
    )

    wandb.finish()

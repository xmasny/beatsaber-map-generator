import os
from dotenv import load_dotenv
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score
import pandas as pd
import wandb
from config import *
from training.class_loader import ClassBaseLoader
from tqdm import tqdm
from dl.models.classes import MultiClassOnsetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()

base_dataset_api = os.getenv("BASE_DATASET_API", "http://localhost:8000/dataset/batch/")


batch_size = 1024  # Adjust as needed


def cycle(dataloader, test_dataset):
    for file in dataloader:
        for batch in test_dataset.process(file):
            yield batch


def evaluate_full(
    model: MultiClassOnsetClassifier,
    test_loader,
    test_dataset,
    class_count: ClassCount,
    device,
    class_names=None,
):
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(
            cycle(test_loader, test_dataset),
            total=class_count["test_iterations"] // batch_size,  # type: ignore
            desc="Evaluating",
        ):
            preds, _ = model.run_on_batch(batch)
            targets = batch["classes"].argmax(-1).to(device)

            all_preds.append(preds)
            all_targets.append(targets)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(19), zero_division=0
    )

    class_labels = class_names if class_names else [f"class_{i}" for i in range(19)]

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
    # {
    #     "difficulty": DifficultyName.NORMAL,
    #     "seed": "bullocks_2060294999",
    #     "wandb_id": "a59tsqsu",
    #     "model_version": "model-epoch-10:v74",
    # },
    {
        "difficulty": DifficultyName.NORMAL,
        "seed": "bullocks_2060294999",
        "wandb_id": "3f3h2j05",
        "model_version": "model-epoch-10:v75",
    },
    {
        "difficulty": DifficultyName.HARD,
        "seed": "pluma_1660179518",
        "wandb_id": "wx0fikkt",
        "model_version": "model-epoch-10:v77",
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

    class_count: ClassCount = np.load(
        f"dataset/batch/{param['difficulty'].value.lower()}/{param['seed']}/class_count.npy",
        allow_pickle=True,
    ).item()

    model = MultiClassOnsetClassifier(class_count["test_class_count"]).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH + "/model_epoch_10.pt", map_location=device)
    )

    common_dataset_args = {
        "difficulty": param["difficulty"],
        "split_seed": param["seed"],
    }

    test_dataset = ClassBaseLoader(split=Split.TEST, **common_dataset_args)

    test_loader = test_dataset.get_dataloader(batch_size=batch_size)

    # Then call evaluation
    results = evaluate_full(model, test_loader, test_dataset, class_count, device)

    wandb.finish()

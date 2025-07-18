# training/ignite_note.py

import os
from matplotlib import pyplot as plt
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Average, ConfusionMatrix, Precision, Recall, Fbeta
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
import wandb
import seaborn as sns

from dl.models.classes import MultiClassOnsetClassifier
from training.class_loader import ClassBaseLoader
from utils import setup_checkpoint_upload


def score_function(engine: Engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def ignite_train(
    train_dataset: ClassBaseLoader,
    valid_dataset: ClassBaseLoader,
    model: MultiClassOnsetClassifier,
    train_loader,
    valid_loader,
    optimizer: Optimizer,
    train_dataset_len,
    valid_dataset_len,
    device,
    lr_scheduler: CyclicLR | CosineAnnealingLR,
    wandb_logger: WandBLogger,
    **run_parameters,
):
    epochs = run_parameters.get("epochs", 100)
    validation_interval = run_parameters.get("validation_interval", 1)
    warmup_steps = run_parameters.get("warmup_steps", 0)
    wandb_mode = run_parameters.get("wandb_mode", "online")
    n_saved_model = run_parameters.get("n_saved_model", 10)
    n_saved_checkpoint = run_parameters.get("n_saved_checkpoint", 10)
    resume_checkpoint = run_parameters.get("resume_checkpoint", None)
    batch_size = run_parameters.get("batch_size", 32)
    patience = run_parameters.get("patience", 10)
    target_lr = optimizer.param_groups[0]["lr"]

    warmup_steps = (
        train_dataset_len // batch_size * warmup_steps if warmup_steps > 0 else 0
    )

    def cycle(dataloader):
        while True:
            for file in dataloader:
                for batch in train_dataset.process(file):
                    yield batch

    def train_step(engine: Engine, batch):
        model.train()
        if warmup_steps > 0 and engine.state.iteration < warmup_steps:
            lr_scale = min(1.0, float(engine.state.iteration + 1) / float(warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * target_lr

        optimizer.zero_grad()
        preds, losses = model.run_on_batch(batch)
        loss = losses["loss"]
        loss.backward()
        optimizer.step()
        if lr_scheduler and engine.state.iteration >= warmup_steps:
            lr_scheduler.step()

        return preds, {k: v.item() for k, v in losses.items()}

    def eval_step(engine: Engine, batch):
        model.eval()
        with torch.no_grad():
            preds, losses = model.run_on_batch(batch)
            true_classes = batch["classes"].argmax(-1)  # shape: (B, 3, 4)
            return preds, {
                **{k: v.item() for k, v in losses.items()},
                "true_classes": true_classes,
            }

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    early_stopping = EarlyStopping(
        patience=patience,
        score_function=score_function,
        trainer=trainer,
    )

    conf_matrix = ConfusionMatrix(
        num_classes=19,
        output_transform=lambda output: (
            output[0]["classes"]
            .permute(0, 3, 1, 2)  # (B, 19, 3, 4)
            .reshape(-1, 19)
            .argmax(dim=1),  # preds
            output[0]["classes"].argmax(dim=-1).reshape(-1),  # targets
        ),
    )

    precision = Precision(
        average=None,
        is_multilabel=True,
        output_transform=lambda output: (
            output[0]["classes"].argmax(-1).flatten(),  # predicted
            output[1]["true_classes"].flatten(),  # true
        ),
    )
    recall = Recall(
        average=None,
        is_multilabel=True,
        output_transform=lambda output: (
            output[0]["classes"].argmax(-1).flatten(),
            output[1]["true_classes"].flatten(),
        ),
    )
    f1 = Fbeta(
        beta=1.0,
        output_transform=lambda output: (
            output[0]["classes"].argmax(-1).flatten(),
            output[1]["true_classes"].flatten(),
        ),
    )

    # precision.attach(evaluator, "precision")
    # recall.attach(evaluator, "recall")
    # f1.attach(evaluator, "f1")
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    # conf_matrix.attach(evaluator, "confusion_matrix")

    # Attach metrics
    avg_loss = Average(output_transform=lambda output: output[1]["loss"])
    avg_loss.attach(trainer, "loss")
    avg_loss.attach(evaluator, "loss")

    # Logging
    @trainer.on(Events.EPOCH_COMPLETED(every=validation_interval))
    def log_validation(engine: Engine):
        evaluator.run(
            cycle(valid_loader),
            epoch_length=valid_dataset_len // batch_size,
        )

        metrics = evaluator.state.metrics
        epoch = engine.state.epoch

        print(f"\n[Validation] Epoch {epoch} - Loss: {metrics['loss']:.4f}\n")
        if wandb_mode != "disabled":
            for k, v in metrics.items():
                if k != "confusion_matrix":
                    wandb_logger.log({f"validation/{k}": v, "epoch": epoch})

            cm = metrics["confusion_matrix"].cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix (Epoch {epoch})")
            wandb_logger.log(
                {"validation/confusion_matrix": wandb.Image(fig), "epoch": epoch}
            )
            plt.close(fig)

    # Checkpointing
    to_save = {"model": model, "optimizer": optimizer}
    if wandb_mode != "disabled":

        if resume_checkpoint:
            checkpoint_path = os.path.join(wandb.run.dir, "checkpoints", f"checkpoint_<X>.pth")  # type: ignore
            Checkpoint.load_objects(
                to_load={"model": model, "optimizer": optimizer},
                checkpoint=torch.load(checkpoint_path),
            )

        handler = Checkpoint(
            to_save,
            DiskSaver(os.path.join(wandb.run.dir, "checkpoints"), create_dir=True, require_empty=False),  # type: ignore
            n_saved=n_saved_checkpoint,
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=validation_interval),
            handler,
        )

        best_checkpoint = Checkpoint(
            to_save,
            DiskSaver(os.path.join(wandb.run.dir), create_dir=False, require_empty=False),  # type: ignore
            n_saved=2,
            score_function=score_function,
            score_name="validation_loss",
            greater_or_equal=True,
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=validation_interval), best_checkpoint
        )

        best_model = ModelCheckpoint(
            dirname=os.path.join(wandb.run.dir),  # type: ignore
            filename_prefix="model",
            n_saved=2,
            create_dir=True,
            require_empty=False,
            score_function=score_function,
            score_name="validation_loss",
            greater_or_equal=True,
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=validation_interval),
            best_model,
            {"mymodel": model},
        )

        model_handler = ModelCheckpoint(
            dirname=os.path.join(wandb.run.dir, "model_checkpoints"),  # type: ignore
            filename_prefix="model",
            n_saved=n_saved_model,
            create_dir=True,
            require_empty=False,
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=validation_interval),
            model_handler,
            {"mymodel": model},
        )

        wandb.watch(model, log="all", criterion=avg_loss)

        setup_checkpoint_upload(trainer, {"model": model, "optimizer": optimizer}, wandb.run.dir, validation_interval=validation_interval)  # type: ignore

    # Progress bars
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: x[1]["loss"])
    ProgressBar(persist=True).attach(evaluator, output_transform=lambda x: x[1]["loss"])

    trainer.run(
        cycle(train_loader),
        max_epochs=epochs,
        epoch_length=train_dataset_len // batch_size,
    )

    if wandb_mode != "disabled":
        wandb_logger.close()

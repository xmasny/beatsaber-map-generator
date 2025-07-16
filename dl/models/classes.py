import typing

import numpy as np
import torch
import wandb
from torch import nn
from torch.nn import functional as F

# from notes_generator.constants import *
from dl.models.merge_labels import merge_labels
from dl.models.util import batch_first


class ClassesBase(nn.Module):
    def __init__(self, class_counts: np.ndarray, enable_condition: bool = False):
        super().__init__()
        self.class_counts = class_counts
        self.enable_condition = enable_condition

    def predict(self, batch: typing.Dict[str, torch.Tensor]):
        """Predict an onset score.

        Parameters
        ----------
        batch : typing.Dict[str, torch.Tensor]
            The Dict containing tensors below:
            * audio
            * onset
            * conditions
            * beats

        Returns
        -------
        pred : torch.Tensor
            The tensor of shape (batch_size, seq_len, output_features)
            containing predicted onset score.
            `output_features` defaults to `1`.

        """
        pred, _ = self.predict_with_probs(batch)
        return pred

    def predict_with_probs(self, batch: typing.Dict[str, torch.Tensor]):
        """Predict an onset score with a probability

        Parameters
        ----------
        batch : typing.Dict[str, torch.Tensor]
            The Dict containing tensors below:
            * audio
            * onset
            * conditions
            * beats

        Returns
        -------
        pred : torch.Tensor
            The tensor of shape (batch_size, seq_len, output_features)
            containing predicted onset score.
        proba : torch.Tensor
            The tensor of shape (batch_size, seq_len, output_features)
            containing predicted probability of onset score on each frame.

        """
        device = next(self.parameters()).device
        mel = batch_first(batch["audio"]).to(device)
        condition = batch["condition"].expand(
            (
                mel.shape[0],
                mel.shape[1],
            )
        )
        condition = condition.reshape(-1, condition.shape[-1], 1).to(device)
        if self.enable_beats:
            beats = batch["beats"].reshape(mel.shape[0], mel.shape[1], -1).to(device)
        else:
            beats = None
        self.eval()
        with torch.no_grad():
            logits = self(mel, condition, beats)
            assert not torch.isnan(logits).any(), "NaNs in logits"

            probs = torch.sigmoid(logits)

            if wandb.run is not None:
                wandb.log({"tracking/onset_probs": probs.max()})
            return probs > 0.5, probs

    def run_on_batch(
        self,
        batch: typing.Dict[str, torch.Tensor],
        merge_scale: typing.Optional[float] = None,
        net: typing.Optional[nn.Module] = None,
    ):
        """Forward training on one minibatch

        Parameters
        ----------
        batch : typing.Dict[str, torch.Tensor]
            The Dict containing minibatch tensors below:
            * audio
            * onset
            * conditions
            * beats
        fuzzy_width : int
            The width of fuzzy labeling applied to notes_label.
            default: `1`
        fuzzy_scale : float
            The scale of fuzzy labeling applied to notes_label.
            The value should be within an interval `[0, 1]`.
            default: `1.0`
        merge_scale : typing.Optional[float] = None,
            If nonzero, mix the label of other conditions in specified scale.
            Formally, at each time step use the label calculated as below:
                max(onset_label, merge_scale * onset_label_in_other_conditions)
            default: `None`
        net : typing.Optional[nn.Module] = None,
            If not `None`, use specified model for forward propagation.
            default: `None`

        Returns
        -------
        g_loss : typing.Dict
            The Dict containing losses evaluated for current iteration.

        """
        device = next(self.parameters()).device

        mel_input = batch_first(batch["mel"]).to(device)
        class_labels = batch["classes"].to(device)
        # reshape batch first
        mel_input = batch_first(mel_input).unsqueeze(1)

        class_counts = self.class_counts.sum(axis=(0, 1)).astype(np.int32)
        class_counts[class_counts == 0] = 1  # avoid div by zero

        # Inverse frequency weighting
        class_weights = 1.0 / class_counts
        # DO NOT normalize

        # Convert to torch
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        if self.enable_condition:
            condition = batch["condition"].expand(
                (
                    mel_input.shape[0],
                    mel_input.shape[1],
                )
            )
            condition = condition.reshape(-1, condition.shape[-1], 1).to(device)
        else:
            condition = None

        if net is None:
            net = self

        classes_pred = net(mel_input)

        predictions = {"classes": classes_pred}  # shape: (B, 3, 4, 19)

        loss = F.cross_entropy(
            input=predictions["classes"].permute(0, 3, 1, 2),  # → (B, 19, 3, 4)
            target=class_labels.argmax(-1).long(),  # (B, 3, 4)
            weight=(class_weights),
        )

        losses = {"loss": loss}
        return predictions, losses


class MultiClassOnsetClassifier(ClassesBase):
    def __init__(self, class_counts, num_classes=19, grid_shape=(3, 4)):
        super().__init__(class_counts=class_counts)
        self.grid_y, self.grid_x = grid_shape

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,32,229,45)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),  # (B,32,57,22)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,57,22)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 2)),  # (B,64,19,11)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B,128,19,11)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.grid_y, self.grid_x)),  # → (B,128,3,4)
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)  # (B,19,3,4)

    def forward(self, x):  # x: (B, 1, 229, 45)
        x = self.features(x)  # (B, 128, 3, 4)
        x = self.classifier(x)  # (B, 19, 3, 4)
        x = x.permute(0, 2, 3, 1)  # (B, 3, 4, 19)
        return x

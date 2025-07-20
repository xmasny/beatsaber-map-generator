import typing

import numpy as np
import torch
import wandb
from torch import nn
from torch.nn import functional as F

# from notes_generator.constants import *
from dl.models.util import batch_first


class ClassesBase(nn.Module):
    def __init__(
        self,
        class_counts: np.ndarray,
        enable_condition: bool = False,
        focal_loss_gamma: float = 2.0,
        focal_loss_alpha: float = 1.0,
        loss_fn: str = "focal_loss",
    ):
        super().__init__()
        self.class_counts = class_counts
        self.enable_condition = enable_condition
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.loss_fn = loss_fn

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
        if self.loss_fn == "focal_loss":
            focal_loss_fn = MultiClassFocalLoss(
                class_counts=class_counts,
                gamma=self.focal_loss_gamma,
                smoothing=0.1,
                device=device,
                reduction="mean",
                normalize_by="batch",
                debug=True,
            )
            loss = focal_loss_fn(classes_pred, class_labels)
        else:
            loss = F.cross_entropy(
                input=predictions["classes"].permute(0, 3, 1, 2),  # → (B, 19, 3, 4)
                target=class_labels.argmax(-1).long(),  # (B, 3, 4)
                weight=(class_weights),
            )

        losses = {"loss": loss}
        return predictions, losses


class MultiClassOnsetClassifier(ClassesBase):
    def __init__(
        self,
        class_counts,
        num_classes=19,
        grid_shape=(3, 4),
        focal_loss_gamma=2.0,
        loss_fn="focal_loss",
    ):
        super().__init__(class_counts=class_counts)
        self.grid_y, self.grid_x = grid_shape
        self.focal_loss_gamma = focal_loss_gamma
        self.loss_fn = loss_fn

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
            nn.Dropout2d(p=0.3),
            nn.AdaptiveAvgPool2d((self.grid_y, self.grid_x)),  # → (B,128,3,4)
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)  # (B,19,3,4)

        self.head = nn.Sequential(
            nn.Flatten(),  # (B, 128, 3, 4) → (B, 1536)
            nn.Linear(128 * 3 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 3 * 4 * num_classes),  # Predicts for all grid cells at once
        )

    def forward(self, x):  # x: (B, 1, 229, 45)
        x = self.features(x)  # (B, 128, 3, 4)
        x = self.head(x)  # (B, 19, 3, 4)
        x = x.view(-1, 3, 4, 19)  # (B, 3, 4, 19)
        return x


class MultiClassFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts=None,
        gamma=2.0,
        smoothing=0.0,
        reduction="mean",
        normalize_by="element",  # 'element' or 'batch'
        device=None,
        debug=False,
    ):
        """
        :param class_counts: Tensor or list of class counts for alpha computation
        :param gamma: Focal loss gamma
        :param smoothing: Label smoothing factor
        :param reduction: 'mean', 'sum', or 'none'
        :param normalize_by: 'element' (default), or 'batch' to normalize by batch size only
        :param device: device to put alpha on
        :param debug: if True, prints detailed internal states
        """
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.normalize_by = normalize_by

        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32)
            counts[counts == 0] = 1
            alpha = 1.0 / counts
            alpha = alpha / alpha.sum()
            self.alpha = alpha.to(device) if device else alpha
        else:
            self.alpha = None

    def forward(self, logits, one_hot):

        if logits.ndim == 4 and logits.shape[-1] != one_hot.shape[-1]:
            logits = logits.permute(0, 2, 3, 1)  # [B, H, W, C]

        B, H, W, C = logits.shape
        N = B * H * W

        logits = logits.reshape(-1, C)
        one_hot = one_hot.reshape(-1, C)

        probs = F.softmax(logits, dim=-1)
        pt = (probs * one_hot).sum(dim=1).clamp(min=1e-6)
        log_pt = pt.log()
        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            class_indices = one_hot.argmax(dim=1)
            alpha_t = self.alpha.to(logits.device)
            loss *= alpha_t[class_indices]

        if self.reduction == "mean":
            if self.normalize_by == "element":
                return loss.mean()
            elif self.normalize_by == "batch":
                return loss.sum() / B
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

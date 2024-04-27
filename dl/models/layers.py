from torch.nn import (
    Conv2d,
    MaxPool2d,
    Linear,
    Sequential,
    ReLU,
    BatchNorm2d,
    Dropout,
    Module,
    LSTM,
    ModuleList,
    AvgPool2d,
    Upsample,
)
import torch
import typing

from config import *

from notes_generator.layers.drop import DropBlock2d


class BaseConvModel(Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = Sequential(
            # layer 0
            Conv2d(1, output_features // 16, (3, 3), padding=1),
            BatchNorm2d(output_features // 16),
            ReLU(),
            # layer 1
            Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            BatchNorm2d(output_features // 16),
            ReLU(),
            # layer 2
            MaxPool2d((1, 2)),
            Dropout(0.25),
            Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            BatchNorm2d(output_features // 8),
            ReLU(),
            # layer 3
            MaxPool2d((1, 2)),
            Dropout(0.25),
        )
        self.fc = Sequential(
            Linear((output_features // 8) * (input_features // 4), output_features),
            Dropout(0.5),
        )

    def forward(self, mel: torch.Tensor):
        """

        Parameters
        ----------
        mel : torch.Tensor
            Tensor of shape (batch_size, seq_len, input_features(frequency))
            containing the log-scaled melspectrogram audio data.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, seq_len, output_features)

        """
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BiLSTM(Module):
    """Bidirectional LSTM Stack

    Parameters
    ----------
    input_features : int
        The number of expected features in the input x
    recurrent_features : int
        The number of features in the hidden state h
    num_layers : int
        Number of recurrent layers. default: `1`
    dropout: float
        The Rate of Dropout
    """

    def __init__(
        self,
        input_features,
        recurrent_features,
        inference_chunk_length: int = 640,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        self.inference_chunk_length = inference_chunk_length
        self.num_layers = num_layers
        self.rnn = LSTM(
            input_features,
            recurrent_features,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        hc: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, input_features)
            containing the features of input sequence.
        hc : typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]
            Tuple of tensors (h_0, c_0).
            * h_0 is a tensor of shape (num_layers * 2, batch, recurrent_features)
            containing the initial hidden state for each element in the batch.
            * c_0 is a tensor of shape (num_layers * 2, batch, recurrent_features)
            containing the initial cell state for each element in the batch.
            If not provided, both hidden state and cell state are initialized with zero.
            default: `None`

        Returns
        -------
        output: torch.Tensor
            Tensor of shape (batch, seq_len, 2 * recurrent_features)
            containing output features (h_t) from the last layer of the LSTM,
            for each t.

        """
        if self.training:
            self.rnn.flatten_parameters()
            val = self.rnn(x, hc)[0]
            return val
        else:
            self.rnn.flatten_parameters()
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            if hc:
                h, c = hc
            else:
                h = torch.zeros(
                    num_directions * self.num_layers,
                    batch_size,
                    hidden_size,
                    device=x.device,
                )
                c = torch.zeros(
                    num_directions * self.num_layers,
                    batch_size,
                    hidden_size,
                    device=x.device,
                )
            output = torch.zeros(
                batch_size,
                sequence_length,
                num_directions * hidden_size,
                device=x.device,
            )

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output


class ConvStack(Module):
    """Convolution stack
    Parameters
    ----------
    input_features : int
        Size of each input sample.
    output_features : int
        Size of each output sample.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        dropout: float = 0.25,
        dropout_last: float = 0.5,
    ):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnns = ModuleList(
            [
                Sequential(
                    # layer 0
                    Conv2d(1, output_features // 16, (3, 3), padding=1),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 1
                    Conv2d(
                        output_features // 16,
                        output_features // 16,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 2
                    MaxPool2d((1, 4)),
                    DropBlock2d(dropout, 5, 0.25),
                    Conv2d(
                        output_features // 16,
                        output_features // 8,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 8),
                    ReLU(),
                    # layer 3
                    MaxPool2d((1, 4)),
                    DropBlock2d(dropout, 3, 1.00),
                    Conv2d(
                        output_features // 8,
                        output_features // 4,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 4),
                    ReLU(),
                    MaxPool2d((1, 4)),
                    AvgPool2d((1, n_mels // 64)),
                ),
                Sequential(  # 16x max pooling in time direction (~0.5s)
                    # layer 0
                    Conv2d(1, output_features // 16, (3, 3), padding=1),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 1
                    Conv2d(
                        output_features // 16,
                        output_features // 16,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 2
                    MaxPool2d((2, 4)),
                    DropBlock2d(dropout, 5, 0.25),
                    Conv2d(
                        output_features // 16,
                        output_features // 8,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 8),
                    ReLU(),
                    # layer 3
                    MaxPool2d((8, 4)),
                    DropBlock2d(dropout, 3, 1.00),
                    Conv2d(
                        output_features // 8,
                        output_features // 4,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 4),
                    ReLU(),
                    MaxPool2d((1, 4)),
                    AvgPool2d((1, n_mels // 64)),
                    Upsample(scale_factor=(16, 1)),
                ),
                Sequential(  # 64x max pooling in time direction (~2s)
                    # layer 0
                    Conv2d(1, output_features // 16, (3, 3), padding=1),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 1
                    Conv2d(
                        output_features // 16,
                        output_features // 16,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 2
                    MaxPool2d((2, 4)),
                    DropBlock2d(dropout, 5, 0.25),
                    Conv2d(
                        output_features // 16,
                        output_features // 8,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 8),
                    ReLU(),
                    # layer 3
                    MaxPool2d((32, 4)),
                    DropBlock2d(dropout, 3, 1.00),
                    Conv2d(
                        output_features // 8,
                        output_features // 4,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 4),
                    ReLU(),
                    MaxPool2d((1, 4)),
                    AvgPool2d((1, n_mels // 64)),
                    Upsample(scale_factor=(64, 1)),
                ),
                Sequential(  # 128x max pooling in time direction (4s)
                    # layer 0
                    Conv2d(1, output_features // 16, (3, 3), padding=1),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 1
                    Conv2d(
                        output_features // 16,
                        output_features // 16,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 16),
                    ReLU(),
                    # layer 2
                    MaxPool2d((2, 4)),
                    DropBlock2d(dropout, 5, 0.25),
                    Conv2d(
                        output_features // 16,
                        output_features // 8,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 8),
                    ReLU(),
                    # layer 3
                    MaxPool2d((64, 4)),
                    DropBlock2d(dropout, 3, 1.00),
                    Conv2d(
                        output_features // 8,
                        output_features // 4,
                        (3, 3),
                        padding=1,
                    ),
                    BatchNorm2d(output_features // 4),
                    ReLU(),
                    MaxPool2d((1, 4)),
                    AvgPool2d((1, n_mels // 64)),
                    Upsample(scale_factor=(128, 1)),
                ),
            ]
        )
        self.dropout = Dropout(dropout_last)

    def forward(self, mel: torch.Tensor):
        """

        Parameters
        ----------
        mel : torch.Tensor
            Tensor of shape (batch_size, seq_len, input_features(frequency))
            containing the log-scaled melspectrogram audio data.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, seq_len, output_features)

        """
        padding = 0
        if mel.shape[1] % 128 != 0:
            padding = 128 - mel.shape[1] % 128
            mel = torch.cat(
                [
                    mel,
                    torch.zeros((mel.shape[0], padding, mel.shape[-1])).to(mel.device),
                ],
                dim=1,
            )
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        xs = []
        for mod in self.cnns:
            xs.append(mod(x))
        x = torch.cat(xs, dim=1)
        if padding > 0:
            x = x[:, :, :-padding, :]
        x = self.dropout(x)
        # x: B, C, H, W
        x = x.transpose(1, 2).flatten(-2)
        return x

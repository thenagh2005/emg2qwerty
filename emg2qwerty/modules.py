# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class TCNResidualBlock(nn.Module):
    def __init__(
        self, 
        num_features: int, # channels
        kernel_size: int = 3, # how many timestamps
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Non-causal padding
        self.padding = ((kernel_size - 1) * dilation) // 2
        
        # First layer
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(num_features, num_features, kernel_size, padding=self.padding, dilation=dilation))
        self.gn1 = nn.GroupNorm(1, num_features)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer  
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(num_features, num_features, kernel_size,padding=self.padding, dilation=dilation))
        self.gn2 = nn.GroupNorm(1, num_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 2, 0)
        residual = x
        
        # First conv 
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv 
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = out + residual
        return out.permute(2, 0, 1) 

# stacks blocks
class TCNEncoder(nn.Module):    
    def __init__(
        self,
        num_features: int,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        blocks = []
        for i in range(num_blocks):
            dilation = dilation_base ** i  
            blocks.append(
                TCNResidualBlock(num_features=num_features, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
        
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)

class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False, 
            bidirectional=True,
        )
        
        self.projection = nn.Linear(hidden_size * 2, num_features)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(inputs)  
        out = self.projection(lstm_out)  
        return out

###Recurrent block with GRU

class RNNLayer(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, num_layers:int, out_ll_size: int = 384):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        
        self.norm = nn.LayerNorm(hidden_size * 2)  # bidirectional GRU doubles hidden size
        
        self.dropout = nn.Dropout(0.6)  # Increased dropout rate to 0.6
        self.mid_layer = nn.Linear(hidden_size * 2, out_ll_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_ll_size, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, N, in_size)
        # x, _ = self.rnn(x)  # (T, N, hidden_size*2)
        # x = self.fc(x)      # (T, N, num_classes)
        #print(f'Input to RNN: {x.shape}')
        x, _ = self.rnn(x)  
        
        
        
        """""
        This part of the code is generated with the help of ChatGPT
        """""
        x = self.norm(x)
        x = self.dropout(x)
        
        #print(f'Shape after GRU: {x.shape}') # (T, N, hidden_size*2)
        #x = self.mid_layer(x) 
        #print(f'Shape after mid layer: {x.shape}')              # (T, N, mid_layer_size)
        x = self.relu(self.mid_layer(x))                # (T, N, mid_fc_size)
        x = self.fc(x) 
        #print(f'Shape before softmax: {x.shape}')          # (T, N, num_classes)  
        return self.log_softmax(x)
    


class RNNBlock(nn.Module):
    """A generic recurrent block capped with a LayerNorm
    
    Supports
     - GRU or LSTM cells
     - optional skip connection (skip connection will not help with vanishing gradients inside the block, only propogation through the block)
     - optional dropout
     - layer normalization
     - optional bidirectionality (editor's note: if this model will ever be used to predict keystrokes based on live data, bidirectionality should not be used during training)

    Inputs must be of shape (T, N, num_features).

    Args:
        num_features (int): Input and output feature dimension. If
            ``hidden_size`` is not set, this is also used as the hidden size.
        rnn_type (str): Type of RNN cell to use — supports "rnn", "gru", or "lstm".
            (default: "rnn")
        hidden_size (int | None): Hidden size of the RNN. If None, defaults
            to ``num_features``. (default: None)
        num_layers (int): Number of stacked RNN layers. (default: 1)
        dropout (float): Dropout probability between RNN layers (only applies
            when num_layers > 1). (default: 0.0)
        bidirectional (bool): Whether to use a bidirectional RNN. Output is
            projected back to ``num_features`` if True. (default: False) (I've never used this)
        skip_connection (bool): Whether to add a residual connection around
            the RNN to reduce vanishing gradient. (default: True)
    """
    def __init__(
        self,
        num_features: int,
        rnn_type: str = "rnn",
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        skip_connection: bool = True,
    ) -> None:
        super().__init__()
        self.skip_connection = skip_connection
        self.num_features = num_features
        hidden_size = hidden_size or num_features

        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}.get(rnn_type.lower())
        assert rnn_cls is not None, f"Unsupported rnn_type: {rnn_type!r}. Use 'rnn', 'gru', or 'lstm'."

        self.rnn = rnn_cls(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,
        )

        rnn_out_size = hidden_size * (2 if bidirectional else 1)

        self.proj = (
            nn.Linear(rnn_out_size, num_features)
            if rnn_out_size != num_features
            else nn.Identity()
        )

        self.layer_norm = nn.LayerNorm(num_features)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(inputs)   # (T, N, rnn_out_size)
        x = self.proj(x)          # (T, N, num_features)
        if self.skip_connection:
            x = x + inputs        
        return self.layer_norm(x) # (T, N, num_features)

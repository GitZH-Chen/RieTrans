"""Fixed layers for converting raw complex Radar signals to SPD matrices."""

import torch.nn as nn

from . import functional


class SplitSignal_cplx(nn.Module):
    def __init__(self, in_channels, window_size, hop_length):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.hop_length = hop_length

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} Radar channels, got {x.shape[1]}."
            )
        return functional.split_signal_cplx(
            x,
            self.window_size,
            self.hop_length,
        )


class CovPool_cplx(nn.Module):
    def forward(self, features):
        return functional.cov_pool_cplx(features)

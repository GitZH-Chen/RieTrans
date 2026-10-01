"""Functional operations for the fixed Radar encoder."""

import torch


def split_signal_cplx(x, window_size, hop_length):
    """Split real/imaginary Radar channels into overlapping windows."""
    windows = x.unfold(-1, window_size, hop_length)
    return windows.permute(0, 1, 3, 2)


def cov_pool_cplx(features):
    """Compute the real SPD representation of complex Radar windows."""
    real, imag = features.split(features.shape[1] // 2, dim=1)
    real = (real - real.mean(dim=-1, keepdim=True)).double()
    imag = (imag - imag.mean(dim=-1, keepdim=True)).double()
    denominator = features.shape[-1] - 1
    covariance_real = (
        real @ real.transpose(-1, -2)
        + imag @ imag.transpose(-1, -2)
    ) / denominator
    covariance_imag = (
        imag @ real.transpose(-1, -2)
        - real @ imag.transpose(-1, -2)
    ) / denominator
    return (covariance_real + covariance_imag) / 2

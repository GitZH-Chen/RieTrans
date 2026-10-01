"""Random SPD input -> SPDLinear -> MSE -> backward."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SPDNN"))

from spnn.SPDLinear import SPDLinear  # noqa: E402


def random_spd(batch_size: int, dim: int) -> torch.Tensor:
    # A A^T + 0.5 I is symmetric positive definite.
    matrix = torch.randn(batch_size, 1, dim, dim, dtype=torch.float64)
    identity = torch.eye(dim, dtype=matrix.dtype).view(1, 1, dim, dim)
    return matrix @ matrix.transpose(-1, -2) + 0.5 * identity


def main() -> None:
    torch.manual_seed(42)
    inputs = random_spd(batch_size=4, dim=4)
    targets = random_spd(batch_size=4, dim=3)
    # Map 4x4 SPD matrices to 3x3 SPD matrices.
    layer = SPDLinear(
        shape_in=[1, 4, 4],
        shape_out=[1, 3, 3],
        # Supported SPD metrics: LEM, LCM, AIM, BWM, PEM.
        metric="LEM",
    ).double()

    outputs = layer(inputs)
    # A simple loss verifies that gradients flow through the layer.
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()

    assert outputs.shape == targets.shape
    assert torch.isfinite(loss)
    assert layer.Z_vec.grad is not None and torch.isfinite(layer.Z_vec.grad).all()
    print(f"SPDLinear: output={tuple(outputs.shape)}, MSE={loss.item():.6f}, backward=ok")


if __name__ == "__main__":
    main()

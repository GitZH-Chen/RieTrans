"""Random hyperboloid input -> HyperbolicLinear -> MSE -> backward."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HNN"))

from HyperbolicNN.modules import HyperbolicLinear  # noqa: E402
from manifolds.hyperboloid import Hyperboloid  # noqa: E402
from manifolds.klein import Klein  # noqa: E402
from manifolds.poincare import PoincareBall  # noqa: E402


def random_hyperbolic(batch_size: int, spatial_dim: int, metric: str) -> torch.Tensor:
    # Hyperboloid has one extra time coordinate; ball models do not.
    if metric == "Hyperboloid":
        tangent = torch.zeros(batch_size, spatial_dim + 1, dtype=torch.float64)
        tangent[:, 1:] = 0.1 * torch.randn(batch_size, spatial_dim, dtype=torch.float64)
    else:
        tangent = 0.1 * torch.randn(batch_size, spatial_dim, dtype=torch.float64)
    manifold = {
        "PoincareBall": PoincareBall,
        "Klein": Klein,
        "Hyperboloid": Hyperboloid,
    }[metric]()
    return manifold.expmap0(tangent, c=1.0)


def main() -> None:
    torch.manual_seed(42)
    # Choose a model: Hyperboloid, PoincareBall, or Klein.
    metric = "Hyperboloid"
    inputs = random_hyperbolic(batch_size=4, spatial_dim=4, metric=metric)
    targets = random_hyperbolic(batch_size=4, spatial_dim=3, metric=metric)
    # Map spatial dimension 4 to 3 in the chosen model.
    layer = HyperbolicLinear(
        in_dim=4,
        out_dim=3,
        metric=metric,
        K=torch.tensor(-1.0, dtype=torch.float64),
        drop_mode="weight",
        normalize_v=(metric == "Hyperboloid"),
    ).double()

    outputs = layer(inputs)
    # A simple loss verifies that gradients flow through the layer.
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()

    assert outputs.shape == targets.shape
    assert torch.isfinite(loss)
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    print(f"HyperbolicLinear: output={tuple(outputs.shape)}, MSE={loss.item():.6f}, backward=ok")


if __name__ == "__main__":
    main()

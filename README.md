[![arXiv](https://img.shields.io/badge/arXiv-2609.35436-b31b1b.svg)](https://arxiv.org/abs/2609.35436)
<!-- [![OpenReview Forum](https://img.shields.io/badge/OpenReview-forum-8c1b13.svg)](https://openreview.net/forum?id=arer1bhGjQ) -->
<!-- [![OpenReview PDF](https://img.shields.io/badge/OpenReview-pdf-8c1b13.svg)](https://openreview.net/pdf?id=arer1bhGjQ) -->

# RieTrans

Code for *Building Transformation Layers for Riemannian Neural Networks* (NeurIPS 2026).

If you find this project helpful, please consider citing:

```bibtex
@inproceedings{chen2026rietrans,
  title={Building Transformation Layers for Riemannian Neural Networks},
  author={Ziheng Chen},
  booktitle={NeurIPS},
  year={2026}
}
```

## Environment

The two experiments have separate environments:

```bash
conda env create -f SPDNN/environment.yaml
conda env create -f HNN/environment.yaml
```

Activate `SPNN` for the SPD demo and FPHA training, or `hnn` for the HNN demo and HFC training.

## Demo code

The following demos show how to use our Riemannian FC layers.

### SPDLinear

```python
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
```

### HyperbolicLinear

```python
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
```

## SPDNN on FPHA

The SPDNN code framework follows our [RMLR](https://github.com/GitZH-Chen/RMLR) implementation.

The `./SPDNN` covers SPDNN with LEM, AIM, PEM, LCM, and BWM, together with the following baselines:

- SPDNet
- SPDNetBN
- RResNet-AIM and RResNet-LEM
- SPDNetLieBN-AIM and SPDNetLieBN-LCM
- SPDNetMLR
- GyroLE, GyroAI, and GyroLC
- GyroSPD++-LEM, GyroSPD++-AIM, and GyroSPD++-LCM

### Data

The two prepared FPHA archives are included in `SPDNN/data/`:

```text
SPDNN/data/
├── FPHA_TPR_no_wrist20_horizontal_lr_vertical_f200_k1_m2.zip
└── global_cov.zip
```

Extract them before training:

```bash
cd SPDNN
unzip data/FPHA_TPR_no_wrist20_horizontal_lr_vertical_f200_k1_m2.zip -d data
unzip data/global_cov.zip -d data

bash scripts/train_fpha_spnn_lem.sh
bash scripts/train_fpha_spnn_aim.sh
bash scripts/train_fpha_spnn_pem.sh
bash scripts/train_fpha_spnn_lcm.sh
bash scripts/train_fpha_spnn_bwm.sh
bash scripts/train_fpha_baselines.sh
cd ..
```

The SPDNN launchers use the prepared FPHA train/test split. The included `scripts/train_fpha_baselines.sh` runs the FPHA comparison methods. The SPDNN and Gyro methods default to CUDA device 0; set `DEVICE` to choose another device. SPDNet-family baselines run on CPU.

### Output

Results are written to `SPDNN/outputs/${DATASET}/`. Final accuracies are recorded in `final_results_${DATASET}` inside that directory.

## HNN-HFC

The HNN code framework follows [HGCN](https://github.com/HazyResearch/hgcn).

### Data

The Disease, Airport, PubMed, and Cora graph datasets are included in `HNN/data/`:

```text
HNN/data/
└── graph_datasets.zip
```

The archive contains `disease_lp/`, `airport/`, `pubmed/`, and `cora/`. Extract it before training:

```bash
cd HNN
unzip data/graph_datasets.zip -d data
bash scripts/train_hnn_hfc_p.sh
bash scripts/train_hnn_hfc_k.sh
bash scripts/train_hnn_hfc_h.sh
```

The graph benchmark files originate from [HGCN](https://github.com/HazyResearch/hgcn/tree/master/data). Each script runs its method on all four datasets with training seed 42, five folds, and the weight decay/dropout settings in `HNN/scripts/HYPERPARAMETERS.md`. Set `DEVICE` to select a CUDA device.

### Output

Results are written to `HNN/outputs/<dataset>/hfc-p/`, `HNN/outputs/<dataset>/hfc-k/`, and `HNN/outputs/<dataset>/hfc-h/` by default. Each run directory contains `final_results_<dataset>` and the saved fold-wise ROC-AUC tensor under `tensor_results/`. Set `OUTPUT_ROOT` to change the output root.

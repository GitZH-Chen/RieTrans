# Hyperparameters

## Common settings

All experiments use the public `HNN.py` entry point and the link-prediction task.

| Category | Setting | Value |
| --- | --- | --- |
| Training | Training seed | `42` |
| Training | Folds | `5` |
| Training | Maximum epochs | `5000` |
| Training | Minimum epochs | `100` |
| Training | Early-stopping patience | `100` |
| Training | Evaluation frequency | Every epoch |
| Training | Numerical precision | Float64 |
| Model | Backbone | Two-layer HNN |
| Model | Hidden/output dimension | `16` |
| Model | Curvature magnitude `c` | `1.0` |
| Model | Bias | Enabled |
| Model | Dropout mode | Weight dropout |
| Optimizer | Optimizer | Adam |
| Optimizer | Learning rate | `1e-2` |
| Optimizer | AMSGrad | Disabled |
| Decoder | Fermi--Dirac parameters | `r=2.0`, `t=1.0` |
| Data | Validation/test proportions | `0.05 / 0.10` |
| Data | Features/feature normalization/adjacency normalization | Enabled |
| Output | TensorBoard writer/model checkpoint | Disabled |
| Output | Per-fold ROC values | Saved |

The scripts do not override `dataset.split_seed`; they use the dataset configuration default. Disease, Airport, and PubMed use ReLU after each transformation layer, whereas Cora uses no activation.

## Baselines

Each dataset entry reports `weight decay / dropout`.

| Method | Manifold | Gyroaddition bias | Disease | Airport | PubMed | Cora |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Möbius | Poincaré ball | Enabled, built into `HypLinear` | `0 / 0` | `1e-3 / 0` | `0 / 0` | `1e-3 / 0.2` |
| Einstein | Klein model | Enabled, built into `HypLinear` | `5e-5 / 0` | `0 / 0` | `0 / 0` | `0 / 0` |
| LorentzTan | Hyperboloid | Enabled, built into `HypLinear` | `1e-4 / 0` | `0 / 0` | `0 / 0` | `0 / 0` |
| LFC | Hyperboloid | Enabled, separate `Gyrobias` layer | `1e-3 / 0` | `0 / 0.1` | `0 / 0.1` | `0 / 0.1` |
| NestFC | Hyperboloid | Disabled | `1e-3 / 0` | `0 / 0.1` | `0 / 0.1` | `0 / 0.1` |
| Poincaré FC | Poincaré ball | Enabled, separate `Gyrobias` layer | `5e-5 / 0` | `5e-4 / 0` | `5e-4 / 0` | `5e-4 / 0.1` |

## HFC

Each dataset entry reports `weight decay / dropout`. `normalize_v` is a fixed method setting rather than a tuned hyperparameter.

| Method | Manifold | Gyroaddition bias | `normalize_v` | Disease | Airport | PubMed | Cora |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HFC-P | Poincaré ball | Enabled, built into `HyperbolicLinear` | `false` | `1e-5 / 0` | `1e-3 / 0.1` | `1e-5 / 0` | `1e-4 / 0` |
| HFC-K | Klein model | Enabled, built into `HyperbolicLinear` | `false` | `5e-5 / 0` | `1e-3 / 0` | `0 / 0` | `5e-5 / 0.1` |
| HFC-H | Hyperboloid | Enabled, built into `HyperbolicLinear` | `true` | `0 / 0` | `1e-3 / 0` | `1e-5 / 0.1` | `1e-3 / 0.1` |

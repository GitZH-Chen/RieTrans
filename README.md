# RieTrans

Official implementation of *Building Transformation Layers for Riemannian Neural Networks*.

## FPHA Reproduction Package

This repo contains the FPHA subset of the SPD experiments. The package covers SPNN with LEM, AIM, PEM, LCM, and BWM, together with the following baselines:

- SPDNet
- SPDNetBN
- RResNet-AIM and RResNet-LEM
- SPDNetLieBN-AIM and SPDNetLieBN-LCM
- SPDNetMLR
- GyroLE, GyroAI, and GyroLC
- GyroSPD++-LEM, GyroSPD++-AIM, and GyroSPD++-LCM

## Environment

Create the provided environment and activate it:

```bash
conda env create -f environment.yaml
conda activate SPNN
```

The SPNN and Gyro methods default to CUDA device `0`; select another CUDA device with `DEVICE`. The SPDNet-family baselines run on CPU.

## Data

The two prepared FPHA archives are included in `data/`:

```text
data/
├── FPHA_TPR_no_wrist20_horizontal_lr_vertical_f200_k1_m2.zip
└── global_cov.zip
```

Extract both archives from the repository root:

```bash
unzip data/FPHA_TPR_no_wrist20_horizontal_lr_vertical_f200_k1_m2.zip -d data
unzip data/global_cov.zip -d data
```

This creates the two directories expected by the launchers. The TPR representation is used by SPNN and the Gyro methods, and `global_cov` is used by the SPDNet-family baselines.

## Run SPNN

```bash
bash scripts/train_fpha_spnn_lem.sh
bash scripts/train_fpha_spnn_aim.sh
bash scripts/train_fpha_spnn_pem.sh
bash scripts/train_fpha_spnn_lcm.sh
bash scripts/train_fpha_spnn_bwm.sh
```

## Run baselines

```bash
bash scripts/train_fpha_baselines.sh
```

## Evaluation protocol

FPHA uses its prepared official training and test split.

## Output

Results can be found in `outputs/${DATASET}/`. The final accuracies are recorded in `final_results_${DATASET}`.

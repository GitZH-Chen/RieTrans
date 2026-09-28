#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FPHA_TPR_DIR="${FPHA_TPR:-${RELEASE_ROOT}/data/FPHA_TPR_no_wrist20_horizontal_lr_vertical_f200_k1_m2}"
DATASET="${DATASET:-FPHA_TPR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"

cd "${RELEASE_ROOT}"

"${PYTHON_BIN}" SPNN.py -m \
  dataset="${DATASET}" \
  dataset.path="${FPHA_TPR_DIR}" \
  nnet=SPNN \
  nnet.model.transform_mode=SPDLinear \
  nnet.model.architecture=[28,22] \
  nnet.model.channels=[9,1] \
  nnet.model.conv_metric=LEM \
  nnet.model.conv_is_phi_inv=False \
  nnet.classifier.classifier=SPDMLR \
  nnet.classifier.clf_metric=LEM \
  nnet.classifier.clf_is_phi=False \
  nnet.optimizer.optimizer_mode=AMSGRAD \
  nnet.optimizer.lr=1e-4 \
  nnet.optimizer.weight_decay=0 \
  fit.threadnum=1 \
  fit.device="${DEVICE}" \
  fit.epochs=150 \
  fit.folds=1 \
  hydra.launcher.n_jobs=-1 \
  hydra.sweep.dir="outputs/${DATASET}" \
  hydra.sweep.subdir='.'

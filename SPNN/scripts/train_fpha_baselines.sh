#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FPHA_GLOBAL_COV_DIR="${FPHA_GLOBAL_COV:-${RELEASE_ROOT}/data/global_cov}"
FPHA_TPR_DIR="${FPHA_TPR:-${RELEASE_ROOT}/data/FPHA_TPR_no_wrist20_horizontal_lr_vertical_f200_k1_m2}"
DATASET="${DATASET:-FPHA}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"

cd "${RELEASE_ROOT}"

run_baseline() {
  local model_config="$1"
  shift
  "${PYTHON_BIN}" SPNN.py -m \
    dataset="${DATASET}" \
    dataset.path="${FPHA_GLOBAL_COV_DIR}" \
    nnet="${model_config}" \
    nnet.model.architecture=[63,33] \
    nnet.optimizer.optimizer_mode=SGD \
    nnet.optimizer.lr=5e-2 \
    nnet.optimizer.weight_decay=0 \
    fit.threadnum=1 \
    fit.device=cpu \
    fit.epochs=200 \
    fit.folds=1 \
    fit.is_writer=False \
    hydra.launcher.n_jobs=-1 \
    hydra.sweep.dir="outputs/${DATASET}" \
    hydra.sweep.subdir='.' \
    "$@"
}

# Each invocation trains the model and evaluates the validation split per epoch.
run_baseline SPDNet
run_baseline SPDNetBN
run_baseline SPDNetLieBN \
  nnet.model.bn_metric=AIM \
  nnet.model.bn_power=1.5 \
  nnet.optimizer.optimizer_mode=AMSGRAD \
  nnet.optimizer.lr=5e-3
run_baseline SPDNetLieBN \
  nnet.model.bn_metric=LCM \
  nnet.model.bn_power=0.5 \
  nnet.optimizer.optimizer_mode=AMSGRAD \
  nnet.optimizer.lr=5e-3
# RResNet validation accuracy can fluctuate sharply between epochs. The results reported in the paper were obtained by selecting the best results in the final several epochs.
run_baseline SPDResNet \
  nnet.model.ResBlockMetric=AIM
run_baseline SPDResNet \
  nnet.model.ResBlockMetric=LEM

"${PYTHON_BIN}" SPNN.py -m \
  dataset="${DATASET}" \
  dataset.path="${FPHA_GLOBAL_COV_DIR}" \
  nnet=SPDNet \
  nnet.model.architecture=[63,33] \
  nnet.classifier.classifier=SPDMLR \
  nnet.classifier.clf_metric=LCM \
  nnet.classifier.clf_power=0.25 \
  nnet.optimizer.optimizer_mode=AMSGRAD \
  nnet.optimizer.lr=1e-2 \
  nnet.optimizer.weight_decay=0 \
  fit.threadnum=1 \
  fit.device=cpu \
  fit.epochs=100 \
  fit.folds=1 \
  fit.is_writer=False \
  hydra.launcher.n_jobs=-1 \
  hydra.sweep.dir="outputs/${DATASET}" \
  hydra.sweep.subdir='.'

run_gyro() {
  local metric="$1"
  local epochs="$2"
  local learning_rate="$3"

  "${PYTHON_BIN}" SPNN.py -m \
    dataset=FPHA_TPR \
    dataset.path="${FPHA_TPR_DIR}" \
    nnet=GyroSPD \
    nnet.model.architecture=[9,28,28] \
    nnet.model.metric="${metric}" \
    nnet.optimizer.optimizer_mode=AMSGRAD \
    nnet.optimizer.lr="${learning_rate}" \
    nnet.optimizer.weight_decay=0 \
    fit.threadnum=1 \
    fit.device="${DEVICE}" \
    fit.epochs="${epochs}" \
    fit.folds=1 \
    fit.is_writer=False \
    hydra.launcher.n_jobs=-1 \
    hydra.sweep.dir="outputs/${DATASET}" \
    hydra.sweep.subdir='.'
}

# GyroLE, GyroAI, and GyroLC.
run_gyro LEM 50 1e-3
run_gyro AIM 15 1e-3
run_gyro LCM 60 5e-3

run_gyrospdpp() {
  local conv_metric="$1"
  local clf_metric="$2"
  local epochs="$3"

  "${PYTHON_BIN}" SPNN.py -m \
    dataset=FPHA_TPR \
    dataset.path="${FPHA_TPR_DIR}" \
    nnet=GyroSPDpp \
    nnet.model.architecture=[28,21] \
    nnet.model.channels=[9,1] \
    nnet.model.conv_metric="${conv_metric}" \
    nnet.classifier.clf_metric="${clf_metric}" \
    nnet.optimizer.optimizer_mode=AMSGRAD \
    nnet.optimizer.lr=1e-3 \
    nnet.optimizer.weight_decay=0 \
    fit.threadnum=1 \
    fit.device="${DEVICE}" \
    fit.epochs="${epochs}" \
    fit.folds=1 \
    fit.is_writer=False \
    hydra.launcher.n_jobs=-1 \
    hydra.sweep.dir="outputs/${DATASET}" \
    hydra.sweep.subdir='.'
}

# GyroSPD++-LEM, GyroSPD++-AIM, and GyroSPD++-LCM. The AIM variant uses the original LEM classifier.
run_gyrospdpp LEM LEM 100
run_gyrospdpp AIM LEM 50
run_gyrospdpp LCM LCM 100

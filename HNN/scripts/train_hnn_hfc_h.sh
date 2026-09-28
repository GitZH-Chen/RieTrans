#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRAPH_DATA_ROOT="${GRAPH_DATA_ROOT:-${RELEASE_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RELEASE_ROOT}/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"

cd "${RELEASE_ROOT}"

run_dataset() {
  local dataset="$1"
  local activation="$2"
  local weight_decay="$3"
  local dropout="$4"

  # Tangent-vector normalization is important for reproducing HFC-H results.
  "${PYTHON_BIN}" HNN.py \
    dataset.dataset="${dataset}" \
    dataset.path="${GRAPH_DATA_ROOT}" \
    training.folds=5 \
    training.seed=42 \
    training.log_freq=50 \
    training.cuda="${DEVICE}" \
    training.is_writer=false \
    training.save_results=1 \
    nnet.model.act="${activation}" \
    nnet.model.dropout="${dropout}" \
    nnet.model.manifold=Hyperboloid \
    nnet.transformation.transform_mode=HFC \
    nnet.transformation.normalize_v=true \
    nnet.optimizer.weight_decay="${weight_decay}" \
    hydra.run.dir="${OUTPUT_ROOT}/${dataset}/hfc-h" \
    hydra.job.chdir=true \
    'hydra.job_logging.handlers.file.filename=${hydra.runtime.output_dir}/train.log'
}

run_dataset disease_lp relu 0 0
run_dataset airport relu 1e-3 0
run_dataset pubmed relu 1e-5 0.1
run_dataset cora null 1e-3 0.1

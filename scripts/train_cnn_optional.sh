#!/usr/bin/env bash
set -euo pipefail
DATA_H5=${1:?Usage: train_cnn_optional.sh DATA_H5 OUTPUT_DIR}
OUT=${2:?Usage: train_cnn_optional.sh DATA_H5 OUTPUT_DIR}
SEED=${SEED:-42}
PYTHONPATH=src python -m icsm_as.training.train_baseline \
  --model CNN-Utility --data_h5 "$DATA_H5" --output "$OUT/CNN_seed${SEED}" \
  --seed "$SEED" --epochs "${EPOCHS:-20}" --lr "${LR:-3e-5}" \
  --selection_metric R_C_HQ --deterministic

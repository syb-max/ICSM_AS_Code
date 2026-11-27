#!/usr/bin/env bash
set -euo pipefail
DATA_H5=${1:?Usage: train_static_baselines.sh DATA_H5 OUTPUT_ROOT}
OUT=${2:?Usage: train_static_baselines.sh DATA_H5 OUTPUT_ROOT}
read -r -a SEED_LIST <<< "${SEEDS:-42}"
for SEED in "${SEED_LIST[@]}"; do
  for MODEL in MLP-Utility ResNet-Utility FT-Transformer MLP-MultiLabel; do
    SAFE=$(echo "$MODEL" | tr ' ()' '___')
    PYTHONPATH=src python -m icsm_as.training.train_baseline --model "$MODEL" --data_h5 "$DATA_H5" --output "$OUT/${SAFE}_seed${SEED}" --seed "$SEED" --epochs "${EPOCHS:-20}" --lr "${LR:-3e-5}" --selection_metric R_C_HQ --deterministic
  done
done

#!/usr/bin/env bash
set -euo pipefail
DATA_H5=${1:?Usage: train_icsm_as_seeds.sh DATA_H5 QWEN_DIR OUTPUT_ROOT [statistical_wideband|fullwave_response]}
QWEN_DIR=${2:?Usage: train_icsm_as_seeds.sh DATA_H5 QWEN_DIR OUTPUT_ROOT [task]}
OUT=${3:?Usage: train_icsm_as_seeds.sh DATA_H5 QWEN_DIR OUTPUT_ROOT [task]}
TASK=${4:-statistical_wideband}
read -r -a SEED_LIST <<< "${SEEDS:-42 2026 1709}"
case "$TASK" in
  statistical_wideband) LR=${LR:-1e-5}; PHYSICS_ARGS=(--allow_proxy_physics) ;;
  fullwave_response) LR=${LR:-1e-5}; PHYSICS_ARGS=() ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
for SEED in "${SEED_LIST[@]}"; do
  PYTHONPATH=src python -m icsm_as.core.train_icsm_as train \
    --architecture icsm_as --data_h5 "$DATA_H5" --q2_dir "$QWEN_DIR" \
    --save_dir "$OUT/ICSM-AS_${TASK}_seed${SEED}" --seed "$SEED" \
    --batch_size "${BATCH_SIZE:-8}" --epochs "${EPOCHS:-50}" \
    --lr "$LR" --freeze_backbone 1 --use_pretrained_weights 1 "${PHYSICS_ARGS[@]}"
done

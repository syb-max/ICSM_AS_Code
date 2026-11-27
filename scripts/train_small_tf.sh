#!/usr/bin/env bash
set -euo pipefail
DATA_H5=${1:?Usage: train_small_tf.sh DATA_H5 QWEN_CONFIG_DIR OUTPUT_DIR}
QWEN_DIR=${2:?Usage: train_small_tf.sh DATA_H5 QWEN_CONFIG_DIR OUTPUT_DIR}
OUT=${3:?Usage: train_small_tf.sh DATA_H5 QWEN_CONFIG_DIR OUTPUT_DIR}
read -r -a SEED_LIST <<< "${SEEDS:-42}"
TASK=${TASK:-statistical_wideband}
if [[ "$TASK" == "statistical_wideband" ]]; then DEFAULT_LR=1e-5; PHYSICS_ARGS=(--allow_proxy_physics);
elif [[ "$TASK" == "fullwave_response" ]]; then DEFAULT_LR=1e-5; PHYSICS_ARGS=();
else echo "Unknown TASK=$TASK" >&2; exit 2; fi
for SEED in "${SEED_LIST[@]}"; do
  PYTHONPATH=src python -m icsm_as.training.train_baseline --model Small-TF --data_h5 "$DATA_H5" \
    --output "$OUT/SmallTF_seed${SEED}" --qwen_dir "$QWEN_DIR" "${PHYSICS_ARGS[@]}" \
    --seed "$SEED" --epochs "${EPOCHS:-50}" --lr "${LR:-$DEFAULT_LR}" --deterministic
done

#!/usr/bin/env bash
set -euo pipefail
DATA_H5=${1:?Usage: train_paper_baselines.sh DATA_H5 QWEN_DIR OUTPUT_ROOT}
QWEN_DIR=${2:?Usage: train_paper_baselines.sh DATA_H5 QWEN_DIR OUTPUT_ROOT}
OUT=${3:?Usage: train_paper_baselines.sh DATA_H5 QWEN_DIR OUTPUT_ROOT}
read -r -a SEED_LIST <<< "${SEEDS:-42}"
TASK=${TASK:-statistical_wideband}
if [[ "$TASK" == "statistical_wideband" ]]; then
  DEFAULT_LR=1e-5; PHYSICS_ARGS=(--allow_proxy_physics)
elif [[ "$TASK" == "fullwave_response" ]]; then
  DEFAULT_LR=1e-5; PHYSICS_ARGS=()
else
  echo "Unknown TASK=$TASK" >&2; exit 2
fi
MODELS=("Small-TF" "TVT-MLL" "ICSM-AS (static)" "SCMG-variant" "Random-frozen")
for SEED in "${SEED_LIST[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr ' ()' '___')
    EXTRA=()
    case "$MODEL" in
      "ICSM-AS (static)"|"Score-only comparator"|"Score-only"|"SCMG-variant") EXTRA+=(--qwen_dir "$QWEN_DIR");;
      "Small-TF"|"Random-frozen") EXTRA+=(--qwen_dir "$QWEN_DIR" "${PHYSICS_ARGS[@]}");;
    esac
    PYTHONPATH=src python -m icsm_as.training.train_baseline --model "$MODEL" \
      --data_h5 "$DATA_H5" --output "$OUT/${SAFE}_seed${SEED}" \
      --seed "$SEED" --epochs "${EPOCHS:-50}" --lr "${LR:-$DEFAULT_LR}" --deterministic "${EXTRA[@]}"
  done
done

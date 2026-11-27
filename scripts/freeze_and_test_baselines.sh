#!/usr/bin/env bash
set -euo pipefail
DATA_H5=${1:?Usage: freeze_and_test_baselines.sh DATA_H5 BASELINE_RUNS_ROOT [dual-CSI arguments]}
ROOT=${2:?Usage: freeze_and_test_baselines.sh DATA_H5 BASELINE_RUNS_ROOT [dual-CSI arguments]}
shift 2
BASELINE_ARGS=()
for RUN in "$ROOT"/*; do
  [[ -d "$RUN" && -f "$RUN/best.pt" && -f "$RUN/run_manifest.json" ]] || continue
  MANIFEST="$RUN/selection_manifest.json"
  if [[ ! -f "$MANIFEST" ]]; then
    PYTHONPATH=src python scripts/freeze_baseline.py --run_dir "$RUN" --output "$MANIFEST"
  fi
  BASELINE_ARGS+=(--neural-baseline "$RUN/best.pt" "$MANIFEST")
done
PYTHONPATH=src python -m icsm_as.evaluation.imperfect_csi --prepared-data "$DATA_H5" "$@" "${BASELINE_ARGS[@]}"

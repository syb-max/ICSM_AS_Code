# Independent imperfect-CSI evaluation

This evaluation extension leaves the ICSM-AS v2 network, training code and original ZIP unchanged. It implements the dual-channel selection/evaluation protocol in the manuscript: use the same estimated effective CSI for every selector and calculate both the selected-subset capacity and the Capacity-Greedy reference capacity on the corresponding true effective channel.

**Reproduction status.** This extension is newly written. It is not a historical experiment log or proof of the reported 0.9820/0.9470 NCRs. To reproduce the reported nominal and robustness results, use the actual saved estimated channels (or the original error generator and its exact settings), the original true channels, data split and frozen checkpoints. No nominal error level or manuscript result is hard-coded. Synthetic mode is an explicitly separate protocol using circular complex-Gaussian directions normalized to the requested error power per base channel; do not substitute it for the original experiment without verifying equivalence.

## Setup

Extract the original archive and install its dependencies. From the repository root:

```sh
unzip ICSM_AS_Code_Framework_20261117_v2.zip
export PYTHONPATH="$PWD/ICSM_AS_Code_Framework_Simple/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s imperfect_csi -p 'test_evaluate.py' -v
python imperfect_csi/evaluate.py --help
```

## Data and checkpoints

- `--prepared-data`: the original prepared v2 HDF5 dataset, used for split membership, channel/SNR IDs and channel-independent geometry/port-response descriptors. Its original Gram matrices, labels and stored capacities are not forwarded to the selectors or used for capacity scoring.
- `--true-bank`: HDF5 with real/imaginary arrays `H_real`, `H_imag` of shape `[N,F,Nr,Nt]`, containing the effective channels before estimation error (after any required physical coupling transformation). No coupling is applied again. Supply unique integer `base_sample_id` values; for the v2 raw bank where the prepared base ID is demonstrably the original row index, explicitly pass `--true-id-key row-index`. Optional `site_id` is cross-checked.
- `--estimated-bank`: HDF5 with `H_est_real`, `H_est_imag`, same effective-channel convention. For one estimate per base channel, use unique `base_sample_id`. For one recorded estimate per channel-SNR decision, use `--estimate-per decision --est-id-key sample_id`. Array keys are configurable. IDs, shapes, frequency grids, port indexing and physical terminations must correspond to the original experiment. The caller must verify that these are the genuine paired channels.
- `--checkpoint`: the frozen final ICSM-AS checkpoint. The supplied selection manifest must contain its SHA-256 and the prepared-data SHA-256 (`data_sha256` or `training_labels_sha256`) for the held-out test split. Optional neural baselines use `--neural-baseline CHECKPOINT MANIFEST` repeatedly; their checkpoints must be compatible with the v2 registry. Do not choose checkpoints or tune settings using held-out results.

## 1. Recorded nominal condition (preferred for reproducing results)

```sh
python imperfect_csi/evaluate.py \
  --prepared-data prepared.h5 --true-bank true_channels.h5 \
  --estimated-bank actual_nominal_estimates.h5 --condition nominal \
  --checkpoint best.pt --selection-manifest selection_manifest.json \
  --split 2 --device cuda --output runs/nominal_recorded
```

This computes the actual nonzero error power from the saved paired channels. For the recorded -30/-20/-10 dB experiments, use their corresponding saved estimate banks and distinct output directories. There is no assumed nominal error value.

## 2. Explicit synthetic sensitivity experiment

```sh
for level in -30 -20 -10; do
  python imperfect_csi/evaluate.py \
    --prepared-data prepared.h5 --true-bank true_channels.h5 \
    --error-db "$level" --condition "synthetic_${level}dB" --seed 42 \
    --checkpoint best.pt --selection-manifest selection_manifest.json \
    --split 2 --device cuda --output "runs/synthetic_${level}dB"
done
```

The seed above is an example for a NEW experiment, not an asserted historical setting. The error is normalized over all frequencies, receive and transmit ports so that `10*log10(sum(|H_est-H|^2)/sum(|H|^2))` equals the requested level. A fixed base-channel seed provides the same estimate across SNRs; the same estimated Gram is supplied to all methods. The selector receives no true Gram, labels or reference capacities. Recorded mode supports decision-specific estimates when that was the original protocol.

## Outputs and scope

- `predictions.csv`: selected ports, actual error level, true-channel capacity, true-channel capacity of the Greedy subset and unclipped NCR for each method and channel-SNR case.
- `summary.json`: site/scene macro averages within each SNR, then equal weighting across SNRs; absolute capacities are retained. Verify the manuscript's original aggregation rules when reproducing results. Near-zero reference capacities raise an error rather than silently dropping cases.
- `manifest.json`: data/checkpoint hashes, parameters, mode, versions and completion status. Output directories must be new; failed runs are not marked complete.

Default selectors are ICSM-AS, Capacity-Greedy, Power-Top16 and Correlation-Greedy. Optional compatible frozen neural checkpoints can be supplied separately. The same runner can evaluate WB and FW data and individual trained ablation checkpoints, subject to their compatible architecture and selection protocol. It does not train models, generate historical data, measure paper latency, regenerate Fig. 4-6, or establish historical numerical results. Unit tests use small constructed channels, not paper data or trained checkpoints.

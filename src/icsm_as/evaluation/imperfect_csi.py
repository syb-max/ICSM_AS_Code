"""Independent dual-CSI evaluation for the ICSM-AS v2 framework.

Use the original recorded estimates to reproduce an existing experiment.
Synthetic mode implements a separately identified, normalized complex-Gaussian
perturbation protocol; it does not claim to reproduce historical results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


STATIC_KEYS = (
    'T_tx_real', 'T_tx_imag', 'positions_wavelength', 'frequency_coord',
    'coupling_edge', 'current_kernel_real', 'current_kernel_imag',
    'accepted_power_per_source',
)


def gram(h):
    h = np.asarray(h, dtype=np.complex128)
    if h.ndim != 3 or not all(h.shape) or not np.isfinite(h).all():
        raise ValueError('Channel must be finite [F, Nr, Nt]')
    return h.conj().swapaxes(-1, -2) @ h


def error_db(h, h_est):
    if h.shape != h_est.shape or not np.isfinite(h_est).all():
        raise ValueError('True and estimated channels must match and be finite')
    p = float(np.vdot(h, h).real)
    q = float(np.vdot(h_est - h, h_est - h).real)
    if not (p > 0 and q > 0 and math.isfinite(p) and math.isfinite(q)):
        raise ValueError('Imperfect CSI requires positive channel and error power')
    return 10.0 * (math.log10(q) - math.log10(p))


def synthetic_estimate(h, level_db, seed, base_id):
    """Circular Gaussian direction, normalized to exact per-channel error power."""
    if not math.isfinite(level_db) or seed < 0 or base_id < 0:
        raise ValueError('Finite error level and nonnegative seed/base ID required')
    if not np.isfinite(h).all() or not np.any(h):
        raise ValueError('Nonzero finite true channel required')
    rng = np.random.default_rng(np.random.SeedSequence([seed, base_id]))
    e = rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)
    e *= (np.linalg.norm(h) / np.linalg.norm(e)) * 10.0 ** (level_db / 20.0)
    estimate = h + e
    if not math.isclose(error_db(h, estimate), level_db, abs_tol=1e-7, rel_tol=0):
        raise ValueError('Requested error level is not representable')
    return estimate


def estimated_gram(h_est):
    """Common finite-precision Gram for all selectors."""
    g = gram(h_est)
    return g.real.astype(np.float32).astype(np.float64) + 1j * g.imag.astype(np.float32).astype(np.float64)


def estimated_relation(g):
    """Same normalized magnitude convention as the v2 lazy bank reader."""
    p = np.maximum(np.diagonal(g, axis1=1, axis2=2).real, 0).mean(0)
    denom = np.sqrt(np.maximum(p[:, None] * p[None, :], 1e-24))
    return np.clip(np.abs(g) / denom[None, :, :], 0, 1).mean(0).astype(np.float32)


def checked_subset(indices, k, nt):
    s = np.asarray(indices)
    if s.shape != (k,) or not np.issubdtype(s.dtype, np.integer):
        raise ValueError('Selector must return exactly k integer port indices')
    if np.unique(s).size != k or np.any(s < 0) or np.any(s >= nt):
        raise ValueError('Repeated or out-of-range selected ports')
    return s


def evaluate_case(h, h_est, rho, k, selectors, capacity_fn):
    """Selectors receive only the estimated Gram, never H or the true Gram."""
    level = error_db(h, h_est)
    gt = gram(h)
    ge = estimated_gram(h_est)
    if not (np.isfinite(ge).all() and math.isfinite(rho) and rho > 0 and 0 < k <= gt.shape[-1]):
        raise ValueError('Invalid Gram, SNR or selection cardinality')
    results = {}
    for name, choose in selectors.items():
        s = checked_subset(choose(ge.copy()), k, gt.shape[-1])
        cap = float(capacity_fn(gt, rho, s, k))
        if not math.isfinite(cap) or cap < 0:
            raise ValueError(f'Invalid true-channel capacity for {name}')
        results[name] = (s, cap)
    ref = results['Capacity-Greedy'][1]
    if ref <= 1e-8:
        raise ValueError('Near-zero Greedy denominator; do not silently exclude this case')
    return [dict(method=name, capacity=cap, greedy_capacity=ref, NCR=cap / ref,
                 actual_error_db=level, selected_ports=' '.join(map(str, s.tolist())))
            for name, (s, cap) in results.items()]


def summarize(rows):
    """Report paper matched-decision averages and legacy site-macro averages separately."""
    by_method = {}
    for row in rows:
        by_method.setdefault(row['method'], {}).setdefault(row['snr_db'], {}).setdefault(row['site_id'], []).append(row)
    result = {}
    for name, snrs in by_method.items():
        per_snr = {}
        for snr, sites in sorted(snrs.items()):
            cases = [r for group in sites.values() for r in group]
            metrics = ('NCR', 'capacity', 'greedy_capacity', 'actual_error_db')
            per_snr[str(snr)] = {metric: float(np.mean([r[metric] for r in cases])) for metric in metrics}
            per_snr[str(snr)]['site_macro_NCR'] = float(np.mean([np.mean([r['NCR'] for r in group]) for group in sites.values()]))
            per_snr[str(snr)]['n_sites'] = len(sites)
            per_snr[str(snr)]['n_decisions'] = len(cases)
        result[name] = {'per_snr': per_snr, **{metric: float(np.mean([s[metric] for s in per_snr.values()]))
                                              for metric in ('NCR', 'capacity', 'greedy_capacity', 'actual_error_db', 'site_macro_NCR')}}
    return result


def paired_cluster_ci(rows, reference, comparator, replicates, seed):
    """Resample matched site/scene clusters; preserve all their SNR decisions."""
    if replicates < 1 or seed < 0:
        raise ValueError('Positive bootstrap replicates and nonnegative seed required')
    def index(name):
        result = {}
        for r in rows:
            if r['method'] != name:
                continue
            key = (r['sample_id'], r['site_id'], r['snr_db'])
            if key in result:
                raise ValueError('Duplicate paired decision')
            result[key] = r['NCR']
        return result
    a, b = index(reference), index(comparator)
    if not a or a.keys() != b.keys():
        raise ValueError('Paired methods must share every evaluation decision')
    sites = sorted({key[1] for key in a})
    snrs = sorted({key[2] for key in a})
    if len(sites) < 2:
        raise ValueError('Cluster CI requires at least two independent clusters')
    si, fi = {v:i for i,v in enumerate(sites)}, {v:i for i,v in enumerate(snrs)}
    sums = np.zeros((len(sites),len(snrs)))
    counts = np.zeros_like(sums)
    for key, value in a.items():
        i, j = si[key[1]], fi[key[2]]
        sums[i,j] += value-b[key]
        counts[i,j] += 1
    if np.any(counts == 0):
        raise ValueError('Paper cluster bootstrap requires each cluster to cover every SNR')
    mean = float(np.mean(sums.sum(0)/counts.sum(0)))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        sample = rng.integers(0,len(sites),len(sites))
        draws[i] = np.mean(sums[sample].sum(0)/counts[sample].sum(0))
    return dict(reference=reference, comparator=comparator, paired_mean_NCR_gain=mean,
                ci95=[float(x) for x in np.quantile(draws,[.025,.975])],
                clusters=len(sites), bootstrap_replicates=replicates, bootstrap_seed=seed)


def check_paper_dimensions(h, k, task):
    expected = {'statistical_wideband': (32,16,128), 'fullwave_response': (1,16,256)}
    if tuple(h.shape) != expected[task] or k != 16:
        raise ValueError('Channel dimensions do not match the selected paper benchmark')


def check_paper_test_rows(rows, task):
    """Reject incomplete or mismatched test grids instead of silently averaging them."""
    expected_channels = 2000 if task == 'statistical_wideband' else 100
    expected_sites = 100
    grid = {0.,5.,10.,15.,20.}
    ref = [r for r in rows if r['method'] == 'Capacity-Greedy']
    keys = {(r['base_sample_id'],r['snr_db']) for r in ref}
    if len(ref) != len(keys) or len(keys) != 5*expected_channels:
        raise ValueError('Unexpected paper test decision count or repeated channel/SNR pair')
    if len({r['site_id'] for r in ref}) != expected_sites:
        raise ValueError('Unexpected number of independent sites/scenes')
    by_channel = {}
    for base,snr in keys:
        by_channel.setdefault(base,set()).add(snr)
    if any(s != grid for s in by_channel.values()):
        raise ValueError('Every test channel must be evaluated at all five paper SNRs')
    for name in {r['method'] for r in rows}:
        subset = [r for r in rows if r['method'] == name]
        if len(subset) != len(keys) or {(r['base_sample_id'],r['snr_db']) for r in subset} != keys:
            raise ValueError('All methods must share the full paper test grid')


def measure_selector(read_estimate, choose, synchronize, warmup, repeat):
    """Stored estimate access through port-index output; exclude true-CSI scoring."""
    for _ in range(warmup):
        choose(estimated_gram(read_estimate()))
    elapsed = []
    for _ in range(repeat):
        synchronize()
        start = time.perf_counter()
        choose(estimated_gram(read_estimate()))
        synchronize()
        elapsed.append((time.perf_counter()-start)*1000.)
    return elapsed


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class ChannelBank:
    """Explicit ID join, rather than assuming row order matches a dataset."""
    def __init__(self, path, real_key, imag_key, id_key):
        import h5py
        self.file = h5py.File(path, 'r')
        self.real = self.file[real_key]
        self.imag = self.file[imag_key]
        if self.real.shape != self.imag.shape or self.real.ndim != 4:
            raise ValueError('Channel bank arrays must match [N, F, Nr, Nt]')
        if id_key == 'row-index':
            ids = np.arange(len(self.real))
        else:
            ids = np.asarray(self.file[id_key][:]).reshape(-1)
        if len(ids) != len(self.real) or not np.issubdtype(ids.dtype, np.integer) or np.unique(ids).size != len(ids):
            raise ValueError('Bank IDs must be unique integers, one per channel')
        self.index = {int(value): i for i, value in enumerate(ids)}

    def get(self, key, site):
        if key not in self.index:
            raise KeyError(f'Missing channel ID {key}')
        i = self.index[key]
        if 'site_id' in self.file and int(self.file['site_id'][i]) != site:
            raise ValueError(f'Site/scene ID mismatch at channel ID {key}')
        return np.asarray(self.real[i], dtype=np.float64) + 1j * np.asarray(self.imag[i], dtype=np.float64)

    def close(self):
        self.file.close()


def check_manifest(path, checkpoint_hash, data_hash):
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    if m.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Frozen manifest checkpoint hash mismatch')
    if data_hash not in (m.get('data_sha256'), m.get('training_labels_sha256')):
        raise ValueError('Frozen manifest prepared-data hash mismatch')
    return m


def main(args):
    import torch
    from icsm_as.core.physics_data import PhysicsDataset
    from icsm_as.core.train_icsm_as import load_checkpoint
    from icsm_as.core.capacity_teacher import capacity_validated
    from icsm_as.models.baselines.capacity_greedy import capacity_greedy
    from icsm_as.models.baselines.power_top16 import power_topk
    from icsm_as.models.baselines.correlation_greedy import correlation_greedy
    from icsm_as.models.baselines.registry import build_baseline

    if not args.estimated_bank and (args.seed is None or args.seed < 0):
        raise ValueError('Synthetic mode requires an explicit nonnegative --seed')
    if args.timing and not args.estimated_bank:
        raise ValueError('Paper timing requires stored estimated CSI; do not time synthetic noise generation')
    if args.bootstrap < 1 or args.bootstrap_seed < 0 or args.timing_repeat < 1 or args.timing_warmup < 0:
        raise ValueError('Invalid bootstrap/timing configuration')
    data_hash = sha256(args.prepared_data)
    ckpt_hash = sha256(args.checkpoint)
    if args.split == 2 and not args.selection_manifest:
        raise ValueError('Test split requires a frozen selection manifest')
    frozen = check_manifest(args.selection_manifest, ckpt_hash, data_hash) if args.selection_manifest else None
    ds = PhysicsDataset(args.prepared_data, split=args.split, require_teacher=False)
    source_hash = None
    if ds.metadata.get('storage_mode') == 'lazy_raw_bank':
        source_hash = sha256(ds.metadata['source'])
        if frozen and frozen.get('source_bank_sha256') and source_hash != frozen['source_bank_sha256']:
            raise ValueError('Prepared raw-bank hash differs from frozen manifest')
    true_bank = ChannelBank(args.true_bank, args.true_real_key, args.true_imag_key, args.true_id_key)
    est_bank = ChannelBank(args.estimated_bank, args.est_real_key, args.est_imag_key, args.est_id_key) if args.estimated_bank else None
    out = Path(args.output)
    try:
        model, _ = load_checkpoint(args.checkpoint, args.device)
        model.to(args.device).eval()
        if (model.n_tx, model.n_sel) != (ds.n_tx, ds.n_sel):
            raise ValueError('Checkpoint dimensions do not match evaluation data')
        if getattr(model, 'decoder', None) != 'marginal' or getattr(model, 'risk_beta', None) != 0.0:
            raise ValueError('Expected the final marginal, mean-only ICSM-AS selector')
        other = {}
        baseline_provenance = []
        reserved = {'ICSM-AS', 'Capacity-Greedy', 'Power-Top16', 'Correlation-Greedy'}
        for ckpt_path, manifest_path in args.neural_baseline:
            h = sha256(ckpt_path)
            check_manifest(manifest_path, h, data_hash)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            name = ckpt['model_name']
            if name in other or name in ('ICSM-AS', 'Capacity-Greedy', 'Power-Top16', 'Correlation-Greedy'):
                raise ValueError(f'Duplicate method name {name}')
            kwargs = dict(ckpt.get('model_kwargs', {}))
            kwargs.update(n_tx=ds.n_tx, n_sel=ds.n_sel)
            baseline = build_baseline(name, **kwargs).to(args.device)
            baseline.load_state_dict(ckpt['state_dict'])
            other[name] = baseline.eval()
            baseline_provenance.append(dict(method=name, checkpoint_sha256=h, manifest_sha256=sha256(manifest_path)))
        # Native ICF/comparator checkpoints use the same framework loader, not the baseline registry.
        for name, ckpt_path, manifest_path in args.native_comparator:
            if name in reserved or name in other:
                raise ValueError(f'Duplicate method name {name}')
            h = sha256(ckpt_path)
            check_manifest(manifest_path, h, data_hash)
            comparator, _ = load_checkpoint(ckpt_path, args.device)
            if (comparator.n_tx, comparator.n_sel) != (ds.n_tx, ds.n_sel):
                raise ValueError('Comparator dimensions do not match data')
            other[name] = comparator.eval()
            baseline_provenance.append(dict(method=name, checkpoint_sha256=h, manifest_sha256=sha256(manifest_path)))
        out.mkdir(parents=True, exist_ok=False)
        provenance = dict(protocol='dual-csi-paper-v2', status='incomplete', arguments=vars(args),
                          prepared_data_sha256=data_hash, prepared_source_bank_sha256=source_hash, checkpoint_sha256=ckpt_hash,
                          true_bank_sha256=sha256(args.true_bank),
                          estimated_bank_sha256=sha256(args.estimated_bank) if args.estimated_bank else None,
                          selection_manifest_sha256=sha256(args.selection_manifest) if args.selection_manifest else None,
                          baseline_checkpoints=baseline_provenance, numpy_version=np.__version__, torch_version=str(torch.__version__),
                          aggregation='matched-decision mean within SNR, equal SNR weights; site-macro NCR separately; no clipping',
                          error_protocol='recorded estimates' if est_bank else 'synthetic normalized circular-Gaussian direction per base channel')
        (out / 'manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        rows = []
        timings = {'ICSM-AS': [], 'Capacity-Greedy': []}
        with torch.no_grad():
            for i in range(len(ds)):
                sample = ds.evaluation_metadata(i)
                base_id, sample_id, site = (int(sample[key]) for key in ('base_sample_id', 'sample_id', 'site_id'))
                snr = float(sample['snr_reference_db'])
                rho = float(sample['snr_reference_linear'])
                h = true_bank.get(base_id, site)
                check_paper_dimensions(h, ds.n_sel, args.task)
                if h.shape[-1] != ds.n_tx:
                    raise ValueError('True channel port dimensions do not match prepared data')
                if est_bank:
                    h_est = est_bank.get(base_id if args.estimate_per == 'base' else sample_id, site)
                else:
                    h_est = synthetic_estimate(h, args.error_db, args.seed, base_id)
                k = ds.n_sel
                static = {key: sample[key].unsqueeze(0) for key in STATIC_KEYS if key in sample}
                # Main, classical and baseline selectors receive estimated-channel features only.
                # Static port-response descriptors are independent of the CSI perturbation.

                def batch_from_estimate(g):
                    # Do not copy labels, original Gram, relation targets or capacities.
                    return {**{key: value.to(args.device) for key,value in static.items()},
                            'G_tx_real': torch.from_numpy(g.real.astype(np.float32)).unsqueeze(0).to(args.device),
                            'G_tx_imag': torch.from_numpy(g.imag.astype(np.float32)).unsqueeze(0).to(args.device),
                            'X_rel': torch.from_numpy(estimated_relation(g)).unsqueeze(0).to(args.device),
                            'effective_snr_db': torch.tensor([snr], dtype=torch.float32, device=args.device),
                            'snr_reference_db': torch.tensor([snr], dtype=torch.float32, device=args.device),
                            'snr_reference_linear': torch.tensor([rho], dtype=torch.float32, device=args.device)}

                selectors = {
                    'ICSM-AS': lambda g: model.select(model(batch_from_estimate(g)))[0].cpu().numpy(),
                    'Capacity-Greedy': lambda g: capacity_greedy(g, rho, k)[0],
                    'Power-Top16': lambda g: power_topk(g, k),
                    'Correlation-Greedy': lambda g: correlation_greedy(g, k),
                }
                for name, baseline in other.items():
                    selectors[name] = lambda g, b=baseline: b.select(b(batch_from_estimate(g)), k=k)[0].cpu().numpy()
                if args.timing:
                    estimate_id = base_id if args.estimate_per == 'base' else sample_id
                    sync = (lambda: torch.cuda.synchronize(torch.device(args.device))) if str(args.device).startswith('cuda') else (lambda: None)
                    for name in timings:
                        timings[name].extend(measure_selector(lambda: est_bank.get(estimate_id,site), selectors[name], sync,
                                                             args.timing_warmup,args.timing_repeat))
                for row in evaluate_case(h, h_est, rho, k, selectors, capacity_validated):
                    rows.append(dict(condition=args.condition, sample_id=sample_id, base_sample_id=base_id,
                                     site_id=site, snr_db=snr, **row))
        if not rows:
            raise ValueError('Empty evaluation split')
        if args.split == 2:
            check_paper_test_rows(rows,args.task)
        with (out / 'predictions.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows)
        paired = {name: paired_cluster_ci(rows,'ICSM-AS',name,args.bootstrap,args.bootstrap_seed)
                  for name in other}
        payload = dict(methods=summary, paired_cluster_ci=paired)
        if args.timing:
            timing = {name: dict(p50_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),n=len(vals))
                      for name,vals in timings.items()}
            timing['median_speedup'] = timing['Capacity-Greedy']['p50_ms']/timing['ICSM-AS']['p50_ms']
            timing['device'] = torch.cuda.get_device_name(torch.device(args.device)) if str(args.device).startswith('cuda') else 'CPU'
            payload['timing'] = timing
        (out / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        provenance.update(status='complete', n_rows=len(rows), predictions_sha256=sha256(out / 'predictions.csv'))
        (out / 'manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        print(json.dumps(payload, indent=2))
    finally:
        ds.close()
        true_bank.close()
        if est_bank:
            est_bank.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['statistical_wideband','fullwave_response'], required=True)
    p.add_argument('--prepared-data', required=True, help='v2 prepare_physics_data.py HDF5 output')
    p.add_argument('--true-bank', required=True, help='Original effective channels before CSI estimation error')
    p.add_argument('--true-real-key', default='H_real')
    p.add_argument('--true-imag-key', default='H_imag')
    p.add_argument('--true-id-key', default='base_sample_id', help='Explicit HDF5 ID key or row-index')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--estimated-bank', help='Recorded effective channel estimates')
    mode.add_argument('--error-db', type=float, help='Synthetic error power (dB), explicit; no default nominal level')
    p.add_argument('--est-real-key', default='H_est_real')
    p.add_argument('--est-imag-key', default='H_est_imag')
    p.add_argument('--est-id-key', default='base_sample_id', help='ID key or explicit row-index')
    p.add_argument('--estimate-per', choices=['base', 'decision'], default='base')
    p.add_argument('--condition', required=True, help='Name of the evaluated CSI condition')
    p.add_argument('--seed', type=int, help='Required in synthetic mode; optional provenance in recorded mode')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selection-manifest', help='Frozen checkpoint/data selection manifest; required for test')
    p.add_argument('--neural-baseline', nargs=2, action='append', default=[], metavar=('CHECKPOINT', 'MANIFEST'))
    p.add_argument('--native-comparator', nargs=3, action='append', default=[], metavar=('NAME','CHECKPOINT','MANIFEST'))
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--bootstrap-seed', type=int, default=2026)
    p.add_argument('--timing', action='store_true', help='Measure batch-one selectors from stored estimated CSI access')
    p.add_argument('--timing-warmup', type=int, default=1)
    p.add_argument('--timing-repeat', type=int, default=1)
    p.add_argument('--split', type=int, choices=[1, 2], default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True, help='New, nonexistent output directory')
    return p


if __name__ == '__main__':
    main(parser().parse_args())

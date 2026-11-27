"""Development-only timing; paper end-to-end timing uses the dual-CSI evaluator --timing.

Measure ready-Gram inference; excludes CSI acquisition/Gram/EM construction.

Run batch_size=1 for request latency. Batch latency / B is amortized throughput,
not single-request latency. Tiny mode is a software smoke test, not Qwen-0.5B.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
import transformers
import h5py
from transformers import Qwen2Config
from torch.utils.data import DataLoader
from .model import count_parameters
from .scmg_variant import SCMGVariant
from .icsm_as_model import ICSMAS,induced_current_kernel
from .physics import synthetic_passive_impedance,terminated_current_transfer
from .physics_data import PhysicsDataset
from .train_icsm_as import load_checkpoint


def dense_mac_estimate(model, nf):
    """Real dense multiply-accumulates, excluding norms/activations/softmax.

    Attention is a dense upper accounting (do not assume a causal kernel saves
    exactly half). This estimates arithmetic, not actual hardware instructions.
    """
    c = model.backbone.config
    n, k, d, p = model.n_tx, model.n_sel, model.hidden_size, model.model_options["adapter_dim"]
    layers, inter = c.num_hidden_layers, c.intermediate_size
    head_dim = getattr(c, "head_dim", None) or d//c.num_attention_heads
    dkv = c.num_key_value_heads*head_dim
    r, h = model.relation_rank, model.model_options["graph_heads"]
    if isinstance(model,ICSMAS):
        # Main real dense MACs. Nonlinear complex magnitudes, masks, pooling
        # reductions and current-kernel construction are separately documented.
        ab = model.adapter.ablations
        costs = dict(
            PCR=nf*n*((2*n+4)*p+p*p),
            CSF=4*nf*n*n*p+nf*n*(2*p+4)*p+n*p*d +
                (0 if "spectral" in ab else 4*n*nf*p*p+2*n*nf*nf*p+nf*n*p),
            ECG=0 if "graph" in ab else 4*n*p*p+2*n*n*p+n*n*(7*32+32*h),
            SFI_cache=n*d*p+3*n*p*p+n*n*p+10*n*n,
            SFI_steps=0 if "feedback" in ab else k*(p*p+2*p+n*n*p+n*(3*p+2)*p),
            DMH=k*n*(p*p+2*p),
            backbone=layers*(2*n*d*d+2*n*d*dkv+2*n*n*d+3*n*d*inter),
            membership=n*(d*(d//2)+d//2),relation=n*d*r+n*n*r)
        costs["total_dense_MAC_per_sample"] = sum(costs.values())
        return costs
    costs = dict(tokenizer=nf*n*((2*n+4)*p+p*p+p),
        graph=0 if model.graph is None else 4*n*p*p+2*n*n*p+n*n*(5*32+32*h),
        to_backbone=n*p*d,
        backbone=layers*(2*n*d*d+2*n*d*dkv+2*n*n*d+3*n*d*inter),
        membership=n*(d*(d//2)+d//2), relation=n*d*r+n*n*r,
        marginal=0 if model.marginal_head is None else n*d*p+k*(p*p+p+n*p*p+n*p))
    costs["total_dense_MAC_per_sample"] = sum(costs.values())
    return costs


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timing_summary(samples):
    return dict(mean_ms=float(np.mean(samples)), p50_ms=float(np.percentile(samples, 50)),
                p95_ms=float(np.percentile(samples, 95)))


@torch.no_grad()
def benchmark_validation(model, data_h5, device, repeat, warmup, snr_db):
    dataset = PhysicsDataset(data_h5, split=1)
    metadata = dataset.metadata
    if metadata.get("storage_mode") != "lazy_raw_bank" or metadata.get("preparation_role") not in (
        "formal_train_validation", "train_sites_internal_pilot"
    ):
        raise ValueError("Validation timing requires a train/validation-only lazy-bank dataset")
    if set(metadata.get("label_splits", [])) != {"train", "val"}:
        raise ValueError("Validation timing requires labels prepared without the test split")
    with h5py.File(data_h5, "r") as source:
        val_indices = dataset.indices
        snrs = np.asarray(source["snr_reference_db"][val_indices], dtype=np.float64)
        selected = val_indices[np.isclose(snrs, snr_db)]
        base_ids = np.asarray(source["base_sample_id"][selected], dtype=np.int64)
        site_ids = np.asarray(source["site_id"][selected], dtype=np.int64)
    if not len(selected) or len(np.unique(base_ids)) != len(selected):
        raise ValueError(f"Expected one validation row per base channel at {snr_db:g} dB")
    dataset.indices = selected
    if len(dataset) < warmup + repeat:
        raise ValueError(f"Need at least {warmup + repeat} unique validation rows; found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    iterator = iter(loader)
    model.eval()
    device_keys = {
        "G_tx_real", "G_tx_imag", "effective_snr_db", "frequency_coord", "positions_wavelength",
        "T_tx_real", "T_tx_imag", "Z_tx_real", "Z_tx_imag", "source_impedance",
        "reference_impedance", "current_kernel_real", "current_kernel_imag",
        "accepted_power_per_source",
    }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run_one():
        host = next(iterator)
        batch = {key: value.to(device) for key, value in host.items()
                 if key in device_keys and torch.is_tensor(value)}
        model.select(model(batch))

    for _ in range(warmup):
        run_one()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data_ms, transfer_ms, inference_ms, end_to_end_ms = [], [], [], []
    for _ in range(repeat):
        sync()
        start = time.perf_counter()
        host = next(iterator)
        after_data = time.perf_counter()
        batch = {key: value.to(device) for key, value in host.items()
                 if key in device_keys and torch.is_tensor(value)}
        sync()
        after_transfer = time.perf_counter()
        model.select(model(batch))
        sync()
        end = time.perf_counter()
        data_ms.append((after_data - start) * 1000)
        transfer_ms.append((after_transfer - after_data) * 1000)
        inference_ms.append((end - after_transfer) * 1000)
        end_to_end_ms.append((end - start) * 1000)
    timing = dict(
        validation_row_read_and_gram_preparation=_timing_summary(data_ms),
        host_to_device=_timing_summary(transfer_ms),
        model_forward_and_decode=_timing_summary(inference_ms),
        end_to_end_request_latency=_timing_summary(end_to_end_ms),
        measured_requests=repeat,
        single_request_latency_measured=True,
    )
    if device.type == "cuda":
        timing["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        timing["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
    return dict(
        timing=timing,
        data=dict(
            path=str(Path(data_h5).resolve()), split_id=1, snr_reference_db=float(snr_db),
            measured_validation_rows=repeat, validation_rows_at_snr=len(selected),
            validation_sites=int(len(np.unique(site_ids))),
            source_bank_sha256=metadata.get("source_file_sha256"),
            preparation_role=metadata["preparation_role"],
        ),
    )


@torch.no_grad()
def benchmark(model, host, device, repeat, warmup):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def measure(fn):
        samples = []
        for _ in range(repeat):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            samples.append((time.perf_counter()-start)*1000)
        return dict(mean_ms=float(np.mean(samples)), p50_ms=float(np.percentile(samples,50)),
                    p95_ms=float(np.percentile(samples,95)))
    def transfer():
        return {key: value.to(device) for key, value in host.items()}
    batch = transfer()
    model.eval()
    for _ in range(warmup):
        model.select(model(batch))
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cached = model(batch)
    stats = dict(host_to_device=measure(transfer), model_forward=measure(lambda:model(batch)),
        decode_from_cached_output=measure(lambda:model.select(cached)),
        ready_host_gram_to_subset=measure(lambda:model.select(model(transfer()))))
    b = host["G_tx_real"].shape[0]
    stats["amortized_ms_per_sample"] = stats["ready_host_gram_to_subset"]["mean_ms"]/b
    stats["single_request_latency_measured"] = b == 1
    if device.type == "cuda":
        stats["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device)/1024**2
        stats["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device)/1024**2
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--data_h5", help="Validation-only prepared lazy-bank data for real-input timing")
    p.add_argument("--snr_db", type=float, default=10.0,
                   help="Validation reference SNR used by --data_h5 (default: 10 dB)")
    p.add_argument("--q2_dir", help="Load a real pretrained backbone with new, untrained heads")
    p.add_argument("--n_tx", type=int, default=128)
    p.add_argument("--n_sel", type=int, default=16)
    p.add_argument("--tiny_smoke", action="store_true")
    p.add_argument("--architecture", choices=["scmg_variant","icsm_as"],default="icsm_as")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_frequencies", type=int, default=8)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", default="runs/benchmark.json")
    a = p.parse_args()
    if min(a.batch_size, a.n_frequencies, a.repeat, a.threads) < 1 or a.warmup < 0:
        p.error("Invalid count")
    if not np.isfinite(a.snr_db):
        p.error("--snr_db must be finite")
    if a.data_h5 and (a.tiny_smoke or a.q2_dir or not a.checkpoint or a.architecture != "icsm_as" or a.batch_size != 1):
        p.error("--data_h5 requires an ICSM-AS --checkpoint, batch_size=1, and no tiny_smoke or q2_dir")
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    device = torch.device(a.device)
    model_class = ICSMAS if a.architecture == "icsm_as" else SCMGVariant
    if a.tiny_smoke:
        c = Qwen2Config(hidden_size=32,intermediate_size=48,num_hidden_layers=1,
            num_attention_heads=4,num_key_value_heads=2,vocab_size=32)
        model = model_class("unused", n_tx=6, n_sel=3, adapter_dim=16, relation_rank=8,
            use_pretrained_weights=False, backbone_config=c.to_dict()).to(device)
    elif a.checkpoint:
        model, _ = load_checkpoint(a.checkpoint, device)
    elif a.q2_dir:
        model = model_class(a.q2_dir, n_tx=a.n_tx, n_sel=a.n_sel).to(device)
    else:
        p.error("Use --checkpoint, --q2_dir, or --tiny_smoke")
    n, b, nf = model.n_tx, a.batch_size, a.n_frequencies
    if a.data_h5:
        measured = benchmark_validation(model, a.data_h5, device, a.repeat, a.warmup, a.snr_db)
        nf = int(PhysicsDataset(a.data_h5, split=1).metadata["n_frequencies"])
        result = dict(
            status="validation-only real-input timing; test split not opened",
            python=platform.python_version(),torch=str(torch.__version__),transformers=transformers.__version__,
            device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
            dtype=str(next(model.parameters()).dtype),threads=a.threads,batch_size=1,n_frequencies=nf,
            parameters=count_parameters(model),method_name=model.method_name,
            checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=file_sha256(a.checkpoint),
            estimated_dense_MACs=dense_mac_estimate(model,nf),
            data=measured["data"],timing=measured["timing"],
            included="validation HDF5 row access, lazy raw-CSI Gram/relation preparation, host-to-device transfer, model forward and subset decoding",
            excluded="CSI acquisition, EM network solve, H*T, disk cache effects, offline teachers, exact capacity evaluation",
            physics_scope=PhysicsDataset(a.data_h5, split=1).metadata.get(
                "physics_status", "unspecified"
            ),
        )
        output = Path(a.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result["timing"],indent=2))
        return
    h = torch.randn(b,nf,min(n,16),n,dtype=torch.complex64)
    g = h.mH@h
    host = dict(G_tx_real=g.real.contiguous(),G_tx_imag=g.imag.contiguous(),
        effective_snr_db=torch.full((b,),10.),coupling_edge=torch.zeros(b,n,n),
        positions_wavelength=torch.zeros(b,n,3))
    host["positions_wavelength"][:,:,0] = torch.arange(n)*.5
    if isinstance(model,ICSMAS):
        frequencies = 28e9+torch.arange(nf,dtype=torch.float64)*120e3
        positions = host["positions_wavelength"][0].double()*(299792458./28e9)
        z = synthetic_passive_impedance(positions,frequencies)
        transfer = terminated_current_transfer(z).to(torch.complex64)
        g = transfer.mH@g@transfer
        kernel = induced_current_kernel(transfer)
        accepted = .5*(transfer.mH@((z.to(torch.complex64)+z.to(torch.complex64).mH)/2)@transfer).diagonal(dim1=-2,dim2=-1).real
        host.update(G_tx_real=g.real.contiguous(),G_tx_imag=g.imag.contiguous(),
            T_tx_real=transfer.real[None].expand(b,-1,-1,-1).contiguous(),
            T_tx_imag=transfer.imag[None].expand(b,-1,-1,-1).contiguous(),
            current_kernel_real=kernel.real[None].expand(b,-1,-1).contiguous(),
            current_kernel_imag=kernel.imag[None].expand(b,-1,-1).contiguous(),
            accepted_power_per_source=accepted[None].expand(b,-1,-1).contiguous())
    result = dict(status="SYNTHETIC TINY FUNCTIONAL CHECK" if a.tiny_smoke else "synthetic-input timing, not accuracy evaluation; new heads untrained if q2_dir used",
        python=platform.python_version(),torch=str(torch.__version__),transformers=transformers.__version__,
        device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
        dtype=str(next(model.parameters()).dtype),threads=a.threads,batch_size=b,n_frequencies=nf,
        parameters=count_parameters(model),
        parameters_by_component={name:count_parameters(module) for name,module in model.named_children()},
        adapter_components={} if not isinstance(model,ICSMAS) else {name:count_parameters(module) for name,module in model.adapter.named_children()},
        method_name=model.method_name,
        config=model.backbone.config.to_dict(),estimated_dense_MACs=dense_mac_estimate(model,nf),
        timing=benchmark(model,host,device,a.repeat,a.warmup),
        excluded="CSI acquisition, EM network solve, H*T, Gram formation, induced-current kernel cache construction, disk IO, offline teachers, exact capacity evaluation",
        physics_input="synthetic sampled-passive diagnostic impedance only; not full-wave or measured" if isinstance(model,ICSMAS) else "none")
    output = Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["timing"],indent=2))


if __name__ == "__main__":
    main()

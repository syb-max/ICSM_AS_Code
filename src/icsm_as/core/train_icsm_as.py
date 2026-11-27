"""Train, validate, or open the locked test split for the ICSM-AS model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import h5py
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .capacity_teacher import capacity_validated
from ..utils.reproducibility import set_reproducible
from .icsm_as_model import ICSMAS
from .model import count_parameters
from .physics_data import PhysicsDataset
from .scmg_variant import SCMGVariant

RATIO_DENOMINATOR_THRESHOLD = 1e-8


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") not in ("q2_physics_v1", "icsm_as_v1"):
        raise ValueError("Expected an ICSM-AS/physics-guided checkpoint")
    architecture = ckpt.get("architecture", "scmg_variant")
    options = dict(ckpt["model_options"])
    options["use_pretrained_weights"] = False
    model_class = ICSMAS if architecture in ("icsm_as", "icf") else SCMGVariant
    model = model_class(**options, backbone_config=ckpt["backbone_config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def loss_terms(model, outputs, batch, alpha_hq=.5, lambda_relation=.2, lambda_marginal=1., lambda_membership=1., lambda_regression=1., lambda_rank=.1):
    membership = alpha_hq * F.binary_cross_entropy_with_logits(
        outputs["utility_logits"], batch["hq_utility_target"]
    )
    membership = membership + (1 - alpha_hq) * F.binary_cross_entropy_with_logits(
        outputs["utility_logits"], batch["lq_utility_target"]
    )
    n = model.n_tx
    mask = ~torch.eye(n, device=outputs["utility_logits"].device, dtype=torch.bool)
    relation = F.mse_loss(
        outputs["relation_logits"].sigmoid()[:, mask], batch["relation_target"][:, mask]
    ) if n > 1 else membership * 0
    marginal = model.marginal_loss(outputs, batch["teacher_order"].long(), batch["teacher_marginals"], regression_weight=lambda_regression, ranking_weight=lambda_rank)
    total = lambda_membership * membership + lambda_relation * relation + lambda_marginal * marginal
    return dict(loss=total, membership=membership, relation=relation, marginal=marginal)


def _macro_site_mean(rows, value_key, snr=None):
    by_site = {}
    for row in rows:
        if snr is not None and not np.isclose(row["snr_reference_db"], snr):
            continue
        value = row.get(value_key)
        if value is None or not np.isfinite(value):
            continue
        by_site.setdefault(row["site_id"], []).append(float(value))
    return float(np.mean([np.mean(values) for values in by_site.values()])) if by_site else None


def _bootstrap_site_ci(rows, snr, value_key, replicates, seed):
    if replicates < 1:
        return None
    by_site = {}
    for row in rows:
        if np.isclose(row["snr_reference_db"], snr):
            value = row.get(value_key)
            if value is not None and np.isfinite(value):
                by_site.setdefault(row["site_id"], []).append(float(value))
    sites = sorted(by_site)
    if not sites:
        return None
    site_means = np.asarray([np.mean(by_site[site]) for site in sites], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        draws[i] = site_means[rng.integers(0, len(sites), size=len(sites))].mean()
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


@torch.no_grad()
def evaluate(model, loader, device, *, predictions_path=None, bootstrap=0, bootstrap_seed=2026):
    """Internal checkpoint-selection diagnostic; not the paper test evaluator."""
    model.eval()
    rows = []
    for host in loader:
        batch = {key: value.to(device) for key, value in host.items() if torch.is_tensor(value)}
        selected = model.select(model(batch)).cpu().numpy()
        gram = host["G_tx_real"].numpy() + 1j * host["G_tx_imag"].numpy()
        for j, subset in enumerate(selected):
            rho = 10.0 ** (float(host["snr_reference_db"][j]) / 10.0)
            cap = capacity_validated(gram[j], rho, subset, final_k=model.n_sel)
            hq = float(host["hq_capacity"][j])
            greedy = float(host["greedy_capacity"][j]) if "greedy_capacity" in host else None
            sample_id = int(host["sample_id"][j])
            site_id = int(host["site_id"][j]) if "site_id" in host else sample_id
            row = dict(
                sample_id=sample_id,
                base_sample_id=int(host["base_sample_id"][j]) if "base_sample_id" in host else sample_id,
                site_id=site_id,
                snr_reference_db=float(host["snr_reference_db"][j]),
                capacity=cap,
                hq_capacity=hq,
                c_over_hq=cap / hq if hq > RATIO_DENOMINATOR_THRESHOLD else None,
                greedy_capacity=greedy,
                c_over_greedy=cap / greedy if greedy is not None and greedy > RATIO_DENOMINATOR_THRESHOLD else None,
                ratio_eligible=greedy is not None and greedy > RATIO_DENOMINATOR_THRESHOLD,
                gap_greedy=cap - greedy if greedy is not None else None,
                los=int(host["los"][j]) if "los" in host else None,
                blocked=int(host["blocked"][j]) if "blocked" in host else None,
                beta_db=float(host["beta_db"][j]) if "beta_db" in host else None,
                selected=" ".join(map(str, subset.tolist())),
            )
            rows.append(row)
    snrs = sorted({row["snr_reference_db"] for row in rows})
    per_snr = {}
    for snr in snrs:
        selected_rows = [row for row in rows if np.isclose(row["snr_reference_db"], snr)]
        per_snr[str(snr)] = dict(
            n_decisions=len(selected_rows),
            n_sites=len({row["site_id"] for row in selected_rows}),
            n_ratio_eligible=sum(bool(row["ratio_eligible"]) for row in selected_rows),
            ratio_excluded_fraction=(
                1.0 - sum(bool(row["ratio_eligible"]) for row in selected_rows) / len(selected_rows)
                if selected_rows else None
            ),
            mean_site_capacity=_macro_site_mean(rows, "capacity", snr),
            mean_site_c_over_hq=_macro_site_mean(rows, "c_over_hq", snr),
            mean_site_c_over_greedy=_macro_site_mean(rows, "c_over_greedy", snr),
            mean_site_gap_greedy=_macro_site_mean(rows, "gap_greedy", snr),
        )
        if bootstrap:
            per_snr[str(snr)]["site_bootstrap_95ci_capacity"] = _bootstrap_site_ci(
                rows, snr, "capacity", bootstrap, bootstrap_seed
            )
            per_snr[str(snr)]["site_bootstrap_95ci_gap_greedy"] = _bootstrap_site_ci(
                rows, snr, "gap_greedy", bootstrap, bootstrap_seed + int(snr * 10)
            )
    result = dict(
        n_decisions=len(rows),
        n_sites=len({row["site_id"] for row in rows}),
        site_macro_mean_capacity=_macro_site_mean(rows, "capacity"),
        mean_capacity=_macro_site_mean(rows, "capacity"),
        site_macro_mean_c_over_hq=float(np.mean([
            value["mean_site_c_over_hq"] for value in per_snr.values()
            if value["mean_site_c_over_hq"] is not None
        ])) if per_snr else None,
        site_macro_mean_capacity_ratio_greedy=float(np.mean([
            value["mean_site_c_over_greedy"] for value in per_snr.values()
            if value["mean_site_c_over_greedy"] is not None
        ])) if per_snr else None,
        per_snr=per_snr,
        ratio_denominator_threshold_bit_s_per_hz=RATIO_DENOMINATOR_THRESHOLD,
        selection_metric="equal-weight mean across SNR of eligible site-macro C_ICSM-AS/C_Capacity-Greedy; ratios below the denominator threshold are undefined",
        reference_note="Greedy and HQ are feasible references, not global optima; ratios are not clipped",
    )
    if predictions_path is not None and rows:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


class BaseSampleGroupSampler(Sampler):
    def __init__(self, dataset, shuffle, one_snr_per_base=False):
        self.dataset = dataset
        self.shuffle = shuffle
        with h5py.File(dataset.path, "r") as f:
            base_ids = np.asarray(f["base_sample_id"][dataset.indices])
            snr_values = np.asarray(f["snr_reference_db"][dataset.indices])
        groups = {}
        for local_index, (base_id, snr) in enumerate(zip(base_ids, snr_values)):
            groups.setdefault(int(base_id), []).append((float(snr), local_index))
        self.groups = [
            [local_index for _, local_index in sorted(groups[base_id])]
            for base_id in sorted(groups)
        ]
        self.one_snr_per_base = one_snr_per_base
        self.epoch = 0
        if one_snr_per_base:
            snr_grids = [
                [float(snr_values[local_index]) for local_index in group]
                for group in self.groups
            ]
            reference = np.asarray(snr_grids[0], dtype=np.float64)
            if not len(reference) or np.unique(reference).size != len(reference) or any(
                not np.array_equal(np.asarray(grid, dtype=np.float64), reference)
                for grid in snr_grids
            ):
                raise ValueError("one-SNR sampling requires the same unique nonempty SNR grid for every base channel")
            self.n_snr = len(reference)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.groups) if self.one_snr_per_base else len(self.dataset)

    def __iter__(self):
        order = torch.randperm(len(self.groups)).tolist() if self.shuffle else range(len(self.groups))
        for group_index in order:
            group = self.groups[group_index]
            if self.one_snr_per_base:
                yield group[(self.epoch + group_index) % self.n_snr]
            else:
                yield from group


def _loader(dataset, batch_size, shuffle, num_workers, one_snr_per_base=False):
    if one_snr_per_base and not dataset._lazy_bank:
        raise ValueError("one-SNR-per-base sampling requires a lazy bank dataset")
    if dataset._lazy_bank:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=BaseSampleGroupSampler(dataset, shuffle, one_snr_per_base),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _write_checkpoint(model, args, dataset, save_dir, name, optimizer, epoch, validation_metric, stale_epochs, training_history=None):
    path = Path(save_dir) / name
    temporary = path.with_suffix(".pt.partial")
    torch.save(dict(
        format="q2_physics_v1",
        architecture="icsm_as",
        method_name=model.method_name,
        model=model.state_dict(),
        model_options=model.model_options,
        backbone_config=model.backbone.config.to_dict(),
        args={key: value for key, value in vars(args).items() if key != "func"},
        dataset_metadata=dataset.metadata,
        parameters=count_parameters(model),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        best_metric=validation_metric,
        stale_epochs=stale_epochs,
        training_history=training_history,
    ), temporary)
    os.replace(temporary, path)


def _has_pretrained_weights(folder):
    folder = Path(folder)
    has_weights = any((folder / name).is_file() for name in (
        "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"
    ))
    has_tokenizer = any((folder / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "vocab.json"
    ))
    return has_weights and has_tokenizer and (folder / "config.json").is_file()


def _training_progress(save_dir, **payload):
    """Atomic heartbeat for long GPU jobs; does not change training numerics."""
    path = Path(save_dir) / "training_progress.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(dict(pid=os.getpid(), updated_unix=time.time(), **payload)), encoding="utf-8")
    os.replace(temporary, path)


def command_train(a):
    if a.epochs < 1 or a.patience < 1 or a.gradient_accumulation_steps < 1 or not 0 <= a.alpha_hq <= 1:
        raise ValueError("epochs, patience, and gradient accumulation must be positive; alpha_hq must be in [0,1]")
    weights = (a.lambda_membership, a.lambda_relation, a.lambda_marginal, a.lambda_regression, a.lambda_rank)
    if not all(np.isfinite(w) and w >= 0 for w in weights) or a.lambda_marginal <= 0:
        raise ValueError("Loss weights must be finite/nonnegative, with positive lambda_marginal")
    if a.lambda_regression == 0 and a.lambda_rank == 0 and not a.use_nll_ablation:
        raise ValueError("The marginal decoder must retain at least one marginal-head objective")
    if a.architecture != "icsm_as":
        raise ValueError("This entry point trains the proposed ICSM-AS architecture")
    if a.use_pretrained_weights and not _has_pretrained_weights(a.q2_dir):
        raise FileNotFoundError(
            "--q2_dir must contain a real Qwen2 checkpoint, tokenizer, and config; "
            "a config-only folder is not pretrained"
        )

    set_reproducible(a.seed, deterministic=True)
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    train_ds = PhysicsDataset(a.data_h5, 0)
    val_ds = PhysicsDataset(a.data_h5, 1)
    if train_ds.metadata.get("gram_basis") == "statistical_proxy" and not a.allow_proxy_physics:
        raise ValueError("Statistical Gram data uses the identity PCR reference; enable --allow_proxy_physics explicitly")
    train = _loader(train_ds, a.batch_size, True, a.num_workers, a.one_snr_per_base)
    val = _loader(val_ds, a.batch_size, False, a.num_workers)
    model = ICSMAS(
        a.q2_dir,
        n_tx=train_ds.n_tx,
        n_sel=train_ds.n_sel,
        adapter_dim=a.adapter_dim,
        graph_heads=a.graph_heads,
        relation_rank=a.relation_rank,
        freeze_backbone=bool(a.freeze_backbone),
        use_pretrained_weights=bool(a.use_pretrained_weights),
        train_last_n_layers=a.train_last_n_layers,
        token_mode="complex",
        use_graph=True,
        decoder="marginal",
        allow_proxy_physics=a.allow_proxy_physics,
        risk_beta=a.risk_beta,
        use_nll_ablation=a.use_nll_ablation,
        ablations=[item for item in (x.strip() for x in a.ablations.split(",")) if item],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=a.lr,
        weight_decay=a.weight_decay,
    )
    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if a.resume and history_path.exists() else []
    start_epoch = 0
    best = -float("inf")
    stale_epochs = 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") not in ("icsm_as", "icf"):
            raise ValueError("--resume must point to an ICSM-AS checkpoint (legacy architecture=icf is accepted)")
        previous_hash = checkpoint.get("dataset_metadata", {}).get("source_file_sha256")
        current_hash = train_ds.metadata.get("source_file_sha256")
        if previous_hash and current_hash and previous_hash != current_hash:
            raise ValueError("Resume checkpoint belongs to a different source dataset")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best = float(checkpoint.get("best_metric", -float("inf")))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
    for epoch in range(start_epoch, a.epochs):
        epoch_started = time.perf_counter()
        if hasattr(train.sampler, "set_epoch"):
            train.sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total = 0.0
        count = 0
        _training_progress(save_dir, phase="train", epoch=epoch+1, total_epochs=a.epochs,
                           batches_completed=0, batches_total=len(train))
        for batch_index, host in enumerate(train):
            group_start = (batch_index // a.gradient_accumulation_steps) * a.gradient_accumulation_steps
            group_size = min(a.gradient_accumulation_steps, len(train) - group_start)
            if batch_index == group_start:
                optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in host.items() if torch.is_tensor(value)}
            terms = loss_terms(
                model, model(batch), batch, a.alpha_hq, a.lambda_relation, a.lambda_marginal,
                 a.lambda_membership, a.lambda_regression, a.lambda_rank
            )
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch + 1}")
            (terms["loss"] / group_size).backward()
            size = len(host["sample_id"])
            count += size
            total += float(terms["loss"].detach()) * size
            end_group = (batch_index + 1) % a.gradient_accumulation_steps == 0 or batch_index + 1 == len(train)
            if end_group:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                optimizer.step()
            if (batch_index+1) % 100 == 0 or batch_index+1 == len(train):
                heartbeat = dict(phase="train", epoch=epoch+1, total_epochs=a.epochs,
                    batches_completed=batch_index+1, batches_total=len(train),
                    decisions_completed=count, mean_loss=total/max(count,1),
                    epoch_elapsed_seconds=time.perf_counter()-epoch_started)
                _training_progress(save_dir, **heartbeat)
                print(json.dumps(dict(progress=heartbeat)), flush=True)
        _training_progress(save_dir, phase="validation", epoch=epoch+1, total_epochs=a.epochs,
                           decisions_total=len(val_ds), epoch_elapsed_seconds=time.perf_counter()-epoch_started)
        validation = evaluate(model, val, device)
        metric = validation["site_macro_mean_capacity_ratio_greedy"]
        if metric is None or not np.isfinite(metric):
            raise ValueError("Validation site-macro C_ICSM-AS/C_Capacity-Greedy is undefined")
        row = dict(
            epoch=epoch + 1,
            train_loss=total / max(count, 1),
            train_decisions=count,
            validation=validation,
            elapsed_seconds=time.perf_counter() - epoch_started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        )
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if metric > best:
            best = metric
            stale_epochs = 0
            _write_checkpoint(model, a, train_ds, save_dir, "best.pt", optimizer, epoch + 1, best, stale_epochs, history)
        else:
            stale_epochs += 1
        _write_checkpoint(model, a, train_ds, save_dir, "last.pt", optimizer, epoch + 1, best, stale_epochs, history)
        history_temporary = history_path.with_suffix('.json.partial')
        history_temporary.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(history_temporary, history_path)
        if stale_epochs >= a.patience:
            break
    if not (save_dir / "best.pt").exists():
        raise RuntimeError("No validation-best checkpoint was written")
    outcome = dict(
        status="trained_and_validated; test split was not loaded or evaluated",
        best_checkpoint=str((save_dir / "best.pt").resolve()),
        best_checkpoint_sha256=sha256_file(save_dir / "best.pt"),
        best_epoch=history[int(np.argmax([
            item["validation"]["site_macro_mean_capacity_ratio_greedy"] for item in history
        ]))]["epoch"],
        best_validation_metric=best,
        selection_metric="equal-weight mean across SNR of eligible site-macro C_ICSM-AS/C_Capacity-Greedy",
        epochs_completed=len(history),
        seed=a.seed,
        python_version=__import__("platform").python_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        deterministic_algorithms=True,
        pretrained=bool(a.use_pretrained_weights),
        model="ICSM-AS",
        config={key: value for key, value in vars(a).items() if key != "func"},
        physical_status=train_ds.metadata.get("physics_status", "unspecified; requires review"),
        source_bank_sha256=train_ds.metadata.get("source_file_sha256"),
        training_labels_sha256=sha256_file(a.data_h5),
        parameters=count_parameters(model),
    )
    (save_dir / "training_status.json").write_text(
        json.dumps(outcome, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _training_progress(save_dir, phase="complete", epoch=len(history), total_epochs=a.epochs,
                       best_validation_metric=best)
    print(json.dumps(outcome, ensure_ascii=False), flush=True)


def _resolve_checkpoint(checkpoint, manifest_path=None):
    if checkpoint != "frozen_best":
        return Path(checkpoint).resolve()
    if manifest_path is None:
        raise ValueError("frozen_best requires --selection_manifest")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = Path(manifest["checkpoint_path"])
    return (target if target.is_absolute() else manifest_path.parent / target).resolve()


def command_freeze(a):
    checkpoint = Path(a.checkpoint).resolve()
    status_path = checkpoint.parent / "training_status.json"
    history_path = checkpoint.parent / "training_history.json"
    if not status_path.is_file() or not history_path.is_file():
        raise FileNotFoundError("Freeze requires training_status.json and training_history.json next to the checkpoint")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    payload = dict(
        protocol_version="icsm-as-frozen-selection-v1",
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        source_bank_sha256=status.get("source_bank_sha256"),
        training_labels_sha256=status.get("training_labels_sha256"),
        architecture="icsm_as",
        config=status.get("config"),
        seed=status.get("seed"),
        best_epoch=status.get("best_epoch"),
        best_validation_metric=status.get("best_validation_metric"),
        selection_metric=status.get("selection_metric"),
        selection_reason="highest validation-only site-macro Greedy capacity ratio among completed epochs",
        pretrained=status.get("pretrained"),
        physical_status=status.get("physical_status"),
        epochs_completed=status.get("epochs_completed"),
        training_history_sha256=sha256_file(history_path),
        train_validation_status_sha256=sha256_file(status_path),
        frozen_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        test_policy="one run only; test mode writes an exclusive test_opened.json before loading split 2",
    )
    destination = Path(a.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def add_runtime_options(parser):
    parser.add_argument("--data_h5", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("test", "validate"):
        from ..evaluation.imperfect_csi import main as evaluate_dual, parser as dual_parser
        args = dual_parser().parse_args(sys.argv[2:])
        args.split = 2 if sys.argv[1] == "test" else 1
        return evaluate_dual(args)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train on train split and select by val only")
    add_runtime_options(train)
    train.add_argument("--q2_dir", required=True)
    train.add_argument("--save_dir", required=True)
    train.add_argument("--resume", help="resume from a saved ICSM-AS checkpoint")
    train.add_argument("--architecture", choices=["icsm_as"], default="icsm_as")
    train.add_argument("--allow_proxy_physics", action="store_true")
    train.add_argument("--risk_beta", type=float, default=0.0)
    train.add_argument("--use_nll_ablation", action="store_true", help="Enable the separate log-variance/NLL loss ablation")
    train.add_argument("--ablations", default="")
    train.add_argument("--use_pretrained_weights", type=int, choices=[0, 1], default=1)
    train.add_argument("--freeze_backbone", type=int, choices=[0, 1], default=1)
    train.add_argument("--train_last_n_layers", type=int, default=0)
    train.add_argument("--alpha_hq", type=float, default=.5)
    train.add_argument("--lambda_relation", type=float, default=.2)
    train.add_argument("--lambda_marginal", type=float, default=1.)
    train.add_argument("--lambda_membership", type=float, default=1., help="set to 0 for no membership BCE")
    train.add_argument("--lambda_regression", type=float, default=1., help="set to 0 for no continuous marginal regression")
    train.add_argument("--lambda_rank", type=float, default=.1, help="set to 0 for no next-action ranking")
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=50)
    train.add_argument("--adapter_dim", type=int, default=64)
    train.add_argument("--graph_heads", type=int, default=4)
    train.add_argument("--relation_rank", type=int, default=64)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--grad_clip", type=float, default=1.0)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--one_snr_per_base", action="store_true",
                       help="sample one rotating SNR replica per base channel each training epoch")
    train.set_defaults(func=command_train)

    commands.add_parser("validate", help="paper dual-CSI validation; see validate --help")

    freeze = commands.add_parser("freeze", help="create the immutable validation selection manifest")
    freeze.add_argument("--checkpoint", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(func=command_freeze)

    commands.add_parser("test", help="paper dual-CSI test; see test --help")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

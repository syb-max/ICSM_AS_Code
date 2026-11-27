"""Seed control and run provenance. Set environment before importing torch."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random, hashlib, json, platform
from pathlib import Path
import numpy as np
import torch

def set_reproducible(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write_run_manifest(path, *, seed, config_path=None, data_path=None, checkpoint_path=None, extra=None):
    payload={"seed":int(seed),"python":platform.python_version(),"platform":platform.platform(),
             "torch":torch.__version__,"cuda":torch.version.cuda,
             "deterministic_algorithms":True}
    for key,p in (("config_sha256",config_path),("data_sha256",data_path),("checkpoint_sha256",checkpoint_path)):
        if p is not None and Path(p).is_file(): payload[key]=sha256(p)
    if extra: payload.update(extra)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')

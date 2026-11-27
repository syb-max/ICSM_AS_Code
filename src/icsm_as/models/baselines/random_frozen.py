from __future__ import annotations
import torch
from torch import nn
from ...core.icsm_as_model import ICSMAS

class RandomFrozenQwen2(ICSMAS):
    """Random-frozen: same ICSM-AS adapter with a randomly initialized frozen Qwen2."""
    model_name="Random-frozen"
    def __init__(self,qwen_dir,n_tx=128,n_sel=16,backbone_seed=918273,**kwargs):
        # Fix the random backbone realization without consuming the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(backbone_seed))
            super().__init__(q2_dir=qwen_dir,n_tx=n_tx,n_sel=n_sel,
                use_pretrained_weights=False,freeze_backbone=True,**kwargs)

from __future__ import annotations
import torch
from torch import nn
from ...core.model import Q2TxSelector, greedy_select_with_relation

class ICSMASStatic(nn.Module):
    """Score-only: static utility/relation prediction from the relation matrix."""
    model_name="ICSM-AS (static)"
    def __init__(self,qwen_dir,n_tx=128,n_sel=16,**kwargs):
        super().__init__(); self.n_tx=n_tx; self.n_sel=n_sel
        self.model=Q2TxSelector(q2_dir=qwen_dir,n_tx=n_tx,rel_dim=n_tx,freeze_backbone=True,
            relation_rank=64,use_pretrained_weights=True)
    def forward(self,batch):
        if "X_rel" not in batch: raise KeyError("Score-only requires X_rel [B,Nt,Nt]")
        return self.model(batch["X_rel"].float())
    @torch.no_grad()
    def select(self,outputs,k=None):
        return greedy_select_with_relation(outputs["utility_logits"],outputs["relation_logits"],
            self.n_sel if k is None else k,.1)
# Backward-compatible import name used by earlier experiment commands.
ScoreOnlyQ2 = ICSMASStatic

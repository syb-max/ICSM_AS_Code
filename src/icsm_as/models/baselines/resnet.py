from __future__ import annotations
import torch
from torch import nn
from .common import port_features, ScoreHead, ResidualMLPBlock

class ResNetUtility(nn.Module):
    """ResNet-Utility: residual per-port feature regressor."""
    model_name="ResNet-Utility"
    def __init__(self,n_tx=128,hidden=256,depth=3,**kwargs):
        super().__init__(); self.n_tx=n_tx; self.stem=nn.Sequential(nn.LazyLinear(hidden),nn.GELU()); self.blocks=nn.Sequential(*[ResidualMLPBlock(hidden) for _ in range(depth)]); self.head=ScoreHead(hidden,hidden)
    def forward(self,batch):
        z=self.blocks(self.stem(port_features(batch))); return {"utility_logits":self.head(z)}
    @torch.no_grad()
    def select(self,outputs,k=16): return outputs["utility_logits"].topk(k,dim=1).indices

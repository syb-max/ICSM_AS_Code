from __future__ import annotations
import torch
from torch import nn
from .common import port_features, ScoreHead

class MLPUtility(nn.Module):
    """MLP-Utility: independent per-port utility regression."""
    model_name="MLP-Utility"
    def __init__(self,n_tx=128,hidden=256,**kwargs):
        super().__init__(); self.n_tx=n_tx; self.encoder=nn.Sequential(nn.LazyLinear(hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,hidden),nn.GELU()); self.head=ScoreHead(hidden,hidden)
    def forward(self,batch):
        z=self.encoder(port_features(batch)); return {"utility_logits":self.head(z)}
    @torch.no_grad()
    def select(self,outputs,k=16): return outputs["utility_logits"].topk(k,dim=1).indices

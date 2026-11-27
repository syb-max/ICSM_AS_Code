from __future__ import annotations
import torch
from torch import nn
from .common import port_features

class MLPMultiLabel(nn.Module):
    """MLP-MultiLabel: predict all port-membership logits jointly."""
    model_name="MLP-MultiLabel"
    def __init__(self,n_tx=128,hidden=256,**kwargs):
        super().__init__(); self.n_tx=n_tx; self.local=nn.Sequential(nn.LazyLinear(hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.head=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,1))
    def forward(self,batch):
        z=self.local(port_features(batch)); ctx=z.mean(1,keepdim=True).expand_as(z)
        return {"utility_logits":self.head(torch.cat((z,ctx),-1)).squeeze(-1)}
    @torch.no_grad()
    def select(self,outputs,k=16): return outputs["utility_logits"].topk(k,dim=1).indices

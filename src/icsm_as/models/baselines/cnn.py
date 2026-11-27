from __future__ import annotations
import torch
from torch import nn
from .common import port_features

class CNNUtility(nn.Module):
    """CNN-Utility: 1-D convolution over the ordered transmit-port axis.

    This is an optional additional baseline; it is not a method reported in the paper tables.
    """
    model_name="CNN-Utility (optional)"
    def __init__(self,n_tx=128,hidden=128,**kwargs):
        super().__init__(); self.n_tx=n_tx
        self.net=nn.Sequential(nn.LazyConv1d(hidden,kernel_size=5,padding=2),nn.GELU(),
            nn.Conv1d(hidden,hidden,kernel_size=3,padding=1),nn.GELU(),
            nn.Conv1d(hidden,1,kernel_size=1))
    def forward(self,batch):
        x=port_features(batch).transpose(1,2); return {"utility_logits":self.net(x).squeeze(1)}
    @torch.no_grad()
    def select(self,outputs,k=16): return outputs["utility_logits"].topk(k,dim=1).indices

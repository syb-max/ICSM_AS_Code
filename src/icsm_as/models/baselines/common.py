"""Shared feature construction for learning-based comparison models."""
from __future__ import annotations
import torch
from torch import nn

def port_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return [B,Nt,D] per-port tokens from frequency-resolved complex Gram rows."""
    real=batch["G_tx_real"].float(); imag=batch["G_tx_imag"].float()
    if real.ndim != 4 or real.shape != imag.shape or real.shape[-1] != real.shape[-2]:
        raise ValueError("Gram inputs must be matching [B,F,Nt,Nt] tensors")
    b,f,n,_=real.shape
    power=real.diagonal(dim1=-2,dim2=-1).clamp_min(0)
    scale=power.mean(-1,keepdim=True).clamp_min(1e-12)
    re=(real/scale.unsqueeze(-1)).permute(0,2,1,3).reshape(b,n,f*n)
    im=(imag/scale.unsqueeze(-1)).permute(0,2,1,3).reshape(b,n,f*n)
    p=torch.log1p(power.mean(1)).unsqueeze(-1)
    snr=batch["effective_snr_db"].float().reshape(b,1,1).expand(-1,n,1)/20.0
    return torch.cat((re,im,p,snr),dim=-1)

class ScoreHead(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(hidden,hidden//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden//2,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class ResidualMLPBlock(nn.Module):
    def __init__(self,width,dropout=.1):
        super().__init__(); self.norm=nn.LayerNorm(width)
        self.net=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,width))
    def forward(self,x): return x+self.net(self.norm(x))

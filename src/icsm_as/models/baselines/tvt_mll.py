from __future__ import annotations
import torch
from torch import nn

class TVTMLLAdapted(nn.Module):
    """Direct-Gram-MLL task adaptation; not an exact reproduction of TVT 2021."""
    model_name="TVT-MLL (adapted)"
    def __init__(self,n_tx=128,n_sel=16,**kwargs):
        super().__init__(); self.n_tx=n_tx; self.n_sel=n_sel
        self.backbone=nn.Sequential(nn.Linear(2*n_tx+2,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU())
        self.head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,batch):
        real=batch["G_tx_real"].float(); imag=batch["G_tx_imag"].float()
        g=torch.complex(real,imag); power=g.diagonal(dim1=-2,dim2=-1).real.clamp_min(0)
        norm=(power.unsqueeze(-1)*power.unsqueeze(-2)).sqrt().clamp_min(1e-12)
        corr=g/norm
        snr=(batch["effective_snr_db"].float()/20).reshape(-1,1,1,1).expand_as(power.unsqueeze(-1))
        features=torch.cat((corr.real,corr.imag,torch.log1p(power).unsqueeze(-1),snr),-1)
        z=self.backbone(features).mean(1); pooled=z.mean(1,keepdim=True).expand_as(z)
        return {"utility_logits":self.head(torch.cat((z,pooled),-1)).squeeze(-1)}
    @torch.no_grad()
    def select(self,outputs,k=None): return outputs["utility_logits"].topk(self.n_sel if k is None else k,dim=1).indices

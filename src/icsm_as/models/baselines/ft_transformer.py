from __future__ import annotations
import torch
from torch import nn
from .common import port_features

class FTTransformer(nn.Module):
    """FT-Transformer-style token encoder over candidate-port features."""
    model_name="FT-Transformer"
    def __init__(self,n_tx=128,d_model=128,nhead=4,layers=3,**kwargs):
        super().__init__(); self.n_tx=n_tx; self.proj=nn.LazyLinear(d_model)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=4*d_model,dropout=.1,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers); self.head=nn.Linear(d_model,1)
    def forward(self,batch): return {"utility_logits":self.head(self.encoder(self.proj(port_features(batch)))).squeeze(-1)}
    @torch.no_grad()
    def select(self,outputs,k=16): return outputs["utility_logits"].topk(k,dim=1).indices

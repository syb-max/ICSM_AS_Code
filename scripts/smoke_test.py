#!/usr/bin/env python3
"""Forward-shape smoke test for named neural baselines; uses synthetic tensors only."""
import torch
from icsm_as.models.baselines.registry import build_baseline

def main():
 torch.manual_seed(42); b,f,n=2,4,16
 x=torch.randn(b,f,n,n); g=(x@x.transpose(-1,-2)).float()
 batch={"G_tx_real":g,"G_tx_imag":torch.zeros_like(g),"effective_snr_db":torch.tensor([0.,10.])}
 for name in ["MLP-Utility","ResNet-Utility","CNN-Utility","FT-Transformer","MLP-MultiLabel","TVT-MLL (adapted)"]:
  model=build_baseline(name,n_tx=n,n_sel=4); out=model(batch); idx=model.select(out,k=4)
  assert idx.shape==(b,4),(name,idx.shape)
  assert torch.unique(idx,dim=1).shape[1]==4
  print(name,tuple(out['utility_logits'].shape),tuple(idx.shape))
 from icsm_as.models.baselines.small_tf import ScratchTransformer
 scratch=ScratchTransformer(32); hidden=scratch(inputs_embeds=torch.randn(b,n,32)).last_hidden_state
 assert hidden.shape==(b,n,32)
 print("Small-TF scratch backbone",tuple(hidden.shape))
 print("SMOKE TEST PASS (synthetic input only; not an experiment result)")
if __name__=='__main__':main()

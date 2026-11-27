import torch
from icsm_as.core.icsm_as_model import ICSMASAdapter

def test_parallel_pcr_csf_ecg_shapes():
    torch.manual_seed(11)
    b,f,n,hidden,width=2,3,10,32,16
    a=torch.randn(b,f,n,n); g=a@a.transpose(-1,-2)
    batch={
        "G_tx_real":g,
        "G_tx_imag":torch.zeros_like(g),
        "effective_snr_db":torch.tensor([0.0,10.0]),
        "positions_wavelength":torch.randn(n,3),
    }
    adapter=ICSMASAdapter(n,hidden,width=width,heads=4,dropout=0.0,require_physics=False)
    tokens,edges,channel_corr=adapter.encode(batch)
    assert tokens.shape==(b,n,hidden)
    assert edges.shape==(b,n,n,7)
    assert channel_corr.shape==(b,n,n)
    cache=adapter.sfi.cache(torch.randn(b,n,hidden),edges)
    selected=torch.zeros(b,n,dtype=torch.bool)
    mean,logvar=adapter.predict(cache,selected,4)
    assert mean.shape==(b,n) and logvar.shape==(b,n)

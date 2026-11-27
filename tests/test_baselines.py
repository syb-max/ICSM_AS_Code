import torch
from icsm_as.models.baselines.registry import build_baseline

def make_batch():
    torch.manual_seed(7); b,f,n=2,3,12
    a=torch.randn(b,f,n,n); g=a@a.transpose(-1,-2)
    return {"G_tx_real":g,"G_tx_imag":torch.zeros_like(g),"effective_snr_db":torch.tensor([0.,5.])}

def test_neural_baselines_return_unique_fixed_k():
    batch=make_batch()
    for name in ["MLP-Utility","ResNet-Utility","CNN-Utility","FT-Transformer","MLP-MultiLabel","TVT-MLL (adapted)"]:
        model=build_baseline(name,n_tx=12,n_sel=4); out=model(batch); idx=model.select(out,k=4)
        assert idx.shape==(2,4)
        assert all(len(set(row.tolist()))==4 for row in idx)

def test_small_transformer_backbone_shape():
    from icsm_as.models.baselines.small_tf import ScratchTransformer
    model=ScratchTransformer(32)
    hidden=model(inputs_embeds=torch.randn(2,12,32)).last_hidden_state
    assert hidden.shape==(2,12,32)


def test_registry_contains_named_paper_comparators():
    from icsm_as.models.baselines.registry import MODEL_REGISTRY, CLASSICAL_REGISTRY
    for name in ["ICSM-AS", "Score-only comparator", "Score-only", "TVT-MLL (adapted)", "Random-frozen", "SCMG-variant", "Small-TF"]:
        if name == "ICSM-AS":
            continue  # constructed with Qwen2 config/weights by the main training entry point
        assert name in MODEL_REGISTRY
    for name in ["Power-Top16", "Correlation-Greedy", "Capacity-Greedy"]:
        assert name in CLASSICAL_REGISTRY


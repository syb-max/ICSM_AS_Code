"""Model factory. Implementations are kept in individual Python files."""
from __future__ import annotations

from .mlp import MLPUtility
from .resnet import ResNetUtility
from .cnn import CNNUtility
from .ft_transformer import FTTransformer
from .mlp_multilabel import MLPMultiLabel
from .tvt_mll import TVTMLLAdapted
from .icsm_as_static import ICSMASStatic
from .scmg_variant import SCMGVariantModel
from .small_tf import SmallTFComparator
from .random_frozen import RandomFrozenQwen2
from .power_top16 import power_topk
from .correlation_greedy import correlation_greedy
from .capacity_greedy import capacity_greedy
from ..icsm_as import ICSMAS

MODEL_REGISTRY = {
    "ICSM-AS": ICSMAS,
    "ICSM-AS (static)": ICSMASStatic,
    "Score-only comparator": ICSMASStatic,
    "Score-only": ICSMASStatic,
    "TVT-MLL": TVTMLLAdapted,
    "TVT-MLL (adapted)": TVTMLLAdapted,
    "Random-frozen": RandomFrozenQwen2,
    "SCMG-variant": SCMGVariantModel,
    "Small-TF": SmallTFComparator,
    "MLP-Utility": MLPUtility,
    "ResNet-Utility": ResNetUtility,
    "CNN-Utility": CNNUtility,
    "FT-Transformer": FTTransformer,
    "MLP-MultiLabel": MLPMultiLabel,
}

CLASSICAL_REGISTRY = {
    "Power-Top16": power_topk,
    "Correlation-Greedy": correlation_greedy,
    "Capacity-Greedy": capacity_greedy,
}

def build_baseline(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)

def build_classical_baseline(name: str, **kwargs):
    if name not in CLASSICAL_REGISTRY:
        raise KeyError(f"Unknown classical model: {name}. Available: {sorted(CLASSICAL_REGISTRY)}")
    return CLASSICAL_REGISTRY[name](**kwargs)

from __future__ import annotations
from ...core.scmg_variant import SCMGVariant as _SCMGVariant

class SCMGVariantModel(_SCMGVariant):
    model_name = "SCMG-variant"

    def __init__(self, qwen_dir, n_tx=128, n_sel=16, **kwargs):
        super().__init__(q2_dir=qwen_dir, n_tx=n_tx, n_sel=n_sel, **kwargs)

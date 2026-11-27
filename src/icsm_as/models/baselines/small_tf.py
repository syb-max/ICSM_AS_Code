from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ...core.icsm_as_model import ICSMAS

class ScratchTransformer(nn.Module):
    """Two-layer bidirectional Transformer backbone used by the Small-TF control."""
    def __init__(self,hidden):
        super().__init__(); self.config=SimpleNamespace(hidden_size=hidden,num_hidden_layers=2)
        self.input=nn.Linear(hidden,256)
        layer=nn.TransformerEncoderLayer(256,4,512,.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Linear(256,hidden)
    def forward(self,inputs_embeds,**kwargs):
        return SimpleNamespace(last_hidden_state=self.output(self.encoder(self.input(inputs_embeds))))

class SmallTFComparator(ICSMAS):
    """Small-TF: same PCR/CSF/ECG/SFI/DMH path with a two-layer scratch Transformer."""
    model_name="Small-TF"
    def __init__(self,qwen_dir,n_tx=128,n_sel=16,allow_proxy_physics=False,**kwargs):
        # Read architecture width from the supplied Qwen config, but instantiate only
        # a one-layer temporary config before replacing it with the scratch backbone.
        cfg=json.loads((Path(qwen_dir)/"config.json").read_text(encoding="utf-8"))
        hidden=int(cfg.get("hidden_size",896))
        scratch_cfg=dict(model_type="qwen2",vocab_size=128,hidden_size=hidden,intermediate_size=512,
            num_hidden_layers=1,num_attention_heads=int(cfg.get("num_attention_heads",14)),
            num_key_value_heads=int(cfg.get("num_key_value_heads",2)),
            max_position_embeddings=256,rms_norm_eps=float(cfg.get("rms_norm_eps",1e-6)),
            rope_theta=float(cfg.get("rope_theta",10000.0)),attention_dropout=0.0)
        super().__init__(q2_dir=qwen_dir,n_tx=n_tx,n_sel=n_sel,use_pretrained_weights=False,
            freeze_backbone=True,backbone_config=scratch_cfg,allow_proxy_physics=allow_proxy_physics,**kwargs)
        self.backbone=ScratchTransformer(self.hidden_size)
        self.model_options.update(backbone_kind="scratch_transformer",use_pretrained_weights=False)

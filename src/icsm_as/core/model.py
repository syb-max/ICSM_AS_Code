"""
Q2-based regression-driven transmit antenna subset selection model.

Paper-aligned task:
    fixed-receiver 128-to-16 single-side Tx antenna selection.

Input:
    x_rel: [B, 128, 128]
        Real-valued transmit-side relation matrix. Each row is a relation-aware
        Tx antenna token.

Outputs:
    utility_logits: [B, 128]
        Antenna membership logits (BCE targets are not measured capacity gains).
    relation_logits: [B, 128, 128]
        Pairwise structural-relation / redundancy logits.

The model does NOT perform language generation. Q2/Qwen2 is used only as a
candidate-set contextual encoder over relation-aware Tx tokens.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoConfig
except Exception as exc:  # pragma: no cover
    AutoModel = None
    AutoConfig = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class Q2TxSelector(nn.Module):
    """Causal Qwen contextual encoder + membership/structural relation heads."""

    def __init__(
        self,
        q2_dir: str,
        n_tx: int = 128,
        rel_dim: int = 128,
        freeze_backbone: bool = True,
        train_last_n_layers: int = 0,
        relation_rank: int = 64,
        dropout: float = 0.1,
        use_pretrained_weights: bool = True,
        backbone_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        if AutoModel is None or AutoConfig is None:
            raise ImportError(
                "transformers is required for Q2TxSelector. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR}"
            )

        self.n_tx = int(n_tx)
        self.rel_dim = int(rel_dim)
        self.relation_rank = int(relation_rank)

        if use_pretrained_weights:
            self.backbone = AutoModel.from_pretrained(q2_dir, trust_remote_code=False)
        else:
            if backbone_config is None:
                config = AutoConfig.from_pretrained(q2_dir, trust_remote_code=False)
            else:
                options = dict(backbone_config)
                model_type = options.pop("model_type")
                config = AutoConfig.for_model(model_type, **options)
            self.backbone = AutoModel.from_config(config)

        # Local Qwen files may load as bf16 while newly constructed adapters
        # default to fp32. Keep a common fp32 base; training may use autocast.
        self.backbone.float()

        hidden_size = int(self.backbone.config.hidden_size)
        self.hidden_size = hidden_size

        # Each row of x_rel is a 128-dimensional relation-aware Tx token.
        self.input_proj = nn.Sequential(
            nn.Linear(self.rel_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.utility_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # Low-rank pairwise relation head. This is lighter than a full pair MLP.
        self.relation_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.relation_rank),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Optional partial fine-tuning of the last transformer blocks.
        if train_last_n_layers > 0:
            layers = None
            if hasattr(self.backbone, "layers"):
                layers = self.backbone.layers
            elif hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
                layers = self.backbone.model.layers
            if layers is not None:
                for layer in layers[-train_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(self, x_rel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_rel: [B, 128, 128]

        Returns:
            dict with utility_logits and relation_logits.
        """
        if x_rel.ndim != 3:
            raise ValueError(f"x_rel must be [B, N_tx, N_tx], got shape {tuple(x_rel.shape)}")
        if x_rel.shape[1] != self.n_tx or x_rel.shape[2] != self.rel_dim:
            raise ValueError(
                f"Expected x_rel shape [B,{self.n_tx},{self.rel_dim}], got {tuple(x_rel.shape)}"
            )

        inputs_embeds = self.input_proj(x_rel)  # [B,128,H]
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state  # [B,128,H]

        utility_logits = self.utility_head(hidden).squeeze(-1)  # [B,128]

        z = self.relation_proj(hidden)  # [B,128,R]
        relation_logits = torch.matmul(z, z.transpose(1, 2)) / math.sqrt(max(self.relation_rank, 1))
        relation_logits = 0.5 * (relation_logits + relation_logits.transpose(1, 2))

        # Avoid self-relation dominating redundancy learning/selection.
        eye = torch.eye(self.n_tx, device=relation_logits.device, dtype=torch.bool).unsqueeze(0)
        relation_logits = relation_logits.masked_fill(eye, 0.0)

        return {
            "utility_logits": utility_logits,
            "relation_logits": relation_logits,
        }


@torch.no_grad()
def greedy_select_with_relation(
    utility_logits: torch.Tensor,
    relation_logits: Optional[torch.Tensor] = None,
    k: int = 16,
    lambda_rel: float = 0.10,
) -> torch.Tensor:
    """
    Greedy explicit fixed-cardinality selector.

    score_i = utility_i - lambda_rel * average_relation(i, selected_set)

    Args:
        utility_logits: [B,128]
        relation_logits: [B,128,128], optional. If None, top-k utility is used.
        k: subset size.
        lambda_rel: redundancy penalty weight.

    Returns:
        selected indices: [B,k]
    """
    bsz, n_tx = utility_logits.shape
    if not 1 <= k <= n_tx:
        raise ValueError("Require 1 <= k <= n_tx")
    utility = torch.sigmoid(utility_logits)

    if relation_logits is None or lambda_rel <= 0:
        return torch.topk(utility, k=k, dim=1).indices

    relation = torch.sigmoid(relation_logits)
    if relation.shape != (bsz, n_tx, n_tx):
        raise ValueError("relation_logits shape does not match utility_logits")
    selected = []
    available_mask = torch.ones(bsz, n_tx, dtype=torch.bool, device=utility.device)
    relation_sum = torch.zeros_like(utility)

    for step in range(k):
        score = utility.clone()
        if step:
            score = score - lambda_rel * relation_sum / step

        score = score.masked_fill(~available_mask, -1e9)
        next_idx = score.argmax(dim=1)
        selected.append(next_idx)
        available_mask[torch.arange(bsz, device=utility.device), next_idx] = False
        # Accumulating one column per step reduces selection O(N K^2) to O(N K).
        relation_sum += relation.gather(2, next_idx[:, None, None].expand(-1, n_tx, 1)).squeeze(2)

    return torch.stack(selected, dim=1)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

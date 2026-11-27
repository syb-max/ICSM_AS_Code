"""SCMG-variant comparison model: FCP + EPG + Qwen2 + SCMG head.

The model uses frequency-resolved complex Gram features, an optional
physical-edge graph adapter, one causal Qwen2 pass, and subset-conditioned
marginal-gain prediction.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import Q2TxSelector, greedy_select_with_relation


class FCPTokenizer(nn.Module):
    """FCP: encode frequency-resolved complex Gram, power, scale and SNR."""
    def __init__(self, n_tx, width, mode="complex", dropout=0.1):
        super().__init__()
        if mode not in ("complex", "magnitude"):
            raise ValueError("token_mode must be complex or magnitude")
        self.n_tx, self.mode = n_tx, mode
        self.row_encoder = nn.Sequential(nn.Linear(2*n_tx+4, width), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(width, width))
        self.frequency_score = nn.Linear(width, 1)

    def features(self, batch):
        real, imag = batch["G_tx_real"].float(), batch["G_tx_imag"].float()
        if real.ndim != 4 or real.shape[-2:] != (self.n_tx, self.n_tx) or imag.shape != real.shape:
            raise ValueError("Gram must have shape [B,F,N,N]")
        b, nf, n, _ = real.shape
        power = real.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        scale = power.mean(-1, keepdim=True).clamp_min(1e-12)
        # Keeping the scale explicitly avoids erasing absolute channel strength.
        nr, ni = real / scale.unsqueeze(-1), imag / scale.unsqueeze(-1)
        denominator = (power.unsqueeze(-1)*power.unsqueeze(-2)).clamp_min(1e-24).sqrt()
        cr, ci = real / denominator, imag / denominator
        ca = torch.sqrt(cr.square()+ci.square()).clamp_max(1)
        if self.mode == "magnitude":
            # Input ablation: equal-size network, mean magnitude correlation only.
            # Also remove phase from graph edges to prevent an information bypass.
            nr = ca.mean(1, keepdim=True).expand(-1, nf, -1, -1)
            ni = torch.zeros_like(nr)
            cr, ci = nr, ni
        snr = batch["effective_snr_db"].float().view(b, 1, 1, 1).expand(-1, nf, n, -1)/20
        fc = batch.get("frequency_coord")
        if fc is None:
            fc = torch.linspace(-1, 1, nf, device=real.device).expand(b, -1)
        fc = fc.float().reshape(b, nf, 1, 1).expand(-1, -1, n, -1)
        log_power = torch.log1p(power).unsqueeze(-1)
        log_scale = torch.log1p(scale).unsqueeze(-1).expand(-1, -1, n, -1)
        if self.mode == "magnitude":
            log_power, log_scale, snr, fc = (torch.zeros_like(x) for x in (log_power, log_scale, snr, fc))
        rows = torch.cat((nr, ni, log_power, log_scale, snr, fc), dim=-1)
        coupling = batch.get("coupling_edge", torch.zeros_like(real[:, 0]))
        positions = batch.get("positions_wavelength")
        distance = torch.zeros_like(coupling) if positions is None else torch.cdist(positions.float(), positions.float())
        # log1p distance retains wavelength scale; no array-wise distance renormalization.
        edges = torch.stack((cr.mean(1), ci.mean(1), ca.mean(1),
                             coupling.float(), torch.log1p(distance)), dim=-1)
        return rows, edges, ca.mean(1)

    def forward(self, batch):
        rows, edges, direct_relation = self.features(batch)
        per_frequency = self.row_encoder(rows)
        weights = torch.softmax(self.frequency_score(per_frequency), dim=1)
        return (weights*per_frequency).sum(1), edges, direct_relation


class EPGAdapter(nn.Module):
    """EPG: bidirectional pre-encoder biased by channel and array prior edges."""
    def __init__(self, width, heads=4, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("adapter_dim must be divisible by graph_heads")
        self.heads, self.width = heads, width
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.edge_bias = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, heads))
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, edges):
        b, n, p = tokens.shape
        q, k, v = self.qkv(self.norm(tokens)).view(b, n, 3, self.heads, p//self.heads).unbind(2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        bias = self.edge_bias(edges).permute(0, 3, 1, 2)
        attention = torch.softmax(q@k.transpose(-1, -2)/math.sqrt(p//self.heads)+bias, dim=-1)
        mixed = (self.dropout(attention)@v).transpose(1, 2).reshape(b, n, p)
        return tokens + self.dropout(self.out(mixed))


class SCMGHead(nn.Module):
    """SCMG: predict bit/s/Hz increments conditioned on the selected set.

No exact logdet is evaluated by this learned head. It has no approximation
guarantee; actual capacities must be computed separately on the physical Gram.
"""
    def __init__(self, hidden, width):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU())
        self.context = nn.Linear(width, width, bias=False)
        self.step = nn.Linear(1, width, bias=False)
        self.score = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, embedding, selected_mask, final_k):
        selected = selected_mask.to(embedding.dtype)
        count = selected.sum(1, keepdim=True)
        pooled = (embedding*selected.unsqueeze(-1)).sum(1)/count.clamp_min(1)
        context = self.context(pooled) + self.step(count/final_k)
        return F.softplus(self.score(embedding+context.unsqueeze(1)).squeeze(-1))


class SCMGVariant(Q2TxSelector):
    """FCP -> EPG -> one causal Qwen2 pass -> SCMG sequential selection."""

    method_name = "SCMG-variant"
    def __init__(self, q2_dir, n_tx=128, n_sel=16, adapter_dim=64, graph_heads=4,
                 relation_rank=64, freeze_backbone=True, use_pretrained_weights=True,
                 train_last_n_layers=0, token_mode="complex", use_graph=True,
                 decoder="marginal", dropout=0.1, backbone_config=None):
        super().__init__(q2_dir, n_tx, n_tx, freeze_backbone, train_last_n_layers,
                         relation_rank, dropout, use_pretrained_weights, backbone_config)
        if not 1 <= n_sel <= n_tx:
            raise ValueError("Require 1 <= n_sel <= n_tx")
        if decoder not in ("marginal", "relation", "membership", "direct_gram"):
            raise ValueError("Unknown decoder")
        del self.input_proj
        self.n_sel, self.decoder = n_sel, decoder
        self.tokenizer = FCPTokenizer(n_tx, adapter_dim, token_mode, dropout)
        self.graph = EPGAdapter(adapter_dim, graph_heads, dropout) if use_graph else None
        self.to_backbone = nn.Linear(adapter_dim, self.hidden_size)
        self.marginal_head = SCMGHead(self.hidden_size, adapter_dim) if decoder == "marginal" else None
        self.model_options = dict(q2_dir=str(q2_dir), n_tx=n_tx, n_sel=n_sel,
            adapter_dim=adapter_dim, graph_heads=graph_heads, relation_rank=relation_rank,
            freeze_backbone=freeze_backbone, use_pretrained_weights=use_pretrained_weights,
            train_last_n_layers=train_last_n_layers, token_mode=token_mode,
            use_graph=use_graph, decoder=decoder, dropout=dropout)

    def forward(self, batch):
        tokens, edges, direct_relation = self.tokenizer(batch)
        if self.graph is not None:
            tokens = self.graph(tokens, edges)
        embedded = self.to_backbone(tokens)
        # This is a padding mask. Qwen2 retains its causal attention and RoPE.
        # Do not use torch.no_grad(): gradients must cross the frozen backbone
        # to train the tokenizer and graph adapter.
        hidden = self.backbone(inputs_embeds=embedded,
            attention_mask=torch.ones(embedded.shape[:2], device=embedded.device, dtype=torch.long),
            use_cache=False, return_dict=True).last_hidden_state
        z = self.relation_proj(hidden)
        relation = z@z.transpose(1, 2)/math.sqrt(self.relation_rank)
        relation = relation.masked_fill(torch.eye(self.n_tx, device=relation.device, dtype=torch.bool)[None], 0)
        out = dict(hidden=hidden, utility_logits=self.utility_head(hidden).squeeze(-1),
                   relation_logits=relation, direct_relation=direct_relation)
        if self.marginal_head is not None:
            out["marginal_embedding"] = self.marginal_head.project(hidden)
        return out

    @torch.no_grad()
    def select(self, outputs, k=None, lambda_rel=0.1):
        k = self.n_sel if k is None else k
        if not 1 <= k <= self.n_tx:
            raise ValueError("Invalid selection size")
        if self.decoder == "marginal":
            if k != self.n_sel:
                raise ValueError("Marginal head is trained with fixed final n_sel; use the same k")
            embedding = outputs["marginal_embedding"]
            mask = torch.zeros(embedding.shape[:2], device=embedding.device, dtype=torch.bool)
            result = []
            for _ in range(k):
                score = self.marginal_head(embedding, mask, k).masked_fill(mask, -torch.inf)
                nxt = score.argmax(1)
                mask.scatter_(1, nxt[:, None], True)
                result.append(nxt)
            return torch.stack(result, 1)
        relation = outputs["relation_logits"]
        if self.decoder == "membership":
            relation = None
        elif self.decoder == "direct_gram":
            relation = torch.logit(outputs["direct_relation"].clamp(1e-6, 1-1e-6))
        return greedy_select_with_relation(outputs["utility_logits"], relation, k, lambda_rel)

    def marginal_loss(self, outputs, teacher_order, teacher_marginals):
        if self.marginal_head is None:
            return outputs["utility_logits"].sum()*0
        embedding = outputs["marginal_embedding"]
        mask = torch.zeros(embedding.shape[:2], device=embedding.device, dtype=torch.bool)
        total = embedding.new_zeros(())
        for step in range(self.n_sel):
            prediction = self.marginal_head(embedding, mask, self.n_sel)
            valid = ~mask
            # Offline exact gains at greedy prefixes, in bit/s/Hz.
            total = total + F.smooth_l1_loss(prediction[valid], teacher_marginals[:, step][valid])
            mask = mask.scatter(1, teacher_order[:, step, None], True)
        return total/self.n_sel


# Public comparison class: SCMGVariant.

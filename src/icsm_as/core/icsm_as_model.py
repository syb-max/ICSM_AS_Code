"""ICSM-AS model: parallel PCR/CSF branches, ECG, one-pass Qwen2, SFI and DMH.

The supplied Gram matrices must follow the fixed-port-load convention used to
construct the effective channel. Do not apply the same coupling transformation twice.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .physics import terminated_current_transfer
from .scmg_variant import SCMGVariant


def induced_current_kernel(transfer):
    """Normalized overlap of induced-only current responses, averaged over F.

    U=T-diag(diag(T)); K=U^H U. This is NOT radiated power or exact capacity.
    Dense O(B F N^3); cache for a fixed array/termination outside inference.
    """
    induced = transfer - torch.diag_embed(transfer.diagonal(dim1=-2, dim2=-1))
    kernel = induced.mH @ induced
    diagonal = kernel.diagonal(dim1=-2, dim2=-1).real.clamp_min(0)
    norm = (diagonal.unsqueeze(-1)*diagonal.unsqueeze(-2)).clamp_min(1e-24).sqrt()
    return (kernel/norm).mean(dim=-3)


class PortCurrentResponse(nn.Module):
    """PCR: encode full-array source-to-current columns, including idle ports."""
    def __init__(self, n_tx, width, require_physics=True):
        super().__init__()
        self.n_tx, self.require_physics = n_tx, require_physics
        self.encoder = nn.Sequential(nn.Linear(2*n_tx+4, width), nn.GELU(), nn.Linear(width,width))

    def forward(self, batch):
        real = batch["G_tx_real"]
        b, nf, n, _ = real.shape
        if "T_tx_real" in batch and "T_tx_imag" in batch:
            transfer = torch.complex(batch["T_tx_real"].float(), batch["T_tx_imag"].float())
        elif "Z_tx_real" in batch and "Z_tx_imag" in batch:
            if "source_impedance" not in batch or "reference_impedance" not in batch:
                raise ValueError("Direct Z input needs explicit source and reference impedances")
            transfer = terminated_current_transfer(torch.complex(batch["Z_tx_real"].float(),
                batch["Z_tx_imag"].float()), batch["source_impedance"], batch["reference_impedance"])
        else:
            if self.require_physics:
                raise ValueError("ICSM-AS requires compatible complex T or Z. dB S parameters are insufficient. "
                                 "Use allow_proxy_physics only for explicitly labelled channel-only diagnostics.")
            transfer = torch.eye(n,device=real.device,dtype=torch.complex64).expand(b,nf,n,n)
        if transfer.ndim == 3:
            transfer = transfer.unsqueeze(0).expand(b,-1,-1,-1)
        if transfer.shape != real.shape:
            raise ValueError("T must have [B,F,N,N] or shared [F,N,N] matching effective Gram")
        if not torch.isfinite(transfer).all():
            raise ValueError("Nonfinite current transfer")
        responses = transfer.transpose(-1,-2)  # candidate source i -> current on every physical port
        energy = responses.abs().square().sum(-1)
        self_energy = transfer.diagonal(dim1=-2,dim2=-1).abs().square()
        induced_fraction = (energy-self_energy).clamp_min(0)/energy.clamp_min(1e-12)
        norm = energy.clamp_min(1e-12).sqrt().unsqueeze(-1)
        accepted = batch.get("accepted_power_per_source", torch.zeros_like(energy)).float()
        available = torch.full_like(energy, float("accepted_power_per_source" in batch))
        features = torch.cat((responses.real/norm,responses.imag/norm,
            torch.stack((torch.log1p(energy),induced_fraction,torch.log1p(accepted.clamp_min(0)),available),-1)), -1)
        if "current_kernel_real" in batch and "current_kernel_imag" in batch:
            kernel = torch.complex(batch["current_kernel_real"].float(),batch["current_kernel_imag"].float())
            if kernel.ndim == 2:
                kernel = kernel[None].expand(b,-1,-1)
        else:
            kernel = induced_current_kernel(transfer)
        if kernel.shape != (b,n,n) or not torch.isfinite(kernel).all():
            raise ValueError("Cached induced-current kernel must be finite [B,N,N]")
        return self.encoder(features).mean(dim=1), kernel


class ComplexSpectralFusion(nn.Module):
    """CSF: complex linear map and antenna-wise cross-frequency attention."""
    def __init__(self,n_tx,width,hidden,heads,dropout):
        super().__init__()
        self.real_weight = nn.Linear(n_tx,width,bias=False)
        self.imag_weight = nn.Linear(n_tx,width,bias=False)
        self.radial_bias = nn.Parameter(torch.zeros(width))
        self.to_real = nn.Linear(2*width+4,width)
        self.norm = nn.LayerNorm(width)
        self.frequency_attention = nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.pool = nn.Linear(width,1)
        self.to_backbone = nn.Linear(width,hidden)

    def forward(self,batch,spectral=True):
        real, imag = batch["G_tx_real"].float(),batch["G_tx_imag"].float()
        b,nf,n,_ = real.shape
        power = real.diagonal(dim1=-2,dim2=-1).clamp_min(0)
        scale = power.mean(-1,keepdim=True).clamp_min(1e-12)
        gr,gi = real/scale.unsqueeze(-1),imag/scale.unsqueeze(-1)
        zr = self.real_weight(gr)-self.imag_weight(gi)
        zi = self.real_weight(gi)+self.imag_weight(gr)
        magnitude = (zr.square()+zi.square()+1e-12).sqrt()
        gate = F.relu(magnitude+self.radial_bias)/magnitude
        zr,zi = zr*gate,zi*gate
        fc = batch.get("frequency_coord", torch.linspace(-1,1,nf,device=real.device)[None].expand(b,-1))
        side = torch.stack((torch.log1p(power),torch.log1p(scale).expand(-1,-1,n),
            batch["effective_snr_db"].float().view(b,1,1).expand(-1,nf,n)/20,
            fc.float().reshape(b,nf,1).expand(-1,-1,n)), -1)
        tokens = self.to_real(torch.cat((zr,zi,side),-1))
        if spectral:
            sequence = tokens.permute(0,2,1,3).reshape(b*n,nf,-1)
            normal = self.norm(sequence)
            attended,_ = self.frequency_attention(normal,normal,normal,need_weights=False)
            tokens = (sequence+attended).reshape(b,n,nf,-1).permute(0,2,1,3)
            token = (self.pool(tokens).softmax(1)*tokens).sum(1)
        else:
            token = tokens.mean(1)
        denom = (power.unsqueeze(-1)*power.unsqueeze(-2)).clamp_min(1e-24).sqrt()
        corr = torch.complex(real/denom,imag/denom)
        return token,corr


class ElectromagneticCurrentGraph(nn.Module):
    """ECG: channel correlations, induced-current overlap and geometry edges."""
    def __init__(self,width,heads,dropout):
        super().__init__()
        if width % heads:
            raise ValueError("adapter_dim must be divisible by graph_heads")
        self.heads,self.width = heads,width
        self.node_fusion = nn.Sequential(nn.Linear(2*width,width), nn.GELU(), nn.LayerNorm(width))
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width,3*width)
        self.edge = nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,heads))
        self.output = nn.Linear(width,width)
        self.dropout = nn.Dropout(dropout)

    def forward(self,channel_tokens,current_tokens,corr,kernel,batch,enabled=True):
        # PCR and CSF are independent branches; ECG is their first fusion point.
        if channel_tokens.shape != current_tokens.shape:
            raise ValueError("PCR and CSF tokens must have matching [B,N,D] shapes")
        tokens = self.node_fusion(torch.cat((channel_tokens,current_tokens),dim=-1))
        b,n,p = tokens.shape
        pos = batch.get("positions_wavelength")
        if pos is None:
            distance = torch.zeros_like(kernel.real)
        else:
            pos = pos.float()
            if pos.ndim == 2:
                pos = pos.unsqueeze(0).expand(b, -1, -1)
            if pos.ndim != 3 or pos.shape[:2] != (b, n):
                raise ValueError("positions_wavelength must have shape [N,3] or [B,N,3]")
            distance = torch.cdist(pos, pos)
        edges = torch.stack((corr.real.mean(1),corr.imag.mean(1),corr.abs().mean(1),
            kernel.real,kernel.imag,kernel.abs(),torch.log1p(distance)),-1)
        if not enabled:
            return tokens,edges
        q,k,v = self.qkv(self.norm(tokens)).view(b,n,3,self.heads,p//self.heads).unbind(2)
        q,k,v = (t.transpose(1,2) for t in (q,k,v))
        weights = (q@k.transpose(-2,-1)/math.sqrt(p//self.heads)+self.edge(edges).permute(0,3,1,2)).softmax(-1)
        mixed = (self.dropout(weights)@v).transpose(1,2).reshape(b,n,p)
        return tokens+self.dropout(self.output(mixed)),edges


class SubsetFeedbackInteraction(nn.Module):
    """SFI: selected-set state changes the gain on physical pairwise attention."""
    def __init__(self,hidden,width):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,width),nn.GELU())
        self.q,self.k,self.v = (nn.Linear(width,width) for _ in range(3))
        self.edge = nn.Linear(7,1)
        self.physical_edge = nn.Linear(3,1,bias=False)
        self.context_gate = nn.Sequential(nn.Linear(width+1,width),nn.GELU(),nn.Linear(width,1),nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(3*width+2,width),nn.GELU(),nn.LayerNorm(width))

    def cache(self,hidden,edges):
        z = self.project(hidden)
        base = self.q(z)@self.k(z).transpose(1,2)/math.sqrt(z.shape[-1])+self.edge(edges).squeeze(-1)
        physical = self.physical_edge(edges[...,3:6]).squeeze(-1)
        return dict(z=z,value=self.v(z),base=base,physical=physical,overlap=edges[...,5].square())

    def forward(self,cache,selected,final_k,enabled=True):
        z = cache["z"]
        if not enabled:
            return z
        selected_f = selected.to(z.dtype)
        count = selected_f.sum(1,keepdim=True)
        context = (z*selected_f.unsqueeze(-1)).sum(1)/count.clamp_min(1)
        step = count/final_k
        physical_gate = self.context_gate(torch.cat((context,step),-1))
        logits = cache["base"]+physical_gate.unsqueeze(-1)*cache["physical"]
        attention = logits.masked_fill(~selected[:,None,:], -1e4).softmax(-1)*selected_f[:,None,:]
        attention = attention/attention.sum(-1,keepdim=True).clamp_min(1e-12)
        message = attention@cache["value"]
        overlap = (cache["overlap"]*selected_f[:,None,:]).sum(-1)/final_k
        features = torch.cat((z,message,context[:,None,:].expand_as(z),overlap.unsqueeze(-1),
                              step[:,None,:].expand(-1,z.shape[1],-1)), -1)
        return z+self.fusion(features)


class DistributionalMarginalHead(nn.Module):
    """DMH: mean and residual variance; variance is not validated calibration."""
    def __init__(self,width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,2))

    def forward(self,state):
        raw = self.network(state)
        return F.softplus(raw[...,0]),raw[...,1].clamp(-6,6)


class ICSMASAdapter(nn.Module):
    """ONE module containing exactly five named interacting submodules."""
    def __init__(self,n_tx,hidden,width=64,heads=4,dropout=.1,require_physics=True,ablations=()):
        super().__init__()
        self.ablations = frozenset(ablations)
        unknown = self.ablations-{"port","spectral","graph","feedback","distribution"}
        if unknown:
            raise ValueError(f"Unknown ICSM-AS ablations: {sorted(unknown)}")
        self.pcr = PortCurrentResponse(n_tx,width,require_physics)
        self.csf = ComplexSpectralFusion(n_tx,width,hidden,heads,dropout)
        self.ecg = ElectromagneticCurrentGraph(width,heads,dropout)
        self.sfi = SubsetFeedbackInteraction(hidden,width)
        self.dmh = DistributionalMarginalHead(width)

    def encode(self,batch):
        current,kernel = self.pcr(batch)
        if "port" in self.ablations:
            current = torch.zeros_like(current)
            kernel = torch.zeros_like(kernel)
        channel,corr = self.csf(batch,"spectral" not in self.ablations)
        token,edges = self.ecg(channel,current,corr,kernel,batch,"graph" not in self.ablations)
        return self.csf.to_backbone(token),edges,corr.abs().mean(1)

    def predict(self,cache,selected,final_k):
        state = self.sfi(cache,selected,final_k,"feedback" not in self.ablations)
        mean,logvar = self.dmh(state)
        if "distribution" in self.ablations:
            logvar = torch.zeros_like(logvar)
        return mean,logvar


class ICSMAS(SCMGVariant):
    """ICSM-AS: parallel PCR/CSF representation, ECG graph encoding, one-pass Qwen2, SFI and DMH."""
    method_name = "ICSM-AS"
    def __init__(self,*args,allow_proxy_physics=False,risk_beta=0.,ablations=(),use_nll_ablation=False,**kwargs):
        super().__init__(*args,**kwargs)
        if risk_beta < 0 or not math.isfinite(risk_beta):
            raise ValueError("risk_beta must be finite and nonnegative")
        # Replace the old three-block path; retain legacy auxiliary outputs.
        self.tokenizer = self.graph = self.to_backbone = self.marginal_head = None
        options = self.model_options
        if options["token_mode"] != "complex":
            raise ValueError("ICSM-AS uses complex input; use its explicit five-block ablations")
        if not options["use_graph"]:
            ablations = sorted(set(ablations)|{"graph"})
        self.adapter = ICSMASAdapter(self.n_tx,self.hidden_size,options["adapter_dim"],
            options["graph_heads"],options["dropout"],not allow_proxy_physics,ablations)
        self.risk_beta = float(risk_beta)
        self.use_nll_ablation = bool(use_nll_ablation)
        self.model_options.update(allow_proxy_physics=allow_proxy_physics,risk_beta=risk_beta,
                                  ablations=list(ablations),use_nll_ablation=self.use_nll_ablation)

    def forward(self,batch):
        embedded,edges,direct = self.adapter.encode(batch)
        # Candidate tokens retain the original transmit-port index order (0,...,Nt-1).
        # This 2-D mask marks valid tokens; Qwen2 applies its internal causal mask and RoPE.
        hidden = self.backbone(inputs_embeds=embedded,attention_mask=torch.ones(embedded.shape[:2],
            device=embedded.device,dtype=torch.long),use_cache=False,return_dict=True).last_hidden_state
        z = self.relation_proj(hidden)
        relation = z@z.transpose(1,2)/math.sqrt(self.relation_rank)
        relation = relation.masked_fill(torch.eye(self.n_tx,device=relation.device,dtype=torch.bool)[None],0)
        return dict(hidden=hidden,utility_logits=self.utility_head(hidden).squeeze(-1),relation_logits=relation,
            direct_relation=direct,icsm_cache=self.adapter.sfi.cache(hidden,edges))

    @torch.no_grad()
    def select(self,outputs,k=None,lambda_rel=.1):
        if self.decoder != "marginal":
            return super().select(outputs,k,lambda_rel)
        k = self.n_sel if k is None else k
        if k != self.n_sel:
            raise ValueError("ICSM-AS teacher and decoder require the same final n_sel")
        mask = torch.zeros(outputs["utility_logits"].shape,dtype=torch.bool,device=outputs["hidden"].device)
        indices = []
        for _ in range(k):
            mean,logvar = self.adapter.predict(outputs["icsm_cache"],mask,k)
            score = mean-self.risk_beta*torch.exp(.5*logvar)
            nxt = score.masked_fill(mask,-torch.inf).argmax(1)
            indices.append(nxt)
            mask.scatter_(1,nxt[:,None],True)
        return torch.stack(indices,1)

    def marginal_loss(self,outputs,teacher_order,teacher_marginals,*,regression_weight=1.0,ranking_weight=0.1):
        cache = outputs["icsm_cache"]
        mask = torch.zeros(cache["z"].shape[:2],device=cache["z"].device,dtype=torch.bool)
        losses = []
        for step in range(self.n_sel):
            mean,logvar = self.adapter.predict(cache,mask,self.n_sel)
            valid = ~mask
            target = teacher_marginals[:,step]
            loss = regression_weight * F.smooth_l1_loss(mean[valid],target[valid])
            # Ranking supervises the next teacher action among currently available ports.
            loss = loss + ranking_weight * F.cross_entropy(mean.masked_fill(mask,-1e4),teacher_order[:,step])
            if self.use_nll_ablation and "distribution" not in self.adapter.ablations:
                nll = .5*(logvar[valid]+(target[valid]-mean[valid]).square()*torch.exp(-logvar[valid]))
                loss = loss + 0.1*nll.mean()
            losses.append(loss)
            mask = mask.scatter(1,teacher_order[:,step,None],True)
        return torch.stack(losses).mean()


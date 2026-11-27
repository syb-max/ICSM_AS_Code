"""Paper-aligned objective terms for membership and sequential-gain models."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def membership_bce(logits,batch,alpha_hq=.5):
    hq=F.binary_cross_entropy_with_logits(logits,batch["hq_utility_target"].float())
    lq=F.binary_cross_entropy_with_logits(logits,batch["lq_utility_target"].float())
    return alpha_hq*hq+(1-alpha_hq)*lq

def relation_mse(logits,target):
    n=logits.shape[-1]; mask=~torch.eye(n,device=logits.device,dtype=torch.bool)
    return F.mse_loss(torch.sigmoid(logits[:,mask]),target[:,mask].float())

def icsm_objective(model,outputs,batch,lambda_relation=.2,lambda_marginal=1.,alpha_hq=.5):
    bce=membership_bce(outputs["utility_logits"],batch,alpha_hq)
    rel=relation_mse(outputs["relation_logits"],batch["relation_target"])
    marginal=model.marginal_loss(outputs,batch["teacher_order"].long(),batch["teacher_marginals"].float())
    total=bce+lambda_relation*rel+lambda_marginal*marginal
    return {"loss":total,"bce":bce,"relation":rel,"marginal":marginal}

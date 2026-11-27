"""Train a named baseline on the train split; select hyperparameters on validation only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from ..core.physics_data import PhysicsDataset
from ..core.capacity_teacher import capacity_validated
from ..models.baselines.registry import build_baseline
from ..utils.reproducibility import set_reproducible, write_run_manifest
from .losses import membership_bce

def _collate(batch):
    keys=batch[0].keys(); return {k:torch.stack([x[k] for x in batch]) if torch.is_tensor(batch[0][k]) else batch[0][k] for k in keys}

def run(args):
    set_reproducible(args.seed,args.deterministic)
    device=torch.device(args.device)
    tr=PhysicsDataset(args.data_h5,split=0); va=PhysicsDataset(args.data_h5,split=1)
    model_kwargs={"n_tx":tr.n_tx,"n_sel":tr.n_sel}
    if args.model in ("ICSM-AS (static)","Score-only comparator","Score-only","SCMG-variant","Random-frozen","Small-TF"):
        if not args.qwen_dir:
            raise ValueError(f"--qwen_dir is required for {args.model}")
        model_kwargs["qwen_dir"]=args.qwen_dir
    if args.model in ("Random-frozen","Small-TF"):
        model_kwargs["allow_proxy_physics"]=args.allow_proxy_physics
    model=build_baseline(args.model,**model_kwargs).to(device)
    train_loader=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=_collate)
    val_loader=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=_collate)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); best=-float('inf'); history=[]
    for epoch in range(args.epochs):
        model.train(); total=0.; n=0
        for host in train_loader:
            batch={k:v.to(device) for k,v in host.items() if torch.is_tensor(v)}
            opt.zero_grad(set_to_none=True); pred=model(batch)
            if hasattr(model,"marginal_loss"):
                loss=model.marginal_loss(pred,batch["teacher_order"].long(),batch["teacher_marginals"].float())
                if "relation_logits" in pred:
                    from .losses import relation_mse
                    loss=loss+0.2*relation_mse(pred["relation_logits"],batch["relation_target"])
                if "utility_logits" in pred:
                    loss=loss+membership_bce(pred["utility_logits"],batch,args.alpha_hq)
            else:
                loss=membership_bce(pred["utility_logits"],batch,args.alpha_hq)
                if "relation_logits" in pred:
                    from .losses import relation_mse
                    loss=loss+0.2*relation_mse(pred["relation_logits"],batch["relation_target"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach())*len(host["sample_id"]); n+=len(host["sample_id"])
        model.eval(); val_total=0.; val_n=0; grouped={}
        with torch.no_grad():
            for host in val_loader:
                batch={k:v.to(device) for k,v in host.items() if torch.is_tensor(v)}
                pred=model(batch)
                if hasattr(model,"marginal_loss"):
                    loss=model.marginal_loss(pred,batch["teacher_order"].long(),batch["teacher_marginals"].float())
                    if "relation_logits" in pred:
                        from .losses import relation_mse
                        loss=loss+0.2*relation_mse(pred["relation_logits"],batch["relation_target"])
                    if "utility_logits" in pred:
                        loss=loss+membership_bce(pred["utility_logits"],batch,args.alpha_hq)
                else:
                    loss=membership_bce(pred["utility_logits"],batch,args.alpha_hq)
                    if "relation_logits" in pred:
                        from .losses import relation_mse
                        loss=loss+0.2*relation_mse(pred["relation_logits"],batch["relation_target"])
                val_total+=float(loss)*len(host["sample_id"]); val_n+=len(host["sample_id"])
                chosen=model.select(pred,k=tr.n_sel).detach().cpu().numpy()
                grams=host["G_tx_real"].numpy()+1j*host["G_tx_imag"].numpy()
                for j,subset in enumerate(chosen):
                    rho=float(host["snr_reference_linear"][j])
                    cap=capacity_validated(grams[j],rho,subset,final_k=tr.n_sel)
                    greedy=float(host["greedy_capacity"][j]); hq=float(host["hq_capacity"][j])
                    ncr=cap/greedy if greedy>1e-8 else float('nan')
                    rc=cap/hq if hq>1e-8 else float('nan')
                    key=(float(host["snr_reference_db"][j]),int(host["site_id"][j]))
                    grouped.setdefault(key,[]).append({"NCR":ncr,"R_C_HQ":rc})
        per_site={key:{metric:float(np.nanmean([x[metric] for x in vals])) for metric in ("NCR","R_C_HQ")} for key,vals in grouped.items()}
        per_snr={}
        for snr in sorted({key[0] for key in per_site}):
            rows_snr=[v for (s,_),v in per_site.items() if s==snr]
            per_snr[snr]={metric:float(np.nanmean([r[metric] for r in rows_snr])) for metric in ("NCR","R_C_HQ")}
        val_metrics={metric:float(np.nanmean([v[metric] for v in per_snr.values()])) for metric in ("NCR","R_C_HQ")}
        selected_metric=val_metrics[args.selection_metric]
        row={"epoch":epoch+1,"train_loss":total/max(n,1),"val_loss":val_total/max(val_n,1),
             "val_NCR":val_metrics["NCR"],"val_R_C_HQ":val_metrics["R_C_HQ"],
             "selection_metric":args.selection_metric,"selection_value":selected_metric}; history.append(row); print(json.dumps(row),flush=True)
        if selected_metric>best:
            best=selected_metric; torch.save({"model_name":args.model,"seed":args.seed,"model_kwargs":model_kwargs,
                "state_dict":model.state_dict(),"epoch":epoch+1,"val_loss":row["val_loss"],
                "val_NCR":val_metrics["NCR"],"val_R_C_HQ":val_metrics["R_C_HQ"],
                "selection_metric":args.selection_metric,"selection_value":best},out/"best.pt")
    (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    write_run_manifest(out/"run_manifest.json",seed=args.seed,data_path=args.data_h5,extra={
        "model":args.model,"model_kwargs":model_kwargs,"selection_metric":args.selection_metric,
        "config":vars(args),"best_validation_value":best,"checkpoint":"best.pt",
        "note":"No test split was opened by this training script."},
        checkpoint_path=out/"best.pt")
    tr.close(); va.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True,choices=["MLP-Utility","ResNet-Utility","CNN-Utility","FT-Transformer","MLP-MultiLabel","TVT-MLL","TVT-MLL (adapted)","Small-TF","ICSM-AS (static)","Score-only comparator","Score-only","SCMG-variant","Random-frozen"])
    p.add_argument("--data_h5",required=True); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch_size",type=int,default=8); p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--weight_decay",type=float,default=1e-4); p.add_argument("--alpha_hq",type=float,default=.5); p.add_argument("--num_workers",type=int,default=0)
    p.add_argument("--selection_metric",choices=["NCR","R_C_HQ"],default="NCR")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--deterministic",action="store_true")
    p.add_argument("--qwen_dir",default=None); p.add_argument("--allow_proxy_physics",action="store_true")
    run(p.parse_args())
if __name__=="__main__": main()

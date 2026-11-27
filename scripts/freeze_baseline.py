#!/usr/bin/env python3
"""Freeze a baseline checkpoint selected on validation; does not open test data."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run_dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 d=Path(a.run_dir); status=json.loads((d/'run_manifest.json').read_text(encoding='utf-8'))
 ckpt=d/status['checkpoint']; payload={"protocol":"icsm-as-baseline-frozen-v1","model":status['model'],"seed":status['seed'],
  "data_sha256":status['data_sha256'],"checkpoint_path":str(ckpt.resolve()),"checkpoint_sha256":sha(ckpt),
  "selection":f"best validation {status.get('selection_metric', 'NCR')}","test_policy":"test split may be opened only after this manifest is created"}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()

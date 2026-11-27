#!/usr/bin/env python3
"""Validate the prepared HDF5 schema and group-level split integrity."""
from __future__ import annotations
import argparse, json
from icsm_as.data.schema import validate_prepared_h5

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_h5',required=True)
    p.add_argument('--require_physical',action='store_true')
    args=p.parse_args()
    result=validate_prepared_h5(args.data_h5,require_physical=args.require_physical)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': main()

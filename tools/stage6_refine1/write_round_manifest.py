#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,subprocess
from pathlib import Path
from tools.stage6_refine1.common import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument('--round',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--round-name',required=True);p.add_argument('--zip',type=Path,required=True);p.add_argument('--next',required=True);a=p.parse_args();r=a.round.resolve();gpu='unknown';
 try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,memory.total','--format=csv,noheader'],text=True).strip()
 except Exception as e:gpu=f'query_failed:{e}'
 files=[p for p in sorted(r.rglob('*')) if p.is_file() and p.name!='run_manifest.md'];(r/'run_manifest.md').write_text(f'stage = {a.stage}\nround = {a.round_name}\nrepo_root = /home/__compress_data/xushijie/CUPID\npython = /home/xushijie/.conda/envs/cupid/bin/python\nGPU inventory:\n{gpu}\noutput ZIP = {a.zip.resolve()}\nZIP SHA256 = {sha256(a.zip) if a.zip.exists() else "pending"}\nlarge files omitted = true\nnext action = {a.next}\nartifacts = {len(files)}\n')
if __name__=='__main__':main()

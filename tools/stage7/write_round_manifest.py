#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
try:
 from .common import digest
except ImportError:
 from common import digest
def main():
    p=argparse.ArgumentParser(); p.add_argument('--round',type=Path,required=True); p.add_argument('--round-name',required=True); p.add_argument('--zip',type=Path,required=True); p.add_argument('--next',required=True); p.add_argument('--mode',default='reward_only'); a=p.parse_args()
    r=a.round.resolve(); r.mkdir(parents=True,exist_ok=True)
    try: gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    except Exception as e: gpu=f'query_failed:{e}'
    files=sum(1 for f in r.rglob('*') if f.is_file() and f.name!='run_manifest.md')
    text=(f'stage = stage7\nround = {a.round_name}\nmode = {a.mode}\nstart_time = unknown\nend_time = unknown\nrepo_root = /home/__compress_data/xushijie/CUPID\npython = /home/xushijie/.conda/envs/cupid/bin/python\nGPU inventory:\n{gpu}\nstatistics_unit = content_group_id\nprovenance = mixed_sources_explicitly_labeled\nlarge files omitted = true\noutput ZIP = {a.zip.resolve()}\nZIP SHA256 = {digest(a.zip) if a.zip.exists() else "pending"}\nnext action = {a.next}\nartifacts = {files}\n')
    (r/'run_manifest.md').write_text(text)
if __name__=='__main__': main()

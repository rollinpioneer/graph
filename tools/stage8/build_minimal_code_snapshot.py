#!/usr/bin/env python3
"""Copy only code exercised by the Stage 8 reproduction path."""
from __future__ import annotations
import argparse,hashlib,shutil
from pathlib import Path
from tools.stage8.common import write_csv
def digest(path:Path)->str:
 h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()
def copy(source:Path,target:Path,rows:list[dict]):
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);rows.append({'source_path':str(source.resolve()),'snapshot_path':str(target),'sha256':digest(target),'role':'minimal_runtime_dependency'})
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--stage8-tools',type=Path,required=True);p.add_argument('--reward-engine',type=Path,required=True);p.add_argument('--model-bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for source in sorted(a.stage8_tools.glob('*.py')):copy(source,a.output/'tools/stage8'/source.name,rows)
 copy(a.repo_root/'tools/__init__.py',a.output/'tools/__init__.py',rows);copy(a.repo_root/'tools/stage4/__init__.py',a.output/'tools/stage4/__init__.py',rows);copy(a.repo_root/'tools/stage4/lib/__init__.py',a.output/'tools/stage4/lib/__init__.py',rows);copy(a.repo_root/'tools/stage4/lib/model.py',a.output/'tools/stage4/lib/model.py',rows)
 for source in sorted(a.reward_engine.parent.glob('*.py')):copy(source,a.output/'artifacts/pathgraph_sarm/stage5/reward_v1/code'/source.name,rows)
 copy(a.model_bundle,a.output/'model_bundle_persistent.json',rows);write_csv(a.manifest,rows,delimiter='\t')
if __name__=='__main__':main()

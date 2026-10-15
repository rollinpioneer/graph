#!/usr/bin/env python3
"""Copy the frozen Stage 5 reward inputs out of /tmp and verify every SHA256."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--stage5',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    src=a.stage5.resolve(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    for rel in ('code/reward_engine.py','code/reward_types.py','configs/reward_config_v1.yaml','configs/reward_selection_lock.json','configs/stage6_weight_schema.json','configs/reward_normalization.json'):
        target=out/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src/rel,target)
    (out/'code/__init__.py').write_text('# frozen Stage 5 reward implementation package\n')
    bundle=json.loads((src/'configs/model_bundle.json').read_text())
    copied=[]
    for ent in bundle['checkpoints']:
        source=Path(ent['path']); got=digest(source)
        if got != ent['sha256']: raise RuntimeError(f'checkpoint sha mismatch: {source}')
        dest=out/'model_checkpoints'/f"seed_{ent['seed']}_best.pt"; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
        if digest(dest) != got: raise RuntimeError(f'copy sha mismatch: {dest}')
        copied.append({**ent,'path':str(dest),'source_path':str(source)})
    for key in ('feature_schema','label_maps','cost_target_spec'):
        source=Path(bundle[key]); target=out/'graph_inputs'/source.name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target); bundle[key]=str(target)
    bundle['checkpoints']=copied; (out/'configs').mkdir(exist_ok=True)
    (out/'configs/model_bundle_persistent.json').write_text(json.dumps(bundle,indent=2)+'\n')
    manifest=[]
    for f in sorted(x for x in out.rglob('*') if x.is_file()): manifest.append(f'{digest(f)}  {f.relative_to(out)}')
    (out/'PERSISTED_INPUTS_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
    print(json.dumps({'status':'PASS','checkpoint_count':len(copied),'out':str(out)},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())

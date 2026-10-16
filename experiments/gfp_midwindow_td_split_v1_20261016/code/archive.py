import pathlib,json,shutil,subprocess,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
wt=ROOT/'archive_worktree';branch='codex/gfp-midwindow-td-split-v1-20261016'
assert subprocess.check_output(['git','branch','--show-current'],cwd=wt,text=True).strip()==branch
selected=[ROOT/'EXP_CARD_L1_SPLIT_20261016.md',ROOT/'protocol.json']+list((ROOT/'code').glob('*.py'))+list((ROOT/'reports').glob('*.md'))+list((ROOT/'reports').glob('*.patch'))+list((ROOT/'reports').glob('*.json'))+list((ROOT/'reports').glob('*.png'))+list((ROOT/'receipts').glob('*.json'))
for d in (ROOT/'runs').glob('*'):selected+=list(d.glob('*.json'))+list(d.glob('eval_*/results.json'))+list(d.glob('eval_*/exit.json'))+list(d.glob('eval_*/cost.json'))
names=[]
for p in selected:
 assert p.stat().st_size<10000000
 rel=p.relative_to(ROOT);out=wt/'experiments'/ROOT.name/rel;out.parent.mkdir(parents=True,exist_ok=True)
 if p.name=='results.json' and p.parent.name.startswith('eval_'):
  v=json.loads(p.read_text())
  for e in v['episodes']:e.pop('path',None)
  v['full_server_path']=str(p);v['full_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();out.write_text(json.dumps(v,indent=2))
 else:shutil.copy2(p,out)
 names.append(str(out.relative_to(wt)))
(ROOT/'receipts/archive_manifest.json').write_text(json.dumps(dict(branch=branch,files=names),indent=2))
subprocess.run(['git','add','-f','--']+names,cwd=wt,check=True)
print(subprocess.check_output(['git','diff','--cached','--stat'],cwd=wt,text=True))

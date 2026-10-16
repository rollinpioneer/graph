from util import *
import shutil,subprocess
wt=ROOT/'archive_worktree';branch='codex/gfp-reference-support-diagnostic-v1-20261016'
assert subprocess.check_output(['git','branch','--show-current'],cwd=wt,text=True).strip()==branch
files=[ROOT/'EXPERIMENT_CARD.md']+list((ROOT/'code').glob('*.py'))
for folder in ['reports','receipts']:
 files += [p for p in (ROOT/folder).iterdir() if p.suffix in ['.json','.md','.png','.pdf','.csv'] and p.name!='archive_manifest.json']
names=[]
for p in sorted(files):
 assert p.stat().st_size<10_000_000
 dest=wt/'experiments'/ROOT.name/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);names.append(str(dest.relative_to(wt)))
write('receipts/archive_manifest.json',dict(branch=branch,files=names))
subprocess.run(['git','add','-f','--']+names,cwd=wt,check=True)
print(subprocess.check_output(['git','diff','--cached','--stat'],cwd=wt,text=True))

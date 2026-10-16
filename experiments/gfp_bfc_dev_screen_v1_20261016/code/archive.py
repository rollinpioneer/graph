import shutil,json,subprocess,hashlib
from common import ROOT,write

def main():
    wt=ROOT/'archive_worktree';branch='codex/gfp-bfc-dev-screen-v1-20261016'
    actual=subprocess.check_output(['git','branch','--show-current'],cwd=wt,text=True).strip();assert actual==branch
    dest=wt/'experiments/gfp_bfc_dev_screen_v1_20261016';dest.mkdir(parents=True,exist_ok=True)
    selected=[ROOT/'EXPERIMENT_CARD.md',ROOT/'protocol.json']
    selected+=list((ROOT/'code').glob('*.py'))+list((ROOT/'reports').glob('*.json'))+list((ROOT/'reports').glob('*.md'))+list((ROOT/'reports').glob('*.png'))
    selected+=list((ROOT/'receipts').glob('*.json'))
    for d in (ROOT/'runs').glob('*') if (ROOT/'runs').exists() else []:
        selected+=list(d.glob('*.json'))+list(d.glob('eval_*/results.json'))
    names=[]
    for p in selected:
        if not p.exists():continue
        assert p.stat().st_size<10000000
        rel=p.relative_to(ROOT);out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True);
        if p.name=='results.json':
            data=json.loads(p.read_text())
            for episode in data.get('episodes',[]):episode.pop('path',None)
            data['full_result_server_path']=str(p)
            data['full_result_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            data['archive_omits']='Per-step path retained only on server; per-episode metrics retained here.'
            out.write_text(json.dumps(data,indent=2)+'\n')
        else:shutil.copy2(p,out)
        names.append(str(out.relative_to(wt)))
    # Explicit lightweight whitelist only; inherited ignore rules cover all runs/.
    subprocess.run(['git','add','-f','--']+names,cwd=wt,check=True)
    write(ROOT/'receipts/archive_manifest.json',dict(branch=branch,files=names))
    print(subprocess.check_output(['git','diff','--cached','--stat'],cwd=wt,text=True))

if __name__=='__main__':main()

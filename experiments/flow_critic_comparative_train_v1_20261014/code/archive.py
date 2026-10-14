"""Explicit lightweight archival only; never include full weights/data/logs."""
from pathlib import Path
import shutil,hashlib,json,subprocess,sys,os,difflib,time
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'archive_worktree';BRANCH='codex/flow-critic-comparative-train-v1-20261014'
PREFIX=Path('experiments')/ROOT.name
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
def main():
    assert git('branch','--show-current')==BRANCH,'Wrong archive branch'
    files=[]
    for base in ['code','configs','reports','receipts']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.lock'] and p.name not in ['manager.lock','evaluation_manager.lock','light_artifact_manifest.json']:
                if p.stat().st_size>5000000:raise RuntimeError('Unexpected large lightweight artifact '+str(p))
                files.append(p)
    for name in ['learning_curves','diagnostics','critic_components']:
        for ext in ['png','pdf']:
            p=ROOT/'reports/figures'/f'{name}.{ext}'
            assert p.exists() and p.stat().st_size<5000000
            files.append(p)
    files.append(ROOT/'EXPERIMENT_CARD.md')
    for host in ['gfp','dfql']:
        work=ROOT/f'workspaces/{host}_floq'
        names=['main.py','training_wiring.py','batch_hooks.py',f'agents/{host}.py','agents/floq_adapter.py',
            'agents/update_runtime.py','utils/evaluation.py','utils/flax_utils.py','utils/networks.py']
        if host=='gfp':names+=['config/main.yaml','config/agent/gfp.yaml','hyperparams/gfp_params.csv','evaluation_process.py']
        names+=[str(p.relative_to(work)) for p in (work/'floq_value_core').glob('*.py')]
        for name in names:
            p=work/name
            if p.exists():files.append(p)
        patch=[]
        previous=ROOT.parent/'flow_critic_dual_host_train_wiring_t0_v1_20261014'/f'workspaces/{host}_floq'
        for name in names:
            p=work/name;q=previous/name
            if p.exists():patch.extend(difflib.unified_diff(q.read_text().splitlines(True) if q.exists() else [],
                p.read_text().splitlines(True),fromfile='T0/'+name,tofile='comparative/'+name))
        p=ROOT/'reports'/f'{host}_changes_from_t0.patch';p.write_text(''.join(patch));files.append(p)
    for p in (ROOT/'runs').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.csv']:
            if p.stat().st_size>5000000:continue
            files.append(p)
    identities=[];staged=[]
    for p in sorted(set(files)):
        relative=p.relative_to(ROOT);dest=REPO/PREFIX/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest);staged.append(str(PREFIX/relative))
        identities.append(dict(path=str(relative),size=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    manifest=ROOT/'reports/light_artifact_manifest.json';manifest.write_text(json.dumps(identities,indent=2))
    dest=REPO/PREFIX/'reports/light_artifact_manifest.json';shutil.copyfile(manifest,dest)
    staged.append(str(dest.relative_to(REPO)))
    # Explicit force-add only for individually enumerated small lockfiles/run metadata.
    for path in staged:
        force=path.endswith('.lock') or path.startswith(str(PREFIX)+'/runs/')
        subprocess.run(['git','-C',str(REPO),'add',*(['-f'] if force else []),'--',path],check=True)
    actual=git('diff','--cached','--name-only').splitlines()
    if any(not p.startswith(str(PREFIX)+'/') for p in actual):raise RuntimeError('Unrelated staged artifact')
    if actual:
        print(git('diff','--cached','--stat'),flush=True)
        subprocess.run(['git','-C',str(REPO),'commit','-m',sys.argv[1] if len(sys.argv)>1 else 'Archive comparative batch progress'],check=True)
    env=dict(os.environ,GIT_SSH_COMMAND='ssh -i /home/xushijie3/.ssh/graph_cp_disr_deploy_ed25519 -o IdentitiesOnly=yes')
    subprocess.run(['git','-C',str(REPO),'-c','pack.threads=4','push','-u','origin',BRANCH],env=env,check=True)
    local=git('rev-parse','HEAD')
    remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','origin','refs/heads/'+BRANCH],env=env,text=True).split()[0]
    assert local==remote,'Remote SHA mismatch'
    receipt=dict(branch=BRANCH,commit=local,remote_commit=remote,verified=True,epoch=time.time(),files=len(identities),
        url='https://github.com/rollinpioneer/graph/tree/'+BRANCH)
    (ROOT/'receipts/git_latest.json').write_text(json.dumps(receipt,indent=2))
    print('REMOTE_VERIFIED',local[:20],local[20:],len(identities),'light files',flush=True)
if __name__=='__main__':main()

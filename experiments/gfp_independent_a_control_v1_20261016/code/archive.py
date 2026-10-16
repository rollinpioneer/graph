import json,hashlib,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BRANCH='codex/gfp-independent-a-control-v1-20261016'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 wt=ROOT/'archive_worktree';assert subprocess.check_output(['git','-C',str(wt),'branch','--show-current'],text=True).strip()==BRANCH
 fixed=['EXPERIMENT_CARD.md','protocol.json','HANDOFF.md']
 fixed+=['code/'+n for n in ['common.py','reference.py','admission.py','continue_train.py','evaluate.py','evaluation_manager.py','launch.py','summarize.py','write_report.py','archive.py']]
 fixed+=['receipts/'+n for n in ['authorization.json','source_identity.json','prelaunch_files.json','provenance.json','prelaunch_git_verified.json','first_event_source_comparison.json','runtime_code_identity.json','process_closure.json','dispatcher.json','launch.json','admission_started.json','zero_update_admission.json','calibration.json','first_four.json','admission_result.json','admission_exit.json','continuation_started.json','continuation_restore.json','anchors.json','training_complete.json','training_exit.json','evaluation_manager_started.json','evaluation_queue.json','evaluation_complete.json','evaluation_manager_exit.json']]
 fixed+=['reports/'+n for n in ['RESULTS.md','summary.json','costs.json','reference_events.csv','evaluation_episodes.csv','evaluation_summary.json','historical_comparison.json','temporal_comparison.json','checkpoint_index.json','regional_paths.csv','full_curves.png']]
 fixed+=['runs/A/'+n for n in ['identity.json','probes.json','movement_audit.json']]
 fixed += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'runs/A').glob('checkpoint_*.json'))]
 for it in (25000,50000):fixed += [f'runs/A/eval_{it}/{n}' for n in ['exit.json','cost.json']]
 assert len(set(fixed))==len(fixed)
 for f in fixed:assert (ROOT/f).is_file() and (ROOT/f).stat().st_size<2000000,f
 manifest=[dict(path=f,bytes=(ROOT/f).stat().st_size,sha256=sha(ROOT/f)) for f in fixed]
 (ROOT/'receipts/archive_manifest.json').write_text(json.dumps(dict(branch=BRANCH,files=manifest,total_bytes=sum(x['bytes'] for x in manifest)),indent=2))
 large=[ROOT/'receipts/reference_events.json']+list((ROOT/'runs/A').glob('eval_*/results.json'))+list((ROOT/'logs').glob('*.log'))+list((ROOT/'logs').glob('*.jsonl'))
 (ROOT/'receipts/server_artifacts.json').write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in large],indent=2))
 fixed+=['receipts/archive_manifest.json','receipts/server_artifacts.json'];prefix=Path('experiments')/ROOT.name
 for f in fixed:
  dest=wt/prefix/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
 subprocess.run(['git','-C',str(wt),'add','-f','--']+[str(prefix/f) for f in fixed],check=True)
 subprocess.run(['git','-C',str(wt),'-c','core.whitespace=cr-at-eol','diff','--cached','--check'],check=True)
 print(json.dumps(dict(files=len(fixed),bytes=sum(x['bytes'] for x in manifest),branch=BRANCH)))
if __name__=='__main__':main()

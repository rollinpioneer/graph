import json,hashlib,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BRANCH='codex/gfp-reference-support-control-ab-v1-20261016'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 wt=ROOT/'archive_worktree';assert subprocess.check_output(['git','-C',str(wt),'branch','--show-current'],text=True).strip()==BRANCH
 fixed=['EXPERIMENT_CARD.md','protocol.json','HANDOFF.md','receipts/source_identity.json','receipts/authorization.json','receipts/calibration.json','receipts/zero_update_admission.json','receipts/first_four.json','receipts/admission_started.json','receipts/admission_result.json','receipts/admission_exit.json','receipts/preflight_interface_fix.json','receipts/preflight_attempt1/admission_started.json','receipts/preflight_attempt1/admission_exit.json','receipts/preflight_attempt1/execution_error.json','receipts/continuation_started.json','receipts/continuation_restore.json','receipts/anchors.json','receipts/training_complete.json','receipts/training_exit.json','receipts/evaluation_manager_started.json','receipts/evaluation_queue.json','receipts/evaluation_complete.json','receipts/evaluation_manager_exit.json','receipts/provenance.json']
 fixed += ['code/'+n for n in ['common.py','reference.py','admission.py','continue_train.py','evaluate.py','evaluation_manager.py','report_admission.py','summarize.py','archive.py']]
 fixed += ['reports/'+n for n in ['RESULTS.md','summary.json','evaluation_summary.json','evaluation_episodes.csv','paired_comparison.json','reference_matching.csv','event_path_windows.csv','coarse_audit_paths.json','costs.json','checkpoint_index.json','full_curves.png','admission_events.csv','admission_curves.png','admission_summary.json','constraint_diagnostics.json']]
 for arm in ('A','B'):
  fixed += [f'runs/{arm}/{n}' for n in ['identity.json','probes.json','support_audit.json']]
  fixed += [str(p.relative_to(ROOT)) for p in sorted((ROOT/f'runs/{arm}').glob('checkpoint_*.json'))]
  for step in (25000,50000):fixed += [f'runs/{arm}/eval_{step}/{n}' for n in ['exit.json','cost.json']]
 assert len(fixed)==len(set(fixed))
 for f in fixed:
  assert (ROOT/f).is_file(),f
  assert (ROOT/f).stat().st_size<2_000_000,f
 manifest=[dict(path=f,bytes=(ROOT/f).stat().st_size,sha256=sha(ROOT/f)) for f in fixed]
 (ROOT/'receipts/archive_manifest.json').write_text(json.dumps(dict(branch=BRANCH,files=manifest,total_bytes=sum(x['bytes'] for x in manifest)),indent=2))
 large=[ROOT/'receipts/reference_events.json']+list((ROOT/'runs').glob('*/eval_*/results.json'))+list((ROOT/'logs').glob('*.log'))+list((ROOT/'logs').glob('*.jsonl'))
 (ROOT/'receipts/server_artifacts.json').write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in large],indent=2))
 fixed+=['receipts/archive_manifest.json','receipts/server_artifacts.json']
 prefix=Path('experiments/gfp_reference_support_control_ab_v1_20261016')
 for f in fixed:
  dest=wt/prefix/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
 subprocess.run(['git','-C',str(wt),'add','-f','--']+[str(prefix/f) for f in fixed],check=True)
 subprocess.run(['git','-C',str(wt),'-c','core.whitespace=cr-at-eol','diff','--cached','--check'],check=True)
 print(json.dumps(dict(files=len(fixed),bytes=sum(x['bytes'] for x in manifest),branch=BRANCH)))
if __name__=='__main__':main()

from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path(__file__).resolve().parent;W=R/'archive_worktree';B='codex/crr-z1a-critic-response-20261017';prefix='experiments/crr_z1a_critic_response_20261017'
assert subprocess.check_output(['git','-C',str(W),'branch','--show-current'],text=True).strip()==B
names=['EXPERIMENT_CARD.md','protocol.json','asset_manifest.json','healthy_generation_budget.json','common.py','prepare.py','healthy.py','critic.py','controller.py','analyze.py','archive.py','finalize.py','data_manifest.json','source_manifest.json','RESULTS.md','HANDOFF.md','metrics.csv','decision.json','gradient_response.png','absolute_gradients.png','q_response.png','server_artifact_index.json','receipts/source_restore_preflight.json','receipts/preparation.json','receipts/dispatch.json','receipts/controller_complete.json','receipts/protocol_freeze.json','receipts/controllers_exit.json','receipts/zero_update_startup_failure.json','receipts/runtime_config_preflight.json','receipts/analysis.json','resume_bad.py','audit_native_rewrite.py','receipts/native_rewrite_audit_s0.json','receipts/zero_update_native_loss_failure.json','receipts/bad_zero_update_resume_dispatch.json','receipts/bad_zero_update_resume_complete.json']
names += ['sources/'+n for n in ['gfp.py','networks.py','flax_utils.py','datasets.py']+[f's{s}_config.json' for s in range(4)]]
for seed in range(4):
 names += [f'runs/critic_s{seed}/'+n for n in ['zero_update_admission.json','first_four_updates.json','complete.json','exit.json']]
for seed in [1,2]:names += [f'runs/healthy_s{seed}/'+n for n in ['admission.json','complete.json','exit.json','probe_history.json']]
manifest=[];paths=[]
for n in names:
 p=R/n
 if not p.is_file():continue
 assert p.stat().st_size<2_000_000
 dest=W/prefix/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 manifest.append(dict(path=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));paths.append(prefix+'/'+n)
(R/'archive_manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(R/'archive_manifest.json',W/prefix/'archive_manifest.json');paths.append(prefix+'/archive_manifest.json')
subprocess.run(['git','-C',str(W),'add','-f','--']+paths,check=True);staged=subprocess.check_output(['git','-C',str(W),'diff','--cached','--name-only'],text=True).splitlines();assert set(staged)<=set(paths)
print('EXPLICIT_LIGHT_FILES',len(staged),'TOTAL_BYTES',sum(m['bytes'] for m in manifest));print('\n'.join(staged))

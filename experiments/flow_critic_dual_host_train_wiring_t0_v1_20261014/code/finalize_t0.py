"""T0-only finalization; does not launch or update any Agent."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,time,difflib
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
start=read(ROOT/'receipts/start.json');repo=Path('/home/xushijie3/work/graph_cp_disr')
for k,args in {'branch':['branch','--show-current'],'head':['rev-parse','HEAD'],'status':['status','--porcelain']}.items():
 assert subprocess.check_output(['git',*args],cwd=repo,text=True)==start[k]
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=OLD/'archive_worktree',text=True).strip()==start['parent_commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=OLD/'archive_worktree',text=True)
oldfiles=read(OLD/'receipts/staging_manifest.json')
# Three old publication documents were revised in the preceding authorized push.
pub=read(OLD/'receipts/publication_sync_manifest.json');published={x['path']:x['sha256'] for x in pub}
for item in oldfiles:
 rel=item['path'].split('flow_critic_dual_host_prep_v1_20261014/',1)[1]
 expected=published.get(rel,item['sha256']);assert sha(OLD/'archive_worktree'/item['path'])==expected
assets=read(OLD/'assets_manifest/asset_manifest.json');old_ck=0
for item in assets['user_assets']:
 if item.get('path','').endswith('.pkl'):
  assert sha(item['path'])==item['sha256'];old_ck+=1
dataset_files={f['path']:f['sha256'] for d in assets['datasets'] for f in d['files']}
for p,digest in dataset_files.items():assert sha(p)==digest
report={'card':'FLOW-CRITIC-DUAL-HOST-TRAIN-WIRING-T0-V1','status':'T0_ADMITTED','actual_optimizer_updates_total':0,
        'baseline_training_runs':0,'short_stability_runs':0,'formal_training_runs':0,'evaluation_episodes':0,
        'old_checkpoints_unchanged':old_ck,'old_preparation_branch_and_files_unchanged':True,
        'source_data_files_unchanged':len(dataset_files),'hosts':{},'timestamp':time.time()}
commands=[];ckindex=[];patches=[]
for host in ['gfp','dfql']:
 rows=[json.loads(s) for s in (ROOT/'reports'/f'{host}_t0_steps.jsonl').read_text().splitlines()]
 assert len(rows)==2 and [x['update'] for x in rows]==[1,2] and all(x['status']=='PASS' for x in rows)
 assert rows[0]['adam_count_before']==0 and rows[1]['adam_count_after']==2
 assert read(ROOT/'receipts'/f'{host}_optimizer_ledger.json')['completed_updates']==2
 supp=read(ROOT/'reports'/f'{host}_supplemental_checks.json');assert supp['status']=='PASS' and supp['actual_optimizer_updates_in_supplement']==0
 assert all(x['status']=='PASS' for x in supp['tests'])
 runtime=read(ROOT/'receipts'/f'{host}_runtime.json');out=Path(runtime['output_dir'])
 with (out/'train.csv').open() as f:logs=list(csv.DictReader(f))
 assert len(logs)==2
 for row in logs:
  for k,v in row.items():
   if v and k!='step':assert v.lower() not in ['nan','inf','-inf']
 for epoch in [1,2]:
  p=out/f'params_{epoch}.pkl';meta=read(out/f'params_{epoch}.identity.json');assert sha(p)==meta['checkpoint_sha256'];assert meta['completed_updates']==epoch
 native_log=(ROOT/'runs'/host/'native.log').read_text();assert 'T0_REAL_UPDATE_PASS '+host+' 2' in native_log and '2/2' in native_log
 assert 'Traceback (most recent call last)' not in native_log
 launch_name=f'{host}_native_launch_attempt2.json' if host=='dfql' else f'{host}_native_launch.json'
 launch=read(ROOT/'receipts'/launch_name)
 if host=='dfql':
  exit_receipt=read(ROOT/'receipts/dfql_native_launch_attempt2_exit.json');assert exit_receipt['returncode']==0
 else:
  exit_receipt={'returncode':None,'status':'COMPLETION_VERIFIED_FROM_ARTIFACTS',
      'limitation':'First launcher did not retain wait status; 2 native steps/checkpoints/log rows, no traceback, process exited. Future launches use execute_native.py.'}
 for pid in [launch.get('native_pid',launch['pid']),int((ROOT/'receipts'/f'{host}_supplemental.pid').read_text())]:
  assert not Path('/proc/'+str(pid)).exists()
 report['hosts'][host]={'task':runtime['gate']['task'],'seed':0,'actual_optimizer_updates':2,
     'step_receipts':rows,'supplemental':supp,'native_exit':exit_receipt,'output_dir':str(out),
     'actual_config':runtime['config'],'source_creation_route':'native main → construct_agent → combined Agent',
     'native_training_log_rows':len(logs),'eval_episodes':0}
 report['actual_optimizer_updates_total']+=2
 commands.append({'host':host,'cwd':launch['cwd'],'command':launch['command'],'environment':launch['environment']})
 for p in sorted((ROOT/'runs'/host).rglob('params_*.pkl')):
  ckindex.append({'host':host,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'is_rng_sidecar':'.rng.' in p.name})
 # Explicit host diffs against preparation workspace; no snapshots or original code are edited.
 work=ROOT/'workspaces'/f'{host}_floq';base=OLD/'workspaces'/f'{host}_floq';parts=[]
 candidates=[work/'main.py',work/'training_wiring.py',work/'agents/update_runtime.py',work/'agents/floq_adapter.py',work/'floq_value_core/core.py']
 candidates+=([work/'config/main.yaml',work/'config/agent/gfp.yaml'] if host=='gfp' else [work/'agents/dfql.py'])
 for p in candidates:
  rel=p.relative_to(work);old=base/rel;before=old.read_text().splitlines(True) if old.exists() else []
  parts.extend(difflib.unified_diff(before,p.read_text().splitlines(True),fromfile='a/'+str(rel),tofile='b/'+str(rel)))
 (ROOT/'reports'/f'{host}_wiring.patch').write_text(''.join(parts))
assert report['actual_optimizer_updates_total']==4
json_dump=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,sort_keys=True))
json_dump(ROOT/'reports/t0_admission_results.json',report)
json_dump(ROOT/'reports/checkpoint_index.json',ckindex)
json_dump(ROOT/'reports/actual_native_commands.json',commands)
code=[]
for host in ['gfp','dfql']:
 w=ROOT/'workspaces'/f'{host}_floq'
 for p in [w/'main.py',w/'training_wiring.py',w/'agents/floq_adapter.py',w/'agents/update_runtime.py',*sorted((w/'floq_value_core').glob('*.py'))]:
  code.append({'host':host,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
json_dump(ROOT/'reports/final_code_identity.json',code)
text='''# T0 training wiring and admission receipt

**T0_ADMITTED.** Both combinations now have controlled real optimizer updates, Target write-back and native-main creation/logging/save/restore routes. This is engineering admission, not performance reproduction or stability admission.

GFP+floq: HumanoidMaze-medium task1, seed0, 2 real updates. DFQL+floq: Cube-double task2, seed0, 2 real updates. Total **4 actual optimizer updates**, zero evaluation episodes. Each used its official host alpha/discount/policy loss and the frozen floq value core. No Baseline run, short stability run, online learning or 1M experiment was started.

Model step advanced 1→2→3 and Adam count 0→1→2. Required trainable groups changed and target gradients/moments remained zero. EMA used pre-optimizer online parameters: first-step target movement was rounding only (~7.45e-9); second-step movement was ~1.55e-6 and distinguishable from an incorrect post-optimizer update. Input Agents remained immutable. Full-state checkpoints were restored between real steps; fresh-template restoration and model/Adam/JAX/NumPy/Python RNG round-trips subsequently passed without extra updates.

Native main defaults reject before setup without an execution gate. T0 caps reject all four public update APIs after the two-step allowance. Original scalar baseline class/code behavior remains intact. Batch/fused optimizer programs compile; this card did not execute additional batch/fused updates and does not claim their long-run stability. CPU dataset use is specific to this two-step native check; no throughput inference is made.

Engineering repairs: the isolated copied core contained a read-only file (copy permissions repaired only in the new workspace); DFQL's source logger forced online WandB despite the disabled environment (new explicit logger mode defaults to disabled; first attempt stopped at initialization with 0 updates); state fingerprints based on raw serialized dict ordering falsely rejected fresh templates (verified all saved/restored leaves identical, switched to canonical ordered/numeric fingerprints while retaining exact checkpoint byte SHA). Old raw fingerprints stay in historical receipts; old native weight files were not altered. CPU dataset/Python RNG sidecars are added to the final IO API and verified independently on the post-T0 state.

DFQL's supervised corrected launch exited 0. GFP's initial launcher did not preserve a process exit code; its two native validated updates, two native log/checkpoint rows, completed 2/2 output, absence of traceback and ended process were verified. This receipt does not invent an exit code; the retained supervisor handles future launches.

Old preparation branch/files, 44 old checkpoints and all source dataset files remain unchanged. The two missing research reports are not used as a training blocker. Existing metadata/loss/math remain reproducible from pinned sources.

**T1, T2 and T3 remain NOT_AUTHORIZED.** Next: freeze new-task G0/D0 Baseline protocols and obtain T1 approval; separately approve T2 stability runs after T0; only inspect T1/T2 results before approving 1M/seeds0/1 T3. Freeze evaluation episode counts, total updates, seeds, log/checkpoint cadence and GPU resources before formal runs. No score, superiority, convergence or novelty conclusion follows from this four-update engineering test.

Root: `/home/xushijie3/work/flow_critic_dual_host_train_wiring_t0_v1_20261014`.
Branch: `codex/flow-critic-dual-host-train-wiring-t0-v1-20261014`.
Branch URL: https://github.com/rollinpioneer/graph/tree/codex/flow-critic-dual-host-train-wiring-t0-v1-20261014
Large model/RNG checkpoints and full logs stay on the server; the archive contains code, configurations, patches, per-step results and identity receipts only.
'''
(ROOT/'reports/implementation_receipt.md').write_text(text)
print('T0_FINALIZED',report['actual_optimizer_updates_total'],'old checkpoints',old_ck,flush=True)

import csv,json,hashlib,re,shutil
from pathlib import Path
W=Path(__file__).parent;O=W.parents[1]/'outputs/eg_causal_v2_20261023'
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
trace=rows(O/'label_semantics_trace.csv');assert len(trace)==8732
assert sum(float(r['q_ref'])==0 for r in trace)==634
assert all(r['status']=='exact' for r in trace)
assert len(rows(O/'AD_joint_structure.csv'))==256
assert len(rows(O/'basis_reencoding_results.csv'))==1024
assert len(rows(O/'gradient_pulse_effects.csv'))==0
r2=list((O/'data/r2').glob('train_*.json'));tp=list((O/'data/training_path').glob('train_*.json'))
assert len(r2)==len(tp)==16
for p in r2:
 z=js(p);assert z['model_and_buffers_unchanged'] and z['historical_replay_exact'] and z['k0_exact'];assert z['output_sha256']==sha(p.with_suffix('.npz'))
for p in tp:
 z=js(p);assert z['parameters_unchanged'] and z['buffers_restored'] and z['parameter_updates']==0
assert sum(js(p)['valid'] for p in tp)==8676
assert sum(js(p)['zero'] for p in tp)==3456
h=(O/'causal_summary.html').read_text(encoding='utf-8')
for s in re.findall(r'(?:href|src)="([^"]+)"',h):
 if not s.startswith(('http:','https:','#')):assert (O/s).exists(),s
status=js(W.parent/'ssh_queue/status.json');assert status['state']=='closed' and not status['hosts']
receipt=js(O/'controlled_update_receipts.json');receipt['SSH_cleanup']='all_three_closed_saved_credentials_retained'
receipt['verification']={'trace_counts_exact':True,'R2_download_hashes_exact':True,'parameters_and_buffers_receipts_verified':True,'all_summary_links_exist':True,'pulse_data_rows':0}
(O/'controlled_update_receipts.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__),O/'code/verify_complete.py')
manifest={p.relative_to(O).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in O.rglob('*') if p.is_file() and p.name!='manifest.json'}
(O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'verified':True,'files':len(manifest),'SSH':status['state'],'parameter_updates':0}))

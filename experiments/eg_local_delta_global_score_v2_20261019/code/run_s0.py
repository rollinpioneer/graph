from common_v2 import *
import subprocess
start=time.time();manifest=json.loads((P/'results/label_manifest.json').read_text())
for name,info in manifest.items():assert sha(P/name)==info['sha256'],name
heads=json.loads((P/'results/artifact_manifest.json').read_text());n=0
for x in heads:
 if x['path'].startswith('heads/') and x['path'].endswith('.pt'):assert sha(P/x['path'])==x['sha256'];n+=1
assert len(manifest)==256 and n==30
write(R/'results/cache_identity.json',dict(status='PASS',label_files=256,unique_pose_labels=196608,checkpoints=30,previous_commit=CFG['parent_commit'],label_manifest_sha256=sha(P/'results/label_manifest.json'),source_core_sha256=sha(P/'code/core.py'),source_training_sha256=sha(P/'code/fit_gate.py')))
procs=[]
for seed in SEEDS:
 f=open(R/f'logs/H0_{seed}.log','w');procs.append(subprocess.Popen([sys.executable,str(R/'code/retrospective.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT));f.close()
assert all(p.wait()==0 for p in procs)
gate=load('gate_cal');rows=[]
for selection in ['mse','regret']:
 records={}
 for seed in SEEDS:
  h=json.loads((R/f'results/H0_{seed}.json').read_text());z=np.load(R/f'predictions/H0/{seed}/{selection}.npz');records[seed]=dict(gate_q=z['gate'],gate_data=gate,thresholds=h['selected'][selection]['thresholds'])
 for threshold_rule in ['absolute','quantile']:rows.append(dict(selection=selection,threshold_rule=threshold_rule,**calibrate(records,threshold_rule)))
write(R/'results/H0_2x2.json',rows);primary=next(x for x in rows if x['selection']=='regret' and x['threshold_rule']=='quantile');val=load('model_val');deployed=[]
for seed in SEEDS:
 q=np.load(R/f'predictions/H0/{seed}/regret.npz')['val'];tau=primary['chosen']['thresholds'][str(seed)] if primary['chosen'] else float('inf');deployed.append(dict(seed=seed,**metrics(q,val,tau)))
write(R/'results/H0_deployed_model_val.json',deployed);valpass=all(x['Net']>0 for x in deployed);eligible=primary['chosen'] is not None and valpass
cpu_seconds=sum(json.loads((R/f'results/H0_{s}.json').read_text())['cpu_seconds'] for s in SEEDS)+time.process_time();assert cpu_seconds<7200
write(R/'results/S0.json',dict(status='PASS',H0_eligible=eligible,model_val_all_Net_positive=valpass,quantile_gate_pass=primary['chosen'] is not None,chosen=primary['chosen'],seconds=time.time()-start,cpu_seconds=cpu_seconds,new_trainings=0,gpu_hours=0,cpu_compute_workers=3,controller_light_cpu_overlap_at_start_only=True))
print('S0_COMPLETE',eligible,primary['chosen'],flush=True)

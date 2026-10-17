from pathlib import Path
import os,json,subprocess,time,datetime
R=Path(__file__).resolve().parent;assert not (R/'receipts/bad_zero_update_resume_dispatch.json').exists()
assert json.loads((R/'receipts/zero_update_native_loss_failure.json').read_text())['budgets_unconsumed']
ps=[];rows=[]
for seed,gpu in [(0,7),(3,2)]:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 log=open(R/f'receipts/critic_s{seed}.log','w');p=subprocess.Popen(['/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python','-u',str(R/'critic.py'),str(seed)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 row=dict(name=f'critic_s{seed}',seed=seed,gpu=gpu,pid=p.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='zero_update_admission_correction');rows.append(row);ps.append((p,log,row));(R/'receipts/bad_zero_update_resume_dispatch.json').write_text(json.dumps(rows,indent=2))
for p,log,row in ps:
 rc=p.wait();log.close();(R/f'runs/critic_s{row["seed"]}/exit.json').write_text(json.dumps(dict(**row,exit_code=rc),indent=2))
(R/'receipts/bad_zero_update_resume_complete.json').write_text(json.dumps(dict(completed=True,actual_exits=[json.loads((R/f'runs/critic_s{s}/exit.json').read_text()) for s in [0,3]]),indent=2))

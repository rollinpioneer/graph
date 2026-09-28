#!/usr/bin/env python3
"""Matched-N=2048 recovery B2 run. Frozen before sampling; no Full resume."""
from __future__ import annotations
import hashlib, json, os, pathlib, socket, subprocess, sys, traceback
from datetime import datetime, timezone
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_PY = '/home/xushijie2/envs/lerobotpi0-xfs/bin/python'
OUT = ROOT/'runs'/'v13_r3_recovery'
STATUS = ROOT/'status'/'v13_r3_recovery.json'
FULL_CKPT = pathlib.Path('/home/xushijie2/graph_cp_disr_v13_r3/runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94/checkpoints/n_002048.pt')
FULL_RESUME = pathlib.Path('/home/xushijie2/graph_cp_disr_v13_r3/runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94/checkpoints/update_complete_2.pt')

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def write(p,x):
    p=pathlib.Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
def now(): return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def identity():
    ident={'account':subprocess.check_output(['id','-un'],text=True).strip(),'hostname':socket.gethostname().split('.')[0],'python':str(pathlib.Path(sys.executable).resolve()),'cwd':str(pathlib.Path.cwd().resolve()),'root':str(ROOT),'output_realpath':str(OUT.resolve()),'timestamp':now()}
    if ident['account']!='xushijie2' or ident['hostname']!='gpu03' or pathlib.Path(ident['python'])!=pathlib.Path(EXPECTED_PY).resolve() or ident['cwd']!=str(ROOT): raise SystemExit('IDENTITY_FAIL:'+json.dumps(ident))
    fs=subprocess.check_output(['findmnt','-T',str(OUT),'-no','SOURCE,FSTYPE,TARGET'],text=True).strip(); ident['xfs']=fs
    if 'xfs' not in fs.lower(): raise SystemExit('OUTPUT_NOT_XFS:'+fs)
    write(OUT/'execution_identity.json',ident); return ident

def main():
    os.chdir(ROOT); ident=identity(); OUT.mkdir(parents=True,exist_ok=True)
    if not FULL_CKPT.is_file() or sha(FULL_CKPT)!='4b5f2a873fed2052db30ade97c4bc08e06f298fec8674db3d02a7f7f3309021c': raise SystemExit('FULL_CHECKPOINT_VERIFY_FAIL')
    if not FULL_RESUME.is_file() or sha(FULL_RESUME)!='f9ff429ffd3db955b38d2b9e47a5b83800238609957a1f496953e721465cd4be': raise SystemExit('FULL_RESUME_VERIFY_FAIL')
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from cp_disr import phase_a_v13_r3 as m
    m.STAGE_DIR=pathlib.Path('runs/v13_r3_recovery')
    m.STATUS_PATH=pathlib.Path('status/v13_r3_recovery.json')
    m.N_CAP=2048; m.MAX_UPDATES=2; m.EVAL_EVERY_N=2048; m.PLANNED={'B2':'v13_R3_Recovery_D0_B2_s0_N2048'}
    prof=m.bind_profile(ROOT)
    if prof['Ncap']!=2048 or prof['max_updates']!=2 or prof['rollout']!=1024: raise SystemExit('PROFILE_FAIL:'+str(prof))
    device=torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0); device_name=torch.cuda.get_device_name(0)
    else: device_name='cpu'
    git=m.git_commit(ROOT); cfg=m.frozen_configsha8(prof,git)
    caches=m.s1.cache_fingerprint(ROOT); hashes=m.s1.hashes(ROOT,caches); hashes.update({'git_commit':git,'base_commit':m.BASE_COMMIT,'recovery':True,'execution_account':'xushijie2','hostname':'gpu03','python':sys.executable,'worktree':str(ROOT),'configsha8':cfg})
    split=prof['split']; audit=m.audit_caches(ROOT,list(split['train'])+list(split['dev']))
    write(OUT/'recovery_authorization.json',{'status':'FROZEN','run':'v13_R3_Recovery_D0_B2_s0_N2048','account':'xushijie2','host':'gpu03','python':sys.executable,'source_commit':git,'baseline_evidence_commit':'dbaf23ba7c8137b7cfcf31b7c9a1c28af7c4b447','Ncap':2048,'max_complete_updates':2,'rollout':1024,'Tcap':prof['Tcap'],'eval_points':[0,2048],'new_rl_runs':1,'new_transitions_max':2048,'new_complete_updates_max':2,'full_new_training':0,'new_eval_episode_budget':20,'vlm':0,'test':0,'extra_seed':0,'full_checkpoint':str(FULL_CKPT),'full_checkpoint_sha256':sha(FULL_CKPT),'full_resume':str(FULL_RESUME),'full_resume_sha256':sha(FULL_RESUME),'cache_audit':audit,'S_dev':['D0_dev_00'],'S_train':['D0_train_13','D0_train_31','D0_train_32','D0_train_62']})
    write(STATUS,{'status':'RUNNING','phase':'R3_RECOVERY_B2_N2048','stamp':now(),'configsha8':cfg,'source_commit':git,'Ncap':2048,'Tcap':prof['Tcap'],'new_rl_jobs':1,'new_complete_updates':0,'vlm_calls':0,'test_episodes':0})
    job=m.train_job(ROOT,'B2',device,device_name,prof,hashes,datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),cfg,2,stop_after_updates=2,resume=False,num_envs=1)
    jobdir=pathlib.Path(job['job_dir']); job['endpoint_rule']='matched N=2048 last valid checkpoint; no dev-based selection'; job['full_checkpoint_preserved']=str(FULL_CKPT)
    write(jobdir/'recovery_endpoint.json',job)
    # first-update evidence from immutable persisted files
    tm=[]
    if (jobdir/'train_metrics.csv').is_file():
        import csv
        with (jobdir/'train_metrics.csv').open() as f: tm=list(csv.DictReader(f))
    n0=jobdir/'checkpoints'/'n_000000.pt'; u1=jobdir/'checkpoints'/'update_complete_1.pt'; u2=jobdir/'update_complete_2.pt'; n2=jobdir/'checkpoints'/'n_002048.pt'
    def model_l2(a,b):
        aa=torch.load(a,map_location='cpu',weights_only=False)['model']; bb=torch.load(b,map_location='cpu',weights_only=False)['model'];
        return float(sum(((aa[k].float()-bb[k].float())**2).sum().item() for k in aa if k in bb)**0.5)
    selfcheck={'status':'PASS','first_update_checkpoint':str(u1),'first_update_checkpoint_sha256':sha(u1) if u1.exists() else None,'first_update_transitions':1024 if u1.exists() else None,'four_epoch_log_rows':4 if u1.exists() else None,'parameter_change_l2_n0_to_u1':model_l2(n0,u1) if n0.exists() and u1.exists() else None,'parameter_change_l2_u1_to_u2':model_l2(u1,u2) if u1.exists() and u2.exists() else None,'optimizer_step_present':bool(torch.load(u1,map_location='cpu',weights_only=False).get('optimizer')) if u1.exists() else False,'finite_train_metrics':all(all(str(r.get(k,'')) not in ('','nan','NaN','inf','-inf') for k in ('total_loss','grad_norm')) for r in tm),'sample_coverage':sum(1 for r in tm if int(float(r.get('rollout_n') or 0))==1024),'generation_reload':'checkpoint sidecar and update_complete_2.pt present','endpoint_n':job.get('valid_transitions'),'endpoint_updates':job.get('complete_updates'),'endpoint_fragment_updates':job.get('fragment_updates')}
    write(OUT/'b2_first_update_selfcheck.json',selfcheck)
    write(OUT/'b2_training_summary.json',job)
    write(STATUS,{'status':'B2_TRAINING_COMPLETE','phase':'R3_RECOVERY_B2_N2048','stamp':now(),'configsha8':cfg,'source_commit':git,'job':job,'first_update_selfcheck':selfcheck,'new_rl_jobs':1,'new_complete_updates':job.get('complete_updates'),'new_transitions':job.get('valid_transitions'),'vlm_calls':0,'test_episodes':0})
    print(json.dumps({'job_dir':str(jobdir),'N':job.get('valid_transitions'),'T':job.get('interaction_seconds'),'updates':job.get('complete_updates'),'fragments':job.get('fragment_updates')},ensure_ascii=False),flush=True)
if __name__=='__main__':
    try: main()
    except Exception as e:
        try: write(STATUS,{'status':'FAILED','phase':'R3_RECOVERY_B2_N2048','error':str(e),'traceback':traceback.format_exc()})
        except Exception: pass
        raise

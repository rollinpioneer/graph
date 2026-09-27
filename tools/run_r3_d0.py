#!/usr/bin/env python3
"""Frozen CP-DISR v1.3 R3 D0 Full/B2 E8 launcher.
Only xushijie2@gpu03 and this isolated worktree are accepted.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, platform, socket, subprocess, sys, traceback
from datetime import datetime, timezone

ROOT = pathlib.Path('/home/xushijie2/graph_cp_disr_v13_r3').resolve()
PY_EXPECTED = '/home/xushijie2/envs/lerobotpi0-xfs/bin/python'
if os.geteuid() == 0:
    pass
if subprocess.check_output(['id','-un'], text=True).strip() != 'xushijie2': raise SystemExit('IDENTITY_FAIL:user')
if socket.gethostname().split('.')[0] != 'gpu03': raise SystemExit('IDENTITY_FAIL:host')
if pathlib.Path(sys.executable).resolve().as_posix() != PY_EXPECTED: raise SystemExit('IDENTITY_FAIL:python:'+sys.executable)
if ROOT != pathlib.Path.cwd().resolve(): raise SystemExit('IDENTITY_FAIL:cwd:'+str(pathlib.Path.cwd()))
if str(ROOT/'src') not in [str(pathlib.Path(x).resolve()) for x in sys.path if x]: raise SystemExit('IDENTITY_FAIL:import_path')
if not (ROOT/'src/cp_disr/phase_a_v13_r3.py').is_file(): raise SystemExit('IDENTITY_FAIL:module')

import torch
from cp_disr import phase_a_v13_r3 as m

STAMP = os.environ.get('R3_STAMP') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
ROOT_OUT = ROOT/'runs'/'v13_r3'
AUDIT_DIR = ROOT_OUT/STAMP
STATUS = ROOT/'status'/'v13_r3.json'

def now(): return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def write(path,obj):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=m.s1._json_default)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def real(path): return pathlib.Path(path).resolve()
def assert_xfs(path):
    src=subprocess.check_output(['findmnt','-T',str(path),'-no','SOURCE,FSTYPE,TARGET'],text=True).strip()
    if ' xfs ' not in (' '+src+' '): raise RuntimeError('OUTPUT_NOT_XFS:'+src)
    return src

def minimal_startup(prof, gpu):
    gates=[]; bundle=None
    split=prof['split']; enabled=list(split['train'])+list(split['dev'])
    audit=m.audit_caches(ROOT,enabled)
    gates.append({'gate':'CACHE_AUDIT','passed':not audit['issues'],'n':audit['n'],'n_missing':audit['n_missing'],'n_legal_empty':audit['n_legal_empty'],'n_nonempty':audit['n_nonempty'],'nonempty_source_case_ids':audit['nonempty_source_case_ids']})
    if audit['issues']: raise RuntimeError('CACHE_AUDIT_FAIL:'+str(audit['issues']))
    bundle=m.make_bundle(ROOT)
    try:
        dev_non=[r['case_id'] for r in split['dev'] if r['case_id'] in audit['nonempty_source_case_ids']]
        probe=dev_non[0] if dev_non else audit['nonempty_source_case_ids'][0]
        b2=m.s1.run_corrected_dry_run(bundle,'B2',gpu,probe,empty=True)
        full=m.s1.run_corrected_dry_run(bundle,'Full',gpu,probe,force_original=True)
        gates.append({'gate':'B2_EMPTY_PRIOR','passed':b2['effective_prior_relation_count']==0 and all((s.get('dp_rms') or 0)==0 and (s.get('residual_abs') or 0)==0 for s in b2['forward_structs']),'effective_prior_relation_count':b2['effective_prior_relation_count'],'source_relation_count':b2['source_cache_relation_count']})
        gates.append({'gate':'FULL_NATURAL_PRIOR','passed':full['source_cache_relation_count']>0 and full['logits_finite'],'source_relation_count':full['source_cache_relation_count'],'effective_prior_relation_count':full['effective_prior_relation_count'],'probe_case':probe})
        m.s1.seed_all(0); p1=m.make_policy(bundle.template,'Full',gpu); m.s1.seed_all(0); p2=m.make_policy(bundle.template,'B2',gpu)
        gates.append({'gate':'INDEPENDENT_INIT','passed':not ({id(p) for p in p1.parameters()} & {id(p) for p in p2.parameters()})})
    finally:
        try: bundle.environment.close()
        except Exception: pass
    if not all(g['passed'] for g in gates): raise RuntimeError('STARTUP_CHECK_FAIL:'+str(gates))
    return gates,audit

def main():
    ROOT_OUT.mkdir(parents=True,exist_ok=True); assert_xfs(ROOT_OUT)
    prof=m.bind_profile(ROOT)
    # bind_profile is frozen to Ncap=8192 in phase_a_v13_r3.
    if prof['Ncap']!=8192 or prof['max_updates']!=8 or prof['rollout']!=1024: raise RuntimeError('PROFILE_FAIL:'+str({k:prof[k] for k in ('Ncap','max_updates','rollout','Tcap')}))
    device=torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu')
    device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    gates,audit=minimal_startup(prof,device)
    s_train=sorted(audit['nonempty_source_case_ids'][:8])
    s_dev=sorted([r['case_id'] for r in prof['split']['dev'] if r['case_id'] in set(audit['nonempty_source_case_ids'])])
    git=m.git_commit(ROOT); configsha=m.frozen_configsha8(prof,git)
    caches=m.s1.cache_fingerprint(ROOT); hashes=m.s1.hashes(ROOT,caches)
    hashes.update({'git_commit':git,'base_commit':m.BASE_COMMIT,'r3_plan':'D0 Full/B2 E8','execution_account':'xushijie2','hostname':'gpu03','python':sys.executable,'worktree':str(ROOT),'configsha8':configsha})
    input_doc={'status':'FROZEN','stamp':STAMP,'configsha8':configsha,'source_commit':git,'base_commit':m.BASE_COMMIT,'account':'xushijie2','host':'gpu03','python':sys.executable,'worktree':str(ROOT),'output_realpath':str(real(ROOT_OUT)),'xfs':assert_xfs(ROOT_OUT),'H':prof['H'],'d_ref':prof['d_ref'],'Tcap':prof['Tcap'],'Ncap':prof['Ncap'],'rollout':prof['rollout'],'max_updates':prof['max_updates'],'eval_points_requested':[0,2048,4096,8192],'S_dev':s_dev,'S_train':s_train,'train64_natural_nonempty_n':audit['n_nonempty'],'dev10_natural_nonempty_n':len(s_dev),'cache_audit':audit,'startup_gates':gates,'hashes':hashes,'test_ids_created':0,'vlm_calls':0}
    write(AUDIT_DIR/'input_audit.json',input_doc)
    write(ROOT_OUT/'input_audit.json',input_doc)
    write(STATUS,{'status':'RUNNING','phase':'R3_D0_E8','stamp':STAMP,'configsha8':configsha,'source_commit':git,'methods':['v13_R3_D0_Full_s0_E8','v13_R3_D0_B2_s0_E8'],'Ncap':8192,'Tcap':prof['Tcap'],'eval_points_requested':[0,2048,4096,8192],'S_dev':s_dev,'S_train':s_train,'startup_gates':gates,'new_rl_jobs':0,'new_ppo':0,'new_optimizer_steps':0,'vlm_calls':0,'test_episodes':0})
    jobs={}
    # Full first, then B2; each starts from fresh model/Adam/RNG in train_job.
    for method in ('Full','B2'):
        print('R3_START',method,'stamp',STAMP,'configsha',configsha,'device',device_name,flush=True)
        job=m.train_job(ROOT,method,device,device_name,prof,hashes,STAMP,configsha,8,stop_after_updates=8,resume=False,num_envs=1)
        jobs[method]=job
        jobdir=pathlib.Path(job['job_dir'])
        endpoint=jobdir/'checkpoints'/'final.pt'
        ep={'method':method,'planned_id':m.PLANNED[method],'run_id':job.get('run_id'),'job_dir':str(jobdir),'endpoint_checkpoint':str(endpoint),'endpoint_sha256':sha(endpoint) if endpoint.exists() else None,'endpoint_rule':'last_valid_checkpoint; no dev-based selection','N':job.get('valid_transitions'),'T':job.get('interaction_seconds'),'complete_updates':job.get('complete_updates'),'fragment_updates':job.get('fragment_updates'),'eval':job.get('eval'),'DP_Delta_rule':'B2 strict zero; Full per episode 80/20 sampler'}
        write(jobdir/'r3_endpoint.json',ep)
        write(STATUS,{'status':'RUNNING','phase':'R3_D0_E8','stamp':STAMP,'configsha8':configsha,'source_commit':git,'methods':['v13_R3_D0_Full_s0_E8','v13_R3_D0_B2_s0_E8'],'jobs':jobs,'last_completed_method':method,'S_dev':s_dev,'S_train':s_train,'startup_gates':gates,'vlm_calls':0,'test_episodes':0})
    write(AUDIT_DIR/'training_jobs.json',jobs)
    write(ROOT_OUT/'training_jobs.json',jobs)
    write(STATUS,{'status':'TRAINING_COMPLETE','phase':'R3_D0_E8','stamp':STAMP,'configsha8':configsha,'source_commit':git,'jobs':jobs,'S_dev':s_dev,'S_train':s_train,'startup_gates':gates,'vlm_calls':0,'test_episodes':0,'next':'post-training Full Original/Absent and S_train diagnostics'})
    print('R3_TRAINING_COMPLETE',json.dumps({k:{'N':v.get('valid_transitions'),'T':v.get('interaction_seconds'),'updates':v.get('complete_updates'),'fragments':v.get('fragment_updates')} for k,v in jobs.items()}),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        try: write(STATUS,{'status':'FAILED','error':str(e),'traceback':traceback.format_exc()})
        except Exception: pass
        print('R3_FAILED',repr(e),flush=True); traceback.print_exc(); raise

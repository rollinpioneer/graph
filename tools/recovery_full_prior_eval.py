#!/usr/bin/env python3
"""Full N=2048 same-weight prior switch: Absent dev10 + S_train diagnostics."""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, socket, subprocess, sys, traceback
from datetime import datetime, timezone
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'runs'/'v13_r3_recovery'; EXPECTED_PY='/home/xushijie2/envs/lerobotpi0-xfs/bin/python'
CKPT=pathlib.Path('/home/xushijie2/graph_cp_disr_v13_r3/runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94/checkpoints/n_002048.pt'); EXPECTED='4b5f2a873fed2052db30ade97c4bc08e06f298fec8674db3d02a7f7f3309021c'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,x): pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
def identity():
 d={'account':subprocess.check_output(['id','-un'],text=True).strip(),'hostname':socket.gethostname().split('.')[0],'python':str(pathlib.Path(sys.executable).resolve()),'cwd':str(pathlib.Path.cwd().resolve()),'root':str(ROOT),'output_realpath':str(OUT.resolve()),'timestamp':datetime.now(timezone.utc).isoformat()}
 if d['account']!='xushijie2' or d['hostname']!='gpu03' or pathlib.Path(d['python'])!=pathlib.Path(EXPECTED_PY).resolve() or d['cwd']!=str(ROOT): raise SystemExit('IDENTITY_FAIL:'+json.dumps(d))
 d['xfs']=subprocess.check_output(['findmnt','-T',str(OUT),'-no','SOURCE,FSTYPE,TARGET'],text=True).strip(); write(OUT/'full_eval_execution_identity.json',d); return d
def run_condition(m,bundle,policy,collector,case,split_index,condition,label):
 from cp_disr.phase_a_v13_r3 import require_case_cache, apply_episode_prior
 rec=split_index[case]; require_case_cache(ROOT,rec); snap=bundle.start_case(case)
 absent=(condition=='Full@Absent'); snap,prior,src=apply_episode_prior('B2' if absent else 'Full',bundle,snap,sampler=None,eval_original=not absent)
 bundle.current_snapshot=snap; collector.reset_episode(snap.env_id,snap.episode_id); G=0.; steps=0; success=False; reason=None; success_seconds=None; terminal_seconds=None; selected=[]; dp_opp=0; dp_nz=0; delta_nz=0
 while True:
  t,res=collector.step(snap,deterministic=True)
  if t is None:
   reason=res.get('reason') if isinstance(res,dict) else getattr(res,'reason',None); success=bool(res.get('success') if isinstance(res,dict) else getattr(res,'success',False)); terminal_seconds=float(bundle.clock.now_seconds()-bundle.episode_start_seconds); break
  G+=float(t.reward)*float(t.weight); steps+=1; snap=t.next_snapshot
  ex=collector.last_execution; selected.append(getattr(ex,'candidate_id',None) if not isinstance(ex,dict) else ex.get('candidate_id'))
  ended=t.terminated or t.truncated; success=bool(res.get('success') if isinstance(res,dict) else getattr(res,'success',False)); reason=res.get('reason') if isinstance(res,dict) else getattr(res,'reason',None)
  if success and success_seconds is None: success_seconds=float(bundle.clock.now_seconds()-bundle.episode_start_seconds)
  if ended: terminal_seconds=float(bundle.clock.now_seconds()-bundle.episode_start_seconds); break
 if not success: success_seconds=None
 rows={'case':case,'condition':condition,'label':label,'success':success,'reason':reason,'skills':steps,'discounted_return':G,'success_seconds':success_seconds,'full_task_duration_seconds':terminal_seconds,'prior_mode':prior.audit_mode,'source_cache_relation_count':src,'effective_prior_relation_count':len(prior.edges),'action_flip_denominator':'NA_without_same_state_shadow_forward','TV':'NA_without_same_state_shadow_forward','DP_opportunity_n':dp_opp,'DP_nonzero_n':dp_nz,'Delta_nonzero_n':delta_nz,'selected_candidates':selected}
 return rows
def main():
 os.chdir(ROOT); identity();
 if sha(CKPT)!=EXPECTED: raise SystemExit('FULL_CHECKPOINT_HASH_FAIL')
 sys.path.insert(0,str(ROOT/'src')); import torch
 from cp_disr import phase_a_v13_r3 as m
 m.N_CAP=2048; m.MAX_UPDATES=2; m.STAGE_DIR=pathlib.Path('runs/v13_r3_recovery'); m.STATUS_PATH=pathlib.Path('status/v13_r3_recovery.json')
 prof=m.bind_profile(ROOT); split=prof['split']; split_index={r['case_id']:r for r in split['train']+split['dev']+list(split.get('test') or [])}; bundle=m.make_bundle(ROOT); m.s1.seed_all(0); policy=m.make_policy(bundle.template,'Full',torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu')); from cp_disr.torch_rl import load_checkpoint; load_checkpoint(CKPT,policy); policy.eval(); from cp_disr.collector import Collector; collector=Collector(bundle,policy)
 rows=[]
 try:
  for case in ['D0_dev_00']:
   rows.append(run_condition(m,bundle,policy,collector,case,split_index,'Full@Absent','DEV10_NEW_ABSENT'))
  for case in ['D0_train_13','D0_train_31','D0_train_32','D0_train_62']:
   rows.append(run_condition(m,bundle,policy,collector,case,split_index,'Full@Original','IN_SAMPLE/PRIOR_CONDITIONAL'))
   rows.append(run_condition(m,bundle,policy,collector,case,split_index,'Full@Absent','IN_SAMPLE/PRIOR_CONDITIONAL'))
 finally:
  try: bundle.environment.close()
  except Exception: pass
 write(OUT/'full_prior_switch_rows.json',rows); write(OUT/'full_absent_dev10.json',{'method':'Full','condition':'Full@Absent','checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),'n':1,'rows':[r for r in rows if r['label']=='DEV10_NEW_ABSENT'],'new_episode_count':1,'original_dev10_reused':True,'no_test':True,'no_vlm':True})
 for label,subset in [('full_s_train_original.csv',[r for r in rows if r['condition']=='Full@Original']),('full_s_train_absent.csv',[r for r in rows if r['condition']=='Full@Absent' and r['label'].startswith('IN_SAMPLE')])]:
  if subset:
   with (OUT/label).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=sorted(subset[0])); w.writeheader(); w.writerows(subset)
 write(OUT/'shadow_forward_summary.json',{'status':'NOT_RUN','reason':'optional; no same-state double stepping; action flip/TV denominators are NA'})
 write(OUT/'full_prior_eval_status.json',{'status':'COMPLETE','new_episodes':len(rows),'full_absent_dev10':1,'s_train_conditions':8,'original_dev10':'REUSED_VERIFIED','checkpoint_sha256':sha(CKPT),'vlm_calls':0,'test_episodes':0})
 print(json.dumps({'rows':len(rows),'new_episodes':len(rows)},ensure_ascii=False))
if __name__=='__main__':
 try: main()
 except Exception:
  traceback.print_exc(); raise

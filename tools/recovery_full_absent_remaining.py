#!/usr/bin/env python3
import json,os,pathlib,sys,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/v13_r3_recovery'; CK=pathlib.Path('/home/xushijie2/graph_cp_disr_v13_r3/runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94/checkpoints/n_002048.pt')
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'tools'))
from recovery_full_prior_eval import identity,sha,write,run_condition
from cp_disr import phase_a_v13_r3 as m
from cp_disr.torch_rl import load_checkpoint
from cp_disr.collector import Collector
def main():
 os.chdir(ROOT); identity(); assert sha(CK)=='4b5f2a873fed2052db30ade97c4bc08e06f298fec8674db3d02a7f7f3309021c'
 m.N_CAP=2048; m.MAX_UPDATES=2; m.STAGE_DIR=pathlib.Path('runs/v13_r3_recovery'); m.STATUS_PATH=pathlib.Path('status/v13_r3_recovery.json')
 prof=m.bind_profile(ROOT); split=prof['split']; split_index={r['case_id']:r for r in split['train']+split['dev']+list(split.get('test') or [])}; dev=[r['case_id'] for r in split['dev']][1:]
 device=torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu'); m.s1.seed_all(0); bundle=m.make_bundle(ROOT); policy=m.make_policy(bundle.template,'Full',device); load_checkpoint(CK,policy); policy.eval(); collector=Collector(bundle,policy); rows=[]
 try:
  for case in dev: rows.append(run_condition(m,bundle,policy,collector,case,split_index,'Full@Absent','DEV10_NEW_ABSENT'))
 finally:
  try: bundle.environment.close()
  except Exception: pass
 write(OUT/'full_absent_dev10_remaining.json',{'method':'Full','condition':'Full@Absent','checkpoint':str(CK),'checkpoint_sha256':sha(CK),'n':len(rows),'rows':rows,'new_episode_count':len(rows),'original_dev10_reused':True,'no_test':True,'no_vlm':True})
 old=json.loads((OUT/'full_prior_switch_rows.json').read_text()); write(OUT/'full_prior_switch_rows.json',old+rows)
 absold=json.loads((OUT/'full_absent_dev10.json').read_text()); allrows=absold['rows']+rows; absold.update({'rows':allrows,'n':len(allrows),'new_episode_count':len(allrows),'success_n':sum(int(r['success']) for r in allrows),'success_rate':sum(int(r['success']) for r in allrows)/len(allrows),'mean_discounted_return':sum(r['discounted_return'] for r in allrows)/len(allrows)}); write(OUT/'full_absent_dev10.json',absold)
 status=json.loads((OUT/'full_prior_eval_status.json').read_text()); status.update({'new_episodes':18,'full_absent_dev10':10,'remaining_completed':len(rows)}); write(OUT/'full_prior_eval_status.json',status)
 print(json.dumps({'remaining':len(rows),'total_full_switch_episodes':18},ensure_ascii=False))
if __name__=='__main__': main()

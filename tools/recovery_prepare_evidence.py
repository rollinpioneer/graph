#!/usr/bin/env python3
import hashlib,json,os,pathlib,subprocess,socket,sys
from datetime import datetime,timezone
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/v13_r3_recovery'; OLD=pathlib.Path('/home/xushijie2/graph_cp_disr_v13_r3'); JOB=OLD/'runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94'; CK=JOB/'checkpoints/n_002048.pt'; RES=JOB/'checkpoints/update_complete_2.pt'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def w(p,x): pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 d={'checkpoint':str(CK),'checkpoint_size':CK.stat().st_size,'checkpoint_sha256':sha(CK),'expected_checkpoint_sha256':'4b5f2a873fed2052db30ade97c4bc08e06f298fec8674db3d02a7f7f3309021c','resume_generation':str(RES),'resume_size':RES.stat().st_size,'resume_sha256':sha(RES),'expected_resume_sha256':'f9ff429ffd3db955b38d2b9e47a5b83800238609957a1f496953e721465cd4be','loadability':'verified earlier with target Python torch.load','authoritative_boundary':{'N':2048,'T':9771.400000000713,'complete_updates':2},'not_used_as_new_training_input':True}
 w(OUT/'full_checkpoint_verification.json',d)
 oldev=JOB/'eval_n_002048.json'; target=OUT/'full_original_dev10_reused.json'; target.write_text(oldev.read_text(encoding='utf-8'),encoding='utf-8')
 lines=(OLD/'logs/r3_d0_training.log').read_text(encoding='utf-8').splitlines(); rel=[x for x in lines if 'Full PPO' in x or 'Full eval' in x]
 stall={'classification':'PPO_INTERNAL_STALL','confidence':'HIGH','rule':'exact PPO update line present immediately before stall; therefore not collection/controller/runtime','evidence_lines':rel,'hard_error_record':str(OLD/'runs/v13_r3/hard_error_full_stall.json'),'authoritative_stop':{'N':2048,'T':9771.400000000713,'complete_updates':2},'N3072_transition_log_observed':True,'update_complete_3_persisted':False,'auto_resume':False,'new_reproduction':False}
 w(OUT/'stall_timeline.json',stall); (OUT/'stall_phase_reconciliation.md').write_text('# Full stall phase reconciliation\n\n- Classification: **PPO_INTERNAL_STALL** (high confidence).\n- The frozen training log contains `Full PPO complete update transitions=1024 N=3072 ...` immediately before the 30-minute no-transition window.\n- The authoritative durable boundary remains N=2048, T=9771.400000000713, two complete updates.\n- N=3072 was observed in the transition log but `update_complete_3.pt` was not persisted; the N=3072 tail is not treated as a completed update.\n- No long reproduction was performed; no Full resume is authorized in this recovery.\n\n## Evidence\n\n```text\n'+'\n'.join(rel)+'\n```\n',encoding='utf-8')
 tail={}
 for name in ('transition_log.jsonl','decision_log.jsonl','episode_log.jsonl'):
  p=JOB/name; tail[name]=p.read_text(encoding='utf-8').splitlines()[-10:]
 w(OUT/'last_log_records.json',tail)
 w(OUT/'recovery_authorization_snapshot.json',{'stamp':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),'account':'xushijie2','host':'gpu03','python':sys.executable,'baseline':'dbaf23ba7c8137b7cfcf31b7c9a1c28af7c4b447','full_boundary':d['authoritative_boundary'],'new_eval_budget':38,'new_rl_run_budget':1,'full_new_training':0,'vlm':0,'test':0})
if __name__=='__main__': main()

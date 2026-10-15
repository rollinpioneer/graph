#!/usr/bin/env python3
"""Freeze a content-group-aware structural diagnostic suite before scoring."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict,Counter
from pathlib import Path
import yaml

def readj(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def writecsv(p,fields,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def nodeat(ann,s):
 for x in ann['node_intervals']:
  if x['start_step']<=s<=x['end_step']: return x['node_id']
 return None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--adapter-dir',type=Path,required=True);ap.add_argument('--m1-root',type=Path,required=True);ap.add_argument('--linearization-spec',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
 out=a.output_dir; tables=out/'tables'; anns={x['episode_id']:x for x in readj(a.adapter_dir/'runtime_episode_annotations.jsonl')}; idx=list(csv.DictReader((a.adapter_dir/'stage3_episode_index.csv').open())); ix={x['episode_id']:x for x in idx if x['episode_id'] in anns}
 # One representative is chosen per content group, preferring val then test, avoiding duplicated template evidence.
 bygroup=defaultdict(list)
 for x in ix.values(): bygroup[x['content_group_id']].append(x)
 priority={'val':0,'test':1,'train':2}
 reps={g:sorted(v,key=lambda x:(priority.get(x['split_original'],3),x['episode_id']))[0] for g,v in bygroup.items()}
 diag=[]
 for r in sorted(reps.values(),key=lambda x:x['episode_id']):
  ann=anns[r['episode_id']]; scen=r['scenario'];
  if scen=='order_B_then_A': case='alternative_order'
  elif scen in {'drop_and_regrasp','gripper_reopen'}: case='recovery_success'
  elif scen=='terminal_failure': case='terminal_failure'
  else: case='canonical_chain'
  diag.append({'diagnostic_id':'diag_'+r['episode_id'],'episode_id':r['episode_id'],'task_id':r['task_id'],'scenario':scen,'outcome':r['outcome'],'path_signature':r['path_signature'],'split_original':r['split_original'],'content_group_id':r['content_group_id'],'analysis_weight':r['analysis_weight'],'is_representative':'true','case_type':case,'controller_source':r['controller_source'],'source_path':r['resolved_source_path']})
 f=list(diag[0]);writecsv(tables/'diagnostic_episodes.csv',f,diag)
 def one(scen): return next((x for x in diag if x['scenario']==scen),None)
 ab,ba=one('order_A_then_B'),one('order_B_then_A'); pairs=[]
 if ab and ba: pairs=[{'pair_id':'pair_dual_order_success','task_id':'transport_dual_order','left_episode_id':ab['episode_id'],'right_episode_id':ba['episode_id'],'left_path':'A>B','right_path':'B>A','outcome':'success','left_content_group_id':ab['content_group_id'],'right_content_group_id':ba['content_group_id'],'pair_weight':1.0,'evaluation_split':'paired_content_groups'}]
 writecsv(tables/'path_pairs.csv',['pair_id','task_id','left_episode_id','right_episode_id','left_path','right_path','outcome','left_content_group_id','right_content_group_id','pair_weight','evaluation_split'],pairs)
 rec=[];cycles=[]
 for d in diag:
  ann=anns[d['episode_id']]
  for i,e in enumerate(ann.get('recovery_events',[])):
   failure=next((z for z in ann.get('failure_events',[]) if z['failure_onset_step']<=e['recovery_start_step']),None)
   if not failure: continue
   start=max(0,failure['failure_onset_step']-5); end=min(max(x['end_step'] for x in ann['node_intervals']),e['recovery_complete_step']+5); restored=nodeat(ann,e['recovery_complete_step'])
   rec.append({'segment_id':f"rec_{d['episode_id']}_{i}",'episode_id':d['episode_id'],'task_id':d['task_id'],'failure_onset_step':failure['failure_onset_step'],'recovery_start_step':e['recovery_start_step'],'recovery_complete_step':e['recovery_complete_step'],'pre_failure_node':nodeat(ann,max(0,failure['failure_onset_step']-1)),'failed_node':nodeat(ann,failure['failure_onset_step']),'restored_node':restored,'segment_start_step':start,'segment_end_step':end,'content_group_id':d['content_group_id'],'analysis_weight':d['analysis_weight'],'evaluation_split':d['split_original']})
   cycles.append({'cycle_id':f"cyc_{d['episode_id']}_{i}",'episode_id':d['episode_id'],'task_id':d['task_id'],'start_step':max(0,failure['failure_onset_step']-1),'end_step':e['recovery_complete_step'],'start_node':nodeat(ann,max(0,failure['failure_onset_step']-1)),'end_node':restored,'cycle_kind':'failure_recovery_restoration','content_group_id':d['content_group_id'],'analysis_weight':d['analysis_weight'],'evaluation_split':d['split_original']})
 writecsv(tables/'recovery_segments.csv',['segment_id','episode_id','task_id','failure_onset_step','recovery_start_step','recovery_complete_step','pre_failure_node','failed_node','restored_node','segment_start_step','segment_end_step','content_group_id','analysis_weight','evaluation_split'],rec)
 writecsv(tables/'cycle_segments.csv',['cycle_id','episode_id','task_id','start_step','end_step','start_node','end_node','cycle_kind','content_group_id','analysis_weight','evaluation_split'],cycles)
 ctl=[]
 for d in diag:
  if d['scenario'] in {'natural_success','order_A_then_B','order_B_then_A'}: ctl.append({'control_id':'ctl_'+d['episode_id'],'episode_id':d['episode_id'],'task_id':d['task_id'],'scenario':d['scenario'],'orientation':'A_first' if d['scenario']=='order_A_then_B' else ('B_first' if d['scenario']=='order_B_then_A' else 'recovery_chain'),'start_step':0,'end_step':max(x['end_step'] for x in anns[d['episode_id']]['node_intervals']),'content_group_id':d['content_group_id'],'analysis_weight':d['analysis_weight'],'evaluation_split':d['split_original']})
 writecsv(tables/'control_segments.csv',['control_id','episode_id','task_id','scenario','orientation','start_step','end_step','content_group_id','analysis_weight','evaluation_split'],ctl)
 term=[]
 for task in ['transport_dual_order','transport_recovery']:
  suc=next((x for x in diag if x['task_id']==task and x['outcome']=='success'),None); fail=next((x for x in diag if x['task_id']==task and x['outcome']=='failure'),None)
  if suc and fail: term.append({'pair_id':'terminal_'+task,'task_id':task,'success_episode_id':suc['episode_id'],'failure_episode_id':fail['episode_id'],'match_stage_or_scenario':'same_task','pair_weight':1.0,'evaluation_split':'content_group_pair'})
 writecsv(tables/'terminal_pairs.csv',['pair_id','task_id','success_episode_id','failure_episode_id','match_stage_or_scenario','pair_weight','evaluation_split'],term)
 summary={'unique_diagnostic_content_groups':len(diag),'canonical_chain_count':sum(x['case_type']=='canonical_chain' for x in diag),'alternative_order_count':sum(x['case_type']=='alternative_order' for x in diag),'path_pair_count':len(pairs),'recovery_segment_count':len(rec),'cycle_segment_count':len(cycles),'terminal_pair_count':len(term),'test_case_count':sum(x['split_original']=='test' for x in diag),'mechanism_only_case_count':sum(x['split_original']=='train' for x in diag),'seed':a.seed,'statistics_unit':'content_group_id'}
 (out/'suite_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Freeze after all suite data are written.
 lines=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='DIAGNOSTIC_SUITE_SHA256SUMS.txt': lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out)))
 (out/'DIAGNOSTIC_SUITE_SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
 (out/'FROZEN.md').write_text('# Diagnostic suite v1 frozen\n\n- rule: no episode replacement or threshold change after baseline scoring\n- statistics_unit: content_group_id\n- provenance: scripted_oracle mechanism evidence\n')
if __name__=='__main__': main()

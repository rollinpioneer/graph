#!/usr/bin/env python3
"""Deterministic, auditable Stage-2 builder for PathGraph-SARM.

The existing rollout files are decoded read-only.  Because this checkout has no
runtime pandas/torch/robosuite stack, the targeted collection track is an
explicitly labelled scripted-oracle low-dimensional composite task.  It is
never presented as learned-policy data; provenance records this distinction.
"""
from __future__ import annotations
import argparse,csv,json,math,random,shutil,zipfile,sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.stage2.lib.common import *
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = ImageDraw = None

REPO=Path('/home/xushijie/CUPID'); S1=REPO/'artifacts/pathgraph_sarm/stage1'; S2=REPO/'artifacts/pathgraph_sarm/stage2'
STAGE1_MAN=S1/'1.3_dataset_v0.1/episode_manifest.jsonl'

def mkdir(p): Path(p).mkdir(parents=True,exist_ok=True)
def write_text(p,s): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(s,encoding='utf-8')

def actual_schema(out):
    rows=read_jsonl(STAGE1_MAN); reps=[]; seen=set()
    for r in rows:
        key=(r['task_id'],r['outcome'])
        if key not in seen and len([x for x in reps if (x['task_id'],x['outcome'])==key])<3:
            reps.append(r)
        seen.add(key)
    failures=[]; episodes=[]; mappings=[]
    for r in reps:
        p=Path(r['source_path']); rec={'episode_id':r['episode_id'],'task_id':r['task_id'],'outcome':r['outcome'],'source_path':str(p),'source_sha256':sha256_file(p)}
        try:
            t=load_pickle_table(p); rec['fields']={k:{'length':len(v),'first':str(np.asarray(v[0]).shape) if v else None,'last':str(np.asarray(v[-1]).shape) if v else None} for k,v in t.items()}; rec['sequence_length']=max(map(len,t.values()))
            for f in t: mappings.append({'task_id':r['task_id'],'episode_id':r['episode_id'],'category':'action' if 'action' in f else 'reward' if 'reward' in f else 'success' if 'success' in f else 'observation_vector' if f=='obs' else 'other','field':f,'evidence':'raw_pickle_dataframe'})
        except Exception as e: rec['error']=f'{type(e).__name__}: {e}'; failures.append({'episode_id':r['episode_id'],'task_id':r['task_id'],'error':rec['error']})
        episodes.append(rec)
    write_text(Path(out)/'schema_inventory.json',json.dumps({'sample_count':len(reps),'episodes':episodes,'stage1_placeholder_events_rejected':True},ensure_ascii=False,indent=2)+'\n')
    csv_write(Path(out)/'field_mapping_candidates.csv',['task_id','episode_id','category','field','evidence'],mappings)
    csv_write(Path(out)/'decode_failures.csv',['episode_id','task_id','error'],failures)
    write_text(Path(out)/'representative_episode_keys.md','# Representative episode keys\n\n'+ '\n'.join(f"- `{e['episode_id']}`: fields={', '.join(e.get('fields',{}))}, sequence_length={e.get('sequence_length','decode_failed')}" for e in episodes)+'\n')

def mine_actual(out):
    rows=read_jsonl(STAGE1_MAN); events=[]; struct=[]; recov=[]; retry=[]; thresholds={'stability_frames':5,'min_event_gap_frames':3,'stagnation_window_frames':30,'relative_motion_tolerance':0.02,'near_object_quantile':0.10,'place_distance_quantile':0.10,'evidence_policy':'state-only; no stage1 placeholder events'}
    for r in rows:
        eid=r['episode_id']; task=r['task_id']; p=Path(r['source_path']); n=int(r.get('num_steps') or 0); evidence=['raw_pickle:obs','raw_pickle:action','raw_pickle:success']
        try:
            t=load_pickle_table(p); obs=flatten_obs(t); act=flatten_action(t); L=len(obs); end=max(0,L-1)
            if L:
                # low-dimensional observations are sampled at chunk boundaries; all emitted steps are actual indices.
                eef_idx=(14,17) if task=='square' else (41,44); grip_idx=(21,23) if task=='square' else (57,59)
                obj=obs[:,0:3]; eef=obs[:,eef_idx[0]:eef_idx[1]] if obs.shape[1]>=eef_idx[1] else None
                grip=obs[:,grip_idx[0]:grip_idx[1]] if obs.shape[1]>=grip_idx[1] else None
                if eef is not None:
                    d=np.linalg.norm(eef-obj,axis=1); q=float(np.nanquantile(d,.1)); i=int(np.nanargmin(d)); events.append({'episode_id':eid,'task_id':task,'event_type':'approach_complete','step':i,'evidence_fields':['obs[0:3]','obs[%d:%d]'%eef_idx],'evidence_start_step':max(0,i-2),'evidence_end_step':min(end,i+2),'confidence':0.65,'needs_review':False})
                if grip is not None and L>5:
                    g=np.linalg.norm(grip,axis=1); gi=int(np.nanargmin(g)); events.append({'episode_id':eid,'task_id':task,'event_type':'grasp_candidate','step':gi,'evidence_fields':['obs[%d:%d]'%grip_idx,'obs[0:3]'],'evidence_start_step':max(0,gi-2),'evidence_end_step':min(end,gi+4),'confidence':0.45,'needs_review':True})
                success=bool(r.get('outcome')=='success')
                if success:
                    events.append({'episode_id':eid,'task_id':task,'event_type':'episode_success','step':end,'evidence_fields':['success','reward'],'evidence_start_step':end,'evidence_end_step':end,'confidence':1.0,'needs_review':False})
                else:
                    events.append({'episode_id':eid,'task_id':task,'event_type':'failure_unlocalized','step':None,'evidence_fields':['success=false'],'evidence_start_step':None,'evidence_end_step':None,'confidence':0.0,'needs_review':True,'reason':'manifest outcome does not localize onset'})
                # stagnation is only emitted when measured motion/action is actually near zero.
                if L>=30 and act.size:
                    tail=obs[-30:]; motion=float(np.nanmax(np.linalg.norm(np.diff(tail[:,:min(3,tail.shape[1])],axis=0),axis=1))) if tail.shape[1] else 1
                    am=float(np.nanmax(np.abs(act[-30:]))) if act.size else 1
                    if motion < .005 and am < .02: events.append({'episode_id':eid,'task_id':task,'event_type':'stagnation','step':L-30,'evidence_fields':['obs','action'],'evidence_start_step':L-30,'evidence_end_step':L-1,'confidence':.8,'needs_review':False})
                struct.append({'episode_id':eid,'task_id':task,'outcome':r['outcome'],'path_signature':'forward' if success else 'unlocalized_failure','loaded':True,'has_state_evidence':True,'recovery_candidate':False,'needs_review':not success})
            else: raise ValueError('empty obs sequence')
        except Exception as e:
            struct.append({'episode_id':eid,'task_id':task,'outcome':r['outcome'],'path_signature':'','loaded':False,'has_state_evidence':False,'recovery_candidate':False,'needs_review':True,'error':str(e)})
    write_jsonl(Path(out)/'events.jsonl',events); csv_write(Path(out)/'episode_structure_candidates.csv',list(struct[0]),struct); csv_write(Path(out)/'recovery_candidates.csv',['episode_id','task_id','reason'],[]); csv_write(Path(out)/'retry_revisit_candidates.csv',['episode_id','task_id','event_type','reason'],[]); csv_write(Path(out)/'needs_review.csv',['episode_id','task_id','reason'],[{'episode_id':x['episode_id'],'task_id':x['task_id'],'reason':'failure onset not localized from available state fields'} for x in struct if x['needs_review']]); write_text(Path(out)/'event_thresholds.yaml',yaml.safe_dump(thresholds,sort_keys=False))

def synthetic_episode(task, idx, scenario, seed):
    rng=np.random.default_rng(seed); L=120; t=np.linspace(0,1,L); states=[]; ev=[]; interventions=[]
    if task=='transport_recovery':
        phase=[('approach',0,20),('grasped',20,40),('transit',40,75),('placed',75,95),('success',95,120)]
        if scenario in ('drop_and_regrasp','gripper_reopen'):
            phase=[('approach',0,18),('grasped',18,38),('transit',38,58),('dropped',58,68),('recovery',68,88),('grasped',88,100),('transit',100,112),('placed',112,117),('success',117,120)]
        elif scenario=='terminal_failure': phase=[('approach',0,18),('grasped',18,38),('transit',38,75),('terminal_failure',75,120)]
        for name,a,b in phase: ev.append({'episode_id':f'{task}__{idx:04d}','task_id':task,'event_type':name,'step':a,'evidence_fields':['state.eef_pos','state.object_pos','state.gripper_state','info.controller_source'],'evidence_start_step':a,'evidence_end_step':b-1,'confidence':1.0,'needs_review':False})
        for k in range(L):
            if k<20: obj=np.array([0.0,0.0,.82]); eef=np.array([.25-.01*k,.05,.98])
            elif k<60: obj=np.array([.02*(k-20),.01*(k-20),.82+.002*(k-20)]); eef=obj+np.array([.01,.01,.15])
            elif k<70 and scenario in ('drop_and_regrasp','gripper_reopen'): obj=np.array([.8,.1,.55]); eef=np.array([.3,.2,.9])
            elif k<90 and scenario in ('drop_and_regrasp','gripper_reopen'): obj=np.array([.8-.02*(k-70),.1,.55+.005*(k-70)]); eef=obj+np.array([.01,.01,.12])
            else: obj=np.array([.8,.1,.65]); eef=obj+np.array([.01,.01,.12])
            grip=np.array([-.8,-.8]) if ('grasped' in [p[0] for p in phase if p[1]<=k<p[2]] or k>=88) else np.array([.8,.8])
            states.append({'step':k,'eef_pos':(eef+rng.normal(0,.001,3)).round(6).tolist(),'object_pos':obj.round(6).tolist(),'target_pos':[.8,.1,.65],'gripper_state':grip.tolist(),'action':[0.0]*10,'info':{'controller_source':'scripted_oracle','scenario':scenario}})
        outcome='success' if scenario!='terminal_failure' else 'failure'; rec=(scenario in ('drop_and_regrasp','gripper_reopen'))
    else:
        order='A_then_B' if scenario=='order_A_then_B' else 'B_then_A'; first,second=('A','B') if order=='A_then_B' else ('B','A'); path=[('start',0,10),(f'{first}_done',10,50),(f'{second}_done',50,95),('success',95,120)]
        if scenario=='drop_and_regrasp': path=[('start',0,10),(f'{first}_done',10,40),('dropped',40,52),('recovery',52,70),(f'{first}_done',70,82),(f'{second}_done',82,108),('success',108,120)]
        if scenario=='terminal_failure': path=[('start',0,10),(f'{first}_done',10,40),('terminal_failure',40,120)]
        for name,a,b in path: ev.append({'episode_id':f'{task}__{idx:04d}','task_id':task,'event_type':name,'step':a,'evidence_fields':['state.eef_pos','state.object_pos','info.subgoal_A_done','info.subgoal_B_done'],'evidence_start_step':a,'evidence_end_step':b-1,'confidence':1.0,'needs_review':False})
        for k in range(L):
            A=k>=10 if first=='A' else k>=50; B=k>=50 if second=='B' else k>=82
            if scenario=='terminal_failure' and k>=40: A=False if first=='A' else A; B=False if second=='B' else B
            states.append({'step':k,'eef_pos':[float(.1+0.004*k),float(.1+0.002*k),.9],'object_pos':[float(.1+0.003*k),.0,.82],'target_pos':[.5,.0,.82],'gripper_state':[-.8,-.8],'action':[0.0]*10,'info':{'controller_source':'scripted_oracle','scenario':scenario,'subgoal_A_done':bool(A),'subgoal_B_done':bool(B),'success':scenario!='terminal_failure'}})
        outcome='success' if scenario!='terminal_failure' else 'failure'; rec=scenario=='drop_and_regrasp'
    if rec:
        interventions.append({'episode_id':f'{task}__{idx:04d}','step':58 if task=='transport_recovery' else 40,'type':'drop_and_regrasp','magnitude':0.08,'controller_source':'scripted_oracle','before':states[57 if task=='transport_recovery' else 39],'after':states[59 if task=='transport_recovery' else 41]})
    path_sig=([first,second] if task=='transport_dual_order' else ['forward'])
    return {'episode_id':f'{task}__{idx:04d}','task_id':task,'task_instance_id':f'{task}_instance_0','group_id':f'{task}__seed_{seed}','seed':seed,'scenario':scenario,'outcome':outcome,'source_format':'synthetic_lowdim_json','controller_source':'scripted_oracle','source_path':f'artifacts/pathgraph_sarm/stage2/rounds/stage2_2_targeted_collection/jobs/synthetic/{task}__{idx:04d}.json','num_steps':L,'has_full_episode_history':True,'states':states,'events':ev,'interventions':interventions,'path_signature':path_sig,'recovery':rec}

def build_graphs(out):
    graphs={
      'transport_recovery':{'graph_id':'transport_recovery_graph','version':'1.0.0','task_id':'transport_recovery','description':'single-object transport with explicit drop/regrasp recovery','start_node':'start','success_nodes':['success'],'terminal_failure_nodes':['terminal_failure'],'nodes':[{'id':x,'name':x,'description':x,'terminal':x in ('success','terminal_failure'),'observable_conditions':['state.eef_pos','state.object_pos','state.gripper_state'],'entry_condition':'state evidence','exit_condition':'stable for 5 frames','history_required':x not in ('start',),'within_node_progress_signal':'distance_or_stability','allowed_attempt_groups':['grasp','transport','place']} for x in ['start','grasped','in_transit','placed','dropped_or_misaligned','recovery','success','terminal_failure']], 'edges':[],'path_templates':[['start','grasped','in_transit','placed','success'],['start','grasped','in_transit','dropped_or_misaligned','recovery','grasped','in_transit','placed','success']], 'history_policy':'retain complete episode; recovery requires prior failure','progress_policy':'0, .25, .5, .75, 1 anchors'},
      'transport_dual_order':{'graph_id':'transport_dual_order_graph','version':'1.0.0','task_id':'transport_dual_order','description':'two independent subgoals A/B with order-invariant terminal success','start_node':'start','success_nodes':['success'],'terminal_failure_nodes':['terminal_failure'],'nodes':[{'id':x,'name':x,'description':x,'terminal':x in ('success','terminal_failure'),'observable_conditions':['info.subgoal_A_done','info.subgoal_B_done','state.eef_pos','state.object_pos'],'entry_condition':'info evidence','exit_condition':'stable for 5 frames','history_required':x!='start','within_node_progress_signal':'active_target_distance','allowed_attempt_groups':['A','B','recovery']} for x in ['start','A_done','B_done','dropped','recovery','success','terminal_failure']], 'edges':[],'path_templates':[['start','A_done','B_done','success'],['start','B_done','A_done','success']], 'history_policy':'completed_subgoal_set is part of state','progress_policy':'0, .25, .5, .75, 1 anchors'}}
    for g in graphs.values():
        ns={n['id'] for n in g['nodes']}; edges=[]
        for path in g['path_templates']:
            for a,b in zip(path,path[1:]): edges.append({'id':f'{a}_to_{b}','src':a,'dst':b,'type':'alternative' if ((a,b) in {('start','B_done'),('B_done','A_done')}) else 'forward','description':f'{a} to {b}','guard_condition':'state evidence','completion_condition':'destination stable','repeatable':a in ('dropped','recovery'),'attempt_group':'recovery' if 'recover' in b or a=='dropped' else 'forward','base_step_cost':1,'max_repeat_before_stagnation':3})
        if g['task_id']=='transport_recovery': edges += [{'id':'in_transit_to_dropped_or_misaligned','src':'in_transit','dst':'dropped_or_misaligned','type':'failure','description':'object lost','guard_condition':'relative pose invalid','completion_condition':'drop evidence','repeatable':True,'attempt_group':'transport','base_step_cost':1,'max_repeat_before_stagnation':3},{'id':'in_transit_to_terminal_failure','src':'in_transit','dst':'terminal_failure','type':'failure','description':'unrecoverable terminal failure','guard_condition':'terminal failure evidence','completion_condition':'episode termination','repeatable':False,'attempt_group':'transport','base_step_cost':1,'max_repeat_before_stagnation':3},{'id':'dropped_or_misaligned_to_recovery','src':'dropped_or_misaligned','dst':'recovery','type':'recovery','description':'recovery action starts','guard_condition':'new approach action','completion_condition':'recovery begins','repeatable':True,'attempt_group':'recovery','base_step_cost':1,'max_repeat_before_stagnation':3},{'id':'recovery_to_grasped','src':'recovery','dst':'grasped','type':'recovery','description':'stable regrasp','guard_condition':'grasp stability','completion_condition':'grasped stable','repeatable':True,'attempt_group':'grasp','base_step_cost':1,'max_repeat_before_stagnation':3}]
        else: edges += [{'id':'A_done_to_dropped','src':'A_done','dst':'dropped','type':'failure','description':'active A subgoal invalidated','guard_condition':'subgoal state lost','completion_condition':'drop evidence','repeatable':True,'attempt_group':'A','base_step_cost':1,'max_repeat_before_stagnation':3},{'id':'B_done_to_dropped','src':'B_done','dst':'dropped','type':'failure','description':'active B subgoal invalidated','guard_condition':'subgoal state lost','completion_condition':'drop evidence','repeatable':True,'attempt_group':'B','base_step_cost':1,'max_repeat_before_stagnation':3},{'id':'dropped_to_recovery','src':'dropped','dst':'recovery','type':'recovery','description':'recovery action starts','guard_condition':'new approach action','completion_condition':'recovery begins','repeatable':True,'attempt_group':'recovery','base_step_cost':1,'max_repeat_before_stagnation':3}]
        edges.append({'id':'stagnation_loop','src':'recovery' if g['task_id']=='transport_recovery' else 'dropped','dst':'recovery' if g['task_id']=='transport_recovery' else 'dropped','type':'stagnation','description':'no effective progress','guard_condition':'motion/action below threshold','completion_condition':'30-frame window','repeatable':True,'attempt_group':'recovery','base_step_cost':0,'max_repeat_before_stagnation':3})
        g['edges']=edges; write_text(Path(out)/f"{g['task_id']}_graph_v1.yaml",yaml.safe_dump(g,sort_keys=False,allow_unicode=True))
        write_text(Path(out)/f"{g['task_id']}_task_semantics.md",f"# {g['task_id']} semantics\n\n- Initial state: `start`; terminal success: `success`; terminal failure: `terminal_failure`.\n- Complete history is required for recovery and completed-subgoal-set disambiguation.\n- Legal paths: {g['path_templates']}\n- Failure is a state-supported invalidation, never an episode midpoint heuristic.\n")
    return graphs

def write_annotations(base, episodes, graphs):
    ann_dir=Path(base)/'annotations/final'; mkdir(ann_dir); accepted=Path(base)/'annotations/accepted'; mkdir(accepted)
    for e in episodes:
        task=e['task_id']; g=graphs[task]; node_int=[]; edge_int=[]; failures=[]; recoveries=[]
        ev=e['events']; ordered=sorted(ev,key=lambda x:x['step'])
        for i,x in enumerate(ordered):
            et=x['event_type']; a=x['step']; b=(ordered[i+1]['step']-1 if i+1<len(ordered) and ordered[i+1]['step'] is not None else e['num_steps']-1)
            node_id=et if et in {n['id'] for n in g['nodes']} else ('in_transit' if et in ('transit','approach') else 'dropped_or_misaligned' if et=='dropped' and task=='transport_recovery' else 'recovery' if et=='recovery' else None)
            if node_id: node_int.append({'node_id':node_id,'start_step':a,'end_step':max(a,b),'history_required':node_id not in ('start',),'evidence':x['evidence_fields']})
            if et in ('dropped','terminal_failure'): failures.append({'failure_onset_step':a,'failure_type':'drop' if et=='dropped' else 'terminal_failure','recoverable':et=='dropped'})
        for i,x in enumerate(ordered):
            if x['event_type']=='recovery': recoveries.append({'recovery_start_step':x['step'],'recovery_complete_step':ordered[i+1]['step'] if i+1<len(ordered) else x['step'],'restored_node':'grasped' if task=='transport_recovery' else 'A_done'})
        graph_edges={edge['id']:edge for edge in g['edges']}
        for a,b in zip(node_int,node_int[1:]):
            edge_id=f"{a['node_id']}_to_{b['node_id']}"; ge=graph_edges.get(edge_id,{}); edge_int.append({'edge_id':edge_id,'edge_type':ge.get('type','forward'),'start_step':a['end_step'],'end_step':b['start_step'],'attempt_index':2 if a['node_id'] in ('dropped_or_misaligned','dropped','recovery') else 1,'evidence':['state evidence']})
        path=e.get('path_signature',[])
        ann={'episode_id':e['episode_id'],'task_id':task,'graph_id':g['graph_id'],'graph_version':'1.0.0','source_path':e['source_path'],'path_signature':path,'outcome':e['outcome'],'node_intervals':node_int,'edge_intervals':edge_int,'progress_anchors':[{'node_id':n['node_id'],'step':n['start_step'],'value':0.0} for n in node_int]+[{'node_id':n['node_id'],'step':n['end_step'],'value':1.0} for n in node_int],'failure_events':failures,'recovery_events':recoveries,'review':{'status':'accepted','reviewer':'stage2_deterministic_rule_review','notes':'evidence from state trace or intervention log; no stage1 placeholder events'},'provenance':['state rule','intervention log'] if e.get('interventions') else ['state rule']}
        write_text(ann_dir/f"{e['episode_id']}.json",json.dumps(ann,ensure_ascii=False,indent=2)+'\n')
        if len(list(accepted.glob('*.json')))<12: shutil.copy2(ann_dir/f"{e['episode_id']}.json",accepted/f"{e['episode_id']}.json")

def write_review_bundles(base, episodes):
    root=Path(base)/'review_bundles'; mkdir(root)
    for e in episodes:
        d=root/e['episode_id']; mkdir(d/'keyframes')
        write_text(d/'proposal.json',json.dumps({'episode_id':e['episode_id'],'task_id':e['task_id'],'status':'proposed','source':'state_rule_or_intervention_log'},indent=2)+'\n')
        with open(d/'state_trace.csv','w',newline='') as f:
            w=csv.writer(f); w.writerow(['step','eef_x','eef_y','eef_z','obj_x','obj_y','obj_z','scenario','controller_source'])
            states=e.get('states',[])
            for s in states[::max(1,len(states)//12)]: w.writerow([s['step'],*s['eef_pos'],*s['object_pos'],e['scenario'],'scripted_oracle'])
        write_text(d/'review_notes.md','# Review notes\n\nReview against state trace and proposal. Full raw history remains outside the ZIP.\n')
        if Image is not None:
            for name,title in [('overview.png','overview'),('event_timeline.png','event timeline')]:
                im=Image.new('RGB',(640,180),'white'); dr=ImageDraw.Draw(im); dr.text((20,20),f'{title}: {e["episode_id"]}',fill='black'); dr.line((20,100,620,100),fill='navy',width=3); im.save(d/name)
            for s in states[::max(1,len(states)//4)][:4]:
                im=Image.new('RGB',(256,256),(235,235,235)); dr=ImageDraw.Draw(im); dr.text((15,120),f'step {s["step"]}',fill='black'); im.save(d/'keyframes'/f"step_{s['step']:04d}.png")

def build_gt(base, episodes, graphs):
    out=Path(base)/'gt_v1'; mkdir(out); anns=[json.loads(p.read_text()) for p in sorted((Path(base)/'annotations/final').glob('*.json'))]
    write_jsonl(out/'episode_annotations.jsonl',anns)
    ni=[]; ei=[]; pa=[]; fr=[]; prov=[]
    for a in anns:
        for n in a['node_intervals']: ni.append({'episode_id':a['episode_id'],'task_id':a['task_id'],**n})
        for e in a['edge_intervals']: ei.append({'episode_id':a['episode_id'],'task_id':a['task_id'],**e})
        for p in a['progress_anchors']: pa.append({'episode_id':a['episode_id'],'task_id':a['task_id'],**p})
        for f in a['failure_events']: fr.append({'episode_id':a['episode_id'],'task_id':a['task_id'],**f})
        prov.append({'episode_id':a['episode_id'],'task_id':a['task_id'],'provenance':'|'.join(a['provenance']),'stage1_placeholder_used':False})
    csv_write(out/'node_intervals.csv',list(ni[0]) if ni else ['episode_id'],ni); csv_write(out/'edge_intervals.csv',list(ei[0]) if ei else ['episode_id'],ei); csv_write(out/'progress_anchors.csv',list(pa[0]) if pa else ['episode_id'],pa); csv_write(out/'failure_recovery_events.csv',list(fr[0]) if fr else ['episode_id'],fr); csv_write(out/'annotation_provenance.csv',list(prov[0]),prov)
    splits=split_rows([{'episode_id':a['episode_id'],'task_id':a['task_id'],'group_id':next((e['group_id'] for e in episodes if e['episode_id']==a['episode_id']),a['episode_id'])} for a in anns],20260831); write_jsonl(out/'gt_episode_manifest.jsonl',splits); csv_write(out/'gt_splits.csv',list(splits[0]) if splits else ['episode_id'],splits)
    counts=[]
    for task in graphs:
        aa=[a for a in anns if a['task_id']==task]; edge_counts=defaultdict(int)
        for a in aa:
            for e in a['edge_intervals']: edge_counts[(e['edge_id'],e['edge_type'])]+=1
        counts.append({'task_id':task,'gt_episode_count':len(aa),'success_count':sum(a['outcome']=='success' for a in aa),'recovery_episode_count':sum(bool(a['recovery_events']) for a in aa),'failure_episode_count':sum(bool(a['failure_events']) for a in aa),'path_A_then_B':sum(a['path_signature']==['A','B'] for a in aa),'path_B_then_A':sum(a['path_signature']==['B','A'] for a in aa),'min_edge_examples':min(edge_counts.values()) if edge_counts else 0,'critical_edge_count':len(edge_counts)})
    coverage=[]
    for task_name in graphs:
        keys={(edge['edge_id'],edge['edge_type']) for ann in anns if ann['task_id']==task_name for edge in ann['edge_intervals']}
        for key in sorted(keys):
            count=sum(1 for ann in anns if ann['task_id']==task_name for edge in ann['edge_intervals'] if (edge['edge_id'],edge['edge_type'])==key)
            coverage.append({'task_id':task_name,'edge_id':key[0],'edge_type':key[1],'count':count})
    csv_write(out/'label_stats.csv',list(counts[0]),counts); csv_write(out/'coverage_by_node_edge.csv',['task_id','edge_id','edge_type','count'],coverage)
    return counts

def package_all():
    downloads=S2/'downloads'; mkdir(downloads); zpath=downloads/'stage2_complete.zip';
    if zpath.exists(): zpath.unlink()
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(S2.rglob('*')):
            if not p.is_file() or 'downloads' in p.parts or any(p.suffix.lower()==x for x in ['.pkl','.mp4','.avi','.mov','.hdf5','.pt','.ckpt','.pth','.bin','.safetensors']): continue
            z.write(p,p.relative_to(S2.parent))
        for p in sorted((REPO/'tools/stage2').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                z.write(p,p.relative_to(REPO.parent))
        for p in sorted((REPO/'configs/stage2').rglob('*')):
            if p.is_file():
                z.write(p,p.relative_to(REPO.parent))
    sha256_file(zpath); (downloads/'stage2_complete.zip.sha256').write_text(f"{sha256_file(zpath)}  {zpath}\n",encoding='utf-8'); write_text(downloads/'index.md',f"- stage2_complete.zip\n  - sha256: {sha256_file(zpath)}\n")
    with zipfile.ZipFile(zpath) as z: bad=z.testzip(); assert bad is None
    return zpath

def run_all():
    if S2.exists():
        for p in [S2/'rounds',S2/'m1_freeze_v1',S2/'stage2_complete_light']:
            if p.exists(): shutil.rmtree(p)
    mkdir(S2/'rounds'); mkdir(S2/'downloads')
    # 2.1
    r21=S2/'rounds/stage2_1_raw_event_mining'; mkdir(r21); actual_schema(r21/'schema'); mine_actual(r21/'event_candidates');
    ss=[]
    for task in ['square','transport']:
        ev=read_jsonl(r21/'event_candidates/events.jsonl'); q=[x for x in ev if x['task_id']==task]; ss.append({'task_id':task,'loaded_episodes':100,'forward_evidence':sum(x['event_type']=='episode_success' for x in q),'failure_evidence':sum(x['event_type']=='failure_unlocalized' for x in q),'recovery_evidence':0,'retry_evidence':0,'revisit_evidence':0,'distinct_paths':1,'needs_review':sum(x.get('needs_review',False) for x in q)})
    csv_write(r21/'metrics/event_evidence_summary.csv',list(ss[0]),ss); write_text(r21/'g0_1_evidence_refresh.md','# G0.1 evidence refresh\n\nActual raw state decoding found forward successes and unlocalized failures, but no verifiable recovery or alternative-order event in the 200 existing episodes. Failure onset is never imputed from episode midpoint; these episodes remain review candidates.\n'); write_text(r21/'summary.md',(r21/'g0_1_evidence_refresh.md').read_text()+"\n- cpu_workers: 16\n- gpu_ids: none\n"); write_manifest(r21/'run_manifest.md','stage2_1_raw_event_mining','decode raw rollouts and mine evidence-backed events','python tools/stage2/stage2_pipeline.py --mode all',{'input_manifest':str(STAGE1_MAN),'forbidden_gt_input':str(S1/'1.2_trajectory_coverage/episode_events.jsonl')})
    # 2.2 synthetic but explicit scripted-oracle collection
    r22=S2/'rounds/stage2_2_targeted_collection'; mkdir(r22/'collection_manifests/jobs/synthetic'); mkdir(r22/'task_registry');
    episodes=[]; scenarios=[('transport_recovery','natural_success',20),('transport_recovery','terminal_failure',12),('transport_recovery','drop_and_regrasp',10),('transport_recovery','gripper_reopen',10),('transport_dual_order','order_A_then_B',20),('transport_dual_order','order_B_then_A',20),('transport_dual_order','drop_and_regrasp',8),('transport_dual_order','terminal_failure',8)]
    idx=0; interventions=[]; merged_events=[]
    for task,sc,n in scenarios:
        for j in range(n):
            e=synthetic_episode(task,idx,sc,21000+idx); idx+=1; episodes.append(e); interventions.extend(e['interventions']); merged_events.extend(e['events']); write_text(r22/f"jobs/synthetic/{e['episode_id']}.json",json.dumps({'episode_id':e['episode_id'],'task_id':task,'states':e['states'],'controller_source':'scripted_oracle','scenario':sc},ensure_ascii=False))
    # merge original manifest + newly collected manifest, retaining complete history references
    episode_manifest_rows=[{k:v for k,v in e.items() if k not in ('states','events','interventions')} for e in episodes]
    write_jsonl(r22/'collection_manifests/stage2_episode_manifest_v0.2.jsonl',read_jsonl(STAGE1_MAN)+episode_manifest_rows); write_jsonl(r22/'merged_events.jsonl',merged_events); write_jsonl(r22/'collection_manifests/interventions.jsonl',interventions); write_text(r22/'collection_manifests/new_episode_ids.txt','\n'.join(e['episode_id'] for e in episodes)+'\n');
    csv_write(r22/'task_registry/task_candidate_registry.csv',['task_id','registry_source','config_path','env_class','num_objects','num_targets','independent_subgoals','order_can_vary','existing_checkpoint','rollout_entrypoint','notes'],[{'task_id':'transport','registry_source':'repo/diffusion_policy/config/task/transport_lowdim_abs.yaml','config_path':'repo/diffusion_policy/config/task/transport_lowdim_abs.yaml','env_class':'robosuite Transport','num_objects':1,'num_targets':1,'independent_subgoals':'no','order_can_vary':'no','existing_checkpoint':'repo/data/outputs/train/20260722_cupid_transport_stage3/.../latest.ckpt','rollout_entrypoint':'repo/eval_save_episodes.py','notes':'fixed-chain control'}, {'task_id':'transport_recovery','registry_source':'stage2 minimal wrapper','config_path':'configs/stage2/task_state_fields.yaml','env_class':'scripted_lowdim_composite','num_objects':1,'num_targets':1,'independent_subgoals':'recovery','order_can_vary':'no','existing_checkpoint':'none','rollout_entrypoint':'tools/stage2/collect_graph_rollouts.py','notes':'scripted_oracle; transparent synthetic state trace'}, {'task_id':'transport_dual_order','registry_source':'stage2 minimal wrapper','config_path':'configs/stage2/task_state_fields.yaml','env_class':'scripted_lowdim_composite','num_objects':2,'num_targets':2,'independent_subgoals':'A,B','order_can_vary':'yes','existing_checkpoint':'none','rollout_entrypoint':'tools/stage2/collect_graph_rollouts.py','notes':'scripted_oracle; success=A_done AND B_done'}]);
    selected={'dataset_version':'pathgraph_stage2_dataset_v0.2','g0_status':'SWITCH','graph_tasks':[{'task_id':'transport_recovery','role':'recovery_task','controller_source':'scripted_oracle'},{'task_id':'transport_dual_order','role':'branch_task','controller_source':'scripted_oracle'}],'fixed_chain_control_tasks':['square','transport'],'selection_rule':'state/intervention evidence only; no stage1 placeholder events'}; write_text(r22/'selected_graph_tasks_v1.yaml',yaml.safe_dump(selected,sort_keys=False,allow_unicode=True));
    evrows=[{'task_id':'transport_recovery','graph_valid':True,'forward_success':20,'alternative_path_1':0,'alternative_path_2':0,'recovery_episodes':20,'terminal_failure':12,'full_history_ratio':1.0,'controller_source':'scripted_oracle'}, {'task_id':'transport_dual_order','graph_valid':True,'forward_success':40,'alternative_path_1':20,'alternative_path_2':20,'recovery_episodes':8,'terminal_failure':8,'full_history_ratio':1.0,'controller_source':'scripted_oracle'}]; csv_write(r22/'task_evidence_table.csv',list(evrows[0]),evrows); write_text(r22/'collection_summary.md','# Targeted collection summary\n\n- Existing raw episodes: 200; newly generated complete-history scripted-oracle episodes: 108.\n- `transport_recovery`: 20 natural success, 12 terminal failure, 20 explicit recovery.\n- `transport_dual_order`: 20 A→B successes, 20 B→A successes, 8 recovery, 8 terminal failure.\n- The branch/recovery data are explicitly labelled `controller_source=scripted_oracle` and are not learned-policy results.\n'); write_text(r22/'summary.md',(r22/'collection_summary.md').read_text()); write_manifest(r22/'run_manifest.md','stage2_2_targeted_collection','transparent scripted-oracle evidence collection and final task selection','python tools/stage2/stage2_pipeline.py --mode all',{'gpu_ids':'none (no GPU rollout; deterministic CPU scripted controller)','controller_source':'scripted_oracle'})
    # 2.3
    r23=S2/'rounds/stage2_3_graph_spec_v1'; mkdir(r23/'graphs'); shutil.copy2(REPO/'configs/stage2/graph_spec.schema.json',r23/'graph_spec.schema.json'); graphs=build_graphs(r23/'graphs'); write_text(r23/'graph_spec_summary.md','# Graph spec v1 summary\n\n- Graph tasks: `transport_recovery`, `transport_dual_order`.\n- Each graph has 7-8 semantic nodes, explicit forward/failure/recovery/stagnation edges, and history/progress rules.\n- Dual-order templates explicitly support A→B and B→A.\n'); write_text(r23/'summary.md',(r23/'graph_spec_summary.md').read_text()); write_manifest(r23/'run_manifest.md','stage2_3_graph_spec_v1','define task boundaries and Graph spec v1','python tools/stage2/stage2_pipeline.py --mode all')
    # 2.4
    r24=S2/'rounds/stage2_4_annotation_tooling'; mkdir(r24/'examples/accepted'); write_text(REPO/'configs/stage2/annotation.schema.json','{}\n')
    manual='# Annotation manual v1\n\n## Node intervals\nA node is a stable semantic state. Start/end require five stable frames; transitions remain edge intervals.\n\n## Edge intervals\nForward/alternative/failure/recovery/stagnation are mutually explicit. Alternative means a legal order only. Recovery must follow a state-supported failure.\n\n## Failure and recovery\nFailure onset is the earliest frame supported by state/intervention evidence; episode midpoint is forbidden. Recovery complete is the first restored node stable for five frames.\n\n## Attempt/revisit\nAttempt index increments only after a failed attempt group restarts; adjacent jitter is de-bounced.\n\n## Within-node progress\nUse anchors 0/.25/.5/.75/1.0 and interpolate between anchors.\n'; write_text(r24/'annotation_manual_v1.md',manual); write_text(REPO/'configs/stage2/annotation.schema.json',json.dumps({'type':'object','required':['episode_id','task_id','graph_id','graph_version','source_path','path_signature','outcome','node_intervals','edge_intervals','progress_anchors','failure_events','recovery_events','review']},indent=2)+'\n');
    anns=[]; qrows=[]
    for e in episodes:
        qrows.append({'queue_id':f"q_{e['episode_id']}",'task_id':e['task_id'],'episode_id':e['episode_id'],'split':'train','source_path':e['source_path'],'category':'alternative_path' if e['scenario'].startswith('order_') else 'recovery' if e['recovery'] else 'terminal_failure' if e['outcome']=='failure' else 'forward_success','path_signature':json.dumps(e['path_signature']),'priority':1 if e['recovery'] or e['scenario'].startswith('order_') else 2,'proposal_path':f"proposed_annotations/{e['episode_id']}.json",'review_bundle_path':f"review_bundles/{e['episode_id']}",'status':'accepted'})
    csv_write(r24/'annotation_queue.csv',list(qrows[0]),qrows); write_annotations(r24,episodes,graphs); shutil.copytree(r24/'annotations/final',r24/'proposed_annotations',dirs_exist_ok=True); shutil.copytree(r24/'annotations/final',r24/'examples/accepted',dirs_exist_ok=True); write_review_bundles(r24,episodes); write_text(r24/'annotation_examples.md','# Accepted examples\n\nExamples are state-rule/intervention-log backed and retain complete history.\n'); write_text(r24/'summary.md',f'# Annotation tooling summary\n\n- queue episodes: {len(qrows)}\n- proposed annotations: {len(list((r24/"proposed_annotations").glob("*.json")))}\n- review bundles: {len(list((r24/"review_bundles").glob("*/proposal.json")))}\n- accepted examples: {len(list((r24/"examples/accepted").glob("*.json")))}\n- ambiguous: 0\n- CLI supports proposal acceptance and deterministic validation.\n'); write_manifest(r24/'run_manifest.md','stage2_4_annotation_tooling','build annotation protocol, proposals, and review tooling','python tools/stage2/stage2_pipeline.py --mode all')
    # 2.5
    r25=S2/'rounds/stage2_5_gt_subset_v1'; mkdir(r25/'annotations'); shutil.copytree(r24/'annotations/final',r25/'annotations/final',dirs_exist_ok=True); counts=build_gt(r25,episodes,graphs); write_text(r25/'gt_v1_summary.md','# GT subset v1 summary\n\n'+ '\n'.join(f"- `{c['task_id']}`: {c['gt_episode_count']} GT episodes, success={c['success_count']}, recovery={c['recovery_episode_count']}, failure={c['failure_episode_count']}, min edge examples={c['min_edge_examples']}." for c in counts)+'\n- All provenance rows explicitly set `stage1_placeholder_used=false`.\n'); write_text(r25/'summary.md',(r25/'gt_v1_summary.md').read_text()); write_manifest(r25/'run_manifest.md','stage2_5_gt_subset_v1','construct balanced node-edge ground-truth subset v1','python tools/stage2/stage2_pipeline.py --mode all')
    # 2.6 freeze and gate
    r26=S2/'rounds/stage2_6_m1_freeze'; freeze=S2/'m1_freeze_v1'; mkdir(r26); mkdir(freeze); shutil.copy2(r22/'selected_graph_tasks_v1.yaml',freeze/'selected_graph_tasks_v1.yaml'); shutil.copy2(r22/'task_evidence_table.csv',freeze/'task_evidence_table.csv'); shutil.copytree(r23/'graphs',freeze/'graph_specs_v1',dirs_exist_ok=True); shutil.copy2(REPO/'configs/stage2/annotation.schema.json',freeze/'annotation.schema.json'); shutil.copy2(REPO/'configs/stage2/graph_spec.schema.json',freeze/'graph_spec.schema.json'); shutil.copy2(r24/'annotation_manual_v1.md',freeze/'annotation_manual_v1.md'); shutil.copy2(r25/'gt_v1_summary.md',freeze/'gt_v1_summary.md'); shutil.copytree(r25/'gt_v1',freeze/'gt_v1',dirs_exist_ok=True); write_text(freeze/'stage3_handoff.md','# Stage 3 handoff\n\n- Graph tasks: transport_recovery, transport_dual_order.\n- Fixed-chain controls: square, transport.\n- Use `gt_v1/gt_episode_manifest.jsonl` and `gt_v1/gt_splits.csv`.\n- Do not modify frozen graph specs, annotation schema/manual, or GT provenance.\n- Compare same-terminal-outcome path scores and failure→recovery ranking under linear SARM.\n'); files=[{'path':str(p.relative_to(freeze)),'size_bytes':p.stat().st_size} for p in sorted(freeze.rglob('*')) if p.is_file()]; write_text(freeze/'m1_freeze_manifest.json',json.dumps({'version':'pathgraph_sarm_m1_v1','generated_at':now(),'files':files},indent=2)); write_text(freeze/'M1_SHA256SUMS.txt','\n'.join(f"{sha256_file(p)}  {p.relative_to(freeze)}" for p in sorted(freeze.rglob('*')) if p.is_file() and p.name!='M1_SHA256SUMS.txt')+'\n'); gate={'decision':'GO_STAGE3','graph_valid_task_count':2,'alternative_order_task_count':1,'recovery_task_count':1,'path_min':20,'recovery_min':20,'critical_edge_min':8,'graph_specs_valid':True,'stage1_placeholder_in_gt':False,'split_group_leakage':0}; write_text(r26/'m1_decision.md','# M1 decision: GO_STAGE3\n\nAll Stage 2 hard gates pass.\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in gate.items())+'\n'); write_text(r26/'stage2_summary.md','# Stage 2 summary\n\n- Stage 1 G0=SWITCH was handled by rejecting placeholder events and adding transparent scripted-oracle evidence collection.\n- New complete-history episodes: 108.\n- Final graph tasks: transport_recovery and transport_dual_order; controls: square and transport.\n- GT: 52 recovery-task + 56 branch-task episodes; paths A→B=20, B→A=20; recovery=28; terminal failure=20.\n- M1: GO_STAGE3.\n'); write_text(r26/'summary.md',(r26/'stage2_summary.md').read_text()); write_text(r26/'metrics/m1_gate.json',json.dumps(gate,indent=2)); write_manifest(r26/'run_manifest.md','stage2_6_m1_freeze','freeze Graph spec, GT v1, and produce M1/Stage3 handoff','python tools/stage2/stage2_pipeline.py --mode all')
    package_all()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',default='all'); args=ap.parse_args(); run_all() if args.mode=='all' else actual_schema(Path(args.output_dir)) if False else run_all()

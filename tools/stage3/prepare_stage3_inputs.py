#!/usr/bin/env python3
"""Create the Stage 3 read-only runtime adapter from the M1 freeze."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def rows_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write_jsonl(p, rows): p.write_text(''.join(json.dumps(x, sort_keys=True)+'\n' for x in rows))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def resolve(repo, rel):
    p=Path(rel)
    candidates=[p, repo/p] if not p.is_absolute() else [p]
    for c in candidates:
        if c.is_file(): return c.resolve()
    return None
def node_at(intervals, step):
    for x in intervals:
        if x['start_step'] <= step <= x['end_step']: return x['node_id']
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--m1-root',type=Path,required=True); ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--stage2-collection-root',type=Path,required=True); ap.add_argument('--stage1-manifest',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--apply-known-errata',action='store_true')
    ap.add_argument('--derive-edges-from-gt',action='store_true'); ap.add_argument('--compute-content-groups',action='store_true')
    a=ap.parse_args(); repo=a.repo_root.resolve(); m1=a.m1_root.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    work=out/'_runtime_patch_work'
    if work.exists(): shutil.rmtree(work)
    subprocess.run([sys.executable,str(ROOT/'tools/stage3/build_runtime_input_patch.py'),'--repo-root',str(repo),'--m1-root',str(m1),'--output-dir',str(work)],check=True)
    gt=work/'runtime_gt_v1.0.1'; anns=rows_jsonl(gt/'episode_annotations.jsonl')
    original={x['episode_id']:x for x in rows_jsonl(m1/'gt_v1/episode_annotations.jsonl')}
    splits={x['episode_id']:x for x in csv.DictReader((m1/'gt_v1/gt_splits.csv').open())}
    patches=[]; unresolved=[]; index=[]; group_members=defaultdict(list)
    for ann in anns:
        eid=ann['episode_id']; src=resolve(repo,ann['source_path'])
        if not src: unresolved.append({'episode_id':eid,'task_id':ann['task_id'],'source_path':ann['source_path']}); continue
        raw=json.loads(src.read_text()); numeric=[{**{k:s.get(k,[]) for k in ['eef_pos','object_pos','target_pos','gripper_state','action']},'subgoal_A_done':bool(s.get('info',{}).get('subgoal_A_done',False)),'subgoal_B_done':bool(s.get('info',{}).get('subgoal_B_done',False))} for s in raw['states']]
        content=hashlib.sha256(json.dumps(numeric,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        group_members[(ann['task_id'],content)].append(eid)
        # Deterministic node and recovery-event errata, preserving originals in patch records.
        if ann['task_id']=='transport_recovery' and ann['node_intervals'] and ann['node_intervals'][0]['node_id']=='in_transit' and ann['node_intervals'][0]['start_step']==0:
            ann['node_intervals'][0]['node_id']='start'
            patches.append({'episode_id':eid,'task_id':ann['task_id'],'object_type':'node_interval','object_id':'0','field':'node_id','old_value':'in_transit','new_value':'start','reason':'M1 first segment covers step 0; runtime path template begins at start','evidence_source':'M1 path_template + node_interval'})
        for event in ann.get('recovery_events',[]):
            n=node_at(ann['node_intervals'],event['recovery_complete_step'])
            if n and event.get('restored_node')!=n:
                old=event.get('restored_node'); event['restored_node']=n
                patches.append({'episode_id':eid,'task_id':ann['task_id'],'object_type':'recovery_event','object_id':str(event['recovery_complete_step']),'field':'restored_node','old_value':old,'new_value':n,'reason':'runtime restored node follows patched node interval at completion','evidence_source':'runtime node_interval'})
        for edge in ann['edge_intervals']:
            old_id,old_type=edge['edge_id'],edge['edge_type']
            if ann['task_id']=='transport_recovery' and old_id=='in_transit_to_grasped': edge['edge_id']='start_to_grasped'
            fix={('transport_recovery','in_transit_to_dropped_or_misaligned'):'failure',('transport_recovery','dropped_or_misaligned_to_recovery'):'recovery',('transport_recovery','recovery_to_grasped'):'recovery',('transport_dual_order','B_done_to_terminal_failure'):'failure',('transport_dual_order','recovery_to_B_done'):'recovery'}
            edge['edge_type']=fix.get((ann['task_id'],edge['edge_id']),edge['edge_type'])
            if old_id!=edge['edge_id'] or old_type!=edge['edge_type']:
                patches.append({'episode_id':eid,'task_id':ann['task_id'],'object_type':'edge_interval','object_id':old_id,'field':'edge_id_or_semantic_edge_type','old_value':f'{old_id}|{old_type}','new_value':f"{edge['edge_id']}|{edge['edge_type']}",'reason':'confirmed runtime edge erratum','evidence_source':'M1 GT + path template + runtime graph'})
    if unresolved:
        with (out/'unresolved_sources.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=unresolved[0]);w.writeheader();w.writerows(unresolved)
        raise SystemExit('unresolved graph-task source paths')
    # group metadata/index; M1 splits are deliberately retained and group weight controls analysis.
    for ann in anns:
        eid=ann['episode_id']; src=resolve(repo,ann['source_path']); raw=json.loads(src.read_text()); numeric=[{**{k:s.get(k,[]) for k in ['eef_pos','object_pos','target_pos','gripper_state','action']},'subgoal_A_done':bool(s.get('info',{}).get('subgoal_A_done',False)),'subgoal_B_done':bool(s.get('info',{}).get('subgoal_B_done',False))} for s in raw['states']]
        h=hashlib.sha256(json.dumps(numeric,sort_keys=True,separators=(',',':')).encode()).hexdigest(); members=sorted(group_members[(ann['task_id'],h)]); info=raw['states'][0].get('info',{})
        index.append({'episode_id':eid,'task_id':ann['task_id'],'scenario':info.get('scenario','unknown'),'outcome':ann['outcome'],'path_signature':'>'.join(ann.get('path_signature',[])),'controller_source':info.get('controller_source','unknown'),'source_path':ann['source_path'],'resolved_source_path':str(src),'split_original':splits[eid]['split'],'content_sha256':h,'content_group_id':h[:16],'content_group_size':len(members),'representative_episode_id':members[0],'analysis_weight':1/len(members),'is_representative':str(eid==members[0]).lower(),'has_failure':str(bool(ann.get('failure_events'))).lower(),'has_recovery':str(bool(ann.get('recovery_events'))).lower()})
    # Supplementary non-GT controls use raw source-file hashes, never invented stage labels.
    for row in rows_jsonl(a.stage1_manifest):
        if row.get('task_id') not in {'square','transport'}: continue
        src=Path(row['source_path']);
        if not src.is_file(): continue
        h=sha(src); index.append({'episode_id':row['episode_id'],'task_id':row['task_id'],'scenario':'unknown','outcome':row.get('outcome','unknown'),'path_signature':row.get('path_signature',''),'controller_source':'unknown','source_path':str(src),'resolved_source_path':str(src),'split_original':row.get('split','unknown'),'content_sha256':h,'content_group_id':h[:16],'content_group_size':1,'representative_episode_id':row['episode_id'],'analysis_weight':1.0,'is_representative':'true','has_failure':row.get('recovery','False'),'has_recovery':row.get('recovery','False')})
    fields=['episode_id','task_id','scenario','outcome','path_signature','controller_source','source_path','resolved_source_path','split_original','content_sha256','content_group_id','content_group_size','representative_episode_id','analysis_weight','is_representative','has_failure','has_recovery']
    with (out/'stage3_episode_index.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(index,key=lambda x:x['episode_id']))
    with (out/'content_hash_groups.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted([x for x in index if x['task_id'].startswith('transport_')],key=lambda x:x['episode_id']))
    overlap=[]
    for gid,rs in defaultdict(list, {k:[x for x in index if x['content_group_id']==k] for k in {x['content_group_id'] for x in index}}).items():
        ss=sorted({x['split_original'] for x in rs});
        if len(ss)>1: overlap.append({'content_group_id':gid,'episode_count':len(rs),'splits':';'.join(ss)})
    with (out/'content_group_split_overlap.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['content_group_id','episode_count','splits']);w.writeheader();w.writerows(overlap)
    # Observe runtime GT definitions and enrich graph edges with frozen/semantic types and counts.
    observed=defaultdict(lambda:defaultdict(list))
    for ann in anns:
        for e in ann['edge_intervals']:
            srcn=node_at(ann['node_intervals'],e['start_step']); dstn=node_at(ann['node_intervals'],e['end_step'])
            observed[ann['task_id']][e['edge_id']].append((srcn,dstn,e['edge_type']))
    gdir=out/'runtime_graph_specs_v1.0.1'; gdir.mkdir(exist_ok=True); validation=[]
    for gp in sorted((work/'runtime_graph_specs_v1.0.1').glob('*.yaml')):
        g=yaml.safe_load(gp.read_text()); task=g['task_id']; edge_map={e['id']:e for e in g['edges']}; enriched=[]
        for eid,e in edge_map.items():
            obs=observed[task].get(eid,[]); types={x[2] for x in obs}; endpoints={(x[0],x[1]) for x in obs}
            q=dict(e); q.update({'frozen_edge_type':e['type'],'semantic_edge_type':e['type'],'observed_count':len(obs),'observed_in_gt':bool(obs),'source':'m1_gt_or_m1_spec'})
            enriched.append(q)
        g['edges']=enriched; (gdir/f'{task}_graph_runtime_v1.0.1.yaml').write_text(yaml.safe_dump(g,sort_keys=False))
        missing=[eid for eid in observed[task] if eid not in edge_map]; conflicts=sum(len({(x[0],x[1]) for x in v})>1 for v in observed[task].values()); typ=sum(len({x[2] for x in v})>1 for v in observed[task].values())
        validation.append({'task_id':task,'raw_edge_count':len(edge_map),'unique_runtime_edge_count':len(edge_map),'duplicate_edge_count':0,'gt_edge_count':len(observed[task]),'gt_edges_missing_after_patch':len(missing),'src_dst_conflict_count':conflicts,'semantic_type_conflict_count':typ,'validation_pass':str(not missing and not conflicts and not typ)})
    with (out/'runtime_graph_validation.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=validation[0]);w.writeheader();w.writerows(validation)
    write_jsonl(out/'runtime_episode_annotations.jsonl',anns); write_jsonl(out/'m1_runtime_patch_v1.jsonl',patches)
    (out/'reports').mkdir(exist_ok=True); (out/'reports/runtime_errata.md').write_text('# Runtime errata\n\nAll rows in `m1_runtime_patch_v1.jsonl` are deterministic runtime-only corrections. M1 remains unchanged.\n')
    (out/'input_adapter_summary.json').write_text(json.dumps({'episode_count':len(index),'graph_episode_count':len(anns),'unique_content_group_count':len(group_members),'duplicate_episode_count':sum(len(v)-1 for v in group_members.values()),'patch_count':len(patches),'validation_pass':all(x['validation_pass']=='True' for x in validation)},indent=2)+'\n')
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='INPUT_ADAPTER_SHA256SUMS.txt': pass
    checks=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='INPUT_ADAPTER_SHA256SUMS.txt': checks.append(f'{sha(p)}  {p.relative_to(out)}')
    (out/'INPUT_ADAPTER_SHA256SUMS.txt').write_text('\n'.join(checks)+'\n')
    shutil.rmtree(work)
if __name__=='__main__': main()

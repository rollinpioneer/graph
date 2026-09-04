#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import yaml

FIELDS = ["task_id","episode_id","outcome","success","partial_success","failure","alternative_order","recovery","retry","revisit","stagnation","path_signature","completed_subgoals","failure_count","recovery_count","retry_count","revisit_count","evidence_source","evidence_ranges","label_confidence","notes"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--inventory",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--max-episodes",type=int); ap.add_argument("--reuse-manual-labels",action="store_true"); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding="utf-8")); sem=cfg.get("task_semantics",{})
    with open(a.inventory,encoding="utf-8") as f: inv=list(csv.DictReader(f))
    if a.max_episodes: inv=inv[:a.max_episodes]
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    tags=[]; events=[]; sigs=[]; queue=[]
    for r in inv:
        task=r.get("task_id",""); eid=r["episode_id"]; subs=list(sem.get(task,{}).get("subgoals",[])); known=r.get("success_label","").lower()
        outcome="success" if known=="true" else "failure" if known=="false" else "unknown"; success=outcome=="success"; failure=outcome=="failure"; complete=subs if success else []
        end=max(0,int(float(r.get("num_steps") or 0))-1); mid=max(0,end//2); ev=[]
        if subs: ev.append({"episode_id":eid,"task_id":task,"step":0,"timestamp":0.0,"event_type":"subgoal_start","target":subs[0],"source":"task_semantics","confidence":0.7})
        if success:
            for i,s in enumerate(subs): ev.append({"episode_id":eid,"task_id":task,"step":max(0,int(end*(i+1)/max(1,len(subs)))),"timestamp":None,"event_type":"subgoal_complete","target":s,"source":"manifest_success_plus_task_semantics","confidence":0.55})
            ev.append({"episode_id":eid,"task_id":task,"step":end,"timestamp":None,"event_type":"episode_success","target":"task","source":"manifest.success","confidence":1.0})
        elif failure: ev.append({"episode_id":eid,"task_id":task,"step":mid,"timestamp":None,"event_type":"failure_onset","target":"task","source":"manifest.success=false","confidence":0.8})
        ev.append({"episode_id":eid,"task_id":task,"step":end,"timestamp":None,"event_type":"episode_end","target":"task","source":"manifest","confidence":1.0}); events.extend(ev)
        path=">".join(complete)
        tags.append({"task_id":task,"episode_id":eid,"outcome":outcome,"success":success,"partial_success":False,"failure":failure,"alternative_order":False,"recovery":False,"retry":False,"revisit":False,"stagnation":False,"path_signature":path,"completed_subgoals":json.dumps(complete,ensure_ascii=False),"failure_count":1 if failure else 0,"recovery_count":0,"retry_count":0,"revisit_count":0,"evidence_source":"manifest_success;task_semantics" if subs else "manifest_success","evidence_ranges":f"0-{end}","label_confidence":"high" if known in {"true","false"} else "low","notes":"No explicit semantic event/recovery annotation in supplied manifest; structural flags conservatively false."})
        sigs.append({"task_id":task,"episode_id":eid,"outcome":outcome,"path_signature":path,"completed_subgoals":complete,"failure_count":1 if failure else 0,"recovery_count":0,"retry_count":0,"revisit_count":0})
        if outcome in {"failure","unknown"}: queue.append({"task_id":task,"episode_id":eid,"reason":"Review for semantic subgoals/recovery; not inferable from manifest"})
    with open(out/"trajectory_tags.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(tags)
    with open(out/"episode_events.jsonl","w",encoding="utf-8") as f:
        for x in events: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    with open(out/"path_signatures.jsonl","w",encoding="utf-8") as f:
        for x in sigs: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    groups=defaultdict(lambda:[0,0,0,0]); bytag={x["episode_id"]:x for x in tags}
    for r in inv:
        t=bytag[r["episode_id"]]; edge="failure" if t["failure"] else "forward"; key=(t["task_id"],t["path_signature"],edge,t["outcome"]); g=groups[key]; g[0]+=1; g[1]+=int(float(r.get("num_steps") or 0)); g[2]+=str(r.get("has_full_episode_history","")).lower()=="true"; g[3]+=str(r.get("has_action_chunks","")).lower()=="true"
    with open(out/"coverage_matrix.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["task_id","path_signature","edge_type","outcome","episode_count","total_steps","full_history_count","action_chunk_count"])
        for k,v in sorted(groups.items()): w.writerow([*k,*v])
    summ=[]
    for task in sorted({r["task_id"] for r in inv}):
        rs=[r for r in inv if r["task_id"]==task]; ts=[t for t in tags if t["task_id"]==task]; paths={t["path_signature"] for t in ts if t["outcome"]=="success" and t["path_signature"]}; n=len(rs)
        summ.append({"task_id":task,"usable_episodes":sum(str(r.get("has_full_episode_history","")).lower()=="true" for r in rs),"success_episodes":sum(t["outcome"]=="success" for t in ts),"failure_episodes":sum(t["outcome"]=="failure" for t in ts),"partial_episodes":0,"distinct_success_paths":len(paths),"alternative_order_episodes":0,"recovery_episodes":0,"retry_episodes":0,"revisit_episodes":0,"full_history_ratio":round(sum(str(r.get("has_full_episode_history","")).lower()=="true" for r in rs)/n,4) if n else 0,"action_chunk_ratio":round(sum(str(r.get("has_action_chunks","")).lower()=="true" for r in rs)/n,4) if n else 0})
    sf=["task_id","usable_episodes","success_episodes","failure_episodes","partial_episodes","distinct_success_paths","alternative_order_episodes","recovery_episodes","retry_episodes","revisit_episodes","full_history_ratio","action_chunk_ratio"]
    with open(out/"task_structure_summary.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=sf); w.writeheader(); w.writerows(summ)
    with open(out/"manual_review_queue.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=["task_id","episode_id","reason"]); w.writeheader(); w.writerows(queue[:40])
    lines=["# Coverage summary","", "Structural flags are conservative: supplied rollout manifests expose outcome and complete-history boundaries but no semantic subgoal/recovery events.",""]
    for s in summ: lines.append(f"- `{s['task_id']}`: {s['usable_episodes']} usable; success={s['success_episodes']}, failure={s['failure_episodes']}, distinct_success_paths={s['distinct_success_paths']}, recovery={s['recovery_episodes']}; action-chunk coverage={s['action_chunk_ratio']:.3f}.")
    lines.append("\nKey gap: alternative-order and recovery evidence must be collected or manually annotated before PathGraph training."); (out/"coverage_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"analyzed {len(tags)} episodes")
if __name__=="__main__": main()

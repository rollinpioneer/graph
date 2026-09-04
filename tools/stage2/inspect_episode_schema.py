#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re, traceback
from pathlib import Path
import numpy as np
from tools.stage2.lib.common import read_jsonl, load_pickle_table, sha256_file

PATTERNS={
    "action": ["action","actions"], "eef_pos":["eef","end_effector","gripper_pos"],
    "gripper_state":["gripper","qpos","grip"], "object_pos":["object","cube","nut","item","obj"],
    "target_pos":["target","goal","site"], "contact":["contact","collision","touch"],
    "reward":["reward"], "done":["done"], "success":["success"],
}

def arr_info(v):
    try:
        a=np.asarray(v)
        d={"shape":list(a.shape),"dtype":str(a.dtype)}
        if a.size and np.issubdtype(a.dtype,np.number):
            z=a.astype(float,copy=False); finite=np.isfinite(z)
            d.update({"min":float(np.nanmin(z)),"max":float(np.nanmax(z)),"mean":float(np.nanmean(z)),"missing_rate":float(1-finite.mean())})
        return d
    except Exception as e: return {"type":type(v).__name__,"error":str(e)}

def field_name_candidates(fields):
    rows=[]
    for f in fields:
        low=f.lower()
        for cat, pats in PATTERNS.items():
            if any(p in low for p in pats):
                rows.append({"field":f,"category":cat,"match":";".join(p for p in pats if p in low),"evidence":"top_level_field_name"})
    if "obs" in fields:
        rows.extend([
            {"field":"obs","category":"observation_vector","match":"obs","evidence":"episode_dataframe_column"},
            {"field":"obs[0:3]","category":"object_pos","match":"known_robomimic_lowdim_layout","evidence":"stage1_config_and_obs_dim"},
            {"field":"obs[14:17]","category":"eef_pos","match":"known_square_layout","evidence":"square_lowdim_obs_keys"},
            {"field":"obs[41:44]","category":"eef_pos","match":"known_transport_layout","evidence":"transport_lowdim_obs_keys"},
            {"field":"obs[21:23]","category":"gripper_state","match":"known_square_layout","evidence":"square_lowdim_obs_keys"},
            {"field":"obs[57:59]","category":"gripper_state","match":"known_transport_layout","evidence":"transport_lowdim_obs_keys"},
        ])
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--per-task-outcome",type=int,default=3); ap.add_argument("--output-dir",required=True); a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(a.manifest)
    selected=[]
    for task in sorted({r.get("task_id","") for r in rows}):
        for outcome in ["success","failure","unknown"]:
            group=[r for r in rows if r.get("task_id")==task and r.get("outcome")==outcome]
            selected.extend(group[:a.per_task_outcome])
    failures=[]; inventory=[]; candidate_hits=[]
    for r in selected:
        p=Path(r["source_path"])
        rec={"episode_id":r["episode_id"],"task_id":r.get("task_id"),"outcome":r.get("outcome"),"source_path":str(p),"source_sha256":sha256_file(p) if p.exists() else None,"manifest_num_steps":r.get("num_steps")}
        try:
            table=load_pickle_table(p)
            rec["fields"]={k:{"length":len(v),"first":arr_info(v[0]) if v else {},"last":arr_info(v[-1]) if v else {}} for k,v in table.items()}
            rec["sequence_length_consensus"]=max((len(v) for v in table.values()),default=0)
            rec["candidate_fields"]=field_name_candidates(list(table))
            candidate_hits.extend([{**c,"episode_id":r["episode_id"],"task_id":r.get("task_id")} for c in rec["candidate_fields"]])
        except Exception as e:
            failures.append({"episode_id":r["episode_id"],"task_id":r.get("task_id"),"outcome":r.get("outcome"),"source_path":str(p),"error":f"{type(e).__name__}: {e}"})
            rec["error"]=f"{type(e).__name__}: {e}"
        inventory.append(rec)
    (out/"schema_inventory.json").write_text(json.dumps({"manifest":a.manifest,"sample_count":len(selected),"selected_episodes":[r["episode_id"] for r in selected],"episodes":inventory,"stage1_placeholder_events_rejected":True},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fields=["task_id","episode_id","category","field","match","evidence"]
    with open(out/"field_mapping_candidates.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(candidate_hits)
    with open(out/"decode_failures.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["episode_id","task_id","outcome","source_path","error"]); w.writeheader(); w.writerows(failures)
    lines=["# Representative episode keys","", "Only metadata, shapes and scalar summary statistics are recorded; raw arrays are not written.",""]
    for r in inventory:
        lines.append(f"## {r['episode_id']} ({r.get('task_id')}, {r.get('outcome')})")
        if "error" in r: lines.append(f"- decode_error: `{r['error']}`"); continue
        lines.append(f"- source: `{r['source_path']}`")
        lines.append(f"- fields: {', '.join(r.get('fields',{}))}")
        lines.append(f"- sequence_length_consensus: {r.get('sequence_length_consensus')}")
        for k,v in r.get("fields",{}).items(): lines.append(f"  - `{k}`: length={v.get('length')}, first={v.get('first')}, last={v.get('last')}")
    (out/"representative_episode_keys.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"inspected {len(selected)} representative episodes; decode_failures={len(failures)}")

if __name__=="__main__": main()

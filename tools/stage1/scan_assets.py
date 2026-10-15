#!/usr/bin/env python3
"""Scan the local rollout manifests into episode-level PathGraph assets."""
from __future__ import annotations
import argparse, csv, json, os, re
from pathlib import Path
from collections import defaultdict

import yaml

CORE = ["task_id", "task_instance_id", "episode_id", "source_path", "source_format", "num_steps", "duration_sec", "fps", "observation_modalities", "camera_names", "action_dim", "has_gripper", "has_proprio", "has_timestamps", "success_label", "original_outcome", "has_subtask_or_stage_labels", "sarm_annotation_path", "has_action_chunks", "action_chunk_size", "has_full_episode_history", "original_split", "metadata_path", "notes"]

def truth(v):
    if isinstance(v, bool): return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "success", "succeeded"}

def int_from_shape(v):
    nums = re.findall(r"-?\d+", str(v or ""))
    return [int(x) for x in nums]

def infer_task(path, row):
    if row.get("task_id"): return str(row["task_id"])
    s = " ".join([str(path), str(row.get("episode_file", "")), str(row.get("video_file", ""))]).lower()
    return "transport" if "transport" in s else "square" if "square" in s else Path(path).stem

def scan_manifest(path):
    out=[]
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task = infer_task(path, row)
            eid = row.get("episode_id") or row.get("episode") or row.get("demo_id")
            eid = f"{task}__episode_{eid}"
            ep = os.path.abspath(os.path.expanduser(row.get("episode_file") or row.get("source_path") or path))
            first = int(float(row.get("first_timestep") or 0))
            last = int(float(row.get("last_timestep") or -1))
            n = max(0, last-first+1) if last >= first else None
            ash = int_from_shape(row.get("action_shape"))
            obs = int_from_shape(row.get("obs_shape"))
            modalities = ["proprio"] if obs else []
            if row.get("video_file") and os.path.exists(os.path.expanduser(row["video_file"])): modalities.insert(0,"RGB")
            out.append({
                "task_id": task, "task_instance_id": "", "episode_id": eid,
                "source_path": ep, "source_format": "pickle_rollout",
                "num_steps": n, "duration_sec": (n/20.0 if n is not None else None), "fps": 20,
                "observation_modalities": ";".join(modalities), "camera_names": "",
                "action_dim": (ash[-1] if ash else None), "has_gripper": "unknown", "has_proprio": bool(obs),
                "has_timestamps": False, "success_label": ("true" if truth(row.get("success")) else "false"),
                "original_outcome": "success" if truth(row.get("success")) else "failure",
                "has_subtask_or_stage_labels": False, "sarm_annotation_path": "",
                "has_action_chunks": bool(ash), "action_chunk_size": (ash[0] if ash else None),
                "has_full_episode_history": bool(os.path.exists(ep) and first == 0 and n and n > 0),
                "original_split": "", "metadata_path": os.path.abspath(path),
                "notes": "CUPID rollout manifest; semantic subgoal/recovery labels absent; decision_points=" + str(row.get("decision_points", "")),
                "_video_path": os.path.abspath(os.path.expanduser(row.get("video_file", ""))) if row.get("video_file") else "",
                "_seed": row.get("seed", ""),
            })
    return out

def scan_hdf5(path):
    import h5py
    out=[]
    task = "square" if "square" in str(path).lower() else Path(path).stem
    with h5py.File(path, "r", locking=False) as f:
        data=f.get("data")
        if data is None: return out
        for name in data.keys():
            g=data[name]; n=int(g.attrs.get("num_samples", g["actions"].shape[0] if "actions" in g else 0))
            obs=g.get("obs"); keys=list(obs.keys()) if obs is not None else []
            out.append({"task_id":task,"task_instance_id":"","episode_id":f"{task}__{name}","source_path":os.path.abspath(path),"source_format":"hdf5","num_steps":n,"duration_sec":n/20.0,"fps":20,"observation_modalities":"proprio" if keys else "","camera_names":"","action_dim":int(g["actions"].shape[-1]) if "actions" in g else None,"has_gripper":"unknown","has_proprio":bool(keys),"has_timestamps":False,"success_label":"true" if "rewards" in g and float(g["rewards"][-1])>0.5 else "unknown","original_outcome":"success" if "rewards" in g and float(g["rewards"][-1])>0.5 else "unknown","has_subtask_or_stage_labels":False,"sarm_annotation_path":"","has_action_chunks":False,"action_chunk_size":None,"has_full_episode_history":True,"original_split":"","metadata_path":os.path.abspath(path),"notes":"Robomimic HDF5 demo group; semantic event labels absent."})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--max-episodes",type=int); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding="utf-8")); roots=cfg["project"]["data_roots"]; rows=[]
    for root in roots:
        p=Path(root)
        if not p.exists(): continue
        rows.extend(scan_manifest(str(p)) if p.suffix.lower()==".csv" else scan_hdf5(str(p)) if p.suffix.lower() in {".h5",".hdf5"} else [])
    if a.max_episodes: rows=rows[:a.max_episodes]
    # Ensure globally unique episode IDs without hiding collisions.
    seen=set()
    for r in rows:
        base=r["episode_id"]; i=1
        while r["episode_id"] in seen: i+=1; r["episode_id"]=f"{base}_{i}"
        seen.add(r["episode_id"])
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fields=CORE
    with open(out/"asset_inventory.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k,"") for k in fields} for r in rows])
    by=defaultdict(list)
    for r in rows: by[r["task_id"]].append(r)
    tf=["task_id","episode_count","success_known_count","failure_known_count","total_steps","full_history_ratio","action_chunk_ratio","sarm_annotation_ratio","modalities"]
    with open(out/"task_summary.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=tf); w.writeheader()
        for t,rs in sorted(by.items()):
            known=[r for r in rs if r["success_label"] in {"true","false"}]; n=len(rs)
            w.writerow({"task_id":t,"episode_count":n,"success_known_count":sum(r["success_label"]=="true" for r in rs),"failure_known_count":sum(r["success_label"]=="false" for r in rs),"total_steps":sum(r["num_steps"] or 0 for r in rs),"full_history_ratio":round(sum(bool(r["has_full_episode_history"]) for r in rs)/n,4) if n else 0,"action_chunk_ratio":round(sum(bool(r["has_action_chunks"]) for r in rs)/n,4) if n else 0,"sarm_annotation_ratio":0,"modalities":";".join(sorted({m for r in rs for m in str(r["observation_modalities"]).split(";") if m}))})
    (out/"data_format_notes.md").write_text("""# Data format notes\n\nThe scan uses the existing CUPID rollout manifest CSVs as the episode index. Each row points to one complete pickle rollout and retains its first/last timestep, success label, video path (in the source manifest), observation shape, and action shape. No video frames were decoded. `episode_file` is the complete-history source; semantic subtask, SARM, timestamp, and explicit recovery fields are absent. The existing robomimic HDF5 loader is documented in `CUPID/repo` and can be scanned by the same CLI when added to `data_roots`.\n\nEpisode boundary: manifest `first_timestep..last_timestep`, requiring first timestep 0 and an existing episode file for `has_full_episode_history=true`.\n""",encoding="utf-8")
    ck=[]
    for root in cfg["project"].get("existing_checkpoint_roots",[]):
        p=Path(root)
        if p.exists(): ck.extend(str(x) for x in p.rglob("*.pt"))
    (out/"code_checkpoint_inventory.md").write_text("# Code and checkpoint inventory\n\n- Dataset/loader: `CUPID/repo` robomimic-compatible data and rollout manifest index.\n- SARM annotations: 未发现。\n- Checkpoints found: %d (`.pt` files under configured roots).\n\nPaths are recorded as source metadata only; weights are not copied.\n%s"%(len(ck),"\n".join("- "+x for x in ck[:100])),encoding="utf-8")
    print(f"scanned {len(rows)} episodes across {len(by)} tasks")
if __name__=="__main__": main()

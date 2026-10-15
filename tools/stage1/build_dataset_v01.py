#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,random
from pathlib import Path
import yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--inventory",required=True); ap.add_argument("--tags",required=True); ap.add_argument("--path-signatures",required=True); ap.add_argument("--output-dir",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding="utf-8")); seed=int(cfg.get("runtime",{}).get("seed",0)); ratios=cfg.get("split",{}).get("ratios",{"train":.8,"val":.1,"test":.1})
    with open(a.inventory,encoding="utf-8") as f: inv=list(csv.DictReader(f))
    with open(a.tags,encoding="utf-8") as f: tags={r["episode_id"]:r for r in csv.DictReader(f)}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]; bytask={}
    for r in inv: bytask.setdefault(r["task_id"],[]).append(r)
    rng=random.Random(seed)
    for task,rs in sorted(bytask.items()):
        rs=sorted(rs,key=lambda x:x["episode_id"]); rng.shuffle(rs); n=len(rs); ntr=int(n*ratios.get("train",.8)); nv=int(n*ratios.get("val",.1))
        for i,r in enumerate(rs):
            split="train" if i<ntr else "val" if i<ntr+nv else "test"; t=tags.get(r["episode_id"],{}); end=max(0,int(float(r.get("num_steps") or 0))-1)
            rows.append({"dataset_version":"pathgraph_stage1_dataset_v0.1","task_id":r["task_id"],"task_instance_id":r.get("task_instance_id", ""),"episode_id":r["episode_id"],"group_id":r["episode_id"],"split":split,"source_path":r["source_path"],"source_format":r["source_format"],"num_steps":r.get("num_steps"),"history_start_step":0,"history_end_step":end,"has_full_episode_history":str(r.get("has_full_episode_history","")).lower()=="true","usable_for_pathgraph":str(r.get("has_full_episode_history","")).lower()=="true","outcome":t.get("outcome","unknown"),"path_signature":t.get("path_signature",""),"alternative_order":t.get("alternative_order",False),"recovery":t.get("recovery",False),"retry":t.get("retry",False),"revisit":t.get("revisit",False),"stagnation":t.get("stagnation",False),"failure_count":t.get("failure_count",0),"recovery_count":t.get("recovery_count",0),"action_chunk_size":r.get("action_chunk_size",""),"sarm_annotation_path":r.get("sarm_annotation_path",""),"metadata_path":r.get("metadata_path","")})
    mf=["dataset_version","task_id","task_instance_id","episode_id","group_id","split","source_path","source_format","num_steps","history_start_step","history_end_step","has_full_episode_history","usable_for_pathgraph","outcome","path_signature","alternative_order","recovery","retry","revisit","stagnation","failure_count","recovery_count","action_chunk_size","sarm_annotation_path","metadata_path"]
    with open(out/"episode_manifest.jsonl","w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    sf=mf[1:6]+["outcome","recovery","alternative_order"]
    with open(out/"splits.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=sf); w.writeheader(); w.writerows([{k:r.get(k,"") for k in sf} for r in rows])
    with open(out/"split_summary.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["task_id","split","episode_count","success","failure","recovery","alternative_order"])
        for task in sorted(bytask):
            for sp in ["train","val","test"]:
                q=[r for r in rows if r["task_id"]==task and r["split"]==sp]; w.writerow([task,sp,len(q),sum(r["outcome"]=="success" for r in q),sum(r["outcome"]=="failure" for r in q),sum(str(r["recovery"]).lower()=="true" for r in q),sum(str(r["alternative_order"]).lower()=="true" for r in q)])
    fps=[]
    for r in rows:
        p=Path(r["source_path"]); st=p.stat() if p.exists() else None; fps.append({"source_path":r["source_path"],"size_bytes":st.st_size if st else "","mtime":st.st_mtime if st else "","episode_id":r["episode_id"]})
    with open(out/"source_fingerprint.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=["source_path","size_bytes","mtime","episode_id"]); w.writeheader(); w.writerows(fps)
    sha=hashlib.sha256((out/"episode_manifest.jsonl").read_bytes()).hexdigest(); (out/"manifest.sha256").write_text(sha+"  episode_manifest.jsonl\n",encoding="utf-8"); leaks={}
    for r in rows: leaks.setdefault(r["group_id"],set()).add(r["split"])
    checks={"episode_id_unique":len({r["episode_id"] for r in rows})==len(rows),"all_rows_have_split":all(r["split"] in {"train","val","test"} for r in rows),"group_leakage_count":sum(len(v)>1 for v in leaks.values()),"manifest_sha256":sha}; (out/"split_checks.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    card=f"""# Dataset card: pathgraph_stage1_dataset_v0.1

- Episode unit: one complete CUPID rollout manifest row; source files are referenced read-only.
- Tasks: {len(bytask)}; episodes: {len(rows)}.
- Labels: outcome from existing `success` field; semantic subgoal/recovery labels are not present.
- Split: deterministic seed {seed}, per-task shuffle with ratios train/val/test={ratios}; group_id=episode_id prevents cross-split episode leakage.
- Known gap: no observed alternative-order paths or explicit recovery events; G0 must gate further collection/annotation.
"""
    (out/"dataset_card.md").write_text(card,encoding="utf-8")
    print(f"built {len(rows)} manifest rows; leakage={checks['group_leakage_count']}")
if __name__=="__main__": main()

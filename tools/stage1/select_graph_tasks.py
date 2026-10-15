#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--task-summary",required=True); ap.add_argument("--coverage",required=True); ap.add_argument("--tags",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--output-dir",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding="utf-8")); sel=cfg.get("selection",{}); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    with open(a.task_summary,encoding="utf-8") as f: ss=list(csv.DictReader(f))
    with open(a.tags,encoding="utf-8") as f: tags=list(csv.DictReader(f))
    with open(a.manifest,encoding="utf-8") as f: man=[json.loads(x) for x in f if x.strip()]
    score=[]
    for s in ss:
        task=s["task_id"]; rec=int(float(s.get("recovery_episodes",0))); paths=int(float(s.get("distinct_success_paths",0))); fh=float(s.get("full_history_ratio",0)); ac=float(s.get("action_chunk_ratio",0)); alt=int(float(s.get("alternative_order_episodes",0))); usable=int(float(s.get("usable_episodes",0)))
        c_paths=3 if paths>=2 else 0; c_rec=3 if rec>=5 else 2 if rec>=2 else 1 if rec==1 else 0; c_fh=2 if fh>=float(sel.get("min_full_history_ratio",.9)) else 0; c_use=1 if usable>=int(sel.get("min_usable_episodes_preferred",20)) else 0; c_ac=1 if ac>=.8 else 0
        score.append({"task_id":task,"paths_score":c_paths,"recovery_score":c_rec,"full_history_score":c_fh,"usable_score":c_use,"action_chunk_score":c_ac,"total_score":c_paths+c_rec+c_fh+c_use+c_ac,"distinct_success_paths":paths,"recovery_episodes":rec,"alternative_order_episodes":alt,"full_history_ratio":fh,"usable_episodes":usable,"action_chunk_ratio":ac,"structural_reason":"multiple_success_paths" if paths>=2 else "recovery" if rec>0 else "no_structure_evidence"})
    score.sort(key=lambda x:(-x["total_score"],-x["usable_episodes"],x["task_id"]))
    fields=list(score[0].keys()) if score else ["task_id","total_score"]
    with open(out/"candidate_task_score.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(score)
    selected=[]
    for s in score[:int(sel.get("num_tasks_to_select",2))]:
        task=s["task_id"]; ts=[t for t in tags if t["task_id"]==task]; paths=sorted({t["path_signature"] for t in ts if t["outcome"]=="success" and t["path_signature"]}); splits={sp:sum(m["task_id"]==task and m["split"]==sp for m in man) for sp in ["train","val","test"]}; structure="alternative_order" if s["alternative_order_episodes"] else "recovery" if s["recovery_episodes"] else "multi_stage_without_observed_branch"; concepts=list(cfg.get("task_semantics",{}).get(task,{}).get("subgoals",[])); selected.append({"task_id":task,"primary_structure":structure,"reason":s["structural_reason"],"usable_episodes":s["usable_episodes"],"split_counts":splits,"success_path_signatures":paths,"recovery_episodes":s["recovery_episodes"],"candidate_node_concepts":concepts})
    status="GO" if len([s for s in score if (s["distinct_success_paths"]>=2 or s["recovery_episodes"]>0) and s["full_history_ratio"]>=.9])>=2 and all(s["total_score"]>=int(sel.get("min_total_score",6)) for s in score[:2]) else "SWITCH"
    with open(out/"targeted_collection_plan.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["task_id","scenario","current_count","target_count","gap","collection_instruction","required_metadata"]); w.writeheader()
        if status=="SWITCH":
            for task in [s["task_id"] for s in score[:2]]: w.writerow({"task_id":task,"scenario":"alternative_order_and_recovery","current_count":0,"target_count":10,"gap":10,"collection_instruction":"Collect 10 complete episodes with a second valid subgoal order and 10 episodes containing one documented failure followed by legal recovery.","required_metadata":"episode_id, semantic subgoal events, failure_onset, recovery_complete, timestamps, full-history source"})
    (out/"selected_tasks.yaml").write_text(yaml.safe_dump({"dataset_version":"pathgraph_stage1_dataset_v0.1","manifest":a.manifest,"g0_status":status,"selected_tasks":selected},sort_keys=False,allow_unicode=True),encoding="utf-8")
    ev=["# Candidate task evidence","","Evidence source: existing CUPID rollout manifests and stage semantics; no semantic event labels were found, so branch/recovery claims are not upgraded from outcome-only data.",""]
    for s in score[:5]: ev.append(f"- `{s['task_id']}`: score={s['total_score']}, usable={s['usable_episodes']}, success paths={s['distinct_success_paths']}, recovery={s['recovery_episodes']}, full history={s['full_history_ratio']:.3f}; reason={s['structural_reason']}.")
    (out/"candidate_task_evidence.md").write_text("\n".join(ev)+"\n",encoding="utf-8")
    decision=f"# G0 路线决定\n\n- 状态：{status}\n- 数据版本：pathgraph_stage1_dataset_v0.1\n- 选定任务：{', '.join(s['task_id'] for s in selected) or 'none'}\n- 结构证据：现有 manifest 提供完整成功/失败 rollout，但没有可验证的不同合法顺序或 episode 内恢复事件。\n- 完整历史：候选 rollout 的起始步为 0，episode 文件存在，已纳入 manifest。\n- 当前关键 edge 数量：forward/failure 已覆盖；alternative/recovery=0（未标注，不等于物理上不存在）。\n- 需要补采：见 targeted_collection_plan.csv。\n- 阶段 2 入口：若补采后达到门控，读取 selected_tasks.yaml，建立 Graph spec v1 和人工 GT 标注协议。\n"; (out/"g0_decision.md").write_text(decision,encoding="utf-8")
    hand=["# Stage 2 handoff","",f"G0 status: **{status}**.","","Direct inputs:",f"- `{a.manifest}`",f"- `{a.tags}`",f"- `{a.coverage}`","", "Candidate semantic nodes (not frozen):"]
    for s in selected: hand.append(f"- `{s['task_id']}`: " + " -> ".join(s["candidate_node_concepts"] + ["success_terminal"]))
    hand += ["", "First command after evidence is available: create Graph spec v1 and the annotation manual, then annotate failure_onset/recovery_complete and alternative path signatures on complete episodes."]
    (out/"stage2_handoff.md").write_text("\n".join(hand)+"\n",encoding="utf-8")
    print(f"selected {len(selected)} candidates; G0={status}")
if __name__=="__main__": main()

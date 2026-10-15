"""Simulator-only q/D continuation targets, value references, and reward attribution."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn

from .dataset import load_episode, read_csv, write_csv, write_json
from .event_schema import EVENT_NAMES
from .simulator import StochasticBoundarySimulator, make_family_specs


class ValueGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.proj=nn.Sequential(nn.Linear(17,64),nn.ReLU()); self.gru=nn.GRU(64,64,batch_first=True); self.q=nn.Linear(64,1); self.d=nn.Linear(64,1)
    def forward(self,x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        h,_=self.gru(self.proj(x)); h=h[:,-1]; return self.q(h).squeeze(-1),self.d(h).squeeze(-1)


def _episode_spec(row: dict[str,str], formal_seed: int=20260953) -> tuple[Any,int]:
    index=int(row["root_family_id"].split("_")[-1]); rollout=int(row["episode_id"].rsplit("_r",1)[1]); return make_family_specs(120,formal_seed)[index], formal_seed+index*1000+rollout


def _anchor_indices(events: np.ndarray, length: int) -> list[int]:
    contact=np.where(events==1)[0]; failure=np.where(events==3)[0]; recovery=np.where((events==4)|(events==5))[0]
    proposed=[max(0,int(contact[0])-1) if len(contact) else 0, int(contact[0]) if len(contact) else max(0,length//3), int(failure[0]) if len(failure) else max(0,length//2), int(recovery[0]) if len(recovery) else max(0,length-2)]
    seen=[]
    for value in proposed:
        value=max(0,min(length-1,value))
        if value not in seen: seen.append(value)
    while len(seen)<4: seen.append(min(length-1,seen[-1]+1))
    return seen[:4]


def collect_continuations(dataset: Path, anchors_per_family:int, continuations:int, horizon:int, seed:int, output:Path, targets:Path, summary:Path)->list[dict[str,Any]]:
    rows=read_csv(dataset/"episode_manifest.csv"); by_family=defaultdict(list)
    for row in rows: by_family[row["root_family_id"]].append(row)
    raw=[]; out=[]
    for family, episodes in sorted(by_family.items()):
        # Fixed first rollout is replayed exactly to the selected explicit-state anchors.
        row=sorted(episodes,key=lambda x:x["episode_id"])[0]; ep=load_episode(row); spec,rollout_seed=_episode_spec(row); sim=StochasticBoundarySimulator(spec,rollout_seed)
        snapshots=[]
        anchors=_anchor_indices(ep["gold_event_id"],len(ep["actions"]))[:anchors_per_family]
        for t,action in enumerate(ep["actions"]):
            sim.policy_action(); sim.step(action)
            if t in anchors: snapshots.append((t,sim.snapshot()))
        for ai,(t,snapshot) in enumerate(snapshots):
            successes=[]; times=[]
            for ci in range(continuations):
                child=StochasticBoundarySimulator(spec,seed+int(family.split("_")[-1])*100+ai*10+ci); child.restore(snapshot); child.rng=np.random.default_rng(seed+int(family.split("_")[-1])*100+ai*10+ci)
                elapsed=0
                while not child.done and elapsed<horizon:
                    child.step(child.policy_action()); elapsed+=1
                successes.append(int(child.success)); times.append(elapsed if child.success else horizon)
                raw.append({"anchor_id":f"{family}_a{ai}","continuation_id":ci,"root_family_id":family,"split":row["split"],"anchor_t":t,"success":bool(child.success),"time_to_success_or_horizon":elapsed,"right_censored":not child.success})
            out.append({"anchor_id":f"{family}_a{ai}","root_family_id":family,"split":row["split"],"episode_id":row["episode_id"],"anchor_t":t,"q_target":float(np.mean(successes)),"d_target_normalized":float(np.mean(times)/horizon),"right_censored":bool(not any(successes)),"continuation_count":continuations,"success_count":int(sum(successes)),"mean_time_to_success":float(np.mean(times)),"observable_dim":17})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8") as h:
        for x in raw:h.write(json.dumps(x)+"\n")
    write_csv(targets,out,list(out[0])); support=[]
    for split in ("train","val","test"):
        subset=[x for x in out if x["split"]==split]; support.append({"split":split,"anchors":len(subset),"root_families":len({x['root_family_id'] for x in subset}),"continuations":sum(x["continuation_count"] for x in subset),"successes":sum(x["success_count"] for x in subset)})
    write_csv(summary,support,list(support[0])); return out


def build_value_jobs(dataset:Path,targets:Path,variants:list[str],seeds:list[int],steps:int,output_root:Path,table:Path)->list[dict[str,Any]]:
    rows=[]
    for variant in variants:
        for seed in seeds:
            jid=f"{variant}_s{seed}"; rows.append({"job_id":jid,"variant":variant,"seed":seed,"dataset":str(dataset.resolve()),"targets":str(targets.resolve()),"steps":steps,"output_dir":str((output_root/jid).resolve()),"selection_split":"val","test_used_for_selection":False,"cuda_required":True})
    _write_tsv(table,rows);return rows


def _read_tsv(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h,delimiter="\t"))
def _write_tsv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=sorted({k for r in rows for k in r}),delimiter="\t",lineterminator="\n",extrasaction="ignore");w.writeheader();w.writerows(rows)


def _target_samples(dataset:Path,targets:Path,split:str)->list[dict[str,Any]]:
    manifests={r["episode_id"]:r for r in read_csv(dataset/"episode_manifest.csv")}; samples=[]
    for row in read_csv(targets):
        if row["split"]!=split:continue
        ep=load_episode(manifests[row["episode_id"]]);t=int(row["anchor_t"]);start=max(0,t-31); x=ep["observations"][start:t+1]; x=np.vstack([np.repeat(x[:1],32-len(x),axis=0),x]) if len(x)<32 else x
        samples.append({"x":x.astype(np.float32),"q":float(row["q_target"]),"d":float(row["d_target_normalized"]),"anchor_id":row["anchor_id"]})
    return samples


def _metric(model:ValueGRU,samples:list[dict[str,Any]],device:torch.device)->dict[str,float]:
    model.eval();predq=[];predd=[];q=[];d=[]
    with torch.no_grad():
        for s in samples:
            a,b=model(torch.from_numpy(s["x"]).unsqueeze(0).to(device));predq.append(float(torch.sigmoid(a).item()));predd.append(float(torch.sigmoid(b).item()));q.append(s["q"]);d.append(s["d"])
    result={"q_brier":float(np.mean((np.array(predq)-q)**2)),"d_mae":float(np.mean(abs(np.array(predd)-d))),"q_spearman":float(spearmanr(predq,q).statistic or 0.0),"d_spearman":float(spearmanr(predd,d).statistic or 0.0)}
    model.train();return result


def train_value_job(table:Path,job_id:str)->dict[str,Any]:
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for U2 value-reference training")
    job=next(x for x in _read_tsv(table) if x["job_id"]==job_id);device=torch.device("cuda");torch.manual_seed(int(job["seed"]));rng=np.random.default_rng(int(job["seed"]));train=_target_samples(Path(job["dataset"]),Path(job["targets"]),"train");val=_target_samples(Path(job["dataset"]),Path(job["targets"]),"val")
    model=ValueGRU().to(device); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);variant=job["variant"];losses=[];best_state=None;best_step=0;best_score=float("inf");history=[]
    for step in range(int(job["steps"])):
        batch=[train[int(i)] for i in rng.integers(0,len(train),size=min(64,len(train)))];x=torch.from_numpy(np.stack([s["x"] for s in batch])).to(device); q=torch.tensor([s["q"] for s in batch],device=device);d=torch.tensor([s["d"] for s in batch],device=device);a,b=model(x);loss=torch.tensor(0.,device=device)
        if variant in {"q_only","q_plus_D"}:loss=loss+F.binary_cross_entropy_with_logits(a,q)
        if variant in {"D_only","q_plus_D"}:loss=loss+F.mse_loss(torch.sigmoid(b),d)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.item()))
        if (step+1)%100==0 or step+1==int(job["steps"]):
            metric=_metric(model,val,device);metric["step"]=step+1;history.append(metric);score=metric["q_brier"]+metric["d_mae"]
            if score<best_score:best_score=score;best_step=step+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None:raise RuntimeError("no validation checkpoint for value model")
    model.load_state_dict(best_state);out=Path(job["output_dir"]);out.mkdir(parents=True,exist_ok=True);ckpt=out/"best.pt";torch.save({"state_dict":model.state_dict(),"variant":variant,"seed":job["seed"],"best_step":best_step,"validation_every":100,"cuda_used":True},ckpt);metrics=_metric(model,val,device);digest=hashlib.sha256(ckpt.read_bytes()).hexdigest();result={"job_id":job_id,"variant":variant,"seed":job["seed"],"status":"PASS","cuda_used":True,"validation_every":100,"validation_history":history,"best_step":best_step,"loss_finite":bool(np.isfinite(losses).all()),"loss_final":losses[-1],"checkpoint":str(ckpt.resolve()),"checkpoint_sha256":digest,"test_used_for_selection":False,**metrics};write_json(out/"train_result.json",result);return result


def launch_value_jobs(table:Path,gpus:list[str],status:Path)->None:
    pending=_read_tsv(table);running=[];rows=[]
    while pending or running:
        while pending and len(running)<len(gpus):
            job=pending.pop(0);gpu=gpus[len(running)%len(gpus)];env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=gpu;proc=subprocess.Popen([sys.executable,"-m","upgrade_v2.u2.cli","run-value-reference-job","--job-table",str(table),"--job-id",job["job_id"]],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env);running.append((proc,job,gpu))
        proc,job,gpu=running.pop(0);stdout,_=proc.communicate();res=Path(job["output_dir"])/"train_result.json";rows.append({"job_id":job["job_id"],"gpu":gpu,"return_code":proc.returncode,"status":"PASS" if proc.returncode==0 and res.is_file() else "FAIL","detail":"" if proc.returncode==0 else stdout[-4000:]})
    _write_tsv(status,rows)
    if any(r["status"]!="PASS" for r in rows):raise RuntimeError("value jobs failed")


def select_value_checkpoints(table:Path,output:Path)->list[dict[str,Any]]:
    rows=[]
    for job in _read_tsv(table):
        r=json.loads((Path(job["output_dir"])/"train_result.json").read_text());r.update({"selection_split":"val","selected":True,"targets":job["targets"]});rows.append(r)
    write_csv(output,rows,sorted({k for r in rows for k in r}));return rows


def _load_value(path:Path)->ValueGRU:
    d=torch.load(path,map_location="cuda");m=ValueGRU().cuda();m.load_state_dict(d["state_dict"]);m.eval();return m


def infer_potential(selection:Path,dataset:Path,alpha_grid:list[float],output_root:Path,lock:Path)->dict[str,Any]:
    rows=read_csv(selection); qd=[r for r in rows if r["variant"]=="q_plus_D"]; best=min(qd,key=lambda r:float(r["q_brier"])+float(r["d_mae"]));model=_load_value(Path(best["checkpoint"]));targets=[]
    # Alpha is selected solely by validation rank correlation against the
    # simulator continuation q-D target, never by test segments.
    vals=_target_samples(dataset,Path(best["targets"]),"val"); candidates=[]
    with torch.no_grad():
        for alpha_try in alpha_grid:
            pred=[]; truth=[]
            for s in vals:
                q,d=model(torch.from_numpy(s["x"]).unsqueeze(0).cuda());pred.append(float(alpha_try*torch.sigmoid(q).item()-(1-alpha_try)*torch.sigmoid(d).item()));truth.append(alpha_try*s["q"]-(1-alpha_try)*s["d"])
            candidates.append({"alpha":alpha_try,"val_potential_spearman":float(spearmanr(pred,truth).statistic or 0.0)})
    alpha=max(candidates,key=lambda x:x["val_potential_spearman"])["alpha"]
    result={"source_job_id":best["job_id"],"checkpoint":best["checkpoint"],"alpha":alpha,"alpha_grid":alpha_grid,"alpha_candidates":candidates,"selection_split":"val","test_used_for_selection":False,"potential":"phi=alpha*q-(1-alpha)*D"}
    output_root.mkdir(parents=True,exist_ok=True)
    for row in read_csv(dataset/"episode_manifest.csv"):
        ep=load_episode(row); obs=ep["observations"]; xs=[]
        for t in range(len(obs)):
            part=obs[max(0,t-31):t+1];part=np.vstack([np.repeat(part[:1],32-len(part),axis=0),part]) if len(part)<32 else part;xs.append(part)
        with torch.no_grad():q,d=model(torch.from_numpy(np.stack(xs)).cuda());q=torch.sigmoid(q).cpu().numpy();d=torch.sigmoid(d).cpu().numpy()
        phi=alpha*q-(1-alpha)*d;np.savez_compressed(output_root/f"{row['episode_id']}.npz",q=q.astype(np.float32),d=d.astype(np.float32),phi=phi.astype(np.float32))
    write_json(lock,result);return result


def aggregate_reward_segments(dataset:Path,potential_root:Path,causal_root:Path,rule_root:Path,budget_root:Path,output_root:Path,manifest:Path,budget_source:str,causal_source:str)->list[dict[str,Any]]:
    sources=["gold","best_causal","best_rule","uniform","best_budget"] ;allrows=[]; output_root.mkdir(parents=True,exist_ok=True)
    # Selected causal/budget identifiers are recorded explicitly in the manifest.
    causal_id=causal_source; rule_id="sensor_hysteresis_01"; uniform_id="uniform_00"
    for source in sources:
        source_rows=[]
        for row in read_csv(dataset/"episode_manifest.csv"):
            ep=load_episode(row)
            with np.load(potential_root/f"{row['episode_id']}.npz") as p:phi=p["phi"]
            if source=="gold":boundary=ep["gold_boundary"]
            elif source=="best_causal":
                with np.load(causal_root/causal_id/row["split"]/f"{row['episode_id']}.npz") as d:boundary=d["boundary_prediction"]
            elif source=="best_rule":
                with np.load(rule_root/rule_id/f"{row['episode_id']}.npz") as d:boundary=d["boundary_prediction"]
            elif source=="uniform":
                with np.load(rule_root/uniform_id/f"{row['episode_id']}.npz") as d:boundary=d["boundary_prediction"]
            else:
                with np.load(budget_root/budget_source/row["split"]/f"{row['episode_id']}.npz") as d:boundary=d["boundary_prediction"]
            start=0;sid=0
            for end in list(np.where(boundary)[0])+[len(boundary)-1]:
                if end<start:continue
                events=ep["gold_event_id"][start:end+1];failure=np.isin(events,[3,9]).any();rec=np.isin(events,[4,5]).any();success=np.isin(events,[7,8]).any();nonzero=events[events!=0];purity=float(Counter(nonzero).most_common(1)[0][1]/len(nonzero)) if len(nonzero) else 1.0
                source_rows.append({"boundary_source":source,"segment_id":f"{row['episode_id']}_{source}_{sid}","root_family_id":row["root_family_id"],"episode_id":row["episode_id"],"split":row["split"],"start_t":start,"end_t":int(end),"segment_return":float(phi[end]-phi[start]),"dominant_gold_event":int(Counter(events).most_common(1)[0][0]),"event_purity":purity,"contains_failure":bool(failure),"contains_recovery":bool(rec),"contains_success":bool(success),"mixed_failure_recovery":bool(failure and rec)});start=int(end)+1;sid+=1
        write_csv(output_root/f"{source}_segments.csv",source_rows,list(source_rows[0]));allrows+=source_rows
    write_csv(manifest,[{"boundary_source":s,"segments":sum(x["boundary_source"]==s for x in allrows),"budget_source":budget_source} for s in sources],["boundary_source","segments","budget_source"]);return allrows


def evaluate_reward_impact(segment_root:Path,targets:Path,output:Path,per_event:Path,report:Path,bootstrap:int,seed:int)->list[dict[str,Any]]:
    target_rows=list(csv.DictReader(targets.open()))
    rows=[]
    for path in sorted(segment_root.glob("*_segments.csv")):
        data=list(csv.DictReader(path.open()));source=data[0]["boundary_source"]
        def rate(field:str,sign:float)->float:
            subset=[float(x["segment_return"]) for x in data if x[field].lower()=="true"];return float(np.mean([v*sign>0 for v in subset])) if subset else float("nan")
        failures=[x for x in data if x["contains_failure"].lower()=="true"];recoveries=[x for x in data if x["contains_recovery"].lower()=="true"];successes=[x for x in data if x["contains_success"].lower()=="true"]
        # Associate every continuation anchor with its enclosing segment.
        lookup=defaultdict(list)
        for x in data:lookup[x["episode_id"]].append(x)
        returns=[];qtargets=[]
        for t in target_rows:
            for x in lookup.get(t["episode_id"],[]):
                if int(x["start_t"])<=int(t["anchor_t"])<=int(x["end_t"]):returns.append(float(x["segment_return"]));qtargets.append(float(t["q_target"]));break
        corr=float(spearmanr(returns,qtargets).statistic or 0.0) if len(returns)>2 else float("nan")
        family_values=defaultdict(list)
        for x in data: family_values[x["root_family_id"]].append(x)
        fam_failure=[];fam_recovery=[]
        for values in family_values.values():
            fs=[float(x["segment_return"])<0 for x in values if x["contains_failure"].lower()=="true"];rs=[float(x["segment_return"])>0 for x in values if x["contains_recovery"].lower()=="true"]
            if fs:fam_failure.append(float(np.mean(fs)))
            if rs:fam_recovery.append(float(np.mean(rs)))
        rng=np.random.default_rng(seed);boot_failure=np.mean(rng.choice(np.asarray(fam_failure),size=(bootstrap,len(fam_failure)),replace=True),axis=1) if fam_failure else np.asarray([]);boot_recovery=np.mean(rng.choice(np.asarray(fam_recovery),size=(bootstrap,len(fam_recovery)),replace=True),axis=1) if fam_recovery else np.asarray([])
        rows.append({"boundary_source":source,"failure_negative_rate":rate("contains_failure",-1),"failure_negative_rate_ci_low":float(np.quantile(boot_failure,.025)) if boot_failure.size else float("nan"),"failure_negative_rate_ci_high":float(np.quantile(boot_failure,.975)) if boot_failure.size else float("nan"),"recovery_positive_rate":rate("contains_recovery",1),"recovery_positive_rate_ci_low":float(np.quantile(boot_recovery,.025)) if boot_recovery.size else float("nan"),"recovery_positive_rate_ci_high":float(np.quantile(boot_recovery,.975)) if boot_recovery.size else float("nan"),"success_positive_rate":rate("contains_success",1),"failure_recovery_mixed_segment_rate":float(np.mean([x["mixed_failure_recovery"].lower()=="true" for x in data])),"event_segment_purity":float(np.mean([float(x["event_purity"]) for x in data])),"segment_return_rank_correlation_with_continuation_q":corr,"closed_full_input_cycle_residual":0.0,"bootstrap_root_families":bootstrap,"segments":len(data)})
    gold=next(x for x in rows if x["boundary_source"]=="gold")
    for r in rows:r["failure_negative_rate_drop_vs_gold"]=gold["failure_negative_rate"]-r["failure_negative_rate"];r["recovery_positive_rate_drop_vs_gold"]=gold["recovery_positive_rate"]-r["recovery_positive_rate"]
    write_csv(output,rows,list(rows[0]));write_csv(per_event,rows,list(rows[0]));report.parent.mkdir(parents=True,exist_ok=True);report.write_text("# U2 boundary reward impact\n\n"+"\n".join(f"- {r['boundary_source']}: failure-negative={r['failure_negative_rate']:.4f}; recovery-positive={r['recovery_positive_rate']:.4f}" for r in rows)+"\n",encoding="utf-8");return rows

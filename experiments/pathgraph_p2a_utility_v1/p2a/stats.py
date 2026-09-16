"""Crossed seed/family paired bootstrap; transition counts are not sample size."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from .io import new_dir,dump
from .weights import METHODS

def paired_interval(diff,seed=7301,n_boot=10000,alpha=.025):
    d=np.asarray(diff,float)
    if d.ndim!=2 or not d.size or not np.isfinite(d).all(): raise ValueError('seed x family finite matrix')
    rng=np.random.default_rng(seed); s,f=d.shape
    si=rng.integers(0,s,size=(n_boot,s)); fi=rng.integers(0,f,size=(n_boot,f))
    estimates=d[si[:,:,None],fi[:,None,:]].mean((1,2))
    lo,hi=np.quantile(estimates,[alpha/2,1-alpha/2])
    return dict(mean=float(d.mean()),low=float(lo),high=float(hi),confidence=1-alpha,
        n_seeds=s,n_families=f,resamples=n_boot,unit='crossed_policy_seed_and_eval_family')

def analyze(root,out,smoke=False):
    root=Path(root); grouped={}; full_rows=[]
    for p in root.glob('**/episodes.csv'):
        for r in csv.DictReader(p.open()):
            key=(r['method'],int(r['policy_seed']),int(r['family']),r['condition'],r['episode_id'])
            if key in grouped: raise ValueError('duplicate eval record: '+str(key))
            grouped[key]=r; full_rows.append(r)
    present={r['method'] for r in full_rows}
    seeds=sorted({int(r['policy_seed']) for r in full_rows}); families=sorted({int(r['family']) for r in full_rows})
    required=set(METHODS) if not smoke else present
    if required-present: raise ValueError('missing methods: '+str(required-present))
    if not smoke:
        if seeds!=[17,29,43,71,101] or families!=list(range(2013000,2013016)):
            raise ValueError('formal requires preregistered seeds and test families')
        if len(full_rows)!=9*5*1024: raise ValueError('incomplete formal test matrix')
        if any(int(r['training_steps'])!=5000 or r['smoke_only'].lower()=='true' for r in full_rows):
            raise ValueError('smoke/short-trained model in formal results')
    expected=None; matrices={}; method_means={}
    for m in sorted(present):
        rr=[r for r in full_rows if r['method']==m]
        keys={(int(r['policy_seed']),int(r['family']),r['condition'],r['episode_id']) for r in rr}
        if expected is None: expected=keys
        if keys!=expected: raise ValueError('unpaired methods; retain failures as rows, do not drop')
        mat=np.empty((len(seeds),len(families)))
        for i,s in enumerate(seeds):
            for j,f in enumerate(families):
                here=[r for r in rr if int(r['policy_seed'])==s and int(r['family'])==f]
                cs=sorted({r['condition'] for r in here})
                mat[i,j]=np.mean([np.mean([int(r['success']) for r in here if r['condition']==c]) for c in cs])
        matrices[m]=mat; method_means[m]=float(mat.mean())
    comparisons={}; selected='V6_CAP_POTENTIAL'
    if selected in matrices:
        for b in sorted(present-{selected}): comparisons[b]=paired_interval(matrices[selected]-matrices[b])
    primary=('BC_UNIFORM','VALID_COUNT_PLUS_MATCHED_EVENTS_V1')
    if smoke: state='SMOKE_ONLY_NOT_RESEARCH_RESULT'
    elif all(comparisons[b]['low']>0 and comparisons[b]['mean']>=.03 for b in primary):
        state='UTILITY_SUPPORTED_VS_TWO_PRESPECIFIED_BASELINES_IN_REFERENCE_MDP'
    elif all(comparisons[b]['high']<0 for b in primary): state='UTILITY_DEGRADED_IN_REFERENCE_MDP'
    else: state='UTILITY_NOT_ESTABLISHED_OR_MIXED'
    out=new_dir(out)
    condition_rows=[]
    for m in sorted(present):
        for condition in sorted({r['condition'] for r in full_rows}):
            rr=[r for r in full_rows if r['method']==m and r['condition']==condition]
            nloss=sum(int(r['observed_losses']) for r in rr); nrestore=sum(int(r['restored_losses']) for r in rr)
            exposed=[r for r in rr if int(r['observed_losses'])>0]
            condition_rows.append(dict(method=m,condition=condition,episodes=len(rr),
                success_rate=float(np.mean([int(r['success']) for r in rr])),
                episodes_with_observed_loss=len(exposed),observed_loss_events=nloss,restored_events=nrestore,
                restore_given_observed_loss=None if not nloss else nrestore/nloss,
                success_given_loss_exposure=None if not exposed else float(np.mean([int(r['success']) for r in exposed])),
                mean_steps=float(np.mean([int(r['steps']) for r in rr])),
                mean_wait_fraction=float(np.mean([float(r['wait_fraction']) for r in rr]))))
    with (out/'per_condition_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(condition_rows[0]));w.writeheader();w.writerows(condition_rows)
    dump(out/'comparison.json',dict(method_means=method_means,v6_minus_baseline=comparisons,
        primary=list(primary),minimum_effect=.03,bonferroni_primary_intervals=True,
        dataset_replications=1,generalization_scope='fixed training dataset; held-out families in declared abstract MDP'))
    dump(out/'decision.json',dict(execution_status='COMPLETE',scientific_status=state,
        policy_utility_evidence=state,robot_policy_gain_claimed=False,visual_grounding_claimed=False,
        physical_cycle_claim='NOT_EVALUATED',confirmation_passed=False,smoke_only=smoke))
    return out

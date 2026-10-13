"""Frozen paired scene and consumed-start bootstrap; no model or physics queries."""
import os,sys,json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/xushijie3/work/graph_saw_goalprobe05_20261013/experiments/saw_goalprobe05_20261013/vendor/analysis_deps')
import numpy as np,pandas as pd
assert len(os.sched_getaffinity(0))<=4
MODE=sys.argv[1];assert MODE in ('I','FINAL')
def write(p,x):
    q=Path(p);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(q)
def export(p,rows):
    assert not Path(p).exists();pd.DataFrame(rows).to_csv(p,index=False)
arms=[0,1] if MODE=='I' else [0,1,2,3]
for r in range(3):
    for a in arms:
        st=json.loads((ROOT/'results'/f'b{r}_a{a}_status.json').read_text());assert st['status']=='COMPLETE_500K' and st['student_updates']==500000
episodes=pd.concat([pd.read_csv(ROOT/'results'/f'b{r}_a{a}_u{u}_episodes.csv') for r in range(3) for a in arms for u in (100000,200000,300000,400000,500000)],ignore_index=True)
assert not episodes.duplicated(['block','arm','student_step','scene_id']).any()
primary=episodes[episodes.student_step.isin([300000,400000,500000])]
score=[];ys={};contrast=[];taskstats=[];curves={};bootstore={}
contrasts=[('a0-a1',0,1)]+([('a2-a1',2,1),('a2-a0',2,0),('a2-a3',2,3),('a3-a1',3,1)] if MODE=='FINAL' else [])
for r in range(3):
    matrix=np.empty((len(arms),3,5,2,25),float)
    for a in arms:
        curves[r,a]=[]
        for u in (100000,200000,300000,400000,500000):
            part=episodes[(episodes.block==r)&(episodes.arm==a)&(episodes.student_step==u)]
            assert len(part)==(100 if u<300000 else 250)
            overall=100*part.groupby('task').success.mean().mean();curves[r,a].append(float(overall))
            for task in range(1,6):
                g=part[part.task==task];score.append(dict(block=r,arm=a,student_step=u,task=task,episodes=len(g),success_rate_percent=float(100*g.success.mean())))
            score.append(dict(block=r,arm=a,student_step=u,task='ALL_EQUAL_WEIGHT',episodes=len(part),success_rate_percent=float(overall)))
        ys[r,a]=float(np.mean(curves[r,a][2:]))
        for ui,u in enumerate((300000,400000,500000)):
            for t in range(1,6):
                for si,s in enumerate((27101307,28101307)):
                    g=primary[(primary.block==r)&(primary.arm==a)&(primary.student_step==u)&(primary.task==t)&(primary.stream==s)].sort_values('episode');assert list(g.episode)==list(range(25))
                    matrix[a,ui,t-1,si]=g.success.to_numpy()
    rng=np.random.default_rng(29101307+r);draw=rng.integers(0,25,size=(5000,5,2,25))
    # The same stratum draw carries all arms and all three primary checkpoints.
    sampled=np.empty((5000,len(arms),5))
    for t in range(5):
        for a in arms:
            means=[]
            for si in range(2):means.append(matrix[a,:,t,si][:,draw[:,t,si,:]].mean(axis=(0,2)))
            sampled[:,a,t]=100*np.mean(means,axis=0)
    bmean=sampled.mean(axis=2);bootstore[r]=bmean
    for a in arms:
        for t in range(5):
            vals=sampled[:,a,t];point=float(100*matrix[a,:,t].mean());lo,hi=np.quantile(vals,[.025,.975]);taskstats.append(dict(block=r,arm=a,task=t+1,window_mean_percent=point,scene_ci_low=float(lo),scene_ci_high=float(hi),bootstrap_seed=29101307+r,bootstrap_draws=5000))
        lo,hi=np.quantile(bmean[:,a],[.025,.975]);taskstats.append(dict(block=r,arm=a,task='ALL_EQUAL_WEIGHT',window_mean_percent=ys[r,a],scene_ci_low=float(lo),scene_ci_high=float(hi),bootstrap_seed=29101307+r,bootstrap_draws=5000))
    for name,a,b in contrasts:
        point=ys[r,a]-ys[r,b];vals=bmean[:,a]-bmean[:,b];lo,hi=np.quantile(vals,[.025,.975])
        contrast.append(dict(block=r,contrast=name,effect_pp=point,effect_unit='percentage_points',scene_ci_low=float(lo),scene_ci_high=float(hi),bootstrap_seed=29101307+r,bootstrap_draws=5000,scope='fixed_models_scene_uncertainty',primary_checkpoints='300k,400k,500k'))
    if MODE=='FINAL':
        denom=ys[r,0]-ys[r,1];assert denom>=20
        valsden=bmean[:,0]-bmean[:,1];valid=np.abs(valsden)>1e-12;values=(bmean[valid,2]-bmean[valid,1])/valsden[valid];lo,hi=np.quantile(values,[.025,.975])
        contrast.append(dict(block=r,contrast='recovery_ratio',effect_pp=None,recovery_ratio=(ys[r,2]-ys[r,1])/denom,effect_unit='ratio',scene_ci_low=float(lo),scene_ci_high=float(hi),bootstrap_seed=29101307+r,bootstrap_draws=int(valid.sum()),scope='ratio_conditional_on_nonzero_bootstrap_denominator',primary_checkpoints='300k,400k,500k'))
gap=[ys[r,0]-ys[r,1] for r in range(3)];hr=all(g>=20 for g in gap)
direction_stable=all(all(curves[r,0][i]-curves[r,1][i]>0 for i in (2,3,4)) for r in range(3))
gate=dict(H_R_pass=hr,raw_gap_pp=gap,raw_window_scores={str(r):{f'a{a}':ys[r,a] for a in arms} for r in range(3)},H_R_threshold_pp=20,primary_checkpoints=[300000,400000,500000],primary_point_directions_stable=direction_stable,verdict=('REEXTRACTION_GAP_WITH_TEMPORAL_INSTABILITY' if hr and not direction_stable else ('REEXTRACTION_GAP_CONFIRMED_IN_THIS_PLATFORM' if hr else 'REEXTRACTION_GAP_NOT_CONFIRMED_IN_THIS_PLATFORM')),a0_learning_diagnostic='Report a0 vs source A and full task curves descriptively; no new threshold introduced.')
if MODE=='I':write(ROOT/'results/I_GATE.json',gate)
out=ROOT/'results'/f'{MODE}_STATISTICS';out.mkdir(exist_ok=False)
export(out/'CHECKPOINT_SCORES.csv',score);export(out/'PRIMARY_WINDOW_SCORES.csv',taskstats);export(out/'PAIRED_BLOCK_CONTRASTS.csv',contrast)
summaries=[]
for a in arms:
    vals=np.array([ys[r,a] for r in range(3)]);summaries.append(dict(series=f'a{a}',n=3,mean_percent=float(vals.mean()),sample_sd=float(vals.std(ddof=1)),minimum=float(vals.min()),maximum=float(vals.max()),raw_values=json.dumps(vals.tolist())))
for name,a,b in contrasts:
    vals=np.array([ys[r,a]-ys[r,b] for r in range(3)]);summaries.append(dict(series=name,n=3,mean_percent=float(vals.mean()),sample_sd=float(vals.std(ddof=1)),minimum=float(vals.min()),maximum=float(vals.max()),raw_values=json.dumps(vals.tolist())))
export(out/'TRAINING_BLOCK_DESCRIPTIVES.csv',summaries)
# Exact original 68 main near/far common-support conditions, with the original
# target->distance->channel->start weighting and start-cluster bootstrap.
distance=pd.concat([pd.read_csv(ROOT/'results'/f'b{r}_a{a}_u500000_distance.csv') for r in range(3) for a in arms],ignore_index=True)
main=distance[distance.bank=='main'].copy();main['channel']=main.firsts.map(lambda z:json.dumps(json.loads(z)[0],separators=(',',':')));main['start_key']=main.start_cell.map(lambda z:json.dumps(json.loads(z),separators=(',',':')))
supports=[]
for (start,ch),g in main.groupby(['start_key','channel']):
    d=set(g.distance);supports.append((start,ch,bool(d&{2,4}) and bool(d&{8,10,12,14})))
allowed={(s,c) for s,c,yes in supports if yes};common=main[[ (s,c) in allowed for s,c in zip(main.start_key,main.channel) ]]
assert common.condition_id.nunique()==68 and len(allowed)==9
starts=sorted(common.start_key.unique());assert len(starts)==8
localpoints={};localrows=[];startrows=[]
for r in range(3):
    rng=np.random.default_rng(29101307+r);draw=rng.integers(0,len(starts),size=(5000,len(starts)))
    for a in arms:
        g=common[(common.block==r)&(common.arm==a)];values=[]
        for st in starts:
            channels=[]
            for ch in sorted(g[g.start_key==st].channel.unique()):
                gg=g[(g.start_key==st)&(g.channel==ch)];bydist=gg.groupby('distance').correct.mean()
                channels.append([float(bydist[bydist.index.isin((2,4))].mean()),float(bydist[bydist.index.isin((8,10,12,14))].mean())])
            v=np.mean(channels,axis=0);values.append(v);startrows.append(dict(block=r,arm=a,start_cell=st,near_CBR=float(v[0]),far_CBR=float(v[1])))
        values=np.asarray(values);localpoints[r,a]=values.mean(axis=0);bo=values[draw].mean(axis=1)
        for i,band in enumerate(('near','far')):
            lo,hi=np.quantile(bo[:,i],[.025,.975]);localrows.append(dict(block=r,arm=a,band=band,CBR=float(values[:,i].mean()),start_cluster_ci_low=float(lo),start_cluster_ci_high=float(hi),start_clusters=8,common_conditions=68))
local_contrasts=[]
for r in range(3):
    rng=np.random.default_rng(29101307+r);draw=rng.integers(0,8,size=(5000,8))
    for name,a,b in contrasts:
        aa=np.asarray([[x['near_CBR'],x['far_CBR']] for x in startrows if x['block']==r and x['arm']==a]);bb=np.asarray([[x['near_CBR'],x['far_CBR']] for x in startrows if x['block']==r and x['arm']==b]);delta=aa-bb;bs=100*delta[draw].mean(axis=1)
        for i,band in enumerate(('near','far')):
            lo,hi=np.quantile(bs[:,i],[.025,.975]);local_contrasts.append(dict(block=r,contrast=name,band=band,effect_pp=float(100*delta[:,i].mean()),start_cluster_ci_low=float(lo),start_cluster_ci_high=float(hi)))
export(out/'DISTANCE_SUMMARY.csv',localrows);export(out/'DISTANCE_START_CLUSTERS.csv',startrows);export(out/'DISTANCE_CONTRASTS.csv',local_contrasts)
if MODE=='FINAL':
    recovery=[(ys[r,2]-ys[r,1])/(ys[r,0]-ys[r,1]) for r in range(3)]
    reverse=[ys[r,2]-ys[r,3] for r in range(3)];low_improvement=[ys[r,3]-ys[r,1] for r in range(3)]
    far_gain=[100*(localpoints[r,2][1]-localpoints[r,1][1]) for r in range(3)]
    support=all(v>=2/3 for v in recovery) and all(v>=15 for v in reverse) and all(v<=5 for v in low_improvement) and all(far_gain[r]>=15 for r in (1,2))
    method=[ys[r,2]-ys[r,0] for r in range(3)];hm=all(v>0 for v in method) and np.mean(method)>=10
    gate.update(H_SEL_complete_pattern=support,recovery_ratio=recovery,a2_minus_a3_pp=reverse,a3_minus_a1_pp=low_improvement,a2_minus_a1_far_CBR_pp=far_gain,H_M_pass=bool(hm),a2_minus_a0_pp=method,selection_verdict='VALUE_CONDITIONAL_SELECTION_EFFECT_SUPPORTED' if support else 'REGISTERED_COMPLETE_SELECTION_PATTERN_NOT_SUPPORTED',method_verdict='PRELIMINARY_METHOD_SIGNAL' if hm else 'NO_REGISTERED_PRELIMINARY_METHOD_SIGNAL',action_averaging_proven=False)
write(out/'HYPOTHESIS_GATE.json',gate);print('ANALYSIS_DONE',MODE,json.dumps(gate),flush=True)

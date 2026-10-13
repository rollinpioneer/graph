"""Preregistered paired clustered estimates, complete counterexamples and figures."""
from core import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def paired_interval(frame, value, seed):
    p=frame.pivot(index=['state_id','cluster','goal_index'],columns='variant',values=value).reset_index();p['delta']=p['A']-p['B']
    clusters=sorted(p.cluster.unique());values=[p.loc[p.cluster==c,'delta'].to_numpy(float) for c in clusters]
    rng=np.random.default_rng(seed);draws=np.empty(10000)
    for k in range(10000):
        sample=rng.integers(len(values),size=len(values));draws[k]=np.concatenate([values[i] for i in sample]).mean()
    return float(p.delta.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),len(clusters)

def main():
    guard();assert json.loads((ROOT/'receipts/EXECUTION_EXIT.json').read_text())['exit_code']==0
    r=pd.read_csv(ROOT/'BRANCH_ROLLOUTS.csv');s=pd.read_csv(ROOT/'GOAL_SENSITIVITY.csv');contrasts=[];gates=[];counter=[];negative=[]
    for seed in (0,1,2):
        for group in ('P1','P2','P3'):
            f=r[(r.seed==seed)&(r.group==group)];a=f[f.variant=='A'];b=f[f.variant=='B']
            for metric in ['correct','reference_progress','initial_channel_projection','displacement','mean_action_norm','small_action_fraction']:
                d,lo,hi,n=paired_interval(f,metric,9051302+seed*10+int(group[-1]))
                contrasts.append(dict(seed=seed,group=group,metric=metric,A=float(a[metric].mean()),B=float(b[metric].mean()),A_minus_B=d,CI95_low=lo,CI95_high=hi,clusters=n,states=int(a.state_id.nunique()),goal_tests_per_model=len(a),training_seeds=1,interval_scope='within-seed clustered states only'))
            sf=s[(s.seed==seed)&(s.group==group)&(s.goal_index==0)].copy();sf['goal_index']=0
            for metric in ['D_clipped','D_raw','goal_action_angle_deg']:
                # Undefined zero-vector angles omitted symmetrically, with denominator reported.
                m=sf.pivot(index=['state_id','cluster','goal_index'],columns='variant',values=metric).dropna();valid=set(m.reset_index().state_id)
                if not valid:continue
                f2=sf[sf.state_id.isin(valid)];d,lo,hi,n=paired_interval(f2,metric,9051402+seed*10+int(group[-1]))
                contrasts.append(dict(seed=seed,group=group,metric=metric,A=float(f2[f2.variant=='A'][metric].mean()),B=float(f2[f2.variant=='B'][metric].mean()),A_minus_B=d,CI95_low=lo,CI95_high=hi,clusters=n,states=len(valid),goal_tests_per_model=len(valid),training_seeds=1,interval_scope='within-seed clustered states only'))
            for variant in ('A','B'):
                sub=f[f.variant==variant]
                negative.append(dict(seed=seed,group=group,variant=variant,tests=len(sub),correct=int((sub.category=='correct').sum()),wrong_channel=int((sub.category=='wrong_channel').sum()),no_motion=int((sub.category=='no_motion').sum()),not_left_region=int((sub.category=='not_left_region').sum()),mean_action_norm=float(sub.mean_action_norm.mean()),mean_reference_progress=float(sub.reference_progress.mean()),small_action_fraction=float(sub.small_action_fraction.mean()),mean_displacement=float(sub.displacement.mean())))
        p1=r[(r.seed==seed)&(r.group=='P1')].pivot(index=['state_id','goal_index'],columns='variant',values='correct');divergent=p1[(p1.A==1)&(p1.B==0)].reset_index();ids=divergent.state_id.tolist()
        sf=s[(s.seed==seed)&(s.group=='P1')&(s.goal_index==0)].pivot(index='state_id',columns='variant',values='D_clipped')
        corresponding=len(ids)>0 and float(sf.loc[ids,'B'].mean())<float(sf.loc[ids,'A'].mean())
        vals={group:next(x for x in contrasts if x['seed']==seed and x['group']==group and x['metric']=='correct') for group in ('P1','P2')}
        d1=vals['P1']['A_minus_B'];d2=vals['P2']['A_minus_B'];sensitivity_B_weaker=float(sf.B.mean())<float(sf.A.mean())
        gate=dict(seed=seed,P1_CBR_delta_pp=100*d1,P2_CBR_delta_pp=100*d2,P1_minus_P2_delta_pp=100*(d1-d2),P1_mean_D_A=float(sf.A.mean()),P1_mean_D_B=float(sf.B.mean()),A_correct_B_incorrect_tests=len(ids),D_A_on_divergence_tests=None if not ids else float(sf.loc[ids,'A'].mean()),D_B_on_divergence_tests=None if not ids else float(sf.loc[ids,'B'].mean()),gate1=d1>=.15-1e-12,gate2=bool(sensitivity_B_weaker and corresponding),gate3=d2<.15-1e-12)
        gates.append(gate)
        # All state-goal pairs, including B>=A and cases that contradict ignored-goal story.
        for (state_id,gi),row in p1.iterrows():
            da=float(sf.loc[state_id,'A']);db=float(sf.loc[state_id,'B']);br=r[(r.seed==seed)&(r.state_id==state_id)&(r.goal_index==gi)&(r.variant=='B')].iloc[0]
            counter.append(dict(seed=seed,state_id=state_id,goal_index=gi,A_correct=int(row.A),B_correct=int(row.B),D_A=da,D_B=db,B_distinguishes_at_least_as_much_as_A=db>=da,B_correct_with_larger_D=bool(row.B and db>=da),B_category=br.category,is_supporting_divergence=bool(row.A and not row.B)))
    rows(ROOT/'PAIRED_CONTRASTS.csv',contrasts);rows(ROOT/'NEGATIVE_CONTROLS.csv',negative);rows(ROOT/'results/ALL_P1_PAIRED_TESTS.csv',counter);write(ROOT/'results/SUPPORT_GATES.json',gates)
    score_uncertainty=[]
    for seed in (0,1,2):
        for group in ('P1','P2','P3'):
            f=r[(r.seed==seed)&(r.group==group)];clusters=sorted(f.cluster.unique());rng=np.random.default_rng(9051302+seed*10+int(group[-1]));draws={'A':[],'B':[]}
            per={v:[f[(f.cluster==c)&(f.variant==v)].correct.to_numpy(float) for c in clusters] for v in ('A','B')}
            for k in range(10000):
                selected=rng.integers(len(clusters),size=len(clusters))
                for v in ('A','B'):draws[v].append(float(np.concatenate([per[v][i] for i in selected]).mean()))
            for v in ('A','B'):score_uncertainty.append(dict(seed=seed,group=group,variant=v,CBR=float(f[f.variant==v].correct.mean()),CI95_low=float(np.quantile(draws[v],.025)),CI95_high=float(np.quantile(draws[v],.975)),clusters=len(clusters),training_seeds=1))
    rows(ROOT/'results/SCORE_UNCERTAINTY.csv',score_uncertainty)
    strat=[]
    for (seed,group,source,variant),f in r.groupby(['seed','group','source','variant']):
        sf=s[(s.seed==seed)&(s.group==group)&(s.source==source)&(s.variant==variant)&(s.goal_index==0)]
        strat.append(dict(seed=int(seed),group=group,source=source,variant=variant,states=f.state_id.nunique(),goal_tests=len(f),CBR=float(f.correct.mean()),mean_goal_pair_D=float(sf.D_clipped.mean()),mean_action_norm=float(f.mean_action_norm.mean()),mean_displacement=float(f.displacement.mean()),wrong_channel=int((f.category=='wrong_channel').sum()),no_motion=int((f.category=='no_motion').sum())))
    rows(ROOT/'results/SOURCE_STRATIFIED_RESULTS.csv',strat)
    offline=s[s.offline_alignment_status!='UNMEASURABLE'].copy();offline.to_csv(ROOT/'results/OFFLINE_ALIGNMENT.csv',index=False)
    descriptive=[]
    for group in ('P1','P2','P3'):
        sub=[x for x in contrasts if x['group']==group and x['metric']=='correct']
        descriptive.append(dict(group=group,mean_A_CBR=float(np.mean([x['A'] for x in sub])),mean_B_CBR=float(np.mean([x['B'] for x in sub])),mean_delta_pp=float(np.mean([100*x['A_minus_B'] for x in sub])),training_seeds=3,scope='descriptive only; no cross-training-seed significance'))
    write(ROOT/'results/THREE_SEED_DESCRIPTIVE.json',descriptive)
    eligible=json.loads((ROOT/'receipts/EXECUTION_STARTED.json').read_text())['eligible_main_mechanism'];new=[g for g in gates if g['seed'] in (1,2)]
    supported=eligible and all(g['gate1'] and g['gate2'] and g['gate3'] for g in new)
    if not eligible: verdict='UNDETERMINED_INSUFFICIENT_ADMISSION'
    elif supported:verdict='SUPPORTIVE_BEHAVIORAL_EVIDENCE_FOR_H0'
    elif all(not g['gate1'] for g in new):verdict='H0_NOT_SUPPORTED_BY_LOCAL_BRANCH_DEFICIT'
    else:verdict='UNDETERMINED_H0_CRITERIA_NOT_MET'
    write(ROOT/'results/VERDICT.json',dict(verdict=verdict,eligible=eligible,all_three_gates_required=True,seed0_exploratory=True,seed1_2_training_outcomes_previously_seen=True,gates=gates,no_algorithm_causality_claim=True))
    # CBR by training seed and P1/P2 control.
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,group in zip(axes,['P1','P2','P3']):
        v=[x for x in contrasts if x['group']==group and x['metric']=='correct'];x=np.arange(3)
        for variant,dx in [('A',-.18),('B',.18)]:
            estimates=[z for z in score_uncertainty if z['group']==group and z['variant']==variant];values=np.array([100*z['CBR'] for z in estimates]);low=np.array([100*z['CI95_low'] for z in estimates]);high=np.array([100*z['CI95_high'] for z in estimates]);ax.bar(x+dx,values,.36,yerr=[values-low,high-values],capsize=3,label=variant)
        ax.set_xticks(x,['seed 0','seed 1','seed 2']);ax.set_title(group);ax.set_ylim(0,105);ax.set_ylabel('Correct first channel (%)');ax.legend()
    fig.suptitle('Local 25-step channel entry; does not measure distant task success');fig.tight_layout();fig.savefig(ROOT/'figures/CBR_BY_SEED_AND_CONTROL.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for j,seed in enumerate((0,1,2)):
        for i,group in enumerate(('P1','P2')):
            ax=axes[i,j]
            for v in ('A','B'):
                f=s[(s.seed==seed)&(s.group==group)&(s.variant==v)];ax.hist(f.direction_deg.dropna(),bins=np.linspace(-180,180,19),alpha=.5,label=v)
            ax.set_title(f'{group}, seed {seed}');ax.set_xlabel('Clipped action angle (degrees)');ax.set_ylabel('state-goal tests');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/GOAL_ACTION_DIRECTION_DISTRIBUTION.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,seed in zip(axes,(0,1,2)):
        for variant,dx in [('A',-.12),('B',.12)]:
            y=[float(s[(s.seed==seed)&(s.group==g)&(s.variant==variant)&(s.goal_index==0)].D_clipped.mean()) for g in ('P1','P2','P3')];ax.bar(np.arange(3)+dx,y,.24,label=variant)
        ax.set_xticks(np.arange(3),['P1','P2','P3']);ax.set_title(f'seed {seed}');ax.set_ylabel('Mean goal-pair clipped action L2');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/GOAL_SENSITIVITY_CONTROLS.png',dpi=180);plt.close(fig)
    choices=[];candidates=r[r.group=='P1'].sort_values(['seed','state_id','goal_index','variant'])
    for category in ('correct','wrong_channel','not_left_region','no_motion'):
        matches=candidates[candidates.category==category]
        if len(matches):choices.append({k:(None if pd.isna(v) else v) for k,v in matches.iloc[0].to_dict().items()})
    fig,axes=plt.subplots(1,4,figsize=(16,4));env=env_new();maze=env.unwrapped.maze_map.copy();env.close()
    for ax,category in zip(axes,('correct','wrong_channel','not_left_region','no_motion')):
        selected=[z for z in choices if z['category']==category]
        if not selected:ax.text(.5,.5,'No observed test in class',ha='center');ax.set_title(category);continue
        c=selected[0];t=pickle.load(open(c['trace_path'],'rb'));xy=t['states'];start=t['initial']['qpos']
        for cell in np.argwhere(maze==1):ax.add_patch(plt.Rectangle((4*cell[1]-6,4*cell[0]-6),4,4,color='.85',zorder=0))
        ax.plot(xy[:,0],xy[:,1],'-o',ms=2,label='policy')
        cells=t['route'];reference=np.array([[4*cell[1]-4,4*cell[0]-4] for cell in cells]);ax.plot(reference[:,0],reference[:,1],'--',alpha=.6,label='BFS reference');ax.scatter(start[0],start[1],s=60,marker='s');ax.set_xlim(start[0]-6,start[0]+6);ax.set_ylim(start[1]-6,start[1]+6);ax.set_aspect('equal');ax.set_title(f'{category}\n{c["variant"]} s{int(c["seed"])} {c["state_id"]} g{int(c["goal_index"])}');ax.legend()
    fig.suptitle('First lexicographic test per outcome class; no selection by best score');fig.tight_layout();fig.savefig(ROOT/'figures/REPRESENTATIVE_TRACES.png',dpi=180);plt.close(fig);write(ROOT/'results/REPRESENTATIVE_SELECTION.json',dict(rule='first lexicographic (seed,state_id,goal_index,variant) per P1 outcome class',selected=choices))
    lines=['# SAW-GOALPROBE-05 mechanism verdict','',f'**{verdict}**','', 'All estimates use the preregistered first cell exit within at most 25 steps. A successful local exit does not mean that the distant official task was completed.','', '| Seed | P1 A | P1 B | P1 A−B, pp (cluster 95% CI) | P2 A−B, pp | Gates 1/2/3 |','|---|---:|---:|---:|---:|---|']
    for g in gates:
        z=next(x for x in contrasts if x['seed']==g['seed'] and x['group']=='P1' and x['metric']=='correct')
        lines.append(f'| {g["seed"]} | {100*z["A"]:.2f}% | {100*z["B"]:.2f}% | {100*z["A_minus_B"]:.2f} [{100*z["CI95_low"]:.2f}, {100*z["CI95_high"]:.2f}] | {g["P2_CBR_delta_pp"]:.2f} | {g["gate1"]}/{g["gate2"]}/{g["gate3"]} |')
    lines.extend(['','Gate 2 requires both smaller mean B sensitivity and smaller sensitivity on A-correct/B-incorrect tests. Gate 3 requires P2 A−B below 15 pp in each repeat seed, a conservative technical operationalization frozen before querying. Passing only gate 1 does not support H0.','', 'Counterexamples: all P1 pairs are retained in results/ALL_P1_PAIRED_TESTS.csv. Outcomes and movement magnitudes for every P1/P2/P3 model are in NEGATIVE_CONTROLS.csv. No excluded state was replaced after model outputs.','',f'Offline alignment: {offline.state_id.nunique()} exact recorded state IDs, {len(offline)} state-goal-model rows. Other states are UNMEASURABLE. Original integration for offline records is unavailable; reconstructed physics and genuine recorded action alignment are explicitly separated.','', 'Limits: three training seeds only; seed 0 exploratory and seed 1/2 prior training scores already observed. Junction-cluster intervals are within-model scene uncertainty, not independent training replication. P1/P2 diagnostics are constructed or offline-record reconstructions, not the official test distribution. P3 original states are official initial states but the second goal is a diagnostic intervention, not a second official task. Geometry BFS is a reference channel, not optimal-actor action truth. Clipping, action amplitude, generic local deficits and wrong goal-dependent directions can coexist; no AWR/Waypoint causal attribution follows.'])
    if any(not g['gate3'] for g in new):lines.extend(['','P2 also shows a comparable general local performance deficit in at least one repeat seed. This favors keeping H-A (general local capability) as a competing explanation; ignoring goals is not established.'])
    if any(not g['gate2'] for g in new):lines.extend(['','The required cross-seed sensitivity-to-behavior correspondence is incomplete. Larger action differences can still point toward wrong channels; action variance alone is insufficient.'])
    (ROOT/'HYPOTHESIS_VERDICT.md').write_text('\n'.join(lines)+'\n')
    next_text='A separately authorized GCIVL A/B experiment without Waypoint could test cross-method generality. This experiment alone establishes only a behavioral association, not the causal loss component.' if supported else 'Do not build a dedicated ignored-goal repair from this result. Retain general local ability, action amplitude, direction quality, offline support and later route execution as alternatives. Any added diagnostic states or cross-method training require a separate card and authorization; no training is triggered.'
    (ROOT/'NEXT_DECISION.md').write_text('# Next decision\n\n'+verdict+'\n\n'+next_text+'\n\nThe scientific task is complete as a frozen diagnostic. Failure to support H0 does not exclude all effects of random goal sampling.\n')
    guard();write(ROOT/'receipts/ANALYSIS_EXIT.json',dict(exit_code=0,epoch=time.time(),verdict=verdict,rows=len(contrasts),training_seeds=3,new_updates=0,GPU_hours=0));print('ANALYSIS_COMPLETE',verdict,flush=True)

if __name__=='__main__':main()

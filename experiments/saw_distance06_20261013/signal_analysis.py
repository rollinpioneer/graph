"""Analyze already saved S probes and their approximate correspondence to D.

No checkpoint loading, new network calls, differentiation, or physical steps.
"""
from core import *
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BINS=('0-1','2-3','4-5','6-7','8-9','10-11','12-13','14-15','16+')
BOOTSTRAPS=500
BOOTSTRAP_SEED=26061303


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def number(value):
    if value is None or value=='':
        return None
    x=float(value)
    return x if np.isfinite(x) else None


def truth(value):
    return str(value).lower() in ('true','1')


def load_npz(path):
    with np.load(path) as f:
        return {k:f[k].copy() for k in f.files}


def interval(values):
    values=np.asarray([x for x in values if x is not None and np.isfinite(x)],float)
    return dict(low=float(np.quantile(values,.025)) if len(values) else None,
                high=float(np.quantile(values,.975)) if len(values) else None,
                valid_replicates=len(values))


class AUC:
    """Exact tie-aware weighted AUROC with a fixed score ordering."""
    def __init__(self,score,positive,trajectory_inverse):
        order=np.argsort(score,kind='stable')
        sorted_score=np.asarray(score)[order]
        self.positive=np.asarray(positive,bool)[order]
        self.trajectory_inverse=np.asarray(trajectory_inverse,int)[order]
        self.starts=np.r_[0,np.flatnonzero(np.diff(sorted_score)!=0)+1] if len(order) else np.asarray([],int)

    def calculate(self,cluster_count):
        if not len(self.starts):
            return None
        weights=cluster_count[self.trajectory_inverse]
        pos=np.add.reduceat(weights*self.positive,self.starts)
        neg=np.add.reduceat(weights*(~self.positive),self.starts)
        den=float(pos.sum()*neg.sum())
        return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/den) if den>0 else None


def reference_stats(tuples,signals,mask,trajectory_inverse):
    d=tuples['delta_w'][mask]
    inv=trajectory_inverse[mask]
    traj=tuples['original_trajectory'][mask]
    positive_negative=d!=0
    eligible=(int(np.sum(d>0))>=30 and int(np.sum(d<0))>=30 and
              len(np.unique(traj[positive_negative]))>=8)
    result=dict(n=int(np.sum(mask)),positive_count=int(np.sum(d>0)),negative_count=int(np.sum(d<0)),
                neutral_count=int(np.sum(d==0)),original_trajectories=len(np.unique(traj)),
                AUROC_status='MEASURED' if eligible else 'INSUFFICIENT_CLASSES_OR_TRAJECTORIES')
    evaluators={}
    for name,key in [('AWR','awr'),('WP','wp')]:
        weights=np.asarray(signals[name+'_weight'][mask],float)
        logs=np.asarray(signals['log_'+key][mask],float)
        result[name+'_weight_mean']=float(weights.mean()) if len(weights) else None
        result[name+'_weight_median']=float(np.median(weights)) if len(weights) else None
        result[name+'_saturated_fraction']=float(np.mean(weights>=100)) if len(weights) else None
        evaluators[name+'_weight_mean']=(weights,inv)
        for kind,score in [('log',logs),('clipped',weights)]:
            evaluators[name+'_'+kind+'_AUROC']=AUC(score[positive_negative],d[positive_negative]>0,inv[positive_negative]) if eligible else None
    wp=np.asarray(signals['WP_weight'][mask],float)
    evaluators['negative_WP_weight_share']=(wp,inv,d<0)
    result['negative_WP_weight_share']=float(wp[d<0].sum()/wp.sum()) if wp.sum()>0 else None
    return result,evaluators


def evaluate(evaluators,cluster_count):
    out={}
    for key,value in evaluators.items():
        if value is None:
            out[key]=None
        elif isinstance(value,AUC):
            out[key]=value.calculate(cluster_count)
        elif len(value)==2:
            weights,inv=value;multiplier=cluster_count[inv]
            den=float(multiplier.sum())
            out[key]=float(np.sum(weights*multiplier)/den) if den>0 else None
        else:
            weights,inv,negative=value;multiplier=cluster_count[inv]
            den=float(np.sum(weights*multiplier))
            out[key]=float(np.sum(weights*multiplier*negative)/den) if den>0 else None
    return out


def near_far_effect(item,tuples,signals):
    distance=tuples['graph_distance']
    good=tuples['geometry_valid']
    near=good&(distance>=2)&(distance<=5)
    far=good&(distance>=8)&(distance<=15)
    union=near|far
    uniq=np.unique(tuples['original_trajectory'][union])
    # All selected samples share a single original-trajectory bootstrap draw.
    inv=np.searchsorted(uniq,tuples['original_trajectory'])
    ns,ne=reference_stats(tuples,signals,near,inv)
    fs,fe=reference_stats(tuples,signals,far,inv)
    counts=np.ones(len(uniq),float)
    npnt=evaluate(ne,counts);fpnt=evaluate(fe,counts)
    ns.update(npnt);fs.update(fpnt)
    keys=list(ne)
    nboot={k:[] for k in keys};fboot={k:[] for k in keys};diff={k:[] for k in keys}
    if len(uniq)>=2:
        rng=np.random.default_rng(BOOTSTRAP_SEED+90000+int(item['seed'])*1000+(item['variant']=='B')*100)
        for j in range(BOOTSTRAPS):
            if j%25==0:
                guard()
            draw=rng.multinomial(len(uniq),np.full(len(uniq),1/len(uniq)))
            n=evaluate(ne,draw);f=evaluate(fe,draw)
            for k in keys:
                if n[k] is not None:nboot[k].append(n[k])
                if f[k] is not None:fboot[k].append(f[k])
                if n[k] is not None and f[k] is not None:diff[k].append(f[k]-n[k])
    effects={}
    for key in keys:
        effects[key]=dict(near=npnt[key],far=fpnt[key],
                         far_minus_near=None if npnt[key] is None or fpnt[key] is None else fpnt[key]-npnt[key],
                         near_CI=interval(nboot[key]),far_CI=interval(fboot[key]),far_minus_near_CI=interval(diff[key]))
    return dict(seed=int(item['seed']),variant=item['variant'],branch='R',
                near_definition='BFS2-5, pooled native tuples',far_definition='BFS8-15, pooled native tuples',
                near=ns,far=fs,effects=effects,bootstrap_repetitions=BOOTSTRAPS,
                bootstrap_unit='same original trajectory draw for near and far',
                uncertainty_scope='sampling over original trajectories for fixed existing models; no training-population claim')


def correlation(x,y):
    keep=[(a,b) for a,b in zip(x,y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(keep)<3:
        return dict(n=len(keep),spearman=None,status='INSUFFICIENT_GROUPS')
    a,b=np.asarray(keep,float).T
    if len(np.unique(a))<2 or len(np.unique(b))<2:
        return dict(n=len(keep),spearman=None,status='CONSTANT_INPUT')
    return dict(n=len(keep),spearman=float(spearmanr(a,b).statistic),status='DESCRIPTIVE_ONLY_NO_P_VALUE')


def conditional_join(conditional,behavior,groups):
    lookup={(int(x['seed']),x['variant'],x['condition_id']):x for x in behavior}
    original={x['group_id']:x for x in groups}
    joined=[]
    for row in conditional:
        c=dict(row);g=original[row['group_id']]
        key=(int(row['seed']),row['variant'],row['condition_id'])
        b=lookup.get(key)
        c.update(S2_status=row['status'],D_join_status='APPROXIMATE_GOAL_CORRESPONDENCE' if b else 'NO_D_ROW',
                 reference_goal=json.dumps(g['reference_goal']),fixed_data_goal=json.dumps(g.get('fixed_goal')),
                 exact_same_goal_claim=False,complete_physical_state_match_claim=False,
                 position_neighbor_radius=.15,goal_record_radius=.5)
        if b:
            c.update(D_correct=b['correct'],D_mean_action_norm=b['mean_action_norm'],D_category=b['category'],
                     D_displacement_reversals=b['displacement_reversals'],D_action_reversals=b['action_reversals'],
                     D_repeated_visit_steps=b['repeated_visit_steps'],D_small_command_candidate=b['small_command_candidate'],
                     D_backtracking_evidence=b['backtracking_evidence'],D_near_stationary=b['near_stationary'])
        joined.append(c)
    return joined


def conditional_summaries(joined):
    result=[]
    for seed in (0,1,2):
        for variant in ('A','B'):
            selected=[x for x in joined if int(x['seed'])==seed and x['variant']==variant]
            eligible=[x for x in selected if x['status']=='ELIGIBLE']
            linked=[x for x in eligible if x['D_join_status']=='APPROXIMATE_GOAL_CORRESPONDENCE']
            ratio=[number(x.get('combined_quadratic_resultant_ratio')) for x in linked]
            actual=[number(x.get('D_mean_action_norm')) for x in linked]
            actor_ref=[number(x.get('reference_actor_raw_norm')) for x in linked]
            equivalent=[number(x.get('equivalent_label_mean_norm')) for x in linked]
            result.append(dict(seed=seed,variant=variant,groups=len(selected),eligible_groups=len(eligible),
                insufficient_or_missing_groups=len(selected)-len(eligible),linked_D_groups=len(linked),
                status_counts={s:sum(x['status']==s for x in selected) for s in sorted(set(x['status'] for x in selected))},
                target_error_range=[min([number(x['goal_record_distance']) for x in eligible],default=None),
                                    max([number(x['goal_record_distance']) for x in eligible],default=None)],
                combined_ratio_vs_D_action_norm=correlation(ratio,actual),
                equivalent_mean_norm_vs_D_action_norm=correlation(equivalent,actual),
                combined_ratio_vs_fixed_record_goal_actor_norm=correlation(ratio,actor_ref),
                linked_D_small_command_count=sum(truth(x['D_small_command_candidate']) for x in linked),
                linked_D_backtracking_count=sum(truth(x['D_backtracking_evidence']) for x in linked),
                limitation='Each neighborhood has different position/velocity records; S2 goal is a nearby recorded goal, not necessarily D center goal.'))
    return result


def auc_figure(rows_out,path):
    fig,axes=plt.subplots(4,3,figsize=(15,14),sharex=True,sharey=True)
    columns=[('WP_log_AUROC','Waypoint / raw log score'),('WP_clipped_AUROC','Waypoint / clipped weight'),
             ('AWR_log_AUROC','AWR / raw log score'),('AWR_clipped_AUROC','AWR / clipped weight')]
    xpos=np.arange(9)
    for rowidx,(field,label) in enumerate(columns):
        for seed in (0,1,2):
            ax=axes[rowidx,seed]
            for variant,color in [('A','#2166ac'),('B','#b2182b')]:
                for branch,line,marker in [('F','-','o'),('R','--','s')]:
                    selected={x['distance_bin']:x for x in rows_out if int(x['seed'])==seed and x['variant']==variant and x['branch']==branch}
                    vals=[number(selected.get(b,{}).get(field)) for b in BINS]
                    numeric=np.asarray([np.nan if v is None else v for v in vals])
                    ax.plot(xpos,numeric,color=color,linestyle=line,marker=marker,markersize=3,label=f'{variant} / {branch}',linewidth=1.5)
                    lows=[number(selected.get(b,{}).get(field+'_CI_low')) for b in BINS]
                    highs=[number(selected.get(b,{}).get(field+'_CI_high')) for b in BINS]
                    low=np.asarray([np.nan if v is None else v for v in lows]);high=np.asarray([np.nan if v is None else v for v in highs])
                    ax.fill_between(xpos,low,high,color=color,alpha=.06)
            ax.axhline(.5,color='#777777',linewidth=.6,linestyle=':')
            ax.set_ylim(-.02,1.02);ax.set_xticks(xpos,BINS,rotation=40);ax.grid(alpha=.15)
            if rowidx==0:ax.set_title(f'Training seed {seed}')
            if seed==0:ax.set_ylabel(label+'\nAUROC: positive vs negative graph progress')
            if rowidx==3:ax.set_xlabel('BFS distance to goal (edges)')
    axes[0,0].legend(ncol=2,fontsize=9,loc='lower right')
    fig.suptitle('Frozen training guidance: every native distance bin, both samplers and configurations',fontsize=15)
    fig.text(.5,.01,'Bands: 95% original-trajectory bootstrap intervals (500 draws). Missing/insufficient classes stay NA; neutral progress is excluded from AUROC.\nScores concern graph progress only, not ground-truth value or causal supervision quality.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.965));fig.savefig(path,dpi=180);plt.close(fig)


def share_and_conditional_figure(rows_out,joined,path):
    fig,axes=plt.subplots(3,3,figsize=(15,12))
    xpos=np.arange(9)
    for seed in (0,1,2):
        ax=axes[0,seed]
        for variant,color in [('A','#2166ac'),('B','#b2182b')]:
            for branch,line,marker in [('F','-','o'),('R','--','s')]:
                selected={x['distance_bin']:x for x in rows_out if int(x['seed'])==seed and x['variant']==variant and x['branch']==branch}
                vals=[number(selected.get(b,{}).get('negative_WP_weight_share')) for b in BINS]
                ax.plot(xpos,[np.nan if v is None else v for v in vals],color=color,linestyle=line,marker=marker,markersize=3,label=f'{variant} / {branch}')
                lo=[number(selected.get(b,{}).get('negative_WP_weight_share_CI_low')) for b in BINS]
                hi=[number(selected.get(b,{}).get('negative_WP_weight_share_CI_high')) for b in BINS]
                ax.fill_between(xpos,[np.nan if v is None else v for v in lo],[np.nan if v is None else v for v in hi],color=color,alpha=.06)
        ax.set_title(f'Training seed {seed}');ax.set_xticks(xpos,BINS,rotation=40);ax.set_ylim(-.02,1.02);ax.grid(alpha=.15)
        if seed==0:ax.set_ylabel('Negative graph progress share\nof total Waypoint weight')
        ax.set_xlabel('BFS distance (edges)')
        selected=[x for x in joined if int(x['seed'])==seed]
        eligible=[x for x in selected if x['status']=='ELIGIBLE']
        for variant,marker in [('A','o'),('B','^')]:
            for band,color in [('near','#1b9e77'),('far','#7570b3')]:
                group=[x for x in eligible if x['variant']==variant and ('near' if int(x['distance']) in (2,4) else 'far')==band]
                linked=[x for x in group if x['D_join_status']=='APPROXIMATE_GOAL_CORRESPONDENCE']
                valid=[x for x in linked if number(x.get('combined_quadratic_resultant_ratio')) is not None and number(x.get('D_mean_action_norm')) is not None]
                axes[1,seed].scatter([number(x['combined_quadratic_resultant_ratio']) for x in valid],
                                     [number(x['D_mean_action_norm']) for x in valid],color=color,marker=marker,s=24,alpha=.75,label=f'{variant} / {band}')
                valid2=[x for x in group if number(x.get('data_quadratic_resultant_ratio')) is not None and number(x.get('teacher_quadratic_resultant_ratio')) is not None]
                axes[2,seed].scatter([number(x['data_quadratic_resultant_ratio']) for x in valid2],
                                     [number(x['teacher_quadratic_resultant_ratio']) for x in valid2],color=color,marker=marker,s=24,alpha=.75,label=f'{variant} / {band}')
        axes[1,seed].set_xlim(-.02,1.02);axes[1,seed].set_ylim(bottom=-.02);axes[1,seed].grid(alpha=.15)
        axes[1,seed].set_xlabel('Combined quadratic-label resultant ratio')
        axes[1,seed].set_title(f'S2 eligible {len(eligible)}/{len(selected)} model-group rows')
        axes[2,seed].set_xlim(-.02,1.02);axes[2,seed].set_ylim(-.02,1.02);axes[2,seed].grid(alpha=.15)
        axes[2,seed].set_xlabel('Data-action quadratic resultant ratio')
        if seed==0:
            axes[1,seed].set_ylabel('D mean clipped action norm\n(nearby recorded-goal correspondence)')
            axes[2,seed].set_ylabel('Teacher-action quadratic resultant ratio')
    axes[0,0].legend(ncol=2,fontsize=9);axes[1,0].legend(ncol=2,fontsize=9)
    fig.suptitle('Graph-progress weighting and fixed-recorded-goal conditional actions',fontsize=15)
    fig.text(.5,.008,'Every frozen group, target offset and missing-support status remains in CONDITIONAL_ACTION_AUDIT.csv and S2_BEHAVIOR_LINK.csv.\nS2 uses one fixed recorded goal per group, at most 0.5 from the D goal. Nearby states differ; these associations do not prove multimodality or a training cause.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(path,dpi=180);plt.close(fig)


def csv_union(path,data):
    rows(path,data,list(dict.fromkeys(k for r in data for k in r)))


def main():
    guard();started=time.time()
    assert json.loads((ROOT/'results/SIGNAL_FORWARD_AUDIT.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'receipts/D_EXIT.json').read_text())['exit_code']==0
    for name in ('SIGNAL_BY_DISTANCE.csv','CONDITIONAL_ACTION_AUDIT.csv','SIGNAL_TUPLES.csv'):
        original=ROOT/'results'/name;target=ROOT/name
        if target.exists():
            assert sha(target)==sha(original), 'ROOT_CSV_ALREADY_EXISTS_WITH_DIFFERENT_CONTENT'
        else:
            shutil.copy2(original,target)
    summary=read_csv(ROOT/'SIGNAL_BY_DISTANCE.csv')
    conditional=read_csv(ROOT/'CONDITIONAL_ACTION_AUDIT.csv')
    behavior=read_csv(ROOT/'BEHAVIOR_ROLLOUTS.csv')
    groups=json.loads((ROOT/'results/S2_GROUPS.json').read_text())
    joined=conditional_join(conditional,behavior,groups)
    csv_union(ROOT/'results/S2_BEHAVIOR_LINK.csv',joined)
    fixed=load_npz(ROOT/'runs/S1_R_TUPLES.npz')
    manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    effects=[]
    for item in manifest['models']:
        guard()
        label=f"{item['variant']}_s{item['seed']}"
        signals=load_npz(ROOT/'runs'/f'SIGNALS_{label}_R.npz')
        effects.append(near_far_effect(item,fixed,signals))
        del signals
    assoc=conditional_summaries(joined)
    equality=[]
    for seed in (0,1,2):
        for branch in ('F','R'):
            a=load_npz(ROOT/'runs'/f'SIGNALS_A_s{seed}_{branch}.npz')
            b=load_npz(ROOT/'runs'/f'SIGNALS_B_s{seed}_{branch}.npz')
            keys=('value_s_head1','value_s_head2','value_next_head1','value_next_head2',
                  'value_waypoint_head1','value_waypoint_head2','log_awr','log_wp','AWR_weight','WP_weight','raw_teacher')
            equality.append(dict(seed=seed,branch=branch,exact_equal_outputs={k:bool(np.array_equal(a[k],b[k])) for k in keys},
                max_abs_output_differences={k:float(np.max(np.abs(a[k]-b[k]))) for k in keys},
                max_abs_actor_difference=float(np.max(np.abs(a['raw_actor']-b['raw_actor']))),
                scope='exact output equality on these frozen probes only; no parameter-identity inference and no cached independent samples'))
            del a,b
    candidates=[]
    for effect in effects:
        wp=effect['effects']['WP_log_AUROC']['far_minus_near']
        neg=effect['effects']['negative_WP_weight_share']['far_minus_near']
        candidates.append(dict(seed=effect['seed'],variant=effect['variant'],
            weighting_direction='R_FAR_GEOMETRY_WEIGHTING_WEAKER_DIRECTION_CANDIDATE' if wp is not None and neg is not None and wp<0 and neg>0 else 'NOT_BOTH_DIRECTIONAL_FEATURES_OR_NOT_ESTIMABLE',
            WP_log_AUROC_far_minus_near=wp,negative_WP_weight_share_far_minus_near=neg,
            threshold_or_significance_gate=False,
            limitation='Direction-only explanatory description, not proof of harmful supervision or B-specific cause.'))
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    auc_path=figures/'SIGNAL_AUROC_BY_DISTANCE.png';share_path=figures/'SIGNAL_WEIGHT_AND_CONDITIONAL_ACTIONS.png'
    assert not auc_path.exists() and not share_path.exists()
    auc_figure(summary,auc_path);share_and_conditional_figure(summary,joined,share_path)
    write(ROOT/'results/SIGNAL_INTERPRETATION.json',dict(status='COMPLETED_READ_ONLY_ANALYSIS',
        R_near_far=effects,S2_summary=assoc,directional_candidates=candidates,
        A_B_frozen_probe_output_equalities=equality,
        every_frozen_S2_group=[dict(group_id=x['group_id'],condition_id=x['condition_id'],status=x['status'],
                                   n=x['n'],original_trajectories=x['original_trajectories'],
                                   goal_record_distance=x.get('goal_record_distance'),fixed_goal_idx=x.get('fixed_goal_idx')) for x in groups],
        figures=[dict(path=str(p),sha256=sha(p)) for p in (auc_path,share_path)],
        no_extra_model_queries=True,no_gradients=True,new_physical_steps=0,new_parameter_updates=0,
        analysis_seconds=time.time()-started,
        limitations=['BFS progress is a proxy, not true value or proof of correct executable teacher action.',
          'Exact A/B value and teacher output equality on the frozen probes, if observed, means their weaker far-goal graph distinction cannot establish a B-only value error. It does not establish parameter identity.',
          'S2 combines different nearby position and velocity records under one recorded goal; it cannot establish conflicting labels at the exact same input.',
          'S2 and D targets can differ by up to 0.5; correspondence is approximate, not a same-goal intervention.',
          'Any graph-score or action-aggregation association remains a mechanism candidate; no automatic training or new loss is authorized.']))
    print('SIGNAL_ANALYSIS_COMPLETE',len(effects),len(joined),flush=True)


if __name__=='__main__':
    main()

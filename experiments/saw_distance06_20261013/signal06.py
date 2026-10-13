"""Read-only six-model training-signal probes; no gradients or updates.

All S1/S2 inputs must already have been frozen by signal_prepare.py. Per-tuple
forward outputs remain in server NPZ files; CSVs contain indices and summaries.
"""
from core import *
from signal_prepare import BINS
import gc
import jax.numpy as jnp
from scipy.stats import rankdata, spearmanr

BOOTSTRAP_SEED = 26061303
BOOTSTRAPS = 500


@jax.jit
def query(agent, batch):
    s, n, g, w, a = [batch[k] for k in ('observations', 'next_observations', 'high_actor_goals', 'high_actor_targets', 'actions')]
    v1, v2 = agent.network.select('value')(s, g)
    n1, n2 = agent.network.select('value')(n, g)
    w1, w2 = agent.network.select('value')(w, g)
    dist = agent.network.select('actor')(s, g)
    teacher = agent.network.select('low_actor')(s, w)
    am, tm = dist.mode(), teacher.mode()
    v, nv, wv = (v1+v2)/2, (n1+n2)/2, (w1+w2)/2
    la = agent.config['awr_alpha']*(nv-v)
    lw = agent.config['kl_alpha']*(wv-v)
    log_cap = jnp.log(jnp.asarray(100., dtype=la.dtype))
    wa = jnp.minimum(jnp.exp(jnp.minimum(la, log_cap)), 100.)
    ww = jnp.minimum(jnp.exp(jnp.minimum(lw, log_cap)), 100.)
    log_prob = dist.log_prob(a)
    variance = dist.scale_diag**2
    return dict(value_s_head1=v1, value_s_head2=v2, value_next_head1=n1, value_next_head2=n2,
        value_waypoint_head1=w1, value_waypoint_head2=w2,
        log_awr=la, log_wp=lw, AWR_weight=wa, WP_weight=ww,
        raw_log_awr_nonfinite=~jnp.isfinite(la), raw_log_wp_nonfinite=~jnp.isfinite(lw),
        unclipped_exp_awr_overflow=la>jnp.log(jnp.finfo(la.dtype).max),
        unclipped_exp_wp_overflow=lw>jnp.log(jnp.finfo(lw.dtype).max),
        raw_actor=am, clipped_actor=jnp.clip(am,-1.,1.), raw_teacher=tm,
        data_action=a, actor_std=dist.scale_diag,
        data_NLL=-log_prob, AWR_contribution=-wa*log_prob,
        WP_contribution=ww*jnp.sum((am-tm)**2, axis=-1),
        data_quadratic_coefficient=wa[:,None]/(2*variance),
        teacher_quadratic_coefficient=jnp.repeat(ww[:,None], a.shape[-1], axis=1))


@jax.jit
def native_losses(agent, batch):
    # No differentiation occurs: grad_params is merely the unchanged parameter tree.
    al = agent.actor_loss(batch, agent.network.params)[0]
    wl = agent.waypoint_loss(batch, agent.network.params)[0]
    return jnp.asarray([al, wl])


def digest_tree(tree):
    h = hashlib.sha256()
    for v in jax.tree_util.tree_leaves(tree):
        a = np.asarray(v)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def weighted_auc(score, labels, weights=None):
    score, labels = np.asarray(score, float), np.asarray(labels, bool)
    weights = np.ones(len(score), float) if weights is None else np.asarray(weights, float)
    if not len(score):
        return None
    order = np.argsort(score, kind='stable')
    sc, la, wt = score[order], labels[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sc) != 0)+1]
    pos = np.add.reduceat(wt*la, starts)
    neg = np.add.reduceat(wt*(~la), starts)
    denominator = pos.sum()*neg.sum()
    if denominator <= 0:
        return None
    return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/denominator)


def spearman(x, y):
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def cluster_intervals(trajectory, delta, weights, logs, seed):
    """Cluster bootstrap AUROCs and weight shares; neutral samples remain separate."""
    uniq, inverse = np.unique(trajectory, return_inverse=True)
    if len(uniq) < 2:
        return {}
    posneg = delta != 0
    labels = delta[posneg] > 0
    usable_auc = int(np.sum(labels)) >= 30 and int(np.sum(~labels)) >= 30 and len(np.unique(trajectory[posneg])) >= 8
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(uniq), np.full(len(uniq), 1/len(uniq)), size=BOOTSTRAPS)
    records = {'negative_WP_weight_share': []}
    for name in ('AWR', 'WP'):
        records[name+'_log_AUROC'] = []
        records[name+'_clipped_AUROC'] = []
    for draw in draws:
        guard()
        m = draw[inverse]
        denominator = float(np.sum(weights['WP']*m))
        if denominator > 0:
            records['negative_WP_weight_share'].append(float(np.sum(weights['WP']*m*(delta<0))/denominator))
        if usable_auc:
            for name in ('AWR', 'WP'):
                for tag, score in [('log', logs[name]), ('clipped', weights[name])]:
                    v = weighted_auc(score[posneg], labels, m[posneg])
                    if v is not None:
                        records[name+'_'+tag+'_AUROC'].append(v)
    out = {}
    for key, vals in records.items():
        out[key+'_bootstrap_valid'] = len(vals)
        out[key+'_CI_low'] = float(np.quantile(vals,.025)) if vals else None
        out[key+'_CI_high'] = float(np.quantile(vals,.975)) if vals else None
    return out


def summary_rows(item, branch, tuples, signals):
    result = []
    d = tuples['graph_distance']
    masks = [(b, (d>=2*k)&(d<=2*k+1)) for k,b in enumerate(BINS[:-1])]
    masks += [('16+', d>=16), ('GEOMETRY_UNAVAILABLE', ~tuples['geometry_valid'])]
    masks += [('S_NEAR_2_5', (d>=2)&(d<=5)),
              ('S_FAR_8_15', (d>=8)&(d<=15))]
    for bin_index, (label, select) in enumerate(masks):
        select = select & tuples['geometry_valid'] if label != 'GEOMETRY_UNAVAILABLE' else select
        n = int(np.sum(select))
        row = dict(seed=item['seed'], variant=item['variant'], branch=branch, distance_bin=label,
                   n=n, original_trajectories=len(np.unique(tuples['original_trajectory'][select])),
                   status='MEASURED' if n and label!='GEOMETRY_UNAVAILABLE' else 'NO_SAMPLES' if not n else 'GEOMETRY_UNAVAILABLE',
                   actually_cross_trajectory_fraction=float(np.mean(tuples['original_trajectory'][select]!=tuples['goal_original_trajectory'][select])) if n else None,
                   bootstrap_unit='original_trajectory', bootstrap_repetitions=BOOTSTRAPS)
        if n:
            delta = tuples['delta_w'][select]
            traj = tuples['original_trajectory'][select]
            geometry_measured=label!='GEOMETRY_UNAVAILABLE'
            row.update(positive_fraction=float(np.mean(delta>=1)) if geometry_measured else None,
                       negative_fraction=float(np.mean(delta<=-1)) if geometry_measured else None,
                       neutral_fraction=float(np.mean(delta==0)) if geometry_measured else None,
                       positive_count=int(np.sum(delta>=1)) if geometry_measured else None,
                       negative_count=int(np.sum(delta<=-1)) if geometry_measured else None,
                       neutral_count=int(np.sum(delta==0)) if geometry_measured else None)
            positive_negative = delta != 0
            labels = delta[positive_negative] > 0
            both = int(np.sum(labels)) >= 30 and int(np.sum(~labels)) >= 30 and len(np.unique(traj[positive_negative])) >= 8
            row['AUROC_status'] = 'MEASURED' if both and label!='GEOMETRY_UNAVAILABLE' else 'INSUFFICIENT_CLASSES_OR_TRAJECTORIES'
            weights, logs = {}, {}
            for name,key in [('AWR','awr'),('WP','wp')]:
                weight = np.asarray(signals[name+'_weight'][select], float)
                log = np.asarray(signals['log_'+key][select], float)
                contribution = np.asarray(signals[name+'_contribution'][select], float)
                weights[name], logs[name] = weight, log
                for tag, array in [('weight',weight),('log',log),('contribution',contribution)]:
                    row[name+'_'+tag+'_mean'] = float(array.mean())
                    for q in (.1,.5,.9,.99):
                        row[name+'_'+tag+'_p'+str(int(q*100))] = float(np.quantile(array,q))
                denom = float(np.sum(weight**2))
                row[name+'_ESS'] = float(weight.sum()**2/denom) if denom else 0.
                row[name+'_saturated_fraction'] = float(np.mean(weight>=100))
                row[name+'_raw_log_nonfinite_count'] = int(np.sum(signals['raw_log_'+key+'_nonfinite'][select]))
                row[name+'_unclipped_exp_overflow_count'] = int(np.sum(signals['unclipped_exp_'+key+'_overflow'][select]))
                row[name+'_log_AUROC'] = weighted_auc(log[positive_negative], labels) if both and label!='GEOMETRY_UNAVAILABLE' else None
                row[name+'_clipped_AUROC'] = weighted_auc(weight[positive_negative], labels) if both and label!='GEOMETRY_UNAVAILABLE' else None
                row[name+'_log_Spearman_delta_w'] = spearman(log,delta) if label!='GEOMETRY_UNAVAILABLE' else None
                row[name+'_clipped_Spearman_delta_w'] = spearman(weight,delta) if label!='GEOMETRY_UNAVAILABLE' else None
                row[name+'_actual_total_contribution'] = float(contribution.sum())
            row['negative_WP_weight_share'] = float(np.sum(weights['WP'][delta<0])/weights['WP'].sum()) if weights['WP'].sum()>0 and label!='GEOMETRY_UNAVAILABLE' else None
            if label!='GEOMETRY_UNAVAILABLE':
                row.update(cluster_intervals(traj,delta,weights,logs,
                    BOOTSTRAP_SEED+item['seed']*10000+(item['variant']=='B')*2000+(branch=='R')*1000+bin_index))
        result.append(row)
    return result


def resultant(actions, weights):
    denominator = float(np.sum(np.asarray(weights)*np.linalg.norm(actions,axis=1)))
    return float(np.linalg.norm(np.sum(actions*np.asarray(weights)[:,None],axis=0))/denominator) if denominator>0 else None


def action_distribution(actions, weights):
    norms = np.linalg.norm(actions,axis=1)
    angles = np.arctan2(actions[:,1],actions[:,0])
    hist = np.histogram(angles[norms>1e-12],bins=np.linspace(-np.pi,np.pi,17),weights=np.asarray(weights)[norms>1e-12])[0]
    nonzero = norms>1e-12
    cutoff = np.quantile(weights,.75) if len(weights) else np.inf
    high = (weights>=cutoff)&(weights>0)&nonzero
    cosine = None
    if np.sum(high)>=2:
        unit=actions[high]/norms[high,None]
        mat=unit@unit.T
        cosine=float(np.min(mat[np.triu_indices(len(unit),1)]))
    return dict(mean_norm=float(norms.mean()), p10_norm=float(np.quantile(norms,.1)),
                median_norm=float(np.median(norms)), p90_norm=float(np.quantile(norms,.9)),
                zero_action_count=int(np.sum(~nonzero)), resultant_ratio=resultant(actions,weights),
                direction_hist_16=json.dumps(hist.tolist()), high_weight_upper_quartile_count=int(np.sum(high)),
                high_weight_min_pair_cosine=cosine,
                high_weight_max_pair_angle=None if cosine is None else float(np.degrees(np.arccos(np.clip(cosine,-1,1)))))


def conditional_rows(item, agent, groups, tuples, signals):
    out=[]
    for group in groups:
        select=tuples['group_index']==group['group_index']
        row=dict(seed=item['seed'],variant=item['variant'],group_id=group['group_id'],
                 condition_id=group['condition_id'],distance=group['distance'],region=group['region'],
                 n=group['n'],original_trajectories=group['original_trajectories'],status=group['status'],
                 fixed_goal_idx=group.get('fixed_goal_idx'),goal_record_distance=group.get('goal_record_distance'),
                 interpretation='conditional R-possible combinations; nearby positions are different inputs')
        if group['status']!='ELIGIBLE':
            out.append(row);continue
        assert np.sum(select)==group['n']
        a=signals['data_action'][select];t=signals['raw_teacher'][select]
        ca=signals['data_quadratic_coefficient'][select];ct=signals['teacher_quadratic_coefficient'][select]
        sigma=signals['actor_std'][select]
        assert np.all(sigma==1.), 'FIXED_SIGMA_REQUIRED_FOR_REGISTERED_EQUIVALENT_MEAN'
        assert np.array_equal(ca[:,0],ca[:,1]) and np.array_equal(ct[:,0],ct[:,1])
        denominator=np.sum(ca+ct,axis=0)
        equivalent=np.sum(ca*a+ct*t,axis=0)/denominator if np.all(denominator>0) else None
        sref=np.asarray(group['reference_position'],np.float32)
        gref=np.asarray(group['fixed_goal'],np.float32)
        actual=np.asarray(raw_mode(agent,sref,gref),float)
        for label,action,weight in [('data_AWR',a,signals['AWR_weight'][select]),
                                    ('teacher_WP',t,signals['WP_weight'][select]),
                                    ('data_quadratic',a,ca[:,0]),('teacher_quadratic',t,ct[:,0])]:
            row.update({label+'_'+key:value for key,value in action_distribution(action,weight).items()})
        mixed=np.concatenate((a,t));mixed_weights=np.concatenate((ca[:,0],ct[:,0]))
        row.update({ 'combined_quadratic_'+key:value for key,value in action_distribution(mixed,mixed_weights).items()})
        row.update(equivalent_mean_status='DEFINED' if equivalent is not None else 'ZERO_TOTAL_QUADRATIC_COEFFICIENT',
                   equivalent_label_mean_x=None if equivalent is None else float(equivalent[0]),
                   equivalent_label_mean_y=None if equivalent is None else float(equivalent[1]),
                   equivalent_label_mean_norm=None if equivalent is None else float(np.linalg.norm(equivalent)),
                   reference_actor_raw_x=float(actual[0]),reference_actor_raw_y=float(actual[1]),
                   reference_actor_raw_norm=float(np.linalg.norm(actual)),
                   reference_actor_clipped_norm=float(np.linalg.norm(np.clip(actual,-1,1))),
                   actual_equivalent_mean_distance=None if equivalent is None else float(np.linalg.norm(actual-equivalent)),
                   data_quadratic_coefficient_total=float(ca[:,0].sum()),
                   teacher_quadratic_coefficient_total=float(ct[:,0].sum()),
                   position_error_max=float(tuples['state_position_error'][select].max()),
                   velocity_error_mean=float(tuples['velocity_error'][select].mean()),
                   velocity_error_max=float(tuples['velocity_error'][select].max()))
        out.append(row)
    return out


def load_arrays(path):
    with np.load(path) as file:
        return {k:file[k].copy() for k in file.files}


def forward_all(agent, tuples, label):
    n=len(tuples['state_idx'])
    if not n:
        return {}
    parts=[]
    keys=('observations','next_observations','actions','high_actor_goals','high_actor_targets')
    for start in range(0,n,1024):
        guard()
        batch={key:tuples[key][start:start+1024] for key in keys}
        res=jax.block_until_ready(query(agent,batch))
        converted={k:np.asarray(v) for k,v in res.items()}
        parts.append(converted)
    result={key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    path=ROOT/'runs'/f'SIGNALS_{label}.npz'
    assert not path.exists()
    np.savez_compressed(path,**result)
    # Preserve raw outputs and nonfinite/overflow markers even on a technical stop.
    assert all(np.isfinite(x).all() for name,x in result.items() if not name.endswith('nonfinite')), 'NONFINITE_SIGNAL_OUTPUT; raw NPZ preserved'
    return result


def csv_union(path, data):
    fields=list(dict.fromkeys(k for row in data for k in row))
    rows(path,data,fields)


def main():
    guard();start=time.time()
    frozen=json.loads((ROOT/'results/SIGNAL_SAMPLE_FREEZE.json').read_text())
    assert frozen['status']=='FROZEN_BEFORE_ANY_CURRENT_MODEL_QUERY'
    for identity in frozen['S1']+[frozen['S2']]:
        assert sha(identity['path'])==identity['sha256']
    f=load_arrays(ROOT/'runs/S1_F_TUPLES.npz')
    r=load_arrays(ROOT/'runs/S1_R_TUPLES.npz')
    s2=load_arrays(ROOT/'runs/S2_TUPLES.npz')
    groups=json.loads((ROOT/'results/S2_GROUPS.json').read_text())
    summaries=[];conditionals=[];checks=[];identities=[]
    manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    frozen_models=manifest['models']
    assert len(frozen_models)==6 and {(int(x['seed']),x['variant']) for x in frozen_models}=={(s,v) for s in (0,1,2) for v in ('A','B')}
    for item in frozen_models:
        guard();label=f"{item['variant']}_s{item['seed']}"
        assert sha(item['path'])==item['sha256'], 'CHECKPOINT_IDENTITY'
        print('SIGNAL_MODEL_BEGIN',label,flush=True)
        agent=load_agent(item)
        assert sha(item['config_path'])==item['config_sha256'], 'CONFIG_IDENTITY'
        source_config=json.loads(Path(item['config_path']).read_text())['config']
        def normal(x):
            if isinstance(x,dict) or hasattr(x,'items'):
                return {key:normal(value) for key,value in x.items()}
            if isinstance(x,(tuple,list)):
                return [normal(value) for value in x]
            return x
        assert normal(agent.config)==normal(source_config), 'COMPLETE_CONFIG_MISMATCH'
        assert agent.config['const_std'] and not agent.config['discrete']
        assert agent.config['awr_alpha']==3 and agent.config['kl_alpha']==3
        before=digest_tree(agent.network)
        for branch,tuples in [('F',f),('R',r)]:
            signals=forward_all(agent,tuples,f'{label}_{branch}')
            check_batch={key:tuples[key][:32] for key in ('observations','next_observations','actions','high_actor_goals','high_actor_targets')}
            native=np.asarray(jax.block_until_ready(native_losses(agent,check_batch)),float)
            reproduced=np.asarray([signals['AWR_contribution'][:32].mean(),signals['WP_contribution'][:32].mean()],float)
            error=np.abs(native-reproduced)
            assert np.all(error<=1e-4+1e-5*np.abs(native)), (label,branch,native,reproduced,error)
            variance=signals['actor_std']**2
            quadratic=np.sum((signals['raw_actor']-signals['data_action'])**2/(2*variance),axis=1)
            normalization=np.sum(np.log(signals['actor_std']),axis=1)+np.log(2*np.pi)
            nll_error=np.max(np.abs(signals['data_NLL']-quadratic-normalization))
            assert nll_error<=1e-4+1e-5*np.max(np.abs(signals['data_NLL']))
            assert np.all(signals['actor_std']==1.)
            checks.append(dict(seed=item['seed'],variant=item['variant'],branch=branch,
                               native_losses=native.tolist(),reproduced_losses=reproduced.tolist(),
                               max_absolute_loss_error=float(error.max()),gaussian_NLL_max_abs_error=float(nll_error),
                               fixed_sigma=1.,new_parameter_updates=0))
            summaries+=summary_rows(item,branch,tuples,signals)
            path=ROOT/'runs'/f'SIGNALS_{label}_{branch}.npz'
            identities.append(dict(path=str(path),sha256=sha(path),n=len(tuples['state_idx']),
                                   variant=item['variant'],seed=item['seed'],branch=branch))
            del signals
        signals=forward_all(agent,s2,f'{label}_S2')
        conditionals+=conditional_rows(item,agent,groups,s2,signals)
        if signals:
            path=ROOT/'runs'/f'SIGNALS_{label}_S2.npz'
            identities.append(dict(path=str(path),sha256=sha(path),n=len(s2['state_idx']),
                                   variant=item['variant'],seed=item['seed'],branch='S2'))
        after=digest_tree(agent.network)
        assert after==before, 'NETWORK_AND_OPTIMIZER_CHANGED'
        checks.append(dict(seed=item['seed'],variant=item['variant'],branch='IMMUTABILITY',
                           full_network_state_before=before,full_network_state_after=after,
                           native_step=int(agent.network.step),parameter_updates=0))
        write(ROOT/'receipts'/f'SIGNAL_{label}_PROGRESS.json',dict(status='COMPLETED',seconds=time.time()-start,
            summary_rows=len(summaries),conditional_rows=len(conditionals),parameter_updates=0,physical_steps=0))
        del agent,signals;gc.collect()
        print('SIGNAL_MODEL_DONE',label,flush=True)
    csv_union(ROOT/'results/SIGNAL_BY_DISTANCE.csv',summaries)
    csv_union(ROOT/'results/CONDITIONAL_ACTION_AUDIT.csv',conditionals)
    write(ROOT/'results/SIGNAL_FORWARD_AUDIT.json',dict(status='PASS',checks=checks,outputs=identities,
        no_gradients=True,new_parameter_updates=0,all_network_and_optimizer_states_unchanged=True,
        cached_model_aliases=[],independent_model_count=6,
        scientific_tuple_count=16384+len(s2['state_idx']),native_loss_check_tuple_evaluations=6*64,
        reference_actor_query_upper_count=6*len(groups),probe_cap=32768,
        interpretation='Frozen probes follow original rules; they are not a replay of observed SGD batches.',
        seconds=time.time()-start,physical_control_steps=0))
    print('SIGNAL_COMPLETED',len(summaries),len(conditionals),flush=True)


if __name__=='__main__':
    main()

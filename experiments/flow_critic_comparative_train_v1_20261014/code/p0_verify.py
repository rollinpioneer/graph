"""Pure P0 identity/config/Actor initialization checks, zero optimizer updates."""
import os,sys,json,ast,hashlib,time
from pathlib import Path
import jax,numpy as np
ROOT=Path(os.environ['COMPARATIVE_ROOT']);host=sys.argv[1]
sys.path.insert(0,str(ROOT/f'workspaces/{host}_floq'))
from agents.floq_adapter import create_agent
from training_wiring import state_digest
task='humanoidmaze-medium-navigate-singletask-task1-v0' if host=='gfp' else 'cube-double-play-singletask-task2-v0'
prefix='G' if host=='gfp' else 'D'
data=np.load(ROOT/'data'/f'{task}.batch.npz')
obs=data['observations'][:1];actions=data['actions'][:1]
report=[]
for seed in [0,1]:
    c0=json.loads((ROOT/'configs'/f'{prefix}0_{task}.json').read_text())['agent']
    c1=json.loads((ROOT/'configs'/f'{prefix}1_{task}.json').read_text())['agent']
    for c in [c0,c1]:c.update(training_authorized=True,optimizer_steps_allowed=1000000,online_learning_allowed=False)
    a=create_agent(host,seed,obs,actions,c0);b=create_agent(host,seed,obs,actions,c1)
    keys=['modules_actor','modules_vabc'] if host=='gfp' else ['modules_actor_bc_flow']
    for key in keys:
        aa=jax.tree.leaves(a.network.params[key]);bb=jax.tree.leaves(b.network.params[key])
        assert len(aa)==len(bb)
        for x,y in zip(aa,bb):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    np.testing.assert_array_equal(a.rng,b.rng)
    np.testing.assert_array_equal(np.asarray(a.sample_actions(data['observations'][:16],jax.random.PRNGKey(14014))),
                                  np.asarray(b.sample_actions(data['observations'][:16],jax.random.PRNGKey(14014))))
    assert int(a.network.step)==int(b.network.step)==1
    assert int(a.network.opt_state[0].count)==int(b.network.opt_state[0].count)==0
    diff={k:[c0.get(k),c1.get(k)] for k in sorted(c0.keys()|c1.keys()) if c0.get(k)!=c1.get(k)}
    allowed={'critic_type','r_min','r_max','q_min','q_max','noise_min','noise_max','block_depth','block_width',
        'activate_before_layernorm','critic_flow_steps','distill_num_ensembles','embed_time','flow_num_ensembles',
        'noise_coverage','noise_samples','num_bins','num_transformations','reward_offset','sigma',
        'time_embed_dim','train_at_zero_only','use_prob_embed','use_resnets'}
    assert set(diff)<=allowed,('Unapproved noncritic differences',diff)
    report.append(dict(seed=seed,status='PASS',actor_parameter_keys=keys,actor_initialization_exact=True,
        initial_actions_exact=True,agent_rng_exact=True,optimizer_updates=0,critic_only_config_diff=diff,
        baseline_state_sha256=state_digest(a),combination_state_sha256=state_digest(b)))
(ROOT/'receipts'/f'{host}_p0_identity.json').write_text(json.dumps(dict(status='PASS',host=host,
    seeds=report,device=str(jax.devices()),optimizer_updates=0,epoch=time.time()),indent=2))
print('P0_PASS',host,'seeds 0/1; same Actor weights/actions/RNG; zero updates',flush=True)

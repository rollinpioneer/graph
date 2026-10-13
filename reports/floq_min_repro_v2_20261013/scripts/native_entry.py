"""Observe the pinned official entry point without changing numerical updates.

The original source checkout is untouched. The hooks log existing values,
force synchronization every 100 updates for finite checks, and switch W&B to
offline. P1 alone performs isolated probes and a checkpoint round trip.
"""
import os
import sys
import json
import time
import pickle
import hashlib
import random
import pathlib
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = pathlib.Path(os.environ['FLOQ_RUN_DIR'])
PHASE = os.environ['FLOQ_PHASE']
SOURCE = ROOT / 'source'
os.chdir(SOURCE)
sys.path.insert(0, str(SOURCE))
RUN.mkdir(parents=True, exist_ok=True)

import numpy as np
import jax
import flax
import main as native
import utils.evaluation as evaluation
from agents.floq import FloQAgent
from utils.flax_utils import restore_agent

state = {'updates': 0, 'start_epoch': time.time()}
captured = {}


def plain(x):
    if hasattr(x, 'to_dict'):
        return plain(x.to_dict())
    if hasattr(x, 'items'):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, bool, int, float)) or x is None:
        return x
    a = np.asarray(jax.device_get(x))
    return a.item() if a.ndim == 0 else a.tolist()


def write(name, obj):
    p = RUN / name
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(plain(obj), indent=2, allow_nan=False))
    t.replace(p)


def append(name, obj):
    with (RUN / name).open('a') as f:
        f.write(json.dumps(plain(obj), allow_nan=False) + '\n')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def tree_sha(tree):
    h = hashlib.sha256()
    for a in jax.tree_util.tree_leaves(tree):
        a = np.asarray(jax.device_get(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


orig_setup = native.setup_wandb


def offline_setup(**kw):
    kw['mode'] = 'offline'
    return orig_setup(**kw)


native.setup_wandb = offline_setup
orig_make = native.make_env_and_datasets


def observe_make(*args, **kw):
    values = orig_make(*args, **kw)
    env, eval_env, data, val = values
    captured.update(env=env, eval_env=eval_env, data=data)
    u = eval_env.unwrapped
    task_keys = ['cur_task_id', 'task_id', 'maze_map', '_target_pos', '_success_radius']
    write('environment.json', dict(
        spec_id=eval_env.spec.id,
        max_episode_steps=eval_env.spec.max_episode_steps,
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        task={k: plain(getattr(u, k)) for k in task_keys if hasattr(u, k)},
        train={k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                   'sha256': sha_bytes(np.asarray(v).tobytes())}
               for k, v in data.items()},
        validation={k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                        'sha256': sha_bytes(np.asarray(v).tobytes())}
                    for k, v in val.items()},
        rewards_min=float(data['rewards'].min()),
        rewards_max=float(data['rewards'].max()),
        eval_rng_initial=plain(u.np_random.bit_generator.state),
    ))
    return values


native.make_env_and_datasets = observe_make
orig_create = FloQAgent.create.__func__
orig_update = FloQAgent.update


def observe_create(cls, *args, **kw):
    agent = orig_create(cls, *args, **kw)
    write('actual_config.json', agent.config)
    write('expanded_flags.json', native.get_flag_dict())
    write('initial_state.json', dict(
        agent_tree_sha256=tree_sha(agent),
        network_param_sha256=tree_sha(agent.network.params),
        config_sha256=sha_bytes(json.dumps(plain(agent.config), sort_keys=True).encode()),
        numpy_state_sha256=sha_bytes(pickle.dumps(np.random.get_state())),
        python_state_sha256=sha_bytes(pickle.dumps(random.getstate())),
        jax_rng=agent.rng,
        source_commit=os.environ['FLOQ_SOURCE_COMMIT'],
        matrix_precision=str(jax.config.jax_default_matmul_precision),
        x64=jax.config.jax_enable_x64,
        leaves=[{'shape': list(np.shape(a)), 'dtype': str(a.dtype)}
                for a in jax.tree_util.tree_leaves(agent.network.params)],
    ))
    if PHASE == 'P1':
        np_state = np.random.get_state()
        batch = captured['data'].sample(agent.config['batch_size'])
        np.random.set_state(np_state)
        loss_grad = jax.jit(jax.value_and_grad(agent.total_loss, argnums=1, has_aux=True))
        (loss, info), grads = loss_grad(batch, agent.network.params)
        assert np.isfinite(plain(loss))
        grad_norms = {}
        for name, tree in grads.items():
            leaves = [np.asarray(jax.device_get(a)) for a in jax.tree_util.tree_leaves(tree)]
            assert all(np.isfinite(a).all() for a in leaves)
            grad_norms[name] = float(sum(np.linalg.norm(a) for a in leaves))
        a1, i1 = orig_update(agent, batch)
        a2, i2 = orig_update(a1, batch)
        expected = jax.tree_util.tree_map(
            lambda p, tp: p * agent.config['tau'] + tp * (1 - agent.config['tau']),
            a1.network.params['modules_floq'], a1.network.params['modules_target_floq'])
        target_error = max(float(np.max(np.abs(np.asarray(jax.device_get(a - b)))))
                           for a, b in zip(jax.tree_util.tree_leaves(expected),
                                           jax.tree_util.tree_leaves(a2.network.params['modules_target_floq'])))
        assert target_error < 1e-6
        obs, act = batch['observations'], batch['actions']
        ratios = jax.random.uniform(jax.random.PRNGKey(912), (obs.shape[0], 1))
        current = agent.compute_current_flow_returns(obs, act, ratios)
        target = agent.compute_target_flow_returns(obs, act, ratios)
        assert current.shape == (2, obs.shape[0], 1)
        assert target.shape == (obs.shape[0], 1)
        assert np.isfinite(np.asarray(current)).all() and np.isfinite(np.asarray(target)).all()
        write('initialization_probes.json', dict(
            loss=loss, info=info, gradient_norms=grad_norms,
            target_update_error=target_error,
            target_update_uses_pre_optimizer_online_parameters=True,
            current_return_shape=current.shape, target_return_shape=target.shape,
            probe_updates=2, applied_to_training_agent=False,
            numpy_sampling_state_restored=True,
        ))
    return agent


FloQAgent.create = classmethod(observe_create)


def observe_update(agent, batch, **kw):
    new, info = orig_update(agent, batch, **kw)
    state['updates'] += 1
    if state['updates'] % 100 == 0:
        values = plain(info)
        if not all(np.isfinite(v) for v in values.values()):
            write('numerical_failure.json', dict(step=state['updates'],
                                                 metrics={k: str(v) for k, v in values.items()}))
            raise ArithmeticError('Nonfinite official loss/gradient; preserved logs, no restart')
        state.update(last_metrics=values, epoch=time.time(), phase='TRAINING_RUNNING')
        write('training_progress.json', state)
        if state['updates'] % 5000 == 0:
            append('loss_samples.jsonl', dict(step=state['updates'], epoch=time.time(), metrics=values))
    return new, info


FloQAgent.update = observe_update
orig_supply = evaluation.supply_rng


def observe_supply(f, rng=jax.random.PRNGKey(0)):
    state['eval_policy_initial_key'] = plain(rng)
    return orig_supply(f, rng)


evaluation.supply_rng = observe_supply
orig_eval = native.evaluate


def observe_eval(*args, **kw):
    begin = time.time()
    env = kw['env']
    state.update(phase='NATIVE_EVALUATING', epoch=begin)
    write('training_progress.json', state)
    pre_rng = plain(env.unwrapped.np_random.bit_generator.state)
    metrics, trajs, renders = orig_eval(*args, **kw)
    for episode, traj in enumerate(trajs):
        end = evaluation.flatten(traj['info'][-1])
        success = end.get('success', end.get('episode.success'))
        assert success is not None, 'Native success metric missing'
        append('native_episodes.jsonl', dict(
            step=state['updates'], episode=episode, success=float(success),
            return_=float(np.sum(traj['reward'])), length=len(traj['reward']),
            initial_observation_sha256=sha_bytes(np.asarray(traj['observation'][0]).tobytes()),
            final_info=end, policy_initial_key=state['eval_policy_initial_key'],
        ))
    append('native_evaluations.jsonl', dict(
        step=state['updates'], duration_seconds=time.time()-begin, metrics=metrics,
        episodes=len(trajs), policy_initial_key=state['eval_policy_initial_key'],
        env_rng_before=pre_rng, env_rng_after=plain(env.unwrapped.np_random.bit_generator.state),
        numpy_state_after_sha256=sha_bytes(pickle.dumps(np.random.get_state())),
    ))
    state.update(phase='TRAINING_RUNNING', epoch=time.time())
    write('training_progress.json', state)
    return metrics, trajs, renders


native.evaluate = observe_eval
orig_save = native.save_agent


def observe_save(agent, save_dir, epoch):
    orig_save(agent, save_dir, epoch)
    path = pathlib.Path(save_dir) / ('params_%d.pkl' % epoch)
    append('checkpoint_index.jsonl', dict(step=epoch, path=str(path), bytes=path.stat().st_size,
                                         sha256=file_sha(path), agent_tree_sha256=tree_sha(agent)))
    supplemental = dict(python=random.getstate(), numpy=np.random.get_state(),
                        eval_env_rng=captured['eval_env'].unwrapped.np_random.bit_generator.state)
    with (RUN / ('supplemental_rng_%d.pkl' % epoch)).open('wb') as f:
        pickle.dump(supplemental, f)
    if PHASE == 'P1':
        restored = restore_agent(agent, save_dir, epoch)
        before, after = tree_sha(agent), tree_sha(restored)
        assert before == after
        obs = captured['data']['observations'][0]
        key = jax.random.PRNGKey(4819)
        a = np.asarray(agent.sample_actions(obs, seed=key, temperature=0))
        b = np.asarray(restored.sample_actions(obs, seed=key, temperature=0))
        assert np.array_equal(a, b) and np.isfinite(a).all()
        assert np.max(np.abs(a)) <= 1
        write('checkpoint_roundtrip.json', dict(passed=True, checkpoint=str(path),
                                               before_sha256=before, after_sha256=after,
                                               policy_actions_equal=True,
                                               action=a, temperature_zero_still_stochastic=True))


native.save_agent = observe_save


if __name__ == '__main__':
    try:
        native.app.run(native.main)
    except SystemExit as e:
        if e.code not in (None, 0):
            write('entry_failure.json', dict(error=str(e), traceback=traceback.format_exc()))
            raise
    except BaseException as e:
        write('entry_failure.json', dict(error=str(e), traceback=traceback.format_exc()))
        raise
    finally:
        write('entry_final_state.json', state)

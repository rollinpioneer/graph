"""SAW-RD-01 instrumentation; original official agent and sampler remain intact."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/saw/impls'))
import jax
import numpy as np
import flax.serialization
import ogbench
from agents.saw import SAWAgent, get_config
from utils.datasets import Dataset, HGCDataset


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def separate_numpy(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def load(name):
    config = get_config().to_dict()
    env, data, val = ogbench.make_env_and_datasets(
        name, dataset_dir=str(ROOT / 'vendor/data'), compact_dataset=True)
    return env, HGCDataset(Dataset.create(**data), config), HGCDataset(Dataset.create(**val), config), config


def init_agent(data, config):
    random.seed(0)
    np.random.seed(0)
    example = data.sample(1)
    return SAWAgent.create(0, example['observations'], example['actions'], config)


def sync(tree):
    return jax.block_until_ready(tree)


def checkpoint(agent, env, step, directory, label='params'):
    sync(agent)
    path = Path(directory) / f'{label}_{step}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(agent=flax.serialization.to_state_dict(jax.device_get(agent)), step=step,
                 numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                 env_rng=env.unwrapped.np_random.bit_generator.state,
                 action_rng=env.action_space.np_random.bit_generator.state,
                 qpos=env.unwrapped.data.qpos.copy(), qvel=env.unwrapped.data.qvel.copy(),
                 current_goal=getattr(env.unwrapped, 'cur_goal_xy', None),
                 elapsed_steps=getattr(env, '_elapsed_steps', None))
    with open(str(path) + '.tmp', 'wb') as f:
        pickle.dump(state, f, protocol=4)
    Path(str(path) + '.tmp').replace(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, step=step)


def evaluate(agent, name, seed, episodes=50, save_dir=None, label='eval', stop_check=None):
    """Official goal threshold, termination, clipping, temperature and task averaging.

    Separate explicit scene/action/JAX seeds remove dependence on training or logging.
    Official reset uses global NumPy for goal noise before Gymnasium reset; isolation
    therefore scopes the entire evaluation, rather than relying on reset(seed) alone.
    """
    start = time.monotonic()
    cpu_agent = jax.device_put(agent, jax.devices('cpu')[0])
    env = ogbench.make_env_and_datasets(name, env_only=True)
    action_fn = jax.jit(lambda a, s, g: a.network.select('actor')(s, g, temperature=0.0).mode(), backend='cpu')
    records, trajectories = [], []
    with separate_numpy(seed):
        for task in range(1, len(env.unwrapped.task_infos) + 1):
            for episode in range(episodes):
                reset_seed = seed + task * 10000 + episode
                env.action_space.seed(reset_seed + 2000000)
                s, info = env.reset(seed=reset_seed, options={'task_id': task, 'render_goal': False})
                g = info['goal'].copy()
                initial = s.copy()
                scene_rng = env.unwrapped.np_random.bit_generator.state
                action_rng = env.action_space.np_random.bit_generator.state
                states, actions, rewards, qvels, contacts = [s.copy()], [], [], [env.unwrapped.data.qvel.copy()], []
                terminated = truncated = False
                while not (terminated or truncated):
                    if stop_check and stop_check():
                        env.close()
                        raise RuntimeError('RESOURCE_STOP_REQUESTED')
                    a = np.clip(np.asarray(action_fn(cpu_agent, s, g)), -1, 1)
                    s, r, terminated, truncated, info = env.step(a)
                    states.append(s.copy()); actions.append(a); rewards.append(r)
                    qvels.append(env.unwrapped.data.qvel.copy()); contacts.append(int(env.unwrapped.data.ncon))
                xy = np.asarray(states)
                rec = dict(task_id=task, episode=episode, reset_seed=reset_seed, seed=seed,
                           task_name=env.unwrapped.task_infos[task-1]['task_name'], success=float(info['success']),
                           steps=len(actions), initial=initial.tolist(), goal=g.tolist(), final=s.tolist(),
                           final_distance=float(np.linalg.norm(s-g)), path_length=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                           terminated=bool(terminated), truncated=bool(truncated),
                           first_success_step=len(actions) if info['success'] else None,
                           scene_rng=scene_rng, action_rng=action_rng)
                records.append(rec)
                trajectories.append(dict(**rec, states=xy, actions=np.asarray(actions), rewards=np.asarray(rewards),
                                         qvel=np.asarray(qvels), contacts=np.asarray(contacts)))
    env.close()
    if save_dir:
        path = Path(save_dir) / f'{label}_trajectories.pkl'
        with open(path, 'wb') as f:
            pickle.dump(trajectories, f, protocol=4)
    result = dict(seed=seed, episodes_per_task=episodes, eval_temperature=0, eval_on_cpu=True,
                  seconds=time.monotonic()-start, overall_success=float(np.mean([r['success'] for r in records])),
                  tasks={str(t): float(np.mean([r['success'] for r in records if r['task_id']==t])) for t in range(1,6)},
                  records=records)
    if save_dir:
        write_json(Path(save_dir) / f'{label}.json', result)
    return result

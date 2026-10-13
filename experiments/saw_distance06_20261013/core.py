"""CPU-only frozen evaluation; no optimizer update or training path."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
import copy, csv, hashlib, json, pickle, random, sys, time
from collections import deque
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GP = Path('/home/xushijie3/work/graph_saw_goalprobe05_20261013/experiments/saw_goalprobe05_20261013')
OLD = Path('/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012')
PREV = Path('/home/xushijie3/work/graph_saw_analysis03b_20261012/experiments/saw_analysis03b_20261012')
MECH = Path('/home/xushijie3/work/graph_saw_mech04_20261013/experiments/saw_mech04_20261013')
sys.path.insert(0, str(OLD / 'vendor/saw/impls'))
import jax, numpy as np, mujoco, gymnasium as gym, ogbench, flax.serialization
from agents.saw import SAWAgent, get_config
assert all(d.platform == 'cpu' for d in jax.devices())
assert len(os.sched_getaffinity(0)) <= 4
STATE = mujoco.mjtState.mjSTATE_INTEGRATION

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def write(p, x):
    p = Path(p); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')
    tmp.replace(p)

def rows(p, data, fields=None):
    if fields is None: fields = list(data[0]) if data else ['status']
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(data)

def dump(p, x):
    with open(p, 'xb') as f: pickle.dump(x, f, protocol=4)

def equal(a, b):
    if isinstance(a, dict): return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))

def guard():
    p = ROOT / 'receipts/RESOURCE_START.json'
    assert p.exists(), 'BUDGET_NOT_STARTED'
    x = json.loads(p.read_text()); assert time.time() - x['start_epoch'] < 14400, 'CPU_WALL_CAP'
    assert len(os.sched_getaffinity(0)) <= 4

def step(env, action, phase):
    guard(); p = ROOT / 'receipts/CONTROL_BUDGET.json'
    x = json.loads(p.read_text()); assert x['used'] < 25000, 'CONTROL_CAP'
    x['used'] += 1; x['by_phase'][phase] = x['by_phase'].get(phase, 0) + 1; write(p, x)
    return env.step(action)

def integration(u):
    s = np.empty(mujoco.mj_stateSize(u.model, STATE)); mujoco.mj_getState(u.model, u.data, s, STATE); return s

def wrappers(env):
    out = []
    while isinstance(env, gym.Wrapper):
        out.append(dict(class_name=type(env).__name__, fields={k: copy.deepcopy(env.__dict__[k]) for k in ('_elapsed_steps', '_has_reset', '_terminated', '_truncated') if k in env.__dict__})); env = env.env
    return out

def snapshot(env, goal, reset_seed):
    u = env.unwrapped
    return dict(integration=integration(u), qpos=u.data.qpos.copy(), qvel=u.data.qvel.copy(), goal=np.asarray(goal).copy(), goal_xy=np.asarray(u.cur_goal_xy).copy(), task=u.cur_task_id,
                env_rng=copy.deepcopy(u.np_random.bit_generator.state), action_rng=copy.deepcopy(env.action_space.np_random.bit_generator.state), numpy_rng=np.random.get_state(), python_rng=random.getstate(), wrappers=wrappers(env), reset_seed=reset_seed)

def restore(env, scene, goal=None):
    goal = np.asarray(scene['goal'] if goal is None else goal).copy()
    u = env.unwrapped; env.reset(seed=scene['reset_seed'], options={'task_id': scene['task'], 'render_goal': False}); u.set_goal(goal_xy=goal)
    mujoco.mj_setState(u.model, u.data, scene['integration'], STATE); mujoco.mj_forward(u.model, u.data); mujoco.mj_setState(u.model, u.data, scene['integration'], STATE)
    u.np_random.bit_generator.state = copy.deepcopy(scene['env_rng']); env.action_space.np_random.bit_generator.state = copy.deepcopy(scene['action_rng'])
    w = env
    for item in scene['wrappers']:
        assert type(w).__name__ == item['class_name']
        for k, v in item['fields'].items(): setattr(w, k, copy.deepcopy(v))
        w = w.env
    np.random.set_state(scene['numpy_rng']); random.setstate(scene['python_rng'])
    actual = snapshot(env, goal, scene['reset_seed'])
    errors = {k: float(np.max(np.abs(actual[k] - scene[k]))) for k in ('integration', 'qpos', 'qvel')}
    assert max(errors.values()) <= 1e-10, errors
    assert np.array_equal(actual['goal_xy'], goal)
    for k in ('env_rng', 'action_rng', 'numpy_rng', 'python_rng', 'wrappers'): assert equal(actual[k], scene[k]), k
    return u.get_ob().copy(), max(errors.values())

def env_new():
    e = ogbench.make_env_and_datasets('pointmaze-large-stitch-v0', env_only=True)
    assert e.spec.max_episode_steps == 1000 and e.unwrapped._goal_tol == 1.0 and e.unwrapped._teleport_info is None
    return e

class Geometry:
    def __init__(self, env):
        self.u = env.unwrapped; self.cells = sorted(tuple(int(v) for v in row) for row in np.argwhere(self.u.maze_map == 0))
        self.free = set(self.cells); self.distances = {c: self.bfs(c) for c in self.cells}
    def xy(self, cell): return np.asarray(self.u.ij_to_xy(cell), dtype=float)
    def cell(self, xy): return self.u.xy_to_ij(xy)
    def neighbors(self, c): return sorted((c[0] + di, c[1] + dj) for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)] if (c[0] + di, c[1] + dj) in self.free)
    def bfs(self, goal):
        d = {goal: 0}; q = deque([goal])
        while q:
            c = q.popleft()
            for n in self.neighbors(c):
                if n not in d: d[n] = d[c] + 1; q.append(n)
        return d
    def firsts(self, start, goal):
        d = self.distances[goal]
        return [n for n in self.neighbors(start) if d[n] == d[start] - 1]
    def route(self, start, goal):
        p = [start]
        while p[-1] != goal: p.append(self.firsts(p[-1], goal)[0])
        return p
    def remaining(self, xy, goal):
        c = self.cell(xy); d = self.distances[goal]
        if c not in d: return None
        if c == goal: return float(np.linalg.norm(xy - self.xy(goal)))
        n = self.firsts(c, goal)[0]
        return float(np.linalg.norm(xy - self.xy(n)) + 4 * d[n])

def angle(a):
    return float(np.degrees(np.arctan2(a[1], a[0]))) if np.linalg.norm(a) > 1e-12 else None

def between(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / den, -1, 1)))) if den > 1e-12 else None

def rollout(env, geom, item, gi, policy, phase):
    g = item['goals'][gi]; start = tuple(item['cell']); valid = [tuple(v) for v in item['firsts'][gi]]
    s, err = restore(env, item['scene'], g); initial = snapshot(env, g, item['scene']['reset_seed'])
    states = [s.copy()]; qvels = [env.unwrapped.data.qvel.copy()]; raw = []; clipped = []; full = [integration(env.unwrapped)]; times = []
    first_exit = None; exit_step = None; terminated = truncated = False; reached = False
    for t in range(25):
        a = np.asarray(policy(s, g), dtype=float); assert a.shape == (2,) and np.isfinite(a).all()
        c = np.clip(a, -1, 1); s, r, terminated, truncated, info = step(env, c, phase)
        raw.append(a); clipped.append(c); states.append(s.copy()); qvels.append(env.unwrapped.data.qvel.copy()); full.append(integration(env.unwrapped)); times.append((bool(terminated), bool(truncated), float(info['success'])))
        cell = geom.cell(s)
        if first_exit is None and cell != start: first_exit = cell; exit_step = t + 1
        reached = reached or bool(info['success'])
        if terminated or truncated: break
    xy = np.asarray(states); actions = np.asarray(clipped); norms = np.linalg.norm(actions, axis=1)
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()); disp = float(np.linalg.norm(xy[-1] - xy[0])); maximum = float(np.linalg.norm(xy - xy[0], axis=1).max())
    correct = first_exit in valid if first_exit is not None else False
    category = 'correct' if correct else ('wrong_channel' if first_exit is not None else ('no_motion' if maximum <= .1 else 'not_left_region'))
    ref = geom.xy(valid[0]) - geom.xy(start); ref /= np.linalg.norm(ref)
    ri = geom.remaining(xy[0], tuple(item['goal_cells'][gi])); rf = geom.remaining(xy[-1], tuple(item['goal_cells'][gi]))
    metric = dict(state_id=item['state_id'], group=item['group'], cluster=item['cluster'], source=item['source'], goal_index=gi, correct=int(correct), category=category, first_exit=json.dumps(first_exit), first_exit_step=exit_step, steps=len(actions), terminated=bool(terminated), truncated=bool(truncated), local_goal_reached=bool(reached), displacement=disp, maximum_displacement=maximum, path_length=path, reference_progress=None if rf is None else ri-rf, initial_channel_projection=float(np.dot(xy[-1]-xy[0], ref)), mean_action_norm=float(norms.mean()), small_action_fraction=float(np.mean(norms <= .1)), saturation_fraction=float(np.mean(np.abs(np.asarray(raw)) >= 1)), restore_error=err)
    trace = dict(initial=initial, states=xy, qvel=np.asarray(qvels), integration=np.asarray(full), raw_actions=np.asarray(raw), actions=actions, termination_success=times, goals=item['goals'], route=item['routes'][gi], reference_firsts=valid, metric=metric)
    return metric, trace

def model_entries():
    ans = []
    for seed in (0, 1, 2):
        for variant in ('A', 'B'):
            source = (OLD if variant == 'A' else PREV) if seed == 0 else MECH
            name = 'stitch_checkpoints.json' if seed == 0 else f'C_stitch_{variant}_s{seed}_checkpoints.json'
            entries = json.loads((source/'results'/name).read_text()); item = next(x for x in entries if x['step'] == 1000000 and Path(x['path']).name == 'params_1000000.pkl')
            assert sha(item['path']) == item['sha256']
            ans.append(dict(**item, seed=seed, variant=variant, source_index=str(source/'results'/name), source_index_sha256=sha(source/'results'/name)))
    return ans

def load_agent(item):
    guard(); config = get_config().to_dict()
    if item['variant'] == 'B': config.update(actor_p_trajgoal=.5, actor_p_randomgoal=.5)
    if 'config_path' in item:
        source_config = json.loads(Path(item['config_path']).read_text())['config']
        normalized = lambda v: json.loads(json.dumps(v, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
        assert normalized(config) == normalized(source_config), 'FROZEN_CONFIG_MISMATCH'
        assert sha(item['config_path']) == item['config_sha256']
    # Shape-only initialization is not a gradient update; no dataset sampling or optimizer step.
    agent = SAWAgent.create(item['seed'], np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), config)
    with open(item['path'], 'rb') as f: saved = pickle.load(f)
    assert saved['step'] == 1000000
    agent = flax.serialization.from_state_dict(agent, saved['agent']); agent = jax.device_put(agent, jax.devices('cpu')[0])
    assert int(agent.network.step) == 1000001
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent.network.params))
    return agent

raw_mode = jax.jit(lambda a, s, g: a.network.select('actor')(s, g, temperature=0.).mode())

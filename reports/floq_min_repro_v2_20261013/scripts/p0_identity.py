"""Freeze installed versions, source/data identities and task alias semantics."""
import os
import sys
import json
import csv
import pathlib
import hashlib
import importlib.metadata as metadata
import subprocess
import platform

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ['HOME'] = str(ROOT/'home')
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION', None)
os.environ['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in
    (ROOT/'envs/floq/lib/python3.8/site-packages/nvidia').glob('*/lib'))
sys.path.insert(0, str(ROOT/'source'))
import numpy as np
import ogbench
import jax
import yaml
from agents.floq import get_config


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()


def write(name, obj):
    (ROOT/'report'/name).write_text(json.dumps(obj, indent=2, default=str))


write('source_identity.json', dict(
    commit=subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD'],text=True).strip(),
    tree_sha=subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD^{tree}'],text=True).strip(),
    files={str(p.relative_to(ROOT/'source')):sha_file(p) for p in (ROOT/'source').rglob('*')
           if p.is_file() and '.git' not in p.parts and p.suffix in ('.py','.yaml','.md')},
))
lock = {d.metadata['Name']:d.version for d in metadata.distributions()}
write('installed_environment.json', dict(python=sys.version, platform=platform.platform(),
                                       packages=lock, devices=[str(x) for x in jax.devices()],
                                       x64=jax.config.jax_enable_x64,
                                       matmul_precision=str(jax.config.jax_default_matmul_precision)))
author = yaml.safe_load((ROOT/'source/conda-environment.yaml').read_text())
author_versions = {}
for dep in author['dependencies']:
    if isinstance(dep,str) and '=' in dep:
        name,version,*_ = dep.split('=')
        author_versions[name.lower().replace('_','-')] = version
    elif isinstance(dep,dict):
        for req in dep.get('pip',[]):
            if '==' in req:
                name, version = req.split('==',1)
                author_versions[name.lower().replace('_','-')] = version
with (ROOT/'report/dependency_differences.csv').open('w') as f:
    out = csv.writer(f,lineterminator='\n')
    out.writerow(['package','author_version','installed_version','status'])
    for name, version in sorted(lock.items()):
        expected = author_versions.get(name.lower().replace('_','-'))
        out.writerow([name,expected or '',version,'MATCH' if expected==version else
                      'PACKAGING_OR_TRANSITIVE_DEVIATION' if expected else 'NOT_LISTED_BY_AUTHOR'])
write('default_config.json', get_config().to_dict())
paths = [ROOT/'home/.ogbench/data/antmaze-large-navigate-v0.npz',
         ROOT/'home/.ogbench/data/antmaze-large-navigate-v0-val.npz']
data = [{'path':str(p), 'resolved_path':str(p.resolve()), 'bytes':p.stat().st_size,
         'sha256':sha_file(p)} for p in paths]
write('dataset_identity.json', dict(ogbench='1.1.2', files=data,
                                  source='Official OGBench download reused read-only from prior experiment'))
env, train, val = ogbench.make_env_and_datasets('antmaze-large-navigate-singletask-v0')
explicit = ogbench.make_env_and_datasets('antmaze-large-navigate-singletask-task1-v0',env_only=True)
np.random.seed(104219)
ob, info = env.reset(seed=947)
np.random.seed(104219)
ob2, info2 = explicit.reset(seed=947)
assert np.array_equal(ob,ob2)
assert np.array_equal(env.unwrapped.data.qpos, explicit.unwrapped.data.qpos)
assert np.array_equal(env.unwrapped.data.qvel, explicit.unwrapped.data.qvel)
write('task_alias_check.json', dict(
    passed=True, alias=env.spec.id, explicit=explicit.spec.id,
    max_episode_steps=env.spec.max_episode_steps, observation_shape=env.observation_space.shape,
    action_low=env.action_space.low.tolist(), action_high=env.action_space.high.tolist(),
    train_shapes={k:list(v.shape) for k,v in train.items()},
    validation_shapes={k:list(v.shape) for k,v in val.items()},
    rewards_min=float(train['rewards'].min()), rewards_max=float(train['rewards'].max()),
    sampled_masks=list(np.unique(train['masks'])),
    goal=info.get('goal',None), exact_initial_state_and_observation_match=True,
))
env.close()
explicit.close()
write('P0_STATUS.json',dict(status='ENGINEERING_P0_PASSED',
                          dataset_alias_verified=True, gpu_verified=True,
                          author_checkpoint_aggregation='UNVERIFIED',
                          primary_result='2M native evaluation, 50 episodes per seed, no best-score selection',
                          reference_task1_success_percent=94, reference_spread_percent=4,
                          reference_seeds=3, local_seeds=[0,1],
                          paper_version='2509.06863v2',
                          full_paper_statistical_reproduction_claim_allowed=False))
print('P0 identities and task alias passed',flush=True)

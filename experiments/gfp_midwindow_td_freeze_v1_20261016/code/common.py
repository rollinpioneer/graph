import sys, json, pickle, hashlib, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/xushijie3/work/gfp_bfc_dev_screen_v1_20261016')
FORMAL = Path('/home/xushijie3/work/gfp_eta0_ablation_20261015')
PYTHON = '/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
CHECKPOINT = SOURCE/'runs/s3_R3/checkpoint_50000.pkl'
sys.path.insert(0, str(SOURCE/'source/gfp'))
spec = importlib.util.spec_from_file_location('source_common', SOURCE/'code/common.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
write, sha, load_base = original.write, original.sha, original.load_base

def make_agent(seed, arm):
    import flax
    from frozen import FrozenTD
    assert seed == 3 and arm in ('L0', 'L1')
    ag, cfg, _ = original.make_agent(3, 'R3')
    with open(CHECKPOINT, 'rb') as f: payload = pickle.load(f)
    ag = flax.serialization.from_state_dict(ag, payload['agent'])
    assert int(ag.network.step) == 350001
    assert int(ag.network.opt_state[0].count) == 350000
    assert ag.tau_pi == 0.00005 and ag.epsilon == 0
    assert not ag.config['target_use_next']
    if arm == 'L1':
        ag = FrozenTD(ag.rng, ag.network, ag.config, reference=ag.reference,
                      tau_pi=0., epsilon=0., frozen_vabc=ag.network.params['modules_vabc'])
    return ag, cfg, CHECKPOINT

def digest(tree):
    import jax, numpy as np
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a = np.asarray(x)
        h.update(str((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()

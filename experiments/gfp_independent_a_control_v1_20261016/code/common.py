import sys, json, pickle, hashlib, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/xushijie3/work/gfp_bfc_dev_screen_v1_20261016')
SUPPORT = Path('/home/xushijie3/work/gfp_reference_support_diagnostic_v1_20261016')
FORMAL = Path('/home/xushijie3/work/gfp_eta0_ablation_20261015')
sys.path.insert(0, str(SOURCE/'code'))
sys.path.insert(0, str(SOURCE/'source/gfp'))
spec = importlib.util.spec_from_file_location('source_common', SOURCE/'code/common.py')
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
write, sha, load_base = original.write, original.sha, original.load_base
def digest(tree):
 import jax,numpy as np
 h=hashlib.sha256()
 for x in jax.tree_util.tree_leaves(tree):
  a=np.asarray(x);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
 return h.hexdigest()
def make_agent():
 import flax
 ag,cfg,_=original.make_agent(3,'R3')
 source=SOURCE/'runs/s3_R3/checkpoint_50000.pkl'
 assert sha(source)==json.loads((ROOT/'protocol.json').read_text())['source_sha256']
 with open(source,'rb') as f:p=pickle.load(f)
 ag=flax.serialization.from_state_dict(ag,p['agent'])
 assert int(ag.network.step)==350001 and int(ag.network.opt_state[0].count)==350000
 assert ag.tau_pi==.00005 and ag.epsilon==0 and not ag.config['target_use_next']
 assert ag.config['target_agg']==ag.config['q_agg']=='mean' and ag.config['discount']==.995
 return ag.replace(tau_pi=0.),cfg,p

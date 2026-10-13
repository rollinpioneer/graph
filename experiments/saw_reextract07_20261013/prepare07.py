"""CPU-only identities, exact fresh initialization, and complete scene freeze."""
from common07 import *
import importlib.util, platform, subprocess
assert all(d.platform=='cpu' for d in jax.devices())
start=time.monotonic()
refs=references(); identities=[]
for r,item in enumerate(refs):
    ref=load_ref(item);p,o=fresh(r)
    assert all(not equal(p[k],ref.network.params[k]) for k in KEYS)
    hashes=[tree_sha(fresh(r)[0]) for arm in range(4)];assert len(set(hashes))==1
    init=ROOT/'runs'/f'INIT_b{r}.pkl';dump(init,dict(params=p,opt=o,student_internal_rng=np.asarray(jax.random.PRNGKey(78001300+r))))
    identities.append(dict(reference=item,reference_native_step=int(ref.network.step),reference_full_state_sha256=tree_sha(flax.serialization.to_state_dict(ref)),reference_parameter_sha256=tree_sha(ref.network.params),reference_optimizer_sha256=tree_sha(ref.network.opt_state),student_seed=71001300+r,student_internal_seed=78001300+r,actor_keys=KEYS,four_arm_parameter_sha256=hashes,init_path=str(init),init_sha256=sha(init),new_adam_count=int(o[0].count),new_adam_moments_zero=True))
write(ROOT/'REFERENCE_AND_INIT_IDENTITY.json',identities)
spec=importlib.util.spec_from_file_location('d06env',D06/'core.py');ec=importlib.util.module_from_spec(spec);spec.loader.exec_module(ec)
env=ec.env_new();scenes=[]
for stream in (27101307,28101307):
    np.random.seed(stream);__import__('random').seed(stream);env.action_space.seed(stream)
    for task in range(1,6):
        for episode in range(25):
            seed=stream if task==1 and episode==0 else None
            s,info=env.reset(seed=seed,options={'task_id':task,'render_goal':False})
            scene=ec.snapshot(env,info['goal'],seed);ec.restore(env,scene)
            scenes.append(dict(scene_id=f'g{stream}_t{task}_e{episode}',stream=stream,task=task,episode=episode,scene=scene))
env.close();path=ROOT/'runs/EVALUATION_SCENES.pkl';dump(path,scenes)
def fingerprint(s):return hashlib.sha256(np.asarray(s['integration']).tobytes()+np.asarray(s['goal']).tobytes()).hexdigest()
old=pickle.load(open(MECH/'runs/C_stitch_SCENES.pkl','rb'));oldhash={fingerprint(x.get('scene',x)) for x in old}
manifest=json.loads((D06/'FROZEN_MANIFEST.json').read_text());bank=D06/'runs/SCENE_GOAL_BANK.pkl'
assert sha(bank)==manifest['bank_sha256']
local=pickle.load(open(bank,'rb'));assert len(local)==93
write(ROOT/'EVALUATION_BANK.json',dict(path=str(path),sha256=sha(path),count=len(scenes),streams=[27101307,28101307],reset='official-style deterministic continuous stream; first actual episode reset(seed=stream), following reset()',overlap_with_mech04=sum(fingerprint(x['scene']) in oldhash for x in scenes),development_episodes_per_task_stream=10,primary_episodes_per_task_stream=25,primary_steps=[300000,400000,500000],distance_bank_path=str(bank),distance_bank_sha256=sha(bank),distance_bank_count=len(local),distance_bank_already_consumed=True,d06_core_path=str(D06/'core.py'),d06_core_sha256=sha(D06/'core.py')))
write(ROOT/'RUN_APPENDIX.json',dict(status='CPU_PREPARATION_COMPLETE_AWAITING_BUDGET_SCOPE',authorization_request='Human user requested 执行; explicit P+I versus conditional II choice pending. No GPU computation authorized by this receipt.',branch='codex/saw-reextract-07-20261013',base='6d16c1271b3fad563c58d9b2e34b48418b910860',official_saw_commit=subprocess.check_output(['git','-C',str(OLD/'vendor/saw'),'rev-parse','HEAD'],text=True).strip(),python=platform.python_version(),versions={m:__import__(m).__version__ for m in ['jax','jaxlib','flax','optax','numpy','mujoco','gymnasium']},roots=dict(student=71001300,calibration=73001300,base=74001300,future=75001300,original_random=76001300,replenishment=77001300,student_internal=78001300,manipulation=79001300),rng_implementation='Independent NumPy default_rng generators, saved state per checkpoint. Rejection replenishment has separate generator; base/F/original R reset to same seeds in each arm. Earliest accepted goal among chunks of four; all evaluated candidates billed.',pilot_order=[0,1,2,3],pilot_block=0,pilots_discarded=True,gpu_max_hours=dict(P=.5,I=3.,II=4.5,total=8.),formal_per_job_hours=dict(I=.5,II=.75),cpu_cores=[72,73,74,75],cpu_preparation_analysis_cap_seconds=14400,cpu_eval_probe_cap_seconds=28800,peak_owned_gpu_memory_cap_gib=12,combined_rss_cap_gib=16,new_raw_cap_gib=50,X_authorized=False,seconds=time.monotonic()-start))
print('PREPARED',len(identities),len(scenes),len(local),flush=True)

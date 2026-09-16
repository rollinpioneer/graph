from __future__ import annotations
import copy,importlib.metadata,random
from pathlib import Path
import numpy as np
from .io_utils import new_dir,dump,sha256,value_hash,ROOT
from .load_sources import load_sources
from .specs import eval_specs

def assert_versions(config):
    observed={}
    for name,expected in config['versions'].items():
        got=importlib.metadata.version(name);observed[name]=got
        if got!=expected:raise RuntimeError(f'{name}: expected {expected}, got {got}; isolate environment, do not silently upgrade')
    return observed

def state_dict_digest(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(model.policy.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def train_job(config,repo,v6_tools,method,policy_seed,out,*,smoke=False):
    # Import actual upstream PPO. No local substitute if dependency is missing.
    versions=assert_versions(config)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure
    from .gym_bridge import SkillGymEnv
    from .evaluate import evaluate_policy
    cfg=copy.deepcopy(config)
    if smoke:
        cfg['families']['train']=cfg['families']['smoke']
        cfg['families']['validation']=cfg['families']['smoke'][:1]
        cfg['evaluation_repeats']=1
        cfg['profiles']={'BASE64':cfg['profiles']['BASE64']}
        cfg['conditions']=['A_STARTED','FREE_ORDER']
        cfg['training'].update(n_envs=2,n_steps=16,batch_size=32,n_epochs=2,
                               total_timesteps=64,validation_milestones=[0,32,64])
    if method not in cfg['methods']:raise ValueError(method)
    t=cfg['training'];steps=t['total_timesteps'];unit=t['n_envs']*t['n_steps']
    if steps%unit or any(x%unit for x in t['validation_milestones']):raise ValueError('timesteps not rollout multiples')
    if not smoke and policy_seed not in t['policy_seeds']:raise ValueError('undeclared training seed')
    out=new_dir(out);sources=load_sources(repo,v6_tools)
    source_identity={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'p2b').glob('*.py'))}
    dump(out/'run_identity.json',{'method':method,'policy_seed':policy_seed,'smoke_only':smoke,
        'protocol_hash':value_hash(cfg),'config':cfg,'sources':sources[2],'p2b_code_hashes':source_identity,
        'versions':dict(versions,torch=torch.__version__,numpy=np.__version__)})
    torch.set_num_threads(t['torch_threads']);torch.use_deterministic_algorithms(True)
    random.seed(policy_seed);np.random.seed(policy_seed);torch.manual_seed(policy_seed)
    check_cfg=copy.deepcopy(cfg)
    check_cfg['families']['train']=check_cfg['families']['smoke']
    sample=SkillGymEnv(sources,check_cfg,method,policy_seed=policy_seed)
    check_env(sample,warn=True);sample.close()
    factories=[lambda rank=rank:SkillGymEnv(sources,cfg,method,policy_seed=policy_seed,rank=rank,
                log_root=out/'training_logs') for rank in range(t['n_envs'])]
    env=DummyVecEnv(factories)
    model=PPO('MlpPolicy',env,learning_rate=t['learning_rate'],n_steps=t['n_steps'],
        batch_size=t['batch_size'],n_epochs=t['n_epochs'],gamma=cfg['gamma'],gae_lambda=t['gae_lambda'],
        clip_range=t['clip_range'],clip_range_vf=t['clip_range_vf'],normalize_advantage=t['normalize_advantage'],
        ent_coef=t['ent_coef'],vf_coef=t['vf_coef'],max_grad_norm=t['max_grad_norm'],
        target_kl=t['target_kl'],seed=policy_seed,device=t['device'],verbose=0,
        policy_kwargs={'net_arch':t['net_arch'],'activation_fn':torch.nn.Tanh,
                       'ortho_init':True,'optimizer_kwargs':{'eps':t['optimizer_eps']}})
    model.set_logger(configure(str(out/'ppo_logs'),['csv']))
    dump(out/'initialization.json',{'policy_and_value_sha256':state_dict_digest(model),
        'same_seed_must_match_across_methods':True,'same_later_trajectories_required':False})
    specs=eval_specs(cfg,'validation');checkpoints=[]
    try:
        for target in t['validation_milestones']:
            if target>model.num_timesteps:
                model.learn(total_timesteps=target-model.num_timesteps,reset_num_timesteps=False,
                            log_interval=1,progress_bar=False)
            if model.num_timesteps!=target:raise RuntimeError('upstream timesteps rounded unexpectedly')
            name=out/f'policy_{target}.zip';model.save(name)
            checkpoints.append({'step':target,'path':str(name),'sha256':sha256(name)})
            evaluate_policy(model,sources,cfg,method,specs,out/'validation'/str(target),
                policy_seed=policy_seed,checkpoint_step=target)
        dump(out/'checkpoint_index.json',checkpoints)
        dump(out/'complete.json',{'status':'SMOKE_PASS_NOT_RESEARCH_RESULT' if smoke else 'TRAINING_COMPLETE',
            'method':method,'policy_seed':policy_seed,'environment_steps':int(model.num_timesteps),
            'checkpoint_rule':'FIXED_LAST_STEP','checkpoint':checkpoints[-1],
            'config_hash':value_hash(cfg),'source_hashes':source_identity,
            'test_evaluated':False,'confirmation_passed':False})
    except BaseException as exc:
        dump(out/'failed.json',{'type':type(exc).__name__,'message':str(exc),
            'environment_steps':int(model.num_timesteps),'retry_hidden':False})
        raise
    finally:env.close()
    return out

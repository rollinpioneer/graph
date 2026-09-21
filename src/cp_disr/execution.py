"""Bound-runtime driver; unbound CLI never imports this torch-dependent module."""
import json
from dataclasses import replace
from pathlib import Path
import torch
from .runtime import load_runtime
from .neural import Policy
from .collector import Collector
from .rl import Rollout
from .torch_rl import PPO,load_checkpoint,save_checkpoint
from .common import BindingError,canonical
from .prior import PriorSampler

def execute(mode,root,config):
    import yaml
    required=('model','runtime_driver','output_directory','seed','budget','task_cases','resource_cap_seconds','max_infrastructure_empty_episodes')
    missing=[k for k in required if k not in config]
    if missing:raise BindingError('MUST_BIND: '+','.join(missing))
    manifest=yaml.safe_load((root/'experiments/manifests/runtime_manifest.yaml').read_text())
    bundle=load_runtime(manifest)
    for name in ('start_case','next_case','current_snapshot','task_id','episode_start_seconds','original_prior_edges'):
        if not hasattr(bundle,name):raise BindingError('MUST_BIND: runtime driver '+name)
    if config['runtime_driver']!='skill_on_policy_v1':raise BindingError('Unsupported runtime driver')
    torch.manual_seed(config['seed']);model=Policy(**config['model']);trainer=PPO(model);collector=Collector(bundle,model)
    output=Path(config['output_directory']);output.mkdir(parents=True,exist_ok=False)
    if config.get('dry_run') or config.get('optimizer_steps')==0:
        snap=bundle.start_case(bundle.next_case(config['task_cases'],config['seed']))
        collector.reset_episode(snap.env_id,snap.episode_id)
        t,result=collector.step(snap)
        payload={'mode':mode,'dry_run':True,'ppo_update':False,'has_transition':t is not None,'duration':None if t is None else t.duration,'reason':getattr(result,'reason',result)}
        (output/'preflight_transition.json').write_text(canonical(payload)+'\n')
        bundle.environment.close()
        return 0
    if mode=='evaluate':
        if not config.get('checkpoint'):raise BindingError('MUST_BIND: trusted trained checkpoint')
        load_checkpoint(config['checkpoint'],model);model.eval()
        from .evaluation import evaluate
        return evaluate(bundle,model,config,output)
    rollout=Rollout();count=0;logs=[];interaction_seconds=0.;empty_episodes=0;priors=PriorSampler(config['seed'])
    try:
        while count<config['budget'] and interaction_seconds<config['resource_cap_seconds']:
            if bundle.current_snapshot is None:
                bundle.start_case(bundle.next_case(config['task_cases'],config['seed']))
                s=bundle.current_snapshot
                ep_prior=priors.start(s.env_id,s.episode_id,bundle.original_prior_edges)
                bundle.current_snapshot=replace(s,prior_edges=ep_prior.edges,prior_hash=ep_prior.hash)
                collector.reset_episode(s.env_id,s.episode_id)
                with (output/'episode_priors.jsonl').open('a') as f:f.write(canonical(ep_prior)+'\n')
            t,result=collector.step(bundle.current_snapshot)
            if t is not None:
                rollout.append(t);count+=1;interaction_seconds+=t.duration;bundle.current_snapshot=t.next_snapshot
                with (output/'transitions.jsonl').open('a') as f:f.write(canonical(t)+'\n')
            ended=result.get('terminated') if isinstance(result,dict) else result.terminated or result.truncated
            if t is None:
                empty_episodes+=1
                if empty_episodes>=config['max_infrastructure_empty_episodes']:raise BindingError('Runtime repeatedly exposes no executable skill; no reward or transition fabricated')
            if ended:bundle.current_snapshot=None
            if len(rollout.transitions)>=1024 or count==config['budget'] or interaction_seconds>=config['resource_cap_seconds']:
                if not rollout.transitions:break
                logs.extend(trainer.update(rollout));save_checkpoint(output/f'step_{count}.pt',model,trainer.optimizer,{'config':config,'skill_transitions':count})
        (output/'updates.json').write_text(canonical(logs)+'\n');return 0
    finally:bundle.environment.close()

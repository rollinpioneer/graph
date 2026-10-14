"""P0 official OGBench asset preparation, never imports a training main."""
import ast, csv, hashlib, importlib.metadata as md, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import ogbench

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xushijie3/work/flow_baselines_preflight_20261013')
FLOQ=Path('/home/xushijie3/work/floq_min_repro_v2_20261013')
TASKS=['humanoidmaze-medium-navigate-singletask-task1-v0',
       'cube-double-play-singletask-task2-v0', 'puzzle-3x3-play-singletask-task4-v0',
       'antmaze-large-navigate-singletask-task1-v0']
SHAS={'gfp':'1abbd5f0e287caf9bda70e7bc655d0a499ac8401',
      'floq':'3d60e638b42bbef018c56cb1199ff37c3470520d',
      'dfql':'8b6b15803cc1e5605ad1524553a2c95e103b0bad',
      'fql':'e8cd16eb490332924dfa2492097219f181765933'}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,sort_keys=True,default=lambda v:v.tolist()))

def source_config(path):
    import ml_collections
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_config')
    scope={'ml_collections':ml_collections}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['get_config']().to_dict()

def main():
    manifest={'card':'FLOW-CRITIC-DUAL-HOST-PREP-V1','optimizer_steps':0,'sources':{},'datasets':[], 'user_assets':[]}
    for name,commit in SHAS.items():
        s=ROOT/'upstream_snapshots'/name
        files=[{'path':str(p.relative_to(s)),'bytes':p.stat().st_size,'sha256':sha(p)}
               for p in sorted(s.rglob('*')) if p.is_file()]
        manifest['sources'][name]={'status':'FETCHED_VERIFIED' if name=='dfql' else 'PRESENT_VERIFIED',
                                  'commit':commit,'path':str(s),'files':files}
    for report in ['COMBINATION_RANKING_20261014.md','MODULE_TRANSPLANT_FEASIBILITY_20261014.md']:
        manifest['user_assets'].append({'name':report,'status':'MISSING_USER_ACCESS',
            'reason':'Not found in local workspace or remote approved work tree (depth5); card supplies implementable contract.'})
    for kind,path in [('ACTUAL_COMMANDS',OLD/'receipts/baseline_actual_commands_20261014.txt'),
                      ('identity',OLD/'receipts/baseline_identity_checkpoint_audit_20261014.json')]:
        dest=ROOT/'assets_manifest'/path.name
        if path.exists():dest.write_bytes(path.read_bytes())
        manifest['user_assets'].append({'name':kind,'path':str(path),'status':'PRESENT_VERIFIED' if path.exists() else 'MISSING_USER_ACCESS',
                                        'sha256':sha(path) if path.exists() else None})
    for base,pattern in [(OLD,'formal_gfp_seed*'),(OLD,'formal_fql_seed*'),(FLOQ,'formal_seed*')]:
        for run in sorted((base/'runs').glob(pattern)):
            for ck in sorted(run.rglob('params_*.pkl')):
                manifest['user_assets'].append({'name':run.name+'/'+ck.name,'status':'PRESENT_VERIFIED',
                    'path':str(ck),'bytes':ck.stat().st_size,'sha256':sha(ck), 'permission':'read_only_existing_asset'})
            for name in ['actual_runtime.json','status.json']:
                p=run/name
                if p.exists():manifest['user_assets'].append({'name':run.name+'/'+name,'path':str(p),
                    'status':'PRESENT_VERIFIED','bytes':p.stat().st_size,'sha256':sha(p)})
    manifest['user_assets'].append({'name':'DFQL trained weights','status':'NOT_YET_TRAINED',
                                    'reason':'User has not confirmed trained DFQL assets; not required for zero-train init.'})
    for task in TASKS:
        base=task.split('-singletask')[0]+'-v0'
        for suffix in ['.npz','-val.npz']:
            dest=ROOT/'data'/(base+suffix)
            old=OLD/'home/.ogbench/data'/dest.name
            if old.exists() and not dest.exists():dest.symlink_to(old)
        print('OFFICIAL_LOAD',task,flush=True)
        env,train,val=ogbench.make_env_and_datasets(task,dataset_dir=str(ROOT/'data'))
        # Both pinned host wrappers do the same final action clipping at 1e-5.
        train['actions']=np.clip(train['actions'],-1+1e-5,1-1e-5)
        val['actions']=np.clip(val['actions'],-1+1e-5,1-1e-5)
        rng=np.random.default_rng(14014)
        indices=rng.integers(0,len(train['actions']),size=256)
        batch={k:v[indices] for k,v in train.items()}
        np.savez(ROOT/'data'/f'{task}.batch.npz',**batch)
        files=[{'path':str(ROOT/'data'/(base+suffix)), 'resolved_path':str((ROOT/'data'/(base+suffix)).resolve()),
                'bytes':(ROOT/'data'/(base+suffix)).stat().st_size,'sha256':sha(ROOT/'data'/(base+suffix))}
               for suffix in ['.npz','-val.npz']]
        d={'task':task,'status':'PRESENT_VERIFIED' if base.startswith('antmaze') else 'FETCHED_VERIFIED',
           'ogbench':md.version('ogbench'),'env_spec':env.spec.id,'episode_limit':env.spec.max_episode_steps,
           'files':files,'train_transitions':len(train['actions']),'val_transitions':len(val['actions']),
           'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'min':float(v.min()),'max':float(v.max())} for k,v in train.items()},
           'reward_range':[float(train['rewards'].min()),float(train['rewards'].max())],
           'action_low':env.action_space.low.tolist(),'action_high':env.action_space.high.tolist(),
           'batch_sha256':sha(ROOT/'data'/f'{task}.batch.npz'),'batch_indices':indices.tolist(),
           'processing':'OGBench1.1.0 singletask task-specific relabel_dataset; regular next-state dataset; host action clip1e-5; no further reward/mask change'}
        manifest['datasets'].append(d);env.close()
        dump(ROOT/'assets_manifest/asset_manifest.json',manifest)
        print('DATA_READY',task,d['train_transitions'],d['reward_range'],flush=True)
        del train,val
    import yaml
    gfp=ROOT/'upstream_snapshots/gfp'
    gbase=yaml.safe_load((gfp/'config/main.yaml').read_text())['agent']
    gbase.update(yaml.safe_load((gfp/'config/agent/gfp.yaml').read_text()));gbase['log_metrics']=True
    with open(gfp/'hyperparams/gfp_params.csv') as f: rows=list(csv.DictReader(f))
    flow=source_config(ROOT/'upstream_snapshots/floq/agents/floq.py')
    allowed=['flow_num_ensembles','distill_num_ensembles','noise_samples','noise_coverage','critic_flow_steps',
             'train_at_zero_only','embed_time','time_embed_dim','use_prob_embed','num_bins','sigma','reward_offset',
             'block_width','block_depth','use_resnets','num_transformations','activate_before_layernorm']
    flow={k:flow[k] for k in allowed}
    dump(ROOT/'configs/floq_inherited_defaults.json',flow)
    db=source_config(ROOT/'upstream_snapshots/dfql/agents/dfql.py')
    guard=dict(execution_stage='prepare_and_verify',training_authorized=False,optimizer_steps_allowed=0,online_learning_allowed=False)
    cfg_index=[]
    for task in TASKS:
        data=next(d for d in manifest['datasets'] if d['task']==task)
        hosts=['dfql'] if task.startswith('puzzle') else ['gfp','dfql'] if not task.startswith('antmaze') else ['gfp']
        for host in hosts:
            if host=='gfp':
                cfg=dict(gbase)
                name=task.replace('-','_'); matches=[r for r in rows if r['benchmark']=='OGBench' and r['env_name'] in name]
                row=max(matches,key=lambda r:len(r['env_name']))
                for k,v in row.items():
                    if k in cfg:cfg[k]=type(cfg[k])(v)
                if cfg['eta_temperature_online'] is None:cfg['eta_temperature_online']=cfg['eta_temperature']
                provenance={'host_parameters':'fixed Hydra main+agent defaults, longest CSV match, no CLI overrides',
                            'csv_row':row,'flow_parameters':'pinned floq get_config mechanically inherited except card depth',
                            'execution_guard':'card/user authorization; not official algorithm parameters'}
            else:
                cfg=dict(db)
                cfg.update(alpha=2 if task.startswith('cube') else 10 if task.startswith('humanoid') else 3.3,
                           discount=.995 if task.startswith('humanoid') else .99)
                provenance={'host_parameters':'pinned dfql get_config + fixed README task flags',
                            'flow_parameters':'pinned floq get_config mechanically inherited except card depth',
                            'execution_guard':'card/user authorization; not official algorithm parameters'}
            cfg.update(ob_dims=data['fields']['observations']['shape'][1:],action_dim=data['fields']['actions']['shape'][-1])
            original=dict(cfg,critic_type='scalar')
            combined=dict(cfg,critic_type='floq',**flow,r_min=data['reward_range'][0],r_max=data['reward_range'][1])
            combined['block_depth']=2 if task.startswith('cube') else 4
            combined.update(noise_min=combined['noise_coverage']*combined['r_min']/(1-combined['discount']),
                            noise_max=combined['noise_coverage']*combined['r_max']/(1-combined['discount']),
                            q_min=(combined['r_min']-combined['reward_offset'])/(1-combined['discount']),
                            q_max=(combined['r_max']+combined['reward_offset'])/(1-combined['discount']))
            prefix='G' if host=='gfp' else 'D'
            for variant,agent_cfg in [('0',original),('1',combined)]:
                out=dict(guard,host=host,task=task,seeds_prepared=[0,1],agent=agent_cfg,provenance=provenance)
                fn=f'{prefix}{variant}_{task}.json';dump(ROOT/'configs'/fn,out)
                cfg_index.append(fn)
            dump(ROOT/'configs'/f'{prefix}_diff_{task}.json',
                 {k:{'original':original.get(k),'combined':v} for k,v in combined.items() if k not in original or original[k]!=v})
    dump(ROOT/'configs/index.json',cfg_index)
    dump(ROOT/'assets_manifest/asset_manifest.json',manifest)
    subprocess.run([str(ROOT/'envs/gfp/bin/python'),str(ROOT/'scripts/resolve_gfp_configs.py')],check=True)
    dump(ROOT/'receipts/p0_complete.json',dict(status='ASSET_READY',timestamp=time.time(),optimizer_steps=0))
    print('P0_DONE',flush=True)

if __name__=='__main__':main()

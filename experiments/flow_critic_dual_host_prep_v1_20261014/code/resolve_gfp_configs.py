"""Resolve pinned GFP OmegaConf + actual official CSV matching, no training import."""
import copy,json,sys
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspaces/gfp_floq'))
import hyperparams.hyperparams as hp
# No command-line overrides were requested for these prepared configurations.
hp.get_agent_cli_args=lambda:([],False)
main=OmegaConf.load(ROOT/'upstream_snapshots/gfp/config/main.yaml')
agent=OmegaConf.merge(main.agent,OmegaConf.load(ROOT/'upstream_snapshots/gfp/config/agent/gfp.yaml'))
agent.log_metrics=True
flow=json.loads((ROOT/'configs/floq_inherited_defaults.json').read_text())
assets=json.loads((ROOT/'assets_manifest/asset_manifest.json').read_text())
for d in assets['datasets']:
 task=d['task']
 if task.startswith('puzzle'):continue
 cfg=OmegaConf.create({'agent':copy.deepcopy(agent),'offline_steps':main.offline_steps})
 hp.override_non_cli_args_with_default(cfg,'OGBench_Env_'+task.replace('-','_'))
 resolved=OmegaConf.to_container(cfg.agent,resolve=True)
 if resolved['eta_temperature_online'] is None:resolved['eta_temperature_online']=resolved['eta_temperature']
 resolved.update(ob_dims=d['fields']['observations']['shape'][1:],action_dim=d['fields']['actions']['shape'][-1])
 assert isinstance(resolved['lr'],float) and isinstance(resolved['eta_temperature'],float)
 original=dict(resolved,critic_type='scalar')
 combo=dict(resolved,critic_type='floq',**flow,r_min=d['reward_range'][0],r_max=d['reward_range'][1])
 combo['block_depth']=2 if task.startswith('cube') else 4
 combo.update(noise_min=combo['noise_coverage']*combo['r_min']/(1-combo['discount']),
              noise_max=combo['noise_coverage']*combo['r_max']/(1-combo['discount']),
              q_min=(combo['r_min']-combo['reward_offset'])/(1-combo['discount']),
              q_max=(combo['r_max']+combo['reward_offset'])/(1-combo['discount']))
 for number,params in [('0',original),('1',combo)]:
  path=ROOT/'configs'/f'G{number}_{task}.json'
  old=json.loads(path.read_text())
  old['agent']=params
  old['provenance']['host_parameters']='Pinned OmegaConf main+agent YAML, actual official override_non_cli_args_with_default; no CLI overrides'
  path.write_text(json.dumps(old,indent=2,sort_keys=True))
 (ROOT/'configs'/f'G_diff_{task}.json').write_text(json.dumps(
    {k:{'original':original.get(k),'combined':v} for k,v in combo.items() if k not in original or original[k]!=v},indent=2))
 print('GFP_CONFIG_RESOLVED',task,resolved['discount'],resolved['eta_temperature'],flush=True)

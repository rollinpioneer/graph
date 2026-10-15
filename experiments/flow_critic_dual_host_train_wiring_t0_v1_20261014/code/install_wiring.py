"""Install into the new isolated T0 workspaces; never edits preparation assets."""
from pathlib import Path
import ast,json,shutil
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'implementation/code'
defaults=json.loads((ROOT/'configs/floq_inherited_defaults.json').read_text())

def replace_once(s,old,new):
    if s.count(old)!=1:raise RuntimeError(f'Expected one match, found {s.count(old)}: {old[:70]}')
    return s.replace(old,new,1)

for host in ['gfp','dfql']:
    work=ROOT/'workspaces'/f'{host}_floq'
    shutil.copy2(CODE/'host_adapter.py',work/'agents/floq_adapter.py')
    shutil.copy2(CODE/'update_runtime.py',work/'agents/update_runtime.py')
    shutil.copy2(CODE/'training_wiring.py',work/'training_wiring.py')
    for target in (work/'floq_value_core').glob('*.py'):
        assert target.resolve().is_relative_to(work.resolve()) and not target.is_symlink()
        target.chmod(target.stat().st_mode | 0o200)
    shutil.copytree(CODE/'floq_value_core',work/'floq_value_core',dirs_exist_ok=True)
    p=work/'main.py';s=p.read_text()
    s=s.replace('from utils.flax_utils import restore_agent, save_agent',
        'from training_wiring import (construct_agent, read_execution_gate, dispatch_update, dispatch_fused,\n'
        '    save_state as save_agent, restore_state as restore_agent, resume_index)')
    if host=='gfp':
        pconfig=work/'config/main.yaml';pconfig.write_text(pconfig.read_text()+'\n# Controlled dispatcher; no run without an explicit approved gate.\nwiring_gate: null\n')
        pconfig=work/'config/agent/gfp.yaml'
        values=dict(critic_type='scalar',**defaults,training_authorized=False,optimizer_steps_allowed=0,online_learning_allowed=False)
        pconfig.write_text(pconfig.read_text()+'\n# New floq-critic option; host name stays gfp for CSV matching.\n'+'\n'.join(
            f'{k}: {json.dumps(v)}' for k,v in values.items())+'\n')
        s=s.replace('from omegaconf import DictConfig, OmegaConf','from omegaconf import DictConfig, OmegaConf, open_dict')
        marker='    if cfg.seed is None:'
        s=replace_once(s,marker,
            '    gate = read_execution_gate(cfg.wiring_gate, "gfp", cfg.env_name,\n'
            '        cfg.agent.get("critic_type", "scalar"), cfg.offline_steps, cfg.online_steps,\n'
            '        cfg.seed, cfg.eval_episodes)\n'
            '    if cfg.dataset_on_gpu and gate["stage"] == "T0":\n'
            '        raise RuntimeError("T0 protocol requires dataset_on_gpu=false")\n\n'+marker)
        old='''    agent_class = agents[cfg_agent["agent_name"]]
    agent = agent_class.create(
        cfg.seed,
        example_batch["observations"],
        example_batch["actions"],
        cfg_agent,
    )'''
        new='''    if cfg_agent["agent_name"] != "gfp":
        raise RuntimeError("This isolated dispatcher supports GFP G0/G1 only")
    agent = construct_agent("gfp", cfg.seed, example_batch["observations"],
        example_batch["actions"], cfg_agent, train_dataset, gate, cfg.save_dir)
    cfg_agent = dict(agent.config)
    with open_dict(cfg):
        cfg.agent = OmegaConf.create(cfg_agent)'''
        s=replace_once(s,old,new)
        s=replace_once(s,'range(1, off_steps + on_steps + 1)',
                       'range(resume_index(agent) + 1, off_steps + on_steps + 1)')
        s=s.replace('agent, update_info = agent.multi_sample_and_update(\n                cfg["n_steps_fused"], train_dataset_gpu, dataset_size, batch_size\n            )',
                    'agent, update_info = dispatch_fused(agent, cfg["n_steps_fused"],\n                train_dataset_gpu, dataset_size, batch_size, gate)')
        s=s.replace('agent, update_info = agent.update(batch)',
                    'agent, update_info = dispatch_update(agent, batch, gate, cfg["save_dir"])')
        s=replace_once(s,'if is_last_iters or (\n            cfg["eval_interval"] != 0',
                       'if cfg["eval_episodes"] > 0 and (is_last_iters or (\n            cfg["eval_interval"] != 0')
        s=replace_once(s,'cfg["eval_interval"] != 0 and i % cfg["eval_interval"] == 0\n        ):',
                       'cfg["eval_interval"] != 0 and i % cfg["eval_interval"] == 0\n        )):')
    else:
        s=replace_once(s,"flags.DEFINE_string('run_group', 'Debug', 'Run group.')",
            "flags.DEFINE_string('wiring_gate', None, 'Explicit approved execution gate JSON.')\n"
            "flags.DEFINE_string('wandb_mode', 'disabled', 'Explicit logger mode; disabled for T0.')\n"
            "flags.DEFINE_string('run_group', 'Debug', 'Run group.')")
        s=replace_once(s,'def main(_):\n    # Set up logger.',
            "def main(_):\n    gate = read_execution_gate(FLAGS.wiring_gate, 'dfql', FLAGS.env_name,\n"
            "        FLAGS.agent.get('critic_type', 'scalar'), FLAGS.offline_steps, FLAGS.online_steps,\n"
            "        FLAGS.seed, FLAGS.eval_episodes)\n"
            "    if gate['stage'] == 'T0' and FLAGS.wandb_mode != 'disabled':\n"
            "        raise RuntimeError('T0 permits local logging only')\n    # Set up logger.")
        s=replace_once(s,"setup_wandb(project='dfql', group=FLAGS.run_group, name=exp_name)",
                       "setup_wandb(project='dfql', group=FLAGS.run_group, name=exp_name, mode=FLAGS.wandb_mode)")
        old='''    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )'''
        new='''    if config['agent_name'] != 'dfql':
        raise RuntimeError('This isolated dispatcher supports DFQL D0/D1 only')
    agent = construct_agent('dfql', FLAGS.seed, example_batch['observations'],
        example_batch['actions'], config.to_dict(), train_dataset, gate, FLAGS.save_dir)
    config = agent.config'''
        s=replace_once(s,old,new)
        s=replace_once(s,'range(1, FLAGS.offline_steps + FLAGS.online_steps + 1)',
                       'range(resume_index(agent) + 1, FLAGS.offline_steps + FLAGS.online_steps + 1)')
        s=s.replace('agent, update_info = agent.update(batch)',
                    'agent, update_info = dispatch_update(agent, batch, gate, FLAGS.save_dir)')
        source=work/'agents/dfql.py';ds=source.read_text()
        ds=replace_once(ds,'    return config',
            "    # Config-only registration; original host networks/losses are unchanged.\n"
            "    from pathlib import Path\n    import json\n"
            "    inherited = json.loads((Path(__file__).resolve().parents[3] / 'configs/floq_inherited_defaults.json').read_text())\n"
            "    config.update(inherited)\n"
            "    config.update(dict(critic_type='scalar', training_authorized=False, optimizer_steps_allowed=0, online_learning_allowed=False))\n"
            "    return config")
        source.write_text(ds)
    ast.parse(s);p.write_text(s)
    print('INSTALLED_CONTROLLED_NATIVE_MAIN',host,flush=True)

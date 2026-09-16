from __future__ import annotations
import importlib.util,sys
from pathlib import Path
from .io_utils import ROOT,load_json,sha256

def _module(name,path):
    path=Path(path).resolve()
    if name in sys.modules:
        old=sys.modules[name]
        if Path(old.__file__).resolve()!=path:raise RuntimeError(f'module collision: {name}')
        return old
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module

def load_sources(repo,v6_tools):
    """Loads exact historical bytes. No reward reimplementation fallback."""
    lock=load_json(ROOT/'contracts/source_lock.json')
    repo=Path(repo).resolve();tools=Path(v6_tools).resolve()
    env_path=repo/lock['p2a_env']['repo_path']
    paths={'p2a_env':env_path}
    expected={'p2a_env':lock['p2a_env']['sha256']}
    for name,h in lock['v6_files'].items():
        paths[name]=tools.parent/'contracts'/name if name.endswith('.json') else tools/name
        expected[name]=h
    for name,p in paths.items():
        if not p.is_file() or sha256(p)!=expected[name]:
            raise RuntimeError(f'frozen source missing or changed: {p}')
    # reward_v6 imports its frozen sibling by this name.
    _module('reeval_core',paths['reeval_core.py'])
    v6=_module('reward_v6',paths['reward_v6.py'])
    env=_module('_p2b_frozen_skill_env',env_path)
    sources={n:{'path':str(p),'sha256':sha256(p)} for n,p in paths.items()}
    return env,v6.CapabilityPotential,sources

"""Offline validations are torch/API independent; execution fails closed on bindings."""
import argparse,json,subprocess,sys
from pathlib import Path
from .common import BindingError,ContractError,canonical
from .runtime import require_runtime,require_cache_configuration

def read(path):
    path=Path(path)
    if path.suffix=='.json':return json.loads(path.read_text())
    import yaml
    return yaml.safe_load(path.read_text())

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd())
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('validate-manifests','validate-contracts','build-graph-fixture','train','evaluate','generate-cache'):sub.add_parser(name)
    cache=sub.add_parser('validate-relation-cache');cache.add_argument('--cache-directory',type=Path)
    t=sub.add_parser('run-unit-tests');t.add_argument('--scope',choices=['pure','full'],required=True)
    a=p.parse_args(argv);root=a.root.resolve()
    try:
        if a.command=='validate-manifests':
            from .validation import validate_repository
            result=validate_repository(root);print(canonical(result));return 0 if result['passed'] else 2
        if a.command=='validate-contracts':
            from .contracts import from_dict,Registry
            doc=read(root/'configs/contracts/skills.yaml');reg=Registry(doc['predicate_types'])
            for c in doc['contracts']:reg.register(from_dict(c))
            from jsonschema import Draft202012Validator
            validator=Draft202012Validator(read(root/'schemas/skill_contract.schema.json'))
            for c in doc['contracts']:validator.validate(c)
            print(canonical({'status':'PASS_TEMPLATE_SCHEMA_ONLY','count':len(doc['contracts']),'runtime_bound':False}));return 0
        if a.command=='build-graph-fixture':
            from .fixtures import build_fixture
            f=build_fixture(root/'tests/fixtures');print(canonical({'synthetic_unit_fixture':True,'paper_performance_eligible':False,'template':f['template']}));return 0
        if a.command=='validate-relation-cache':
            from .fixtures import build_fixture
            from .vlm import validate_relations,load_cache
            if a.cache_directory:
                manifest,edges=load_cache(a.cache_directory)
                print(canonical({'cache_identity_valid':True,'edges':len(edges),'semantic_binding_validated':False,'note':'Supply registered task template/ID/effect context to validate_relations for semantic validation'}));return 0
            f=build_fixture(root/'tests/fixtures');result=validate_relations(f['relations'],f['template'])
            print(canonical({'synthetic_unit_fixture':True,'paper_performance_eligible':False,'result':result}));return 0
        if a.command=='run-unit-tests':
            args=[sys.executable,'-m','pytest',str(root/'tests'),'-m','pure' if a.scope=='pure' else 'pure or torch_runtime','-ra']
            return subprocess.call(args,cwd=root)
        if a.command in ('train','evaluate'):
            require_runtime(read(root/'experiments/manifests/runtime_manifest.yaml'))
            # The complete application configuration is also required before any driver loads.
            config=root/'configs/run_resolved.json'
            if not config.exists():raise BindingError('MUST_BIND: configs/run_resolved.json (model dimensions, task cases, budget, trusted runtime factory)')
            from .execution import execute
            return execute(a.command,root,read(config))
        if a.command=='generate-cache':require_cache_configuration(read(root/'experiments/manifests/vlm_manifest.yaml'))
    except (BindingError,ContractError,ValueError,FileNotFoundError) as error:
        print(canonical({'status':'BLOCKED','error':str(error)}));return 2

if __name__=='__main__':raise SystemExit(main())

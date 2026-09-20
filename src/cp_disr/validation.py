"""Static repository checks; do not import neural modules or contact an API."""
import ast,json
from pathlib import Path

def validate_repository(root):
    import yaml
    from jsonschema import Draft202012Validator
    from .contracts import Registry,from_dict
    root=Path(root);errors=[];parsed=0;graph={}
    paths=[]
    for folder in ['configs','schemas','tests/fixtures','experiments/manifests','experiments/configs','experiments/schemas','experiments/stage_status','experiments/sources','experiments/templates']:
        paths.extend(p for p in (root/folder).rglob('*') if p.suffix in ('.yaml','.yml','.json'))
    for p in paths:
        try:yaml.safe_load(p.read_text()) if p.suffix!='.json' else json.loads(p.read_text());parsed+=1
        except Exception as e:errors.append({'path':str(p),'error':str(e)})
    for p in (root/'schemas').glob('*.json'):
        try:Draft202012Validator.check_schema(json.loads(p.read_text()))
        except Exception as e:errors.append({'path':str(p),'error':str(e)})
    try:
        schema=json.loads((root/'schemas/skill_contract.schema.json').read_text());d=yaml.safe_load((root/'configs/contracts/skills.yaml').read_text());reg=Registry(d['predicate_types'])
        for c in d['contracts']:Draft202012Validator(schema).validate(c);reg.register(from_dict(c))
        fs=Draft202012Validator(json.loads((root/'schemas/synthetic_fixture.schema.json').read_text()))
        for p in (root/'tests/fixtures').glob('*.json'):fs.validate(json.loads(p.read_text()))
    except Exception as e:errors.append({'schema_validation':str(e)})
    forbidden='SOFT_'+'ORDER'
    for p in (root/'src/cp_disr').glob('*.py'):
        try:
            text=p.read_text();tree=ast.parse(text)
            if forbidden in text:errors.append({'forbidden_production_relation':str(p)})
            imports=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom):
                    imports.append(('.'*n.level)+(n.module or ''))
                    if n.level and n.module and not (root/'src/cp_disr'/(n.module.split('.')[0]+'.py')).exists():errors.append({'missing_local_import':n.module,'file':str(p)})
                elif isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
            graph[str(p.relative_to(root))]=sorted(set(imports))
        except Exception as e:errors.append({'syntax':str(e),'file':str(p)})
    for p in (root/'configs').rglob('*.yaml'):
        if 'relation_'+'rewrite' in p.read_text():errors.append({'forbidden_train_config':str(p)})
    entries=yaml.safe_load((root/'experiments/manifests/entrypoints.yaml').read_text())
    for name,entry in entries.get('commands',{}).items():
        if not (root/entry['source_path']).is_file():errors.append({'entrypoint_missing':name})
    # Explicit reference policy: unresolved runtime IDs remain placeholders, not paths.
    tasks=yaml.safe_load((root/'experiments/manifests/task_manifest.yaml').read_text())
    for name,ref in tasks.get('template_refs',{}).items():
        if not (root/'experiments/manifests'/ref).is_file():errors.append({'task_template_ref_missing':name,'ref':ref})
    contracts=yaml.safe_load((root/'experiments/manifests/skill_contracts.yaml').read_text())
    for key in ('schema_ref','production_registry_ref'):
        if key in contracts and not (root/'experiments/manifests'/contracts[key]).is_file():errors.append({'contract_reference_missing':contracts[key]})
    task_validator=Draft202012Validator(json.loads((root/'schemas/task_template.schema.json').read_text()))
    for p in (root/'configs/tasks').glob('*.yaml'):
        for error in task_validator.iter_errors(yaml.safe_load(p.read_text())):errors.append({'task_schema':str(p),'error':error.message})
    return {'passed':not errors,'parsed_files':parsed,'errors':errors,'import_free_dependency_graph':graph,'api_called':False,'torch_imported':False}

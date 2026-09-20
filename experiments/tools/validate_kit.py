#!/usr/bin/env python3
"""Validate the handbook package, not CP-DISR production tests or performance."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[dict[str, object]] = []
    def check(name: str, condition: bool, detail: object = '') -> None:
        checks.append({'check': name, 'pass': bool(condition), 'detail': detail})
    try:
        import yaml
        import jsonschema
    except ImportError as error:
        print(f'Install PyYAML and jsonschema before validating the kit: {error}', file=sys.stderr)
        return 2
    try:
        registry = json.loads((root/'configs/stage_registry.json').read_text(encoding='utf-8'))
        expected = ['0A','0B','0C','1A','1B','2A','2B','2C','3A','3B','3C','4A','4B','5A','6A','6B','6C']
        check('17 unique ordered Stages', [s['id'] for s in registry] == expected)
        fields = ['id','part','name','type','purpose','paper','prerequisites','inputs','frozen','tunable',
                  'run_matrix','training_budget','evaluation','logging','required_files','plots','tables',
                  'criteria','failure_handling','rerun_allowed','tuning_allowed','writing_block','next_stage','execution_steps']
        for stage in registry:
            sid = stage['id']
            check(f'{sid}: complete execution specification', all(f in stage and stage[f] != '' for f in fields))
            check(f'{sid}: 8–15 concrete execution steps', 8 <= len(stage['execution_steps']) <= 15,
                  len(stage['execution_steps']))
            card = root/'stage_cards'/f'stage_{sid.lower()}.md'
            check(f'{sid}: offline self-contained card', card.is_file() and card.stat().st_size > 30000)
            stage_yaml = yaml.safe_load((root/'configs/stages'/f'stage_{sid.lower()}.yaml').read_text(encoding='utf-8'))
            check(f'{sid}: JSON/YAML registry consistency', stage_yaml == stage)
        schema = json.loads((root/'schemas/stage_status.schema.json').read_text(encoding='utf-8'))
        for sid in expected:
            data = json.loads((root/'stage_status'/f'stage_{sid.lower()}.json').read_text(encoding='utf-8'))
            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(data))
            check(f'{sid}: valid status schema', not errors, [e.message for e in errors])
        # Parse every structured file, including copied source interfaces.
        for path in sorted(root.rglob('*')):
            if not path.is_file() or 'preflight_history' in path.parts:
                continue
            if path.suffix in {'.yaml','.yml'}:
                yaml.safe_load(path.read_text(encoding='utf-8'))
            elif path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8'))
        check('All YAML/JSON files parse', True)
        sources = json.loads((root/'manifests/source_manifest.json').read_text(encoding='utf-8'))
        for item in sources:
            digest = hashlib.sha256((root/item['relative_path']).read_bytes()).hexdigest()
            check(f"{item['id']}: source preserved byte-for-byte", digest == item['sha256'])
        jobs = json.loads((root/'configs/planned_jobs.json').read_text(encoding='utf-8'))
        new = [j for j in jobs if j['new_training_job']]
        first = [j for j in new if j['stage'] in {'1A','2A','3A'}]
        counter = dict(collections.Counter(j['stage'] for j in new))
        check('Default new job counts', counter == {'1A':2,'2A':24,'2B':36,'3A':3,'3B':3,'3C':3},counter)
        check('First batch 29 / complete route 71 new jobs',len(first)==29 and len(new)==71)
        check('9 reused Full references, no duplicate new job IDs',len(jobs)-len(new)==9 and len({(j['stage'],j['task'],j['method'],j['seed']) for j in new})==len(new))
        check('First batch transition cap 1,802,240',sum(j['skill_transition_cap'] for j in first)==1802240)
        check('Complete route transition cap 7,307,264',sum(j['skill_transition_cap'] for j in new)==7307264)
        check('Planned jobs are not fabricated results',all(j['status']=='PLANNED_NOT_RUN' for j in jobs))
        manifest = yaml.safe_load((root/'manifests/experiment_manifest_v0.yaml').read_text(encoding='utf-8'))
        check('Method v2.1 / pure DP / no trainer claim', manifest['method']['version']=='v2.1' and manifest['method']['no_static_prior'] and not manifest['rl']['trainer_implemented_in_this_kit'])
        check('Only two soft relation types',manifest['graph']['soft_relations']==['SOFT_SUPPORTS','SOFT_RELEVANT_TO_GOAL'])
        check('80/20 and no training artificial edges',manifest['prior_training']['original']==.8 and manifest['prior_training']['absent']==.2 and not manifest['prior_training']['edge_corruption_enabled'])
        doc = (root/'docs/CP_DISR_v2.1_Minimal_Experimental_Execution_Plan.md').read_text(encoding='utf-8')
        check('Handbook includes all17 core Stage sections',all(f'## Stage {sid} —' in doc for sid in expected))
        check('T01–T25 detailed tests included',all(re.search(rf'\bT{i:02d}\b',doc) for i in range(1,26)))
        check('Final five requested answers included',all(f'## {letter}.' in doc for letter in 'ABCDE'))
        check('Balanced Markdown fenced blocks',len(re.findall(r'^```',doc,re.M))%2==0)
        check('No transport citation tokens accidentally in portable document','' not in doc)
        check('Preflight distinction explicit','local_inventory_only' in doc and 'NOT_STARTED' in doc)
        overall = all(item['pass'] for item in checks)
        result = {'scope':'DOCUMENT_AND_PACKAGE_VALIDATION_ONLY', 'status':'PASS' if overall else 'NEEDS_FIX',
                  'production_unit_tests_executed':False, 'rl_jobs_executed':0,
                  'checks_total':len(checks), 'checks_passed':sum(bool(i['pass']) for i in checks), 'checks':checks}
        (root/'docs/kit_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        text = '# Handbook and kit validation\n\n'
        text += f"**{result['status']} — {result['checks_passed']}/{len(checks)} document/package checks.**\n\n"
        text += 'This does not execute T01–T25 production tests, VLM calls, RL training, or robot actions.\n\n'
        text += '| Check | Result | Detail |\n|---|---|---|\n'
        for item in checks:
            detail = str(item['detail']).replace('|','/').replace('\n',' ')
            text += f"| {item['check']} | {'PASS' if item['pass'] else 'FAIL'} | {detail} |\n"
        (root/'docs/kit_validation.md').write_text(text,encoding='utf-8')
        print(f"{result['status']}: {result['checks_passed']}/{len(checks)} package checks; production tests not run.")
        for item in checks:
            if not item['pass']:
                print(item)
        return 0 if overall else 1
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f'Kit validation error: {error}',file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())

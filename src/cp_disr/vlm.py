"""Local-only relation validation and content-addressed cache. No API client."""
import json
from dataclasses import dataclass
from pathlib import Path
from .common import ContractError,digest,canonical

ALLOWED_TYPES=('SOFT_SUPPORTS','SOFT_RELEVANT_TO_GOAL')
RELATION_FIELDS={'relation_id','type','source_ref','target_ref','effect_fact_ref'}
CACHE_FIELDS=('split','task_definition_hash','initial_RGB_content_hash','preprocessing_hash','object_binding_hash','allowed_ID_hash','contract_version','predicate_version','model_snapshot','sdk_api_version','region','endpoint','prompt_hash','fewshot_hash','schema_hash','decoding_config')

@dataclass(frozen=True)
class ValidatedRelations:
    final_edges:tuple
    accepted:tuple
    rejected:tuple
    dedup_log:tuple

def validate_relations(payload,template,goal_aliases=None):
    if set(payload)!={'schema_version','relations'} or payload['schema_version']!='m1_soft_relations_v2':raise ContractError('Invalid relation envelope')
    if not isinstance(payload['relations'],list) or len(payload['relations'])>8:raise ContractError('Relation list limit/type')
    nodes={n.id:n for n in template.nodes}; goals={g.fact_id for g in template.goals}; edges=set(template.edges)
    aliases=goal_aliases or {}
    if not set(aliases.values())<=goals:raise ContractError('Goal aliases must reference registered signed goals')
    accepted=[];rejected=[];dedup=[];seen_ids=set()
    for original in payload['relations']:
        row=dict(original)
        reason=None
        if set(row)!=RELATION_FIELDS or any(not isinstance(v,str) or not v for v in row.values()):reason='SCHEMA_FIELDS'
        elif row['type'] not in ALLOWED_TYPES:reason='RELATION_TYPE'
        elif row['relation_id'] in seen_ids:reason='DUPLICATE_RELATION_ID'
        else:
            seen_ids.add(row['relation_id']); row['target_ref']=aliases.get(row['target_ref'],row['target_ref'])
            s,t,r,p=(row[k] for k in ['source_ref','target_ref','type','effect_fact_ref'])
            if s not in nodes or nodes[s].kind!='ACTION':reason='SOURCE_TYPE_OR_ID'
            elif t not in nodes:reason='TARGET_ID'
            elif r=='SOFT_SUPPORTS' and (nodes[t].kind!='ACTION' or s==t):reason='TARGET_TYPE_OR_SELF'
            elif r=='SOFT_RELEVANT_TO_GOAL' and t not in goals:reason='TARGET_NOT_GOAL'
            elif (s,p,'ADD') not in edges and (s,p,'DEL') not in edges:reason='UNREGISTERED_EFFECT_ANCHOR'
            elif r=='SOFT_SUPPORTS' and (((s,p,'ADD') in edges and (p,t,'PRE_POS') in edges) or ((s,p,'DEL') in edges and (p,t,'PRE_NEG') in edges)):reason='CONTRACT_REDUNDANCY'
            elif r=='SOFT_RELEVANT_TO_GOAL' and p==t:reason='CONTRACT_REDUNDANCY'
        if reason:rejected.append((original,reason))
        else:accepted.append(row)
    kept={}
    for row in sorted(accepted,key=lambda x:(x['type'],x['source_ref'],x['target_ref'],x['effect_fact_ref'],x['relation_id'])):
        key=(row['source_ref'],row['target_ref'],row['type'])
        if key in kept:dedup.append((row,'EXACT_EDGE_DUPLICATE'))
        else:kept[key]=row
    return ValidatedRelations(tuple(sorted(kept)),tuple(kept.values()),tuple(rejected),tuple(dedup))

def cache_key(manifest):
    missing=set(CACHE_FIELDS)-set(manifest)
    if missing:raise ContractError('Missing cache key fields: '+','.join(sorted(missing)))
    if any(manifest[k] is None or (isinstance(manifest[k],str) and manifest[k].startswith('MUST_')) for k in CACHE_FIELDS):raise ContractError('Unbound cache identity')
    return digest({k:manifest[k] for k in CACHE_FIELDS})

def save_cache(root,manifest,raw_response,validated):
    key=cache_key(manifest); split=manifest['split']
    if split not in ('train','dev','test','test_id','generalization'):raise ContractError('Unsupported cache split')
    path=Path(root)/split/key; path.mkdir(parents=True,exist_ok=False)
    files={'manifest.json':{**manifest,'cache_key':key},'raw_response_attempt_0.json':raw_response,'parsed_relations.json':validated.accepted,'rejected_relations.json':validated.rejected,'dedup_log.json':validated.dedup_log,'final_edges.json':validated.final_edges}
    for name,value in files.items():(path/name).write_text(canonical(value)+'\n')
    return path

def load_cache(path):
    path=Path(path); manifest=json.loads((path/'manifest.json').read_text())
    if cache_key(manifest)!=manifest.get('cache_key') or path.name!=manifest['cache_key']:raise ContractError('Cache identity mismatch')
    return manifest,json.loads((path/'final_edges.json').read_text())

"""One production parser/audit/cache path for offline tests and frozen requests."""
import copy,json,hashlib,time
from pathlib import Path
from jsonschema import Draft202012Validator
from .common import ContractError,canonical
from .vlm import validate_relations,cache_key
from .vlm_provider import redact,validate_payload

FORMAT_FEEDBACK='Return only one JSON object conforming to the supplied m1_soft_relations_v2 schema. Do not add prose or markdown.'

def process_response(text,template,schema,forbidden_edges=()):
    """Invalid semantics are isolated, never re-queried. No graph/fact writes."""
    payload=json.loads(text)
    if not isinstance(payload,dict):raise ValueError('JSON object required')
    validator=Draft202012Validator(schema)
    schema_errors=[{'path':list(e.absolute_path),'validator':e.validator} for e in validator.iter_errors(payload)]
    envelope=set(payload)=={'schema_version','relations'} and payload.get('schema_version')=='m1_soft_relations_v2' and isinstance(payload.get('relations'),list) and len(payload['relations'])<=8
    if not envelope:
        return {'parsed':payload,'accepted':[],'rejected':[{'relation':payload,'reason':'SCHEMA_ENVELOPE'}],'dedup':[],'final_edges':[],'schema_errors':schema_errors,'relation_checks':[],'status':'PROCESSED_INVALID_SCHEMA'}
    nodes={n.id:n for n in template.nodes};goals={g.fact_id for g in template.goals};edges=set(template.edges)
    rows=[];valid=[];invalid=[];item_schema=schema['properties']['relations']['items'];item_validator=Draft202012Validator(item_schema)
    for index,row in enumerate(payload['relations']):
        schema_ok=not list(item_validator.iter_errors(row));s=row.get('source_ref') if isinstance(row,dict) else None;t=row.get('target_ref') if isinstance(row,dict) else None;p=row.get('effect_fact_ref') if isinstance(row,dict) else None;typ=row.get('type') if isinstance(row,dict) else None
        id_ok=all(isinstance(v,str) and v in nodes for v in (s,t,p))
        endpoint_ok=id_ok and nodes[s].kind=='ACTION' and ((typ=='SOFT_SUPPORTS' and nodes[t].kind=='ACTION' and s!=t) or (typ=='SOFT_RELEVANT_TO_GOAL' and t in goals))
        effect_ok=id_ok and nodes[p].kind=='PROPOSITION' and ((s,p,'ADD') in edges or (s,p,'DEL') in edges)
        rows.append({'index':index,'schema_valid':schema_ok,'ID_valid':id_ok,'endpoint_type_valid':endpoint_ok,'effect_fact_ref_valid':effect_ok})
        if schema_ok:valid.append(row)
        else:invalid.append({'relation':row,'reason':'RELATION_SCHEMA'})
    checked=validate_relations({'schema_version':'m1_soft_relations_v2','relations':valid},template)
    prohibited={tuple(x) for x in forbidden_edges};accepted=[];conflicts=[]
    for row in checked.accepted:
        if (row['source_ref'],row['target_ref'],row['type']) in prohibited:conflicts.append({'relation':row,'reason':'EXPLICIT_TASK_PROHIBITION'})
        else:accepted.append(row)
    final=sorted({(x['source_ref'],x['target_ref'],x['type']) for x in accepted})
    rejected=invalid+[{'relation':row,'reason':reason} for row,reason in checked.rejected]+conflicts
    return {'parsed':payload,'accepted':accepted,'rejected':rejected,'dedup':[{'relation':row,'reason':reason} for row,reason in checked.dedup_log],
      'final_edges':final,'schema_errors':schema_errors,'relation_checks':rows,'status':'SUCCESS','empty_reason':('RAW_EMPTY' if not payload['relations'] else 'ALL_ISOLATED_OR_REDUNDANT') if not final else None}

def response_text(result):
    raw=result['raw'];choices=raw.get('output',{}).get('choices',[])
    if len(choices)!=1:raise ValueError('Exactly one response choice required')
    choice=choices[0]
    if choice.get('finish_reason') in ('length','max_tokens'):raise ValueError('TRUNCATED_JSON')
    content=choice.get('message',{}).get('content')
    if not isinstance(content,list) or any(set(x)!={'text'} for x in content):raise ValueError('Text-only assistant response required')
    return ''.join(x['text'] for x in content)

def request_and_process(provider,payload,template,schema,forbidden_edges=()):
    validate_payload(payload);attempts=[];processing=None
    for index in range(2):
        request=copy.deepcopy(payload)
        if index:request['messages'].append({'role':'user','content':[{'text':FORMAT_FEEDBACK}]})
        result=provider.send(request);result=redact(result)
        attempt={'attempt':index,'request_payload_redacted':redact(request),'response':result};attempts.append(attempt)
        if result['error_type']!='OK':
            attempt['processing_error']=result['error_type']
            if index==0 and result['error_type'] in ('TIMEOUT','TRANSPORT'):continue
            break
        try:processing=process_response(response_text(result),template,schema,forbidden_edges)
        except (ValueError,TypeError,KeyError) as error:
            attempt['processing_error']='JSON_FORMAT';attempt['exception_type']=type(error).__name__
            if index==0:continue
            break
        attempt['processing_status']=processing['status']
        # Empty, redundant, invalid effects and semantic issues NEVER trigger retries.
        break
    return {'attempts':attempts,'processing':processing,'status':processing['status'] if processing else 'API_ERROR' if attempts[-1]['response']['error_type']!='OK' else 'FORMAT_ERROR'}

def write_audit_cache(root,manifest,prompt,input_refs,execution):
    """Exclusive creation, completion marker and content hashes; no overwrite path."""
    key=cache_key(manifest)
    split=manifest['split']
    if split not in ('dev','train','test'):raise ContractError('Cache split must be train, dev or test')
    path=Path(root)/split/key;path.mkdir(parents=True,exist_ok=False)
    (path/'INCOMPLETE').write_text('Cache cannot be consumed until COMPLETE exists.\n')
    processing=execution.get('processing')
    files={'manifest.json':{**manifest,'cache_key':key,'processing_status':execution['status'],'synthetic_unit_fixture':bool(manifest.get('synthetic_unit_fixture',False))},
      'input_refs.json':input_refs,'request_payload_redacted.json':[a['request_payload_redacted'] for a in execution['attempts']],
      'raw_response.json':[a['response'] for a in execution['attempts']],
      'parsed_relations.json':processing['parsed'] if processing else None,
      'accepted_relations.json':processing['accepted'] if processing else [],
      'rejected_relations.json':processing['rejected'] if processing else [],
      'processing_log.json':{'status':execution['status'],'attempts':execution['attempts'],'validation':processing,'api_failure_is_empty_prior':False},
      'final_edges.json':processing['final_edges'] if processing else None}
    for name,value in files.items():(path/name).write_text(canonical(redact(value))+'\n')
    (path/'prompt.txt').write_text(redact(prompt))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.name!='INCOMPLETE'}
    (path/'content_hashes.json').write_text(canonical(hashes)+'\n');(path/'COMPLETE').write_text('complete\n');(path/'INCOMPLETE').unlink()
    for p in path.iterdir():p.chmod(0o444)
    return path

def verify_audit_cache(path):
    path=Path(path)
    if not (path/'COMPLETE').is_file() or (path/'INCOMPLETE').exists():raise ContractError('Incomplete cache')
    hashes=json.loads((path/'content_hashes.json').read_text())
    required={'manifest.json','prompt.txt','input_refs.json','request_payload_redacted.json','raw_response.json','parsed_relations.json','accepted_relations.json','rejected_relations.json','processing_log.json','final_edges.json'}
    if set(hashes)!=required:raise ContractError('Incomplete cache content inventory')
    for name,h in hashes.items():
        if Path(name).name!=name or hashlib.sha256((path/name).read_bytes()).hexdigest()!=h:raise ContractError('Cache content changed: '+name)
    manifest=json.loads((path/'manifest.json').read_text())
    if cache_key(manifest)!=path.name:raise ContractError('Cache identity changed')
    if manifest['processing_status']!='SUCCESS':raise ContractError('Failed request is not EMPTY_PRIOR')
    return manifest,json.loads((path/'final_edges.json').read_text())

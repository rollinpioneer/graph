"""Fail-closed formal Stage 0C input binding and 24-scene production runner."""
import base64,json,hashlib,os,time
from pathlib import Path
from .common import BindingError,ContractError,canonical,digest,unresolved
from .contracts import Registry,from_dict
from .graph import Goal,build_template
from .vlm import cache_key
from .vlm_provider import ProviderConfig,DashScopeProvider,validate_payload
from .vlm_cache_pipeline import process_response,request_and_process,write_audit_cache,verify_audit_cache

TASKS=('D0','T_A','T_C')
PREPROCESSING={'version':'raw-rgb-inline-v1','resize':False,'crop':False,'content_bytes':'original'}
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def bound_file(root,ref,expected):
    path=Path(ref);path=path if path.is_absolute() else Path(root)/path
    if not path.is_file() or file_hash(path)!=expected:raise BindingError('Missing/hash-mismatched input: '+str(path))
    return path

def prepare_scene(root,scene,fewshot=False):
    if scene.get('synthetic_unit_fixture') is not False:raise BindingError('Formal input must explicitly be non-synthetic')
    if scene.get('split')!=('independent_dev' if fewshot else 'dev'):raise BindingError('Only approved dev split; no train/test reuse')
    needed=['scene_id','task_id','image_ref','image_sha256','provenance','permission','preprocessing','object_table','signed_goals','initial_facts','contracts_ref','contracts_sha256','task_definition_ref','task_definition_sha256','asset_binding_ref','asset_binding_sha256','allowed_action_ids','allowed_proposition_ids','predicate_version']
    if any(k not in scene for k in needed) or unresolved({k:scene.get(k) for k in needed}):raise BindingError('Scene input fields MUST_BIND')
    if scene['task_id'] not in TASKS:raise BindingError('Unknown task')
    if scene['permission'].get('vlm_upload_allowed') is not True or not scene['permission'].get('evidence_ref'):raise BindingError('Image use/upload permission not certified')
    if scene['provenance'].get('capture_kind') not in ('real_camera','recorded_simulator_rgb') or scene['provenance'].get('initial_frame_verified') is not True or not all(scene['provenance'].get(k) for k in ['source_dataset','version','split_evidence_ref']):raise BindingError('Unverified initial RGB provenance/split')
    if scene['preprocessing']!=PREPROCESSING:raise BindingError('Unfrozen preprocessing')
    image=bound_file(root,scene['image_ref'],scene['image_sha256'])
    from PIL import Image
    with Image.open(image) as im:
        if im.mode!='RGB' or im.format not in ('PNG','JPEG','WEBP') or min(im.size)<32:raise BindingError('Require raw valid RGB image')
        mime={'PNG':'png','JPEG':'jpeg','WEBP':'webp'}[im.format];im.verify()
    task=read(bound_file(root,scene['task_definition_ref'],scene['task_definition_sha256']))
    binding=read(bound_file(root,scene['asset_binding_ref'],scene['asset_binding_sha256']))
    if task.get('task_id')!=scene['task_id'] or binding.get('scene_id')!=scene['scene_id'] or binding.get('task_id')!=scene['task_id']:raise BindingError('Task/asset identity mismatch')
    if binding.get('status')!='VERIFIED' or not all(binding.get(k) for k in ['reviewer','reviewed_at','structural_template_ref','source_assets','controller_refs','object_binding_evidence']):raise BindingError('Task-template/real-asset mapping not verified')
    if unresolved(binding):raise BindingError('Unresolved asset/controller binding')
    contract_doc=read(bound_file(root,scene['contracts_ref'],scene['contracts_sha256']))
    registry=Registry(contract_doc['predicate_types'])
    for c in contract_doc['contracts']:
        if unresolved(c):raise BindingError('Unbound Skill Contract in formal input')
        registry.register(from_dict(c))
    objects=scene['object_table']
    if not isinstance(objects,list) or any(set(x)!={'id','type','binding_status','visual_evidence_ref'} or x['binding_status']!='VERIFIED' or not x['visual_evidence_ref'] for x in objects):raise BindingError('Typed visual object binding incomplete')
    typed={x['id']:x['type'] for x in objects}
    if len(typed)!=len(objects):raise BindingError('Duplicate object ID')
    contracts=registry.ground(typed);template=build_template(contracts,tuple(Goal(**g) for g in scene['signed_goals']),registry.predicate_types,typed)
    actions=sorted(c.id for c in contracts);facts=sorted(n.id for n in template.nodes if n.kind=='PROPOSITION')
    if sorted(scene['allowed_action_ids'])!=actions or sorted(scene['allowed_proposition_ids'])!=facts:raise BindingError('Allowed IDs do not match grounded registry')
    if set(scene['initial_facts'])!=set(facts) or any(v not in ('TRUE','FALSE','UNKNOWN') for v in scene['initial_facts'].values()):raise BindingError('Initial facts mismatch or non-three-valued data')
    context={k:scene[k] for k in ['scene_id','task_id','object_table','signed_goals','initial_facts','allowed_action_ids','allowed_proposition_ids']}
    context['object_table']=sorted(context['object_table'],key=lambda x:x['id'])
    context['skill_contracts']=contract_doc;context['registered_effect_edges']=[e for e in template.edges if e[2] in ('ADD','DEL')]
    content=[{'image':'data:image/'+mime+';base64,'+base64.b64encode(image.read_bytes()).decode()},{'text':canonical(context)}]
    return {'scene':scene,'template':template,'content':content,'context':context,'binding':binding,'contract_doc':contract_doc}

def prepare_matrix(root,provider=None):
    root=Path(root);folder=root/'experiments/part_0_validation/stage_0c'
    import yaml
    m=yaml.safe_load((root/'experiments/manifests/vlm_manifest.yaml').read_text())
    fixed={'temperature':0,'max_output_tokens':2048,'max_relations':8,'thinking':False,'response_format':'json_object','max_retries':1,'semantic_retry':False,'allow_web_search':False,'tools_enabled':False,'relation_schema':'m1_soft_relations_v2'}
    if any(m.get(k)!=v for k,v in fixed.items()):raise BindingError('Frozen VLM manifest settings mismatch')
    config=ProviderConfig(model=m['model_snapshot'],region=m['region'],endpoint=m['base_http_api_url'],sdk_version=m['sdk_version'])
    provider=provider or DashScopeProvider(config)
    input_root=root/'experiments/stage_0c_inputs'
    scene_manifest=input_root/'formal_scene_manifest.json'
    few_manifest=input_root/'fewshots/few_shot_manifest.json'
    if not scene_manifest.exists(): scene_manifest=folder/'task_scene_manifest.json'
    if not few_manifest.exists(): few_manifest=folder/'few_shot_manifest.json'
    scenes_doc=read(scene_manifest);few_doc=read(few_manifest)
    scenes=scenes_doc.get('scenes',[]);few=few_doc.get('examples',[]);issues=[]
    if len(scenes)!=24:issues.append('task_scene_manifest: require exactly 24 bound dev scenes')
    for task in TASKS:
        if sorted(s.get('case_index',-1) for s in scenes if s.get('task_id')==task)!=list(range(8)):issues.append(task+': dev case_index 0..7 not bound')
    if len(few)!=3 or few_doc.get('status')!='FROZEN':issues.append('few_shot_manifest: three independent examples not frozen')
    if not provider.key_present():issues.append('DASHSCOPE_API_KEY: SDK environment not provisioned')
    if issues:raise BindingError('MUST_BIND: '+'; '.join(issues))
    bound=[prepare_scene(root,s) for s in scenes];examples=[prepare_scene(root,s,True) for s in few]
    identities=[s['scene_id'] for s in scenes+few];hashes=[s['image_sha256'] for s in scenes+few]
    if len(set(identities))!=27 or len(set(hashes))!=27:raise BindingError('Scene/fewshot duplicate identity or image; independence required')
    schema_path=root/'schemas/relation_schema.json';schema=read(schema_path)
    prompt_path=root/'experiments/sources/v2.1_interfaces/system_prompt.txt';prompt=prompt_path.read_text()
    if scenes_doc.get('prompt_sha256')!=file_hash(prompt_path) or few_doc.get('prompt_sha256')!=file_hash(prompt_path):raise BindingError('Prompt hash not frozen')
    for i,(ex,record) in enumerate(zip(examples,few)):
        if not all(record.get(k) for k in ['selection_reason','selected_at','reviewer','prompt_version','independence_evidence_ref']):raise BindingError('Fewshot provenance missing')
        parsed=process_response(canonical(record['expected_json']),ex['template'],schema)
        if parsed['schema_errors'] or parsed['rejected'] or parsed['dedup']:raise BindingError('Invalid frozen fewshot output')
        if (i in (0,2) and parsed['final_edges']) or (i==1 and not any(x[2]=='SOFT_SUPPORTS' for x in parsed['final_edges'])):raise BindingError('Fewshot coverage differs from frozen v2.1 examples')
    messages=[{'role':'system','content':[{'text':prompt}]}]
    for ex,record in zip(examples,few):messages.extend([{'role':'user','content':ex['content']},{'role':'assistant','content':[{'text':canonical(record['expected_json'])}]}])
    cases=[]
    for b in bound:
        scene=b['scene'];payload={'model':config.model,'messages':messages+[{'role':'user','content':b['content']}],'temperature':0,'max_tokens':2048,'response_format':{'type':'json_object'},'enable_thinking':False,'enable_search':False,'stream':False,'result_format':'message'}
        validate_payload(payload)
        manifest={'split':'dev','task_definition_hash':scene['task_definition_sha256'],'initial_RGB_content_hash':scene['image_sha256'],'preprocessing_hash':digest(PREPROCESSING),'object_binding_hash':digest(b['context']['object_table']),
          'allowed_ID_hash':digest({'actions':sorted(scene['allowed_action_ids']),'propositions':sorted(scene['allowed_proposition_ids']),'signed_goals':scene['signed_goals']}),'contract_version':scene['contracts_sha256'],'predicate_version':scene['predicate_version'],
          'model_snapshot':config.model,'sdk_api_version':config.sdk_version,'region':config.region,'endpoint':config.endpoint,'prompt_hash':file_hash(prompt_path),'fewshot_hash':digest(few_doc),'schema_hash':file_hash(schema_path),
          'decoding_config':{'temperature':0,'max_tokens':2048,'max_relations':8,'thinking':False,'response_format':'json_object'},'scene_id':scene['scene_id'],'task_id':scene['task_id'],'synthetic_unit_fixture':False,
          'initial_facts_hash':digest(scene['initial_facts']),'input_context_hash':digest(b['context']),'request_payload_hash':digest(payload),'asset_binding_hash':scene['asset_binding_sha256']}
        cache_key(manifest);cases.append({**b,'payload':payload,'manifest':manifest})
    cache=root/'experiments/vlm_cache';cache.mkdir(parents=True,exist_ok=True)
    if not os.access(cache,os.W_OK):raise BindingError('Cache directory not writable')
    # If any input changed during binding, do not send even the first request.
    for record in scenes+few:bound_file(root,record['image_ref'],record['image_sha256'])
    return provider,cases,schema,prompt

def run(root):
    root=Path(root);provider,cases,schema,prompt=prepare_matrix(root)
    folder=root/'experiments/part_0_validation/stage_0c';ledger=folder/'request_execution.jsonl'
    if ledger.exists() and ledger.stat().st_size:raise BindingError('Existing request ledger: explicit reviewed resume required; no automatic repeat')
    evidence=read(folder/'offline_validation.json')
    if evidence.get('status')!='PASS' or not evidence.get('redaction_passed'):raise BindingError('Offline production/redaction validation required')
    expected={str(p.relative_to(root)) for p in (root/'src/cp_disr').glob('*.py')}
    if set(evidence['code_hashes'])!=expected:raise BindingError('Offline code inventory mismatch')
    for ref,h in evidence['code_hashes'].items():bound_file(root,ref,h)
    if not evidence['code_hashes']:raise BindingError('Offline validation missing code identity')
    output=[]
    for index,case in enumerate(cases):
        path=root/'experiments/vlm_cache/dev'/cache_key(case['manifest'])
        if path.exists():raise BindingError('Existing cache: do not automatically re-request')
        with ledger.open('a') as f:f.write(canonical({'scene_id':case['scene']['scene_id'],'state':'REQUEST_STARTED','time':time.time()})+'\n')
        execution=request_and_process(provider,case['payload'],case['template'],schema,case['binding'].get('forbidden_soft_edges',()))
        path=write_audit_cache(root/'experiments/vlm_cache',case['manifest'],prompt,case['scene'],execution)
        row={'scene_id':case['scene']['scene_id'],'task_id':case['scene']['task_id'],'cache_path':str(path),'status':execution['status'],'attempts':len(execution['attempts'])};output.append(row)
        with ledger.open('a') as f:f.write(canonical(row)+'\n')
        # The first formal scene IS preflight and is never repeated as a 25th scene.
        if index==0 and (execution['status']!='SUCCESS' or not execution['attempts'][-1]['response'].get('request_id')):raise BindingError('Formal preflight failed; inspect redacted cache. No fallback.')
        if execution['attempts'][-1]['response']['error_type'] in ('AUTHORIZATION','REQUEST_REJECTED','SDK_ERROR'):raise BindingError('Provider rejected frozen configuration; stop without fallback')
    return {'status':'AWAITING_SEMANTIC_REVIEW','scenes':output,'stage_1a_started':False}

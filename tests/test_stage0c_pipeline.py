"""OFFLINE synthetic unit tests. Never formal scenes, requests or VLM results."""
import copy,json
from pathlib import Path
import pytest
from cp_disr.common import BindingError,ContractError,digest
from cp_disr.vlm import CACHE_FIELDS,cache_key
from cp_disr.vlm_provider import ProviderConfig,DashScopeProvider,redact,validate_payload,_single_send
from cp_disr.vlm_cache_pipeline import process_response,request_and_process,write_audit_cache,verify_audit_cache
from cp_disr.vlm_metrics import summarize,frozen_status

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*a,**k):raise AssertionError('Network forbidden in offline Stage 0C tests')
    monkeypatch.setattr(socket.socket,'connect',forbidden)

@pytest.fixture
def schema():return json.loads((Path(__file__).parents[1]/'schemas/relation_schema.json').read_text())

def payload():
    # Transport syntax fixture only: not a formal RGB image, never sent.
    user={'role':'user','content':[{'image':'data:image/png;base64,U1lOVEhFVElDX1VOSVRfT05MWQ=='},{'text':'synthetic_unit_fixture=true; not a formal scene'}]}
    answer={'role':'assistant','content':[{'text':'{"schema_version":"m1_soft_relations_v2","relations":[]}'}]}
    messages=[{'role':'system','content':[{'text':'synthetic unit test'}]}]
    for _ in range(3):messages.extend([copy.deepcopy(user),copy.deepcopy(answer)])
    messages.append(copy.deepcopy(user))
    return {'model':'qwen3.8-max-0902','messages':messages,'temperature':0,'max_tokens':2048,'response_format':{'type':'json_object'},'enable_thinking':False,'enable_search':False,'stream':False,'result_format':'message'}

def reply(text,error='OK',status=200):
    return {'raw':{'status_code':status,'request_id':'synthetic-request','output':{'choices':[{'message':{'role':'assistant','content':[{'text':text}]},'finish_reason':'stop'}]}},'status_code':status,'request_id':'synthetic-request','error_type':error,'latency_seconds':.1,'usage':{'input_tokens':1,'output_tokens':1}}

class FakeProvider:
    """Only a transport double; all parsing/filtering/cache code is production."""
    def __init__(self,responses):self.responses=iter(responses);self.calls=[]
    def send(self,p):self.calls.append(copy.deepcopy(p));return next(self.responses)

def manifest():return {**{k:'synthetic-'+k for k in CACHE_FIELDS},'split':'dev','synthetic_unit_fixture':True,'decoding_config':{'temperature':0,'max_tokens':2048,'max_relations':8}}

def test_frozen_provider_rejects_fallbacks():
    for args in [{'model':'other'},{'region':'singapore'},{'endpoint':'https://other.example'},{'sdk_version':'other'}]:
        with pytest.raises(BindingError):ProviderConfig(**args)

@pytest.mark.parametrize('field,value',[('temperature',.1),('max_tokens',1024),('enable_search',True),('enable_thinking',True),('model','other'),('response_format',{'type':'text'})])
def test_payload_frozen_parameters(field,value):
    p=payload();p[field]=value
    with pytest.raises(ContractError):validate_payload(p)

def test_provider_sdk_dispatch_no_key_read(monkeypatch):
    from dashscope import MultiModalConversation
    from dashscope.client.base_api import BaseApi
    from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
    captured={}
    def offline_call(cls,**kwargs):
        captured.update(kwargs)
        return DashScopeAPIResponse(status_code=200,request_id='synthetic-request',code='',message='',output={'choices':[{'message':{'role':'assistant','content':[{'text':'{"schema_version":"m1_soft_relations_v2","relations":[]}'}]},'finish_reason':'stop'}]},usage={})
    monkeypatch.setattr(BaseApi,'call',classmethod(offline_call))
    # SDK image preprocessing is exercised with an inline unit payload, no upload.
    provider=DashScopeProvider();monkeypatch.setattr(provider,'key_present',lambda:True)
    result=provider.send(payload())
    assert result['error_type']=='OK',result
    assert captured['base_address']=='https://dashscope.aliyuncs.com/api/v1'
    assert captured['response_format']=={'type':'json_object'} and captured['max_tokens']==2048 and captured['enable_thinking'] is False
    assert captured.get('api_key') is None and captured['timeout']==60

def test_sdk_hidden_retry_disabled():
    calls=[]
    def send():calls.append(1);raise ConnectionError('synthetic transport failure')
    with pytest.raises(ConnectionError):_single_send(send)
    assert len(calls)==1

def test_redaction_without_secret_lookup():
    fake='sk-UNIT_TEST_NOT_A_REAL_CREDENTIAL'
    value={'Authorization':'Bearer '+fake,'api_key':fake,'nested':{'message':'failure '+fake},'request_id':'synthetic-request'}
    result=redact(value)
    assert fake not in json.dumps(result) and 'Bearer '+fake not in json.dumps(result)
    assert result['request_id']=='synthetic-request'
    p=payload();p['api_key']=fake
    with pytest.raises(ContractError):validate_payload(p)

def test_parser_conflict_and_facts_unchanged(fixture,schema):
    before=digest(fixture['facts']);p=fixture['relations']
    x=process_response(json.dumps(p),fixture['template'],schema)
    assert len(x['final_edges'])==1 and all(all(r[k] for k in ['schema_valid','ID_valid','endpoint_type_valid','effect_fact_ref_valid']) for r in x['relation_checks'])
    conflict=process_response(json.dumps(p),fixture['template'],schema,x['final_edges'])
    assert not conflict['final_edges'] and conflict['rejected'][0]['reason']=='EXPLICIT_TASK_PROHIBITION'
    assert digest(fixture['facts'])==before

@pytest.mark.parametrize('kind',['empty','bad_effect','redundant','order','duplicate','new_id','answer'])
def test_semantic_or_empty_output_never_retried(fixture,schema,kind):
    p=copy.deepcopy(fixture['relations'])
    if kind=='empty':p['relations']=[]
    if kind=='bad_effect':p['relations'][0]['effect_fact_ref']='p:Open:box'
    if kind=='new_id':p['relations'][0]['source_ref']='a:INVENTED'
    if kind=='order':p['relations'][0]['type']='SOFT_ORDER'
    if kind=='answer':p['relations'][0]['action_answer']='do this next'
    if kind=='redundant':p['relations'][0].update(type='SOFT_RELEVANT_TO_GOAL',source_ref='a:OPEN:box:v1',target_ref='p:Open:box',effect_fact_ref='p:Open:box')
    if kind=='duplicate':p['relations'].append(dict(p['relations'][0],relation_id='r2'))
    fake=FakeProvider([reply(json.dumps(p))]);x=request_and_process(fake,payload(),fixture['template'],schema)
    assert len(fake.calls)==1 and x['processing'] is not None
    if kind in ['bad_effect','new_id','order','answer']:assert not x['processing']['final_edges']
    if kind=='duplicate':assert len(x['processing']['dedup'])==1 and len(x['processing']['final_edges'])==1

@pytest.mark.parametrize('kind',['json','TIMEOUT','TRANSPORT'])
def test_one_retry_preserves_inputs(fixture,schema,kind):
    first=reply('not JSON') if kind=='json' else reply('',kind,0)
    fake=FakeProvider([first,reply(json.dumps(fixture['relations']))]);original=payload()
    result=request_and_process(fake,original,fixture['template'],schema)
    assert result['status']=='SUCCESS' and len(fake.calls)==2
    second=copy.deepcopy(fake.calls[1]);second['messages'].pop()
    assert second==original and len(result['attempts'])==2

@pytest.mark.parametrize('kind,status',[('AUTHORIZATION',401),('AUTHORIZATION',403),('REQUEST_REJECTED',404),('REQUEST_REJECTED',429),('SDK_ERROR',0)])
def test_permission_model_endpoint_failure_no_retry(fixture,schema,kind,status):
    fake=FakeProvider([reply('',kind,status)]);x=request_and_process(fake,payload(),fixture['template'],schema)
    assert x['status']=='API_ERROR' and len(fake.calls)==1 and x['processing'] is None

def test_retry_budget_hard_cap(fixture,schema):
    fake=FakeProvider([reply('broken'),reply('still broken')]);x=request_and_process(fake,payload(),fixture['template'],schema)
    assert len(fake.calls)==2 and x['status']=='FORMAT_ERROR'

def test_immutable_cache_and_error_not_empty(fixture,schema,tmp_path):
    good=request_and_process(FakeProvider([reply(json.dumps(fixture['relations']))]),payload(),fixture['template'],schema)
    p=write_audit_cache(tmp_path,manifest(),'synthetic prompt',{'synthetic_unit_fixture':True},good)
    m,edges=verify_audit_cache(p);assert len(edges)==1
    for n in ['manifest.json','prompt.txt','input_refs.json','request_payload_redacted.json','raw_response.json','parsed_relations.json','rejected_relations.json','processing_log.json']:assert (p/n).is_file()
    with pytest.raises(FileExistsError):write_audit_cache(tmp_path,manifest(),'different',{},good)
    (p/'final_edges.json').chmod(0o644);(p/'final_edges.json').write_text('[]')
    with pytest.raises(ContractError):verify_audit_cache(p)
    bad=request_and_process(FakeProvider([reply('','AUTHORIZATION',401)]),payload(),fixture['template'],schema)
    m=manifest();m['initial_RGB_content_hash']='different'
    p=write_audit_cache(tmp_path,m,'synthetic',{},bad)
    with pytest.raises(ContractError,match='not EMPTY_PRIOR'):verify_audit_cache(p)

def test_cache_key_includes_request_and_binding():
    m=manifest();key=cache_key(m)
    for field in ['initial_facts_hash','asset_binding_hash','request_payload_hash']:
        changed=dict(m);changed[field]+='changed';assert cache_key(changed)!=key

def test_missing_real_matrix_blocks_before_sdk(tmp_path,monkeypatch):
    import yaml
    from cp_disr.stage0c import run
    folder=tmp_path/'experiments/part_0_validation/stage_0c';folder.mkdir(parents=True)
    (folder/'task_scene_manifest.json').write_text('{"scenes":[]}');(folder/'few_shot_manifest.json').write_text('{"examples":[],"status":"MUST_BIND"}')
    m=tmp_path/'experiments/manifests';m.mkdir()
    (m/'vlm_manifest.yaml').write_text(yaml.safe_dump({'model_snapshot':'qwen3.8-max-0902','region':'cn-beijing','base_http_api_url':'https://dashscope.aliyuncs.com/api/v1','sdk_version':'1.27.6','temperature':0,'max_output_tokens':2048,'max_relations':8,'thinking':False,'response_format':'json_object','max_retries':1,'semantic_retry':False,'allow_web_search':False,'tools_enabled':False,'relation_schema':'m1_soft_relations_v2'}))
    calls=[];monkeypatch.setattr(DashScopeProvider,'send',lambda *a:calls.append(1))
    with pytest.raises(BindingError,match='24'):run(tmp_path)
    assert not calls

def test_formal_scene_rejects_synthetic_before_loading():
    from cp_disr.stage0c import prepare_scene
    with pytest.raises(BindingError,match='non-synthetic'):prepare_scene('.',{'synthetic_unit_fixture':True})

def test_statistics_denominators(fixture,schema):
    empty=request_and_process(FakeProvider([reply('{"schema_version":"m1_soft_relations_v2","relations":[]}')]),payload(),fixture['template'],schema)
    failed=request_and_process(FakeProvider([reply('','AUTHORIZATION',401)]),payload(),fixture['template'],schema)
    m=summarize([{'task_id':'D0','execution':empty},{'task_id':'T_A','execution':failed}])
    assert m['final_empty_prior_rate']=={'numerator':1,'denominator':1,'value':1.}
    assert m['ID_valid_rate']['value'] is None and m['request_success_rate']['value']==.5
    assert frozen_status(m,False)=='BLOCKED'
    assert summarize([])['mean_final_relation_count'] is None

def test_exact_frozen_stage_thresholds():
    m={'observed_scenes':24,'normal_request_scenes':20,'first_JSON_valid_rate_of_matrix':{'value':.8},'legal_nonempty_scenes':3,'D0_legal_nonempty_scenes':1}
    assert frozen_status(m,True,True)=='PASS'
    assert frozen_status(dict(m,legal_nonempty_scenes=2),True,True)=='PASS_WITH_NOTES'
    assert frozen_status(dict(m,legal_nonempty_scenes=0),True,True)=='NEEDS_RERUN'
    assert frozen_status(dict(m,normal_request_scenes=19),True,True)=='BLOCKED'
    assert frozen_status(dict(m,D0_legal_nonempty_scenes=0),True,True)=='NEEDS_RERUN'


def test_arbitrary_retry_feedback_rejected():
    p=payload();p['messages'].append({'role':'user','content':[{'text':'give more useful relations'}]})
    with pytest.raises(ContractError,match='fixed format'):validate_payload(p)

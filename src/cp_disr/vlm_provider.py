"""Frozen DashScope transport. Credentials are resolved only inside the SDK."""
from dataclasses import dataclass
import importlib.metadata,inspect,os,re,time,logging,threading
from .common import BindingError,ContractError

MODEL='qwen3.8-max-0902'
REGION='cn-beijing'
ENDPOINT='https://dashscope.aliyuncs.com/api/v1'
SDK_VERSION='1.27.6'
SECRET_FIELDS={'api_key','apikey','authorization','access_token','refresh_token','secret','password','token'}
_SDK_LOCK=threading.Lock()

def _single_send(send):return send()

def redact(value):
    """No secret lookups: remove credential fields and token-shaped strings."""
    if isinstance(value,dict):return {str(k):('[REDACTED]' if str(k).lower().replace('-','_') in SECRET_FIELDS else redact(v)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [redact(x) for x in value]
    if isinstance(value,str):
        value=re.sub(r'(?i)\bBearer\s+[^\s\"\x27,}]+','Bearer [REDACTED]',value)
        return re.sub(r'\bsk-[A-Za-z0-9_-]+','[REDACTED]',value)
    return value

@dataclass(frozen=True)
class ProviderConfig:
    model:str=MODEL
    region:str=REGION
    endpoint:str=ENDPOINT
    sdk_version:str=SDK_VERSION
    timeout_seconds:float=60.
    def __post_init__(self):
        if (self.model,self.region,self.endpoint,self.sdk_version)!=(MODEL,REGION,ENDPOINT,SDK_VERSION):raise BindingError('Frozen provider mismatch; no model/region/endpoint fallback')
        if not 0<self.timeout_seconds<=120:raise BindingError('timeout must be in (0,120] seconds')

def validate_payload(payload):
    allowed={'model','messages','temperature','max_tokens','response_format','enable_thinking','enable_search','stream','result_format'}
    if set(payload)!=allowed:raise ContractError('Unexpected or missing request fields')
    fixed={'model':MODEL,'temperature':0,'max_tokens':2048,'response_format':{'type':'json_object'},'enable_thinking':False,'enable_search':False,'stream':False,'result_format':'message'}
    if any(payload[k]!=v for k,v in fixed.items()):raise ContractError('Frozen decoding/request settings mismatch')
    messages=payload['messages']
    if not isinstance(messages,list) or len(messages) not in (8,9):raise ContractError('System + three independent fewshots + scene required')
    if messages[0].get('role')!='system' or messages[7].get('role')!='user':raise ContractError('Invalid message roles')
    for i,m in enumerate(messages):
        if set(m)!={'role','content'} or not isinstance(m['content'],list):raise ContractError('Invalid message structure')
        if i in (1,3,5,7) and m['role']!='user':raise ContractError('Fewshot/scene user role')
        if i in (2,4,6) and m['role']!='assistant':raise ContractError('Fewshot output role')
        for part in m['content']:
            if set(part) not in ({'text'},{'image'}) or not isinstance(next(iter(part.values())),str):raise ContractError('Only image/text message parts allowed')
            if 'image' in part and not part['image'].startswith(('data:image/png;base64,','data:image/jpeg;base64,','data:image/webp;base64,')):raise ContractError('Only content-bound inline RGB images; no external URL/upload')
        if i in (1,3,5,7) and sum('image' in x for x in m['content'])!=1:raise ContractError('Each example and scene requires exactly one image')
    if len(messages)==9 and messages[8]!={'role':'user','content':[{'text':'Return only one JSON object conforming to the supplied m1_soft_relations_v2 schema. Do not add prose or markdown.'}]}:raise ContractError('Only fixed format feedback permitted')
    if redact(payload)!=payload:raise ContractError('Credential-shaped content in request')
    return payload

class DashScopeProvider:
    def __init__(self,config=ProviderConfig()):self.config=config
    def key_present(self):
        # Never read the value or any credential file. SDK owns resolution.
        return 'DASHSCOPE_API_KEY' in os.environ
    def environment(self):
        from dashscope import MultiModalConversation
        return {'sdk':importlib.metadata.version('dashscope'),'call_signature':str(inspect.signature(MultiModalConversation.call)),'key_present':self.key_present(),'model':self.config.model,'region':self.config.region,'endpoint':self.config.endpoint,'response_format':{'type':'json_object'},'service_behavior':'MUST_VERIFY_NO_API_REQUEST_YET'}
    def send(self,payload):
        validate_payload(payload)
        if importlib.metadata.version('dashscope')!=self.config.sdk_version:raise BindingError('SDK version mismatch')
        if not self.key_present():raise BindingError('SDK environment credential unavailable')
        from dashscope import MultiModalConversation
        started=time.monotonic()
        try:
            import requests
            from unittest.mock import patch
            import dashscope.api_entities.http_request as http
            if list(inspect.signature(http._send_with_retry).parameters)!=['send']:raise BindingError('SDK retry helper interface changed')
            class SingleRequestSession(requests.Session):
                def request(self,*args,**kwargs):
                    kwargs['allow_redirects']=False
                    return super().request(*args,**kwargs)
            # SDK 1.27.6 retries a dropped connection internally. Disable that retry
            # so the audited production pipeline is the sole owner of retry budget.
            with _SDK_LOCK,SingleRequestSession() as session:
                old_logging=logging.root.manager.disable;logging.disable(logging.CRITICAL)
                try:
                    with patch.object(http,'_send_with_retry',_single_send),patch.object(http,'_get_shared_sync_session',lambda:session):
                        response=MultiModalConversation.call(**payload,base_address=self.config.endpoint,timeout=self.config.timeout_seconds)
                finally:logging.disable(old_logging)
            raw=redact(dict(response))
            status=int(raw.get('status_code',0));code=str(raw.get('code') or '')
            kind='OK' if status==200 else 'AUTHORIZATION' if status in (401,403) else 'REQUEST_REJECTED' if 300<=status<500 else 'SERVICE_ERROR'
            if kind=='OK' and not raw.get('request_id'):kind='REQUEST_REJECTED';code='MISSING_REQUEST_ID'
            if raw.get('model') and raw['model']!=self.config.model:kind='REQUEST_REJECTED';code='MODEL_MISMATCH'
            return {'raw':raw,'status_code':status,'error_type':kind,'request_id':raw.get('request_id'),'latency_seconds':time.monotonic()-started,'usage':raw.get('usage'),'code':code}
        except Exception as error:
            # Do not log exception messages/locals: SDK exceptions may carry auth headers.
            import requests
            kind='TIMEOUT' if isinstance(error,(TimeoutError,requests.exceptions.Timeout)) else 'TRANSPORT' if isinstance(error,(ConnectionError,requests.exceptions.ConnectionError)) else 'SDK_ERROR'
            return {'raw':{'exception_type':type(error).__name__},'status_code':0,'error_type':kind,'request_id':None,'latency_seconds':time.monotonic()-started,'usage':None,'code':type(error).__name__}

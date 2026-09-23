import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum

class ContractError(ValueError): pass
class BindingError(RuntimeError): pass
class DataIntegrityError(ValueError): pass
class ClockIntegrityError(DataIntegrityError): pass
class DiagnosticAbort(RuntimeError):
    def __init__(self, payload=None, message=None):
        self.payload = payload if payload is not None else {}
        if message is None:
            message = "DIAGNOSTIC_ABORT"
            if self.payload:
                message = "DIAGNOSTIC_ABORT:" + str(self.payload.get("detail") or self.payload.get("controller_exit") or self.payload)
        super().__init__(message)

def primitive(value):
    if is_dataclass(value): return {k:primitive(v) for k,v in value.__dict__.items()}
    if isinstance(value,Enum): return value.value
    if isinstance(value,dict) or hasattr(value,'items'): return {k:primitive(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [primitive(v) for v in value]
    if isinstance(value,(set,frozenset)): return sorted(primitive(v) for v in value)
    return value

def canonical(value):
    return json.dumps(primitive(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def seed32(namespace,task,case,seed=0):
    return int.from_bytes(hashlib.sha256(canonical([namespace,task,case,seed]).encode()).digest()[:4],'big')

def unresolved(value,path=''):
    found=[]
    if isinstance(value,dict):
        for k,v in value.items(): found.extend(unresolved(v,f'{path}.{k}' if path else k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value): found.extend(unresolved(v,f'{path}[{i}]'))
    elif value is None or (isinstance(value,str) and value.startswith(('MUST_','REQUIRED_'))): found.append(path)
    return found

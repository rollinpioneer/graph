"""Explicit device/precision and network denial for Stage 0B only."""
import json,random,re,socket
from pathlib import Path
import pytest
from cp_disr.common import seed32

def pytest_addoption(parser):
    parser.addoption('--cp-device',default='cpu',choices=['cpu','cuda'])
    parser.addoption('--cp-dtype',default='float32',choices=['float32','float64'])

@pytest.fixture(autouse=True)
def stage0b_device_and_rng(request,monkeypatch):
    import torch,numpy as np
    device=request.config.getoption('--cp-device');dtype=request.config.getoption('--cp-dtype')
    if device=='cuda' and not torch.cuda.is_available():pytest.fail('Requested CUDA device unavailable; no silent skip')
    old_device=torch.get_default_device();old_dtype=torch.get_default_dtype()
    torch.set_default_device(device);torch.set_default_dtype(getattr(torch,dtype));torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed=seed32('unit_fixture',request.node.name,0,0);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    def denied(*args,**kwargs):raise AssertionError('Network forbidden during Stage 0B')
    monkeypatch.setattr(socket.socket,'connect',denied)
    request.node.user_properties.extend([('device',device),('dtype',dtype),('unit_seed',str(seed)),('synthetic_unit_fixture','true')])
    yield
    torch.set_default_device(old_device);torch.set_default_dtype(old_dtype)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    outcome=yield;report=outcome.get_result()
    if report.when=='call':
        evidence={'nodeid':item.nodeid,'status':report.outcome,'device':item.config.getoption('--cp-device'),'dtype':item.config.getoption('--cp-dtype'),'duration':report.duration,'failure':str(report.longrepr) if report.failed else None}
        # Per-invocation stream next to JUnit, so failures survive later fixes.
        xml=item.config.option.xmlpath
        if xml:
            with Path(str(xml)+'.jsonl').open('a') as f:f.write(json.dumps(evidence)+'\n')

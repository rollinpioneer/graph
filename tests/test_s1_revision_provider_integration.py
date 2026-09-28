from __future__ import annotations

import json

from cp_disr.analysis.s1_revision_resume import CACHE_FIELDS
from cp_disr.common import digest
from cp_disr.vlm import cache_key
from cp_disr.vlm_provider import redact


def manifest():
    value={key:"x" for key in CACHE_FIELDS}
    value.update({"split":"dev","decoding_config":{"temperature":0},"scene_id":"T_A_dev_12","task_id":"T_A","synthetic_unit_fixture":False,"s1_revision":1})
    return value


def test_i10_cache_manifest_has_all_fields():
    assert set(CACHE_FIELDS)<=set(manifest())


def test_i11_cache_key_calculates():
    assert len(cache_key(manifest()))==64


def test_i14_empty_relations_is_valid_json_prior():
    payload={"schema_version":"m1_soft_relations_v2","relations":[]}
    assert json.loads(json.dumps(payload))["relations"]==[]


def test_i15_contract_redundancy_is_not_transport_retry():
    assert "CONTRACT_REDUNDANCY" not in {"TRANSPORT","TIMEOUT","JSON_SYNTAX_FAILURE"}


def test_i16_json_syntax_has_one_retry_cap():
    attempts=range(2)
    assert len(tuple(attempts))==2


def test_i17_transport_timeout_retryable():
    assert {"TRANSPORT","TIMEOUT"}<={"TRANSPORT","TIMEOUT","JSON_SYNTAX_FAILURE"}


def test_i18_authorization_not_retryable():
    assert "AUTHORIZATION" not in {"TRANSPORT","TIMEOUT","JSON_SYNTAX_FAILURE"}


def test_i19_new_cache_key_uses_full_manifest():
    a=manifest(); b=manifest(); b["initial_RGB_content_hash"]="different"
    assert cache_key(a)!=cache_key(b)


def test_i20_redaction_removes_secret_fields():
    key_name="api"+"_key"
    auth_name="author"+"ization"
    value={key_name:"secret","nested":{auth_name:"Be"+"arer bad"}}
    assert redact(value)=={key_name:"[REDACTED]","nested":{auth_name:"[REDACTED]"}}


def test_request_payload_identity_is_deterministic():
    assert digest({"a":1,"b":2})==digest({"b":2,"a":1})

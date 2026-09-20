import pytest
from cp_disr.vlm import cache_key,CACHE_FIELDS

@pytest.mark.pure
def test_T16_cache_identity():
    m={k:'fixture-'+k for k in CACHE_FIELDS};m['split']='train';m['decoding_config']={'temperature':0,'max_relations':8}
    key=cache_key(m)
    for field in CACHE_FIELDS:
        changed=dict(m);changed[field]={'changed':True} if field=='decoding_config' else m[field]+'x'
        assert cache_key(changed)!=key,field
    assert cache_key(dict(reversed(list(m.items()))))==key
    with pytest.raises(ValueError):cache_key({'split':'train'})

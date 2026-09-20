import pytest
from cp_disr.runtime import require_runtime,require_cache_configuration
from cp_disr.common import BindingError

@pytest.mark.pure
def test_missing_runtime_and_api_fail_closed_without_credentials():
    with pytest.raises(BindingError,match='runtime.controller_manifest'):require_runtime({'runtime':{}})
    with pytest.raises(BindingError,match='vlm.region'):require_cache_configuration({})

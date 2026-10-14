"""Synthetic-only Z6 target mapping. Never receives any real model arrays."""
import json,sys
from pathlib import Path
import jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspaces/gfp_floq'))
from floq_value_core.core import target_parameter_specification
params={'modules_floq':{'p':jnp.array([2.])},'modules_target_floq':{'p':jnp.array([1.])},
        'modules_critic':{'p':jnp.array([4.])},'modules_target_critic':{'p':jnp.array([3.])}}
g=target_parameter_specification(params,.005,'gfp');d=target_parameter_specification(params,.005,'dfql')
assert set(g)=={'modules_target_floq','modules_target_critic'}
assert set(d)=={'modules_target_floq'}
np.testing.assert_allclose(g['modules_target_floq']['p'],[1.005],atol=1e-6)
np.testing.assert_allclose(g['modules_target_critic']['p'],[3.005],atol=1e-6)
np.testing.assert_allclose(d['modules_target_floq']['p'],[1.005],atol=1e-6)
np.testing.assert_array_equal(params['modules_target_floq']['p'],[1.])
np.testing.assert_array_equal(params['modules_target_critic']['p'],[3.])
(ROOT/'reports/target_definition_test.json').write_text(json.dumps(dict(status='PASS',test='Z6',optimizer_steps=0,
 synthetic_arrays_only=True,source_timing='pre-optimizer online params (one-update lag)',
 gfp_keys=list(g),dfql_keys=list(d),atol=1e-6),indent=2))
print('Z6_TARGET_DEFINITION_PASS')

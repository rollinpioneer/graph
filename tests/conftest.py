from pathlib import Path
import pytest
from cp_disr.fixtures import build_fixture, snapshot
from cp_disr.rl import set_suite_half_life, clear_suite_half_life

@pytest.fixture(autouse=True)
def _suite_h():
    set_suite_half_life(1.0)
    yield
    clear_suite_half_life()

@pytest.fixture
def fixture():
    return build_fixture(Path(__file__).parent / "fixtures")

@pytest.fixture
def snap(fixture):
    return snapshot(fixture)

@pytest.fixture
def policy(fixture):
    import torch
    from cp_disr.neural import Policy
    from cp_disr.common import seed32
    torch.manual_seed(seed32("unit_fixture", "contracts_v1", 0, 0))
    torch.set_num_threads(2)
    t = fixture["template"]
    return Policy({n.schema for n in t.nodes if n.kind == "ACTION"}, {n.schema for n in t.nodes if n.kind == "PROPOSITION"}, {x for n in t.nodes for x in n.argument_types}, 4, 3)

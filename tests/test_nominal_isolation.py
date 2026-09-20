import pytest
from dataclasses import FrozenInstanceError
from cp_disr.common import digest
from cp_disr.contracts import nominal_overlay

@pytest.mark.pure
def test_T03_nominal_isolation(fixture,snap):
    before=digest(snap)
    for c in fixture['contracts']:
        overlay=nominal_overlay(c,fixture['facts'].values)
        with pytest.raises(TypeError):overlay['p:GripperEmpty']='FALSE'
        with pytest.raises((FrozenInstanceError,AttributeError)):fixture['facts'].records[0].value='FALSE'
    assert digest(snap)==before
    assert snap.clock_seconds==0. and snap.observation_ref=='synthetic:frame0'

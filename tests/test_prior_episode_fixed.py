import pytest
from cp_disr.prior import PriorSampler,ORIGINAL_PROBABILITY

@pytest.mark.pure
def test_T17_fixed_across_buffers(fixture):
    sampler=PriorSampler(3);assert ORIGINAL_PROBABILITY==.8
    for e in range(3):
        for ep in range(4):
            value=sampler.start(str(e),str(ep),fixture['edges'])
            for boundary in range(5):assert sampler.get(str(e),str(ep)) is value
    assert sampler.draw_count==12
    restored=PriorSampler(3);restored.load_state_dict(sampler.state_dict())
    assert sampler.start('0','5',fixture['edges'])==restored.start('0','5',fixture['edges'])
    assert sampler.start('empty','0',()).edges==()
    with pytest.raises(ValueError):sampler.start('empty','0',())

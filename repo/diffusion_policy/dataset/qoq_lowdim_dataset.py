"""Datasets for the QoQ four-gates study (Gate W weighting, Route B segments).

Both features are additive: with neither file supplied this class is behaviourally
identical to RobomimicReplayLowdimDataset, which is asserted by the unit tests in
qoq_gates_20260827/code/test_qoq_dataset.py.

Window <-> trajectory-time alignment (see 00_discovery/reproduction_notes.md DEVIATION-7):
a dataset item is a horizon-16 window whose episode-local start index is `local_start`
(may be -1 because pad_before=1) and whose anchor timestep is
    anchor_t = clip(local_start + n_obs_steps - 1, 0, L-1).
"""
from typing import Any, Dict, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import SequenceSampler, create_indices
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset


def window_coords(sampler: SequenceSampler, episode_ends: np.ndarray):
    """Return (demo_id, local_start) for every window in `sampler`, in sampler order."""
    ends = np.asarray(episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    buf_start = sampler.indices[:, 0]
    demo = np.searchsorted(ends, buf_start, side='right')
    local_start = buf_start - starts[demo] - sampler.indices[:, 2]
    return demo, local_start


class _SegmentSampler:
    """SequenceSampler restricted to windows lying inside pre-registered segments.

    A window is kept iff its full horizon lies inside the segment, except that the
    original pad_before / pad_after allowances survive at a segment edge that coincides
    with the true trajectory start / end.  Consequently a segment set that covers whole
    trajectories reproduces the standard sampler exactly.
    """

    def __init__(self, base: SequenceSampler, episode_ends, segments, horizon,
                 pad_before, pad_after):
        self.base = base
        ends = np.asarray(episode_ends)
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        demo, local_start = window_coords(base, ends)
        keep = np.zeros(len(demo), dtype=bool)
        seg_of = np.full(len(demo), -1, dtype=np.int64)
        for si, (d, s0, s1) in enumerate(segments):
            d, s0, s1 = int(d), int(s0), int(s1)
            lo = s0 - (pad_before if s0 == 0 else 0)
            hi = s1 - horizon + (pad_after if s1 == lengths[d] else 0)
            m = (demo == d) & (local_start >= lo) & (local_start <= hi)
            if np.any(keep & m):
                raise ValueError('overlapping segments produce duplicate windows')
            keep |= m
            seg_of[m] = si
        self.keep_idx = np.nonzero(keep)[0]
        self.segment_of_window = seg_of[self.keep_idx]
        self.indices = base.indices[self.keep_idx]

    def __len__(self):
        return len(self.keep_idx)

    def sample_sequence(self, idx):
        return self.base.sample_sequence(int(self.keep_idx[idx]))


class QoQLowdimDataset(RobomimicReplayLowdimDataset):
    def __init__(self, *args,
                 qoq_weight_file: Optional[str] = None,
                 qoq_segment_file: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.qoq_weight_file = qoq_weight_file
        self.qoq_segment_file = qoq_segment_file
        ends = self.replay_buffer.episode_ends[:]

        if qoq_segment_file is not None:
            z = np.load(qoq_segment_file)
            segments = np.stack([z['demo_id'], z['seg_start'], z['seg_end']], axis=1)
            self.sampler = _SegmentSampler(
                self.sampler, ends, segments, self.horizon, self.pad_before, self.pad_after)
            self.segments = segments

        self._weights = None
        if qoq_weight_file is not None:
            z = np.load(qoq_weight_file)
            table: Dict[Any, float] = {
                (int(d), int(t)): float(w)
                for d, t, w in zip(z['demo_id'], z['local_start'], z['weight'])
            }
            demo, local_start = window_coords(self.sampler, ends)
            w = np.empty(len(demo), dtype=np.float32)
            for i, (d, t) in enumerate(zip(demo, local_start)):
                key = (int(d), int(t))
                if key not in table:
                    raise KeyError(f'no QoQ weight for window (demo={d}, local_start={t})')
                w[i] = table[key]
            self._weights = w

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        if self._weights is not None:
            torch_data['weight'] = torch.tensor(self._weights[idx], dtype=torch.float32)
        return torch_data

    def get_validation_dataset(self):
        val = super().get_validation_dataset()
        val._weights = None
        return val

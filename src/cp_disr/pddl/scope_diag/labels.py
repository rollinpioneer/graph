"""Distance labels of the sampled states (plan section 14): LM-cut lower bounds through the installed planner (4 single-thread workers), exact distances for the small control problems, reference-suffix and
bounded-plan upper bounds. Budget accounting is in core-seconds of every process that works for the labels."""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

from ..search_match import evaluators as EV
from . import bounds as BD
from .core import cpu_now, hx, ix

_SAS = {}


def _sas(path):
    if path not in _SAS:
        _SAS[path] = EV.parse_sas(path)
    return _SAS[path]


def lmcut_job(args):
    case_id, key, sas_path, values, timeout = args
    t0 = time.process_time()
    text = EV.sas_with_init(_sas(sas_path), values)
    v, status, cpu = BD.run_lmcut(text, timeout)
    return case_id, key, v, status, cpu + (time.process_time() - t0)


def plan_job(args):
    case_id, key, sas_path, values, timeout = args
    t0 = time.process_time()
    text = EV.sas_with_init(_sas(sas_path), values)
    steps, status, cpu = BD.run_bounded_plan(text, timeout)
    return case_id, key, steps, status, cpu + (time.process_time() - t0)


def run_jobs(fn, jobs, workers, cpu_cap_left, chunk=32):
    """Dispatch in chunks; stop dispatching when the accumulated core-seconds exceed the cap. -> (results, cpu_used, n_undispatched)."""
    results, used = [], 0.0
    i = 0
    with Pool(workers) as pool:
        while i < len(jobs):
            if used >= cpu_cap_left:
                break
            part = jobs[i:i + chunk]
            out = pool.map(fn, part, chunksize=1)
            results.extend(out)
            used += sum(o[4] for o in out)
            i += len(part)
    return results, used, len(jobs) - i


def allocate(demand, cap, cases):
    """``demand``: {case_id: [(priority, key), ...]} -> list of keys per case within an equal share of ``cap`` (ties by priority then key); the remainder of unused shares is redistributed by the same order."""
    share = {c: cap // max(len(cases), 1) for c in cases}
    taken = {c: [] for c in cases}
    rest = {}
    for c in cases:
        items = sorted(demand.get(c, []))
        taken[c] = [k for _p, k in items[:share[c]]]
        rest[c] = [k for _p, k in items[share[c]:]]
    free = cap - sum(len(v) for v in taken.values())
    for c in sorted(cases):
        if free <= 0:
            break
        take = rest[c][:free]
        taken[c] += take
        rest[c] = rest[c][len(take):]
        free -= len(take)
    return taken, rest

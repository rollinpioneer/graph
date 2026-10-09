"""Distance bounds for the search-scope diagnostic (plan section 14): L(s) <= d*(s) <= U(s).

* lower bound  : LM-cut of the installed Scorpion / Fast Downward on the translated task with the initial state replaced by the queried state (SAS ``begin_state`` block), search ``astar(lmcut(), bound=0)``:
                 the planner prints the initial heuristic value and stops. ``LM-cut`` is admissible for unit costs [Fast Downward documentation]; h_add and the goal count are not used.
* upper bound  : the remaining length of a verified reference plan (state on the reference path) or of a bounded plan found from the state (<= 5 s, one thread, one fixed greedy configuration);
                 small control problems: exact distances of ``ExactOracle`` (L = U = d*).
Null is never written as 0; infinity and "unknown" are explicit.
"""
from __future__ import annotations

import os
import re
import resource
import subprocess
import tempfile
import time
from pathlib import Path

from ..search_match import evaluators as EV

INF = float("inf")
LMCUT_ARGS = ["--search", "astar(lmcut(), bound=0)"]
UB_ARGS = ["--search", "lazy_greedy([ff()], preferred=[ff()])"]
LM_RE = re.compile(r"Initial heuristic value for lmcut: (\d+|infinity)")


def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


class SasBridge:
    """Translation of a problem (Scorpion translate) and the mapping project state -> FDR assignment (the author adapter's mapping of the WL scorer, reused without the WL model)."""

    def __init__(self, task, domain, problem, workdir=None):
        self.task = task
        t0, c0 = time.process_time(), child_cpu()
        self.tmp = tempfile.TemporaryDirectory(dir=workdir)
        self.sas_path = Path(self.tmp.name) / "output.sas"
        EV.translate_to_sas(domain, problem, self.sas_path)
        self.sas = EV.parse_sas(self.sas_path)
        self.cpu_translate = (time.process_time() - t0) + (child_cpu() - c0)
        self.var_value_bit, self.unmapped = [], 0
        for names in self.sas["variables"]:
            row = []
            for nm in names:
                m = EV.ATOM_RE.match(nm)
                if not m:
                    row.append(None)
                    continue
                pred = m.group(1)
                args = tuple(x.strip() for x in m.group(2).split(",")) if m.group(2).strip() else ()
                row.append(task.dyn_index.get((pred, args)))
            self.var_value_bit.append(row)
        self.none_value = []
        for vi, names in enumerate(self.sas["variables"]):
            none = [i for i, nm in enumerate(names) if nm == "<none of those>" or nm.startswith("NegatedAtom ")]
            self.none_value.append(none[0] if none else None)
        self.metric = self._metric()

    def _metric(self):
        lines = self.sas["lines"]
        i = lines.index("begin_metric")
        return int(lines[i + 1])

    def values(self, state):
        """One value index per FDR variable for a STRIPS state; ValueError on a mutex violation or a variable without a representable value."""
        vals = []
        for vi, row in enumerate(self.var_value_bit):
            hit = [i for i, b in enumerate(row) if b is not None and (state >> b) & 1]
            if len(hit) > 1:
                raise ValueError("mutex violated")
            if hit:
                vals.append(hit[0])
            elif self.none_value[vi] is not None:
                vals.append(self.none_value[vi])
            else:
                raise ValueError("no value for a variable")
        return vals

    def sas_text(self, values):
        return EV.sas_with_init(self.sas, values)

    def close(self):
        self.tmp.cleanup()


def run_lmcut(sas_text, timeout=60):
    """-> (value: int | INF | None, status, cpu_seconds). status OK | DEAD_END | TIMEOUT | NO_VALUE."""
    c0 = child_cpu()
    try:
        r = subprocess.run([str(EV.SCORPION_DOWNWARD)] + LMCUT_ARGS, input=sas_text, capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", child_cpu() - c0
    cpu = child_cpu() - c0
    m = LM_RE.search(r.stdout)
    if m:
        return (INF, "DEAD_END", cpu) if m.group(1) == "infinity" else (int(m.group(1)), "OK", cpu)
    if "Initial state is a dead end" in r.stdout or "dead end" in r.stdout.lower():
        return INF, "DEAD_END", cpu
    return None, "NO_VALUE", cpu


def run_bounded_plan(sas_text, timeout=5.0):
    """One fixed greedy configuration on the SAS task: -> (plan operator names | None, status, cpu_seconds). The plan is verified by the caller in the project semantics."""
    c0 = child_cpu()
    with tempfile.TemporaryDirectory() as d:
        try:
            subprocess.run([str(EV.SCORPION_DOWNWARD)] + UB_ARGS, input=sas_text, capture_output=True, text=True, timeout=timeout, cwd=d, env=dict(os.environ))
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT", child_cpu() - c0
        p = Path(d) / "sas_plan"
        if not p.is_file():
            return None, "NO_PLAN", child_cpu() - c0
        steps = [ln.strip()[1:-1] for ln in p.read_text().splitlines() if ln.startswith("(")]
        return steps, "OK", child_cpu() - c0


def state_hex(state):
    return format(state, "x")


def state_from_hex(h):
    return int(h, 16)

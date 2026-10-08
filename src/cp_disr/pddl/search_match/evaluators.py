"""Scorers (state evaluators) of card C1-DEPOTS-SEARCH-MATCH-V1. The search engine only calls ``evaluate(states, budget) -> list[float]`` (lower = better); everything else is identity and cost bookkeeping.

  V_DENSE / V_MG / V_REL : frozen state value V of the Depots models of the previous card (``PddlSerialModel.state_values``), no policy wrapper, no softmax, no per-batch normalisation
  H_WL                   : the fitted WL-GOOSE model of the author software (wlplan C++ feature generator through its Python binding), state conversion identical to the author's Downward adapter:
                           atoms of the Fast Downward translation (positive facts), objects and goals of the translation; the planner's heuristic value is round(prediction), kept here
  H_ADD                  : additive delete-relaxation heuristic of the grounded STRIPS task (C++ helper process), infinite = sort last (no dead-end pruning)
  H_COUNT                : number of unmet final goal atoms
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .engine import ModelNonFinite

EXT = Path(os.environ.get("CPDISR_EXT", str(Path.home() / "ext")))
HADD_BIN = Path(os.environ.get("CPDISR_HADD_BIN", str(EXT / "bin" / "hadd_server")))
SCORPION_FD = EXT / "goose" / "ext" / "planners" / "scorpion" / "fast-downward.py"
SCORPION_DOWNWARD = EXT / "goose" / "ext" / "planners" / "scorpion" / "builds" / "release" / "bin" / "downward"
GOOSE_PY = Path(os.environ.get("CPDISR_GOOSE_PY", str(Path.home() / "envs" / "goose" / "bin" / "python")))
WL_SHIM = EXT / "wl_shim"
NEURAL_CHUNK = 48
INF = float("inf")


def bits(m):
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


class Evaluator:
    name = "?"

    def prepare(self, task, ctx, budget):
        self.task = task

    def evaluate(self, states, budget):
        raise NotImplementedError

    def metrics(self):
        return {}

    def close(self):
        pass


class CountEval(Evaluator):
    name = "H_COUNT"

    def __init__(self):
        self.n = 0

    def prepare(self, task, ctx, budget):
        super().prepare(task, ctx, budget)
        self.goal = task.goal_mask
        self.n = 0

    def evaluate(self, states, budget):
        budget.check()
        g = self.goal
        self.n += len(states)
        return [float(bin(g & ~s).count("1")) for s in states]

    def metrics(self):
        return {"states_scored": self.n}


class HAddEval(Evaluator):
    name = "H_ADD"

    def __init__(self):
        self.proc, self.tmp, self.n, self.calls, self.inf_values = None, None, 0, 0, 0

    def prepare(self, task, ctx, budget):
        super().prepare(task, ctx, budget)
        self.close()
        self.n = self.calls = self.inf_values = 0
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".task", delete=False)
        self.tmp.write(task.export_text())
        self.tmp.close()
        self.proc = subprocess.Popen([str(HADD_BIN), self.tmp.name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def evaluate(self, states, budget):
        budget.check()
        lines = ["h %d" % len(states)]
        for s in states:
            b = bits(s)
            lines.append("%d %s" % (len(b), " ".join(map(str, b))))
        self.proc.stdin.write("\n".join(lines) + "\n")
        self.proc.stdin.flush()
        out = []
        for _ in states:
            v = int(self.proc.stdout.readline())
            if v < 0:
                self.inf_values += 1
                out.append(INF)
            else:
                out.append(float(v))
        self.n += len(states)
        self.calls += 1
        return out

    def metrics(self):
        return {"states_scored": self.n, "batches": self.calls, "infinite_values": self.inf_values}

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
        if self.tmp is not None:
            try:
                os.unlink(self.tmp.name)
            except OSError:
                pass
            self.tmp = None


class NeuralEval(Evaluator):
    def __init__(self, name, model, device):
        self.name, self.model, self.device = name, model, device
        self.n = self.calls = 0
        self.infer_seconds = 0.0

    def prepare(self, task, ctx, budget):
        super().prepare(task, ctx, budget)
        self.n = self.calls = 0
        self.infer_seconds = 0.0
        self.template = task.template()                       # graph template of the grounded task (counted in the problem's wall clock)
        budget.check()

    def evaluate(self, states, budget):
        import torch
        out = []
        for i in range(0, len(states), NEURAL_CHUNK):
            budget.check()
            t0 = time.perf_counter()
            chunk = states[i:i + NEURAL_CHUNK]
            v = self.model.state_values(self.template, self.task, chunk, None, chunk=NEURAL_CHUNK)
            vals = v.double().cpu().tolist()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.infer_seconds += time.perf_counter() - t0
            self.calls += 1
            self.n += len(chunk)
            for x in vals:
                if not math.isfinite(x):
                    raise ModelNonFinite(str(x))
            out.extend(vals)
        return out

    def metrics(self):
        return {"states_scored": self.n, "graph_encodings": self.n, "forward_calls": self.calls, "inference_seconds": round(self.infer_seconds, 3)}


# ------------------------------------------------------------------------------------------------ WL-GOOSE
ATOM_RE = re.compile(r"^Atom ([^\s(]+)\((.*)\)$")


def translate_to_sas(domain, problem, out_sas):
    """Fast Downward (Scorpion fork of the author software) translation: the same call the author's data creator uses to decide which atoms exist."""
    cmd = [str(GOOSE_PY), str(SCORPION_FD), "--sas-file", str(out_sas), "--translate", str(domain), str(problem)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_sas).is_file():
        raise RuntimeError("translation failed: " + r.stdout[-300:] + r.stderr[-300:])


def parse_sas(path):
    """-> dict(variables=[[fact name, ...] per variable], init=[value per variable], goals=[(var, val)])."""
    lines = Path(path).read_text().splitlines()
    i, variables = 0, []
    while lines[i] != "begin_variable" and i < len(lines):
        i += 1
    while lines[i] == "begin_variable":
        rng = int(lines[i + 3])
        variables.append(lines[i + 4:i + 4 + rng])
        i += 4 + rng + 1
    while lines[i] != "begin_state":
        i += 1
    init = [int(lines[i + 1 + k]) for k in range(len(variables))]
    while lines[i] != "begin_goal":
        i += 1
    ng = int(lines[i + 1])
    goals = [tuple(map(int, lines[i + 2 + k].split())) for k in range(ng)]
    return {"variables": variables, "init": init, "goals": goals, "lines": lines}


def sas_with_init(sas, values):
    """The SAS text with the initial state replaced by ``values`` (one value index per variable)."""
    lines = list(sas["lines"])
    i = lines.index("begin_state")
    for k, v in enumerate(values):
        lines[i + 1 + k] = str(v)
    return "\n".join(lines) + "\n"


def cpp_round(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


class WlEval(Evaluator):
    name = "H_WL"

    def __init__(self, params_path):
        import sys
        if str(WL_SHIM) not in sys.path:
            sys.path.append(str(WL_SHIM))
        import json
        from wlplan.feature_generator import load_feature_generator
        from wlplan.planning import Atom, Domain, Predicate, Problem, State
        self.W = dict(Atom=Atom, Domain=Domain, Predicate=Predicate, Problem=Problem, State=State)
        self.params_path = str(params_path)
        self.fg = load_feature_generator(self.params_path)
        dom = json.loads(Path(self.params_path).read_text())["domain"]
        self.preds = {n: Predicate(n, a) for n, a in dom["predicates"]}
        self.domain = Domain("domain", list(self.preds.values()), [], [], [])
        self.n = self.negative = 0
        self.sas = None

    def prepare(self, task, ctx, budget):
        super().prepare(task, ctx, budget)
        W = self.W
        self.n = self.negative = 0
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            sas_path = Path(d) / "output.sas"
            translate_to_sas(ctx["domain"], ctx["problem"], sas_path)
            sas = parse_sas(sas_path)
        self.translate_seconds = time.perf_counter() - t0
        self.sas = sas
        budget.check()
        by_bit, objects, self.unmapped = {}, set(), 0
        self.var_value_bit = []                                # per variable: [bit index or None per value] (None: not an Atom / not in the model domain)
        for names in sas["variables"]:
            row = []
            for nm in names:
                m = ATOM_RE.match(nm)
                if not m:
                    row.append(None)
                    continue
                pred = m.group(1)
                args = tuple(x.strip() for x in m.group(2).split(",")) if m.group(2).strip() else ()
                objects.update(args)
                bit = task.dyn_index.get((pred, args))
                if bit is None:
                    row.append(None)
                    if pred in self.preds:
                        self.unmapped += 1                     # translation atom that our grounding does not know (diagnostic, must stay 0)
                    continue
                if pred in self.preds:
                    by_bit[bit] = W["Atom"](self.preds[pred], list(args))
                row.append(bit)
            self.var_value_bit.append(row)
        self.atom_of_bit = by_bit
        self.kept_mask = sum(1 << b for b in by_bit)
        goals = []
        for var, val in sas["goals"]:
            m = ATOM_RE.match(sas["variables"][var][val])
            if m and m.group(1) in self.preds:
                args = tuple(x.strip() for x in m.group(2).split(",")) if m.group(2).strip() else ()
                goals.append(W["Atom"](self.preds[m.group(1)], list(args)))
        self.problem = W["Problem"](self.domain, sorted(objects), goals, [])
        self.fg.set_problem(self.problem)

    def raw(self, state):
        atoms = [self.atom_of_bit[b] for b in bits(state & self.kept_mask)]
        return self.fg.predict(self.W["State"](atoms))

    def evaluate(self, states, budget):
        budget.check()
        out = []
        for s in states:
            x = self.raw(s)
            if not math.isfinite(x):
                raise ModelNonFinite(str(x))
            h = cpp_round(x)
            if h < 0:
                self.negative += 1
            out.append(float(h))
        self.n += len(states)
        return out

    def sas_values(self, state):
        """FDR assignment (one value per variable) of a STRIPS state, for the native cross-check."""
        vals = []
        for row in self.var_value_bit:
            hit = [i for i, b in enumerate(row) if b is not None and (state >> b) & 1]
            if len(hit) > 1:
                raise ValueError("mutex violated")
            if hit:
                vals.append(hit[0])
            else:
                none = [i for i, nm in enumerate(self.sas["variables"][len(vals)]) if nm == "<none of those>" or nm.startswith("NegatedAtom ")]
                if not none:
                    raise ValueError("no value for a variable")
                vals.append(none[0])
        return vals

    def metrics(self):
        return {"states_scored": self.n, "translate_seconds": round(getattr(self, "translate_seconds", 0.0), 3), "negative_values": self.negative, "unmapped_translation_atoms": getattr(self, "unmapped", 0)}


def native_initial_h(sas_text, params_path, timeout=60):
    """h value of the INITIAL state of a SAS task by the native Scorpion planner with the author's wlgoose heuristic (cross-check oracle)."""
    cmd = [str(SCORPION_DOWNWARD), "--search", 'eager_greedy([wlgoose(model_file="%s")])' % params_path]
    env = dict(os.environ, LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run(cmd, input=sas_text, capture_output=True, text=True, timeout=timeout, env=env)
    m = re.search(r"Initial heuristic value for wlgoose: (-?\d+)", r.stdout)
    if not m:
        raise RuntimeError("native planner gave no initial value: " + r.stdout[-300:])
    return int(m.group(1))

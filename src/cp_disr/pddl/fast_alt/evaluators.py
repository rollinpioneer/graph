"""Neural scorers of the FAST / ALT card: the reference scorer (unchanged ``NeuralEval`` plus call counters) and the FAST scorer (same V, cheaper implementation)."""
from __future__ import annotations

import math
import time

import torch

from ..search_match import evaluators as EV
from ..search_match.engine import ModelNonFinite
from .fast_value import FastValue


def encoder_calls(model, E, n_states):
    """Number of encoder forwards the reference splits ``n_states`` into (edge budget of ``PddlSerialModel.zv``)."""
    ch = max(1, type(model).EDGE_BUDGET // max(E, 1))
    return -(-n_states // ch)


class RefNeuralEval(EV.NeuralEval):
    """The previous card's scorer, bit for bit; only the counters of physical encoder calls are added."""

    def prepare(self, task, ctx, budget):
        super().prepare(task, ctx, budget)
        self.E = len(self.template.edges)
        self.encoder_forwards = 0

    def evaluate(self, states, budget):
        for i in range(0, len(states), EV.NEURAL_CHUNK):
            self.encoder_forwards += encoder_calls(self.model, self._edges_of_static(), min(EV.NEURAL_CHUNK, len(states) - i))
        return super().evaluate(states, budget)

    def _edges_of_static(self):
        st = self.model.mg.goal_free_static(self.template)[0]
        return int(st.ei.shape[1])

    def metrics(self):
        m = super().metrics()
        m.update({"outer_physical_calls": m.get("forward_calls"), "encoder_forwards": self.encoder_forwards})
        return m


class FastNeuralEval(EV.Evaluator):
    def __init__(self, name, model, device):
        self.name, self.model, self.device = name, model, device
        self.fv = FastValue(model, device)
        self.n = self.calls = self.encoder_forwards = 0
        self.infer_seconds = 0.0
        self.prepare_seconds = 0.0

    def prepare(self, task, ctx, budget):
        super().prepare(task, ctx, budget)
        self.n = self.calls = self.encoder_forwards = 0
        self.infer_seconds = 0.0
        self.template = task.template()
        self.prepare_seconds = self.fv.prepare(task, self.template)
        budget.check()

    def evaluate(self, states, budget):
        out = []
        for i in range(0, len(states), EV.NEURAL_CHUNK):
            budget.check()
            t0 = time.perf_counter()
            chunk = states[i:i + EV.NEURAL_CHUNK]
            vals = self.fv.values(chunk)
            self.infer_seconds += time.perf_counter() - t0
            self.calls += 1
            self.n += len(chunk)
            self.encoder_forwards += encoder_calls(self.model, self.fv.E, len(chunk))
            for x in vals:
                if not math.isfinite(x):
                    raise ModelNonFinite(str(x))
            out.extend(vals)
        return out

    def metrics(self):
        return {"states_scored": self.n, "graph_encodings": self.n, "forward_calls": self.calls, "outer_physical_calls": self.calls, "encoder_forwards": self.encoder_forwards,
                "inference_seconds": round(self.infer_seconds, 3), "prepare_seconds": round(self.prepare_seconds, 3), "cache_bytes": self.fv.cache_bytes}


class TracingEval(EV.Evaluator):
    """Transparent wrapper recording the sequence of evaluated batches (generation order) of any scorer; used by the equivalence fixtures."""

    def __init__(self, inner):
        self.inner, self.batches = inner, []
        self.name = getattr(inner, "name", "?")

    def prepare(self, task, ctx, budget):
        self.batches = []
        self.inner.prepare(task, ctx, budget)

    def evaluate(self, states, budget):
        self.batches.append(list(states))
        return self.inner.evaluate(states, budget)

    def metrics(self):
        return self.inner.metrics()

    def close(self):
        self.inner.close()

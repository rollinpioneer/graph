"""Additional mainline methods. Not a reimplementation of frozen V6_CAP_POTENTIAL."""
from __future__ import annotations

def n_subgoals(task: str) -> int:
    return 2 if task == "dual_order" else 1

def valid_count(state: dict) -> int:
    if state["task"] == "dual_order":
        return int(bool(state["objects"]["A"]["valid"])) + int(bool(state["objects"]["B"]["valid"]))
    return int(bool(state["objects"]["obj"]["valid"]) or state.get("success") is True)

def sparse_terminal(prev: dict, nxt: dict) -> float:
    if nxt.get("success") is True and prev.get("success") is not True:
        return 1.0
    if nxt.get("terminal_failure") is True and prev.get("terminal_failure") is not True:
        return -1.0
    return 0.0

class MatchedEvents:
    def __init__(self):
        self.seen = set()
        self.open = set()
    def step(self, events, n: int) -> float:
        a = 1.0 / n
        r = 0.0
        for e in events or []:
            oid, lid, kind = e.get("object_id"), e.get("loss_id"), e.get("kind")
            if not oid or not lid:
                continue
            key = (oid, lid)
            if kind == "LOSS":
                if key not in self.seen:
                    self.seen.add(key)
                    self.open.add(key)
                    r -= a
            elif kind == "HOLD_REESTABLISHED":
                if key in self.open:
                    self.open.remove(key)
                    r += a
        return r

def event_plus_count(prev, nxt, bank: MatchedEvents) -> tuple[float, float, float]:
    n = n_subgoals(prev["task"])
    dv = (valid_count(nxt) - valid_count(prev)) / n
    ev = bank.step(nxt.get("events"), n)
    return dv + ev, dv, ev

def unordered_valid_count(prev, nxt) -> float:
    n = n_subgoals(prev["task"])
    return (valid_count(nxt) - valid_count(prev)) / n

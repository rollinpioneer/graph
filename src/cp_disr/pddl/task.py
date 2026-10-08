"""Grounded STRIPS tasks from PDDL, the production CP-DISR graph template for them, a snapshot-producing environment, and the exact-distance oracle (small tasks).

The graph is built from the same relational objects as the Blocksworld card: one ACTION node per grounded action (a ``SkillContract`` with PRE_POS / PRE_NEG / ADD / DEL edges, plus reverse edges) and one
PROPOSITION node per atom; goal atoms carry a +1 mark. Node features are (schema / predicate name, leaf object types, truth) only: no object name, no hash, no planner value.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..common import digest
from ..contracts import Atom, Effects, SkillContract, TypedArgument
from ..facts import FactRecord, FactStore, Truth
from ..graph import GraphTemplate, Goal, Node, reverse_edges
from ..rl import Snapshot
from .parse import parse_domain, parse_problem

DEPOTS_TYPE_MAP = {"depot": "place", "distributor": "place"}
DEPOTS_TYPE_PREDS = ("crate", "pallet", "truck", "hoist", "place", "depot", "distributor", "surface", "locatable")      # untyped IPC encoding: static unary facts that only declare a type
DEPOTS_TYPE_ORDER = ("crate", "pallet", "truck", "hoist", "depot", "distributor", "place")
OBS_DIM, CAND_DIM = 48, 8


@dataclass(frozen=True)
class GroundAction:
    index: int
    schema: str
    args: tuple
    aid: str
    pre: tuple
    neg: tuple
    add: tuple
    dele: tuple
    static_pre: tuple              # static non-type atoms (pred, args) that the action requires (always true)
    pre_mask: int = 0
    neg_mask: int = 0
    add_mask: int = 0
    del_mask: int = 0


def atom_id(pred, args):
    return "p:" + pred + (":" + ":".join(args) if args else "")


class StripsTask:
    """Reachable-relaxation grounding of a (typed or untyped) STRIPS domain / problem pair. State = integer bit mask over the dynamic atoms that can ever become true."""

    def __init__(self, domain_text, problem_text, type_map=None, type_preds=(), type_order=(), name=None):
        self.domain = parse_domain(domain_text)
        self.problem = parse_problem(problem_text)
        self.name = name or self.problem.name
        d, p = self.domain, self.problem
        self.type_preds = set(type_preds) if not d.typed else set()
        eff_preds = {a[0] for act in d.actions.values() for a in act.add + act.delete}
        self.dyn_preds = eff_preds
        init_static = {a for a in p.init if a[0] not in eff_preds}
        init_dyn = {a for a in p.init if a[0] in eff_preds}
        # leaf object types (never the object name)
        tm = type_map or {}
        self.obj_type = {}
        for o, t in p.objects.items():
            if d.typed:
                self.obj_type[o] = tm.get(t, t)
            else:
                lt = next((tp for tp in type_order if (tp, (o,)) in init_static), "object")
                self.obj_type[o] = tm.get(lt, lt)
        self.static = {a for a in init_static if a[0] not in self.type_preds}
        static_all = init_static
        pools = {}

        def pool(t):
            if t not in pools:
                pools[t] = tuple(sorted(o for o, ot in p.objects.items() if d.subtype_of(p.objects[o], t))) if d.typed else tuple(sorted(p.objects))
            return pools[t]
        R = set(init_dyn)
        grounded = {}
        plans = {}
        for act in d.actions.values():
            vars_ = [v for v, _t in act.params]
            order = {v: i for i, v in enumerate(vars_)}
            check = [[] for _ in vars_]
            for pred, args in act.pre_pos:
                last = max((order[v] for v in args), default=0)
                check[last].append((pred, args))
            plans[act.name] = (vars_, [pool(t) for _v, t in act.params], check, order)

        def bindings(act):
            vars_, pls, check, order = plans[act.name]
            n = len(vars_)
            binding = []

            def rec(i):
                if i == n:
                    yield tuple(binding)
                    return
                for obj in pls[i]:
                    binding.append(obj)
                    ok = True
                    for pred, args in check[i]:
                        atom = (pred, tuple(binding[order[v]] for v in args))
                        if (atom not in R) if pred in eff_preds else (atom not in static_all):
                            ok = False
                            break
                    if ok:
                        yield from rec(i + 1)
                    binding.pop()
            return rec(0)
        changed = True
        while changed:
            changed = False
            for act in d.actions.values():
                for b in bindings(act):
                    key = (act.name, b)
                    if key in grounded:
                        continue
                    grounded[key] = act
                    sub = dict(zip((v for v, _t in act.params), b))
                    for pred, args in act.add:
                        atom = (pred, tuple(sub[v] for v in args))
                        if atom not in R:
                            R.add(atom)
                            changed = True
        self.dyn_atoms = tuple(sorted(R))
        self.dyn_index = {a: i for i, a in enumerate(self.dyn_atoms)}
        goal_dyn = [g for g in p.goal if g[0] in eff_preds]
        self.goal_static_ok = all(g in static_all for g in p.goal if g[0] not in eff_preds)
        self.solvable_by_relaxation = self.goal_static_ok and all(g in R for g in goal_dyn)
        acts, dropped = [], 0
        for (name, b), act in grounded.items():
            sub = dict(zip((v for v, _t in act.params), b))
            g = lambda atoms: tuple((pred, tuple(sub[v] for v in args)) for pred, args in atoms)
            pre = g(act.pre_pos)
            neg = g(act.pre_neg)
            add = g(act.add)
            dele = tuple(x for x in g(act.delete) if x not in set(g(act.add)))
            pre_idx = tuple(sorted(self.dyn_index[a] for a in pre if a[0] in eff_preds))
            neg_idx = tuple(sorted(self.dyn_index[a] for a in neg if a in self.dyn_index))
            add_idx = tuple(sorted(self.dyn_index[a] for a in add))
            del_idx = tuple(sorted(self.dyn_index[a] for a in dele if a in self.dyn_index))
            if set(add_idx) <= set(pre_idx) and not del_idx:
                dropped += 1                                               # no-op in every state (e.g. Drive from a place to itself)
                continue
            static_pre = tuple(a for a in pre if a[0] not in eff_preds and a[0] not in self.type_preds)
            aid = "a:" + name + ":" + ":".join(b) + ":v1"
            acts.append((aid, name, b, pre_idx, neg_idx, add_idx, del_idx, static_pre))
        acts.sort()
        mk = lambda idxs: sum(1 << i for i in idxs)
        self.actions = tuple(GroundAction(i, nm, b, aid, pre, neg, add, dl, sp, mk(pre), mk(neg), mk(add), mk(dl)) for i, (aid, nm, b, pre, neg, add, dl, sp) in enumerate(acts))
        self.noop_actions_dropped = dropped
        self.action_by_id = {a.aid: a for a in self.actions}
        self.init_mask = mk(self.dyn_index[a] for a in init_dyn)
        self.goal_mask = mk(self.dyn_index[a] for a in goal_dyn if a in self.dyn_index)
        self.goal_atoms = tuple(g for g in p.goal)
        self._template = None

    # ---- semantics
    def legal(self, state):
        return [a for a in self.actions if (state & a.pre_mask) == a.pre_mask and not (state & a.neg_mask)]

    @staticmethod
    def apply(state, a):
        return (state & ~a.del_mask) | a.add_mask

    def goal_satisfied(self, state):
        return self.solvable_by_relaxation and (state & self.goal_mask) == self.goal_mask

    def unmet_goals(self, state):
        return bin(self.goal_mask & ~state).count("1")

    def state_list(self, state):
        return [i for i in range(len(self.dyn_atoms)) if (state >> i) & 1]

    @staticmethod
    def state_from_list(lst):
        return sum(1 << int(i) for i in lst)

    def atom_true_ids(self, state):
        return {atom_id(*self.dyn_atoms[i]) for i in range(len(self.dyn_atoms)) if (state >> i) & 1}

    def replay(self, plan_ids, state=None):
        """Execute a list of action ids; returns the final state or None if a step is illegal."""
        s = self.init_mask if state is None else state
        for aid in plan_ids:
            a = self.action_by_id.get(aid)
            if a is None or (s & a.pre_mask) != a.pre_mask or (s & a.neg_mask):
                return None
            s = self.apply(s, a)
        return s

    # ---- graph template
    def template(self):
        if self._template is None:
            self._template = build_task_template(self)
        return self._template

    # ---- exact-solver export
    def export_text(self):
        lines = ["%d %d %d" % (len(self.actions), len(self.dyn_atoms), bin(self.goal_mask).count("1"))]
        lines.append("init: %d %s" % (bin(self.init_mask).count("1"), " ".join(map(str, self.state_list(self.init_mask)))))
        lines.append("goal: %d %s" % (bin(self.goal_mask).count("1"), " ".join(map(str, self.state_list(self.goal_mask)))))
        for a in self.actions:
            parts = []
            for grp in (a.pre, a.neg, a.add, a.dele):
                parts.append("%d %s" % (len(grp), " ".join(map(str, grp))))
            lines.append(" ".join(parts))
        return "\n".join(lines) + "\n"


def build_task_template(task):
    """GraphTemplate whose node argument types are the LEAF object types of the grounded arguments."""
    otype = task.obj_type
    contracts, nodes, edges = [], [], []
    atoms = {}
    for a in task.actions:
        def at(pred_args):
            return Atom(pred_args[0], tuple(pred_args[1]))
        pre_pos = tuple(Atom(task.dyn_atoms[i][0], task.dyn_atoms[i][1]) for i in a.pre) + tuple(at(x) for x in a.static_pre)
        pre_neg = tuple(Atom(task.dyn_atoms[i][0], task.dyn_atoms[i][1]) for i in a.neg)
        add = tuple(Atom(task.dyn_atoms[i][0], task.dyn_atoms[i][1]) for i in a.add)
        dele = tuple(Atom(task.dyn_atoms[i][0], task.dyn_atoms[i][1]) for i in a.dele)
        c = SkillContract(name=a.schema, arguments=tuple(TypedArgument("x%d" % k, otype[o]) for k, o in enumerate(a.args)), pre_pos=pre_pos, pre_neg=pre_neg, effects=Effects(add, dele, ()), version="pddl-v1",
                          provenance="pddl-domain", timeout_seconds=1.0, bound_arguments=a.args)
        assert c.id == a.aid, (c.id, a.aid)
        contracts.append(c)
        nodes.append(Node(c.id, "ACTION", c.name, c.bound_arguments, tuple(x.type for x in c.arguments)))
        for atom in pre_pos:
            edges.append((atom.id, c.id, "PRE_POS"))
        for atom in pre_neg:
            edges.append((atom.id, c.id, "PRE_NEG"))
        for atom in add:
            edges.append((c.id, atom.id, "ADD"))
        for atom in dele:
            edges.append((c.id, atom.id, "DEL"))
        for atom in pre_pos + pre_neg + add + dele:
            atoms[atom.id] = atom
    for pred, args in task.dyn_atoms:
        atoms[atom_id(pred, args)] = Atom(pred, args)
    goals = tuple(Goal(atom_id(*g), 1) for g in task.goal_atoms if g in task.dyn_index)
    for a in atoms.values():
        nodes.append(Node(a.id, "PROPOSITION", a.predicate, a.arguments, tuple(otype[o] for o in a.arguments)))
    contracts = tuple(sorted(contracts, key=lambda c: c.id))
    return GraphTemplate(tuple(sorted(nodes, key=lambda n: n.id)), reverse_edges(edges), goals, contracts, (), ())


# ------------------------------------------------------------------------------------------------ environment
def base_input(step_index, step_cap, previous_ok):
    row = [0.0] * OBS_DIM
    row[0] = step_index / step_cap
    row[1] = max(0, step_cap - step_index) / step_cap
    row[2] = 1.0 if previous_ok else 0.0
    return tuple(row)


def candidate_feature(schema, schemas, arity):
    row = [0.0] * CAND_DIM
    row[schemas.index(schema)] = 1.0
    row[len(schemas)] = arity / 4.0
    return tuple(row)


@dataclass(frozen=True)
class PddlCase:
    case_id: str
    split: str
    task: object
    optimal_length: int                 # exact optimum, or the best known plan length (see ``length_kind``)
    step_cap: int
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def step_cap_for(length):
    return 2 * length + 4


_FEATS = {}


def _candidate_features(task):
    hit = _FEATS.get(id(task))
    if hit is not None and hit[0] is task:
        return hit[1]
    t = task.template()
    ids = tuple(c.id for c in sorted(t.contracts, key=lambda c: c.id))
    schemas = tuple(sorted({a.schema for a in task.actions}))
    by = {a.aid: a for a in task.actions}
    feats = tuple(candidate_feature(by[i].schema, SCHEMA_ORDER, len(by[i].args)) for i in ids)
    _FEATS[id(task)] = (task, (ids, feats))
    return ids, feats


SCHEMA_ORDER = ("drive", "lift", "drop", "load", "unload")


class PddlEpisode:
    def __init__(self, case, env_id="pddl-env-0", episode_id="ep-1"):
        self.case, self.task = case, case.task
        self.env_id, self.episode_id = env_id, episode_id
        self.template = self.task.template()
        self.state = self.task.init_mask
        self.step_index, self.previous_ok, self.done, self.success, self.reason = 0, False, False, False, "CONTINUE"
        self._ids, self._feats = _candidate_features(self.task)
        self._static_true = {atom_id(p, a) for p, a in self.task.static}
        self._prop_ids = tuple(n.id for n in self.template.nodes if n.kind == "PROPOSITION")

    def legal_ids(self, state=None):
        return {a.aid for a in self.task.legal(self.state if state is None else state)}

    def facts_for(self, state):
        true = self.task.atom_true_ids(state) | self._static_true
        return {k: (k in true) for k in self._prop_ids}

    def snapshot(self, state=None, step_index=None):
        state = self.state if state is None else state
        facts = self.facts_for(state)
        records = tuple(FactRecord(k, Truth.TRUE if v else Truth.FALSE, 0.0, 0.0) for k, v in facts.items())
        legal = self.legal_ids(state)
        mask = tuple(i in legal for i in self._ids)
        si = self.step_index if step_index is None else step_index
        return Snapshot(self.env_id, self.episode_id, si, self.template, FactStore(records), self._ids, mask, (), digest(()), base_input(si, self.case.step_cap, self.previous_ok or si > 0), self._feats,
                        "pddl:%s:%d" % (self.episode_id, si), float(si))

    def step(self, action_id):
        if self.done:
            raise RuntimeError("episode finished")
        a = self.task.action_by_id[action_id]
        if (self.state & a.pre_mask) != a.pre_mask or (self.state & a.neg_mask):
            raise RuntimeError("masked action executed: %s" % action_id)
        self.state = self.task.apply(self.state, a)
        self.step_index += 1
        self.previous_ok = True
        if self.task.goal_satisfied(self.state):
            self.done, self.success, self.reason = True, True, "TASK_SUCCESS"
        elif self.step_index >= self.case.step_cap:
            self.done, self.reason = True, "DEADLINE"
        return self.reason


def snapshot_at(case, state_list, t):
    ep = PddlEpisode(case)
    ep.step_index = t
    return ep.snapshot(StripsTask.state_from_list(state_list), t)


def codes_for_state_general(st, task, state, static_true):
    """(P,3) TRUE/FALSE one-hot per template proposition for a state bit mask (same arithmetic as the snapshot path)."""
    import torch
    true = task.atom_true_ids(state) | static_true
    return torch.tensor([[1.0, 0.0, 0.0] if p in true else [0.0, 1.0, 0.0] for p in st.prop_ids], device=st.prop_pos.device)


# ------------------------------------------------------------------------------------------------ exact oracle
EXACT_BIN = Path(os.environ.get("CPDISR_EXACT_BIN", str(Path.home() / "ext" / "bin" / "exact_dist")))


class ExactOracle:
    """Complete reachable state space + exact unit-cost distances of one task (subprocess; closed explicitly)."""

    def __init__(self, task, max_states=30_000_000, timeout=600):
        self.task, self.ok = task, False
        self._tmp = tempfile.NamedTemporaryFile("w", suffix=".task", delete=False)
        self._tmp.write(task.export_text())
        self._tmp.close()
        self.proc = subprocess.Popen([str(EXACT_BIN), self._tmp.name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send("build %d" % max_states)
        r = self.proc.stdout.readline().split()
        if r and r[0] == "OK":
            self.ok, self.n_states, self.n_goal_states, self.max_dist = True, int(r[1]), int(r[2]), int(r[3])
        else:
            self.n_states = int(r[1]) if len(r) > 1 else -1

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def query(self, state):
        """(d*(state) or -1, [(action index, d*(successor))...]) for a reachable state bit mask."""
        lst = self.task.state_list(state)
        self._send("q %d %s" % (len(lst), " ".join(map(str, lst))))
        r = list(map(int, self.proc.stdout.readline().split()))
        d, m = r[0], r[1]
        return d, [(r[2 + 2 * i], r[3 + 2 * i]) for i in range(m)]

    def plan(self):
        self._send("plan")
        line = self.proc.stdout.readline().strip()
        if line == "NOPLAN":
            return None
        return [self.task.actions[int(x)].aid for x in line.split()]

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            try:
                os.unlink(self._tmp.name)
            except OSError:
                pass


def load_task(domain_path, problem_path, **kw):
    return StripsTask(Path(domain_path).read_text(), Path(problem_path).read_text(), **kw)


def depots_task(domain_path, problem_path):
    """Depots with the project's normalisation: leaf types {crate, pallet, truck, hoist, place}."""
    text = Path(domain_path).read_text()
    return StripsTask(text, Path(problem_path).read_text(), type_map=DEPOTS_TYPE_MAP, type_preds=DEPOTS_TYPE_PREDS, type_order=DEPOTS_TYPE_ORDER)

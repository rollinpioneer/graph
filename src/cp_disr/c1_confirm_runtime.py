"""CP-DISR-C1-MECH-CONFIRM-V1 evaluation runtime: one real episode loop for neural checkpoints and B_PLAN with decision-level evidence.

Everything the policy sees and everything the environment does is the frozen T_B runtime (the shared ``Collector.step`` executes, verifies and evaluates every
decision for every method). The reference optimal-first-action set, completed-goal tracking and divergence labels are post-hoc analysis fields computed from the
public facts after the policy has chosen; they are never an input to any policy. No optimizer, no provider, no hidden truth to any policy or planner.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from . import c1_fresh_confirm
from . import struct_gen as G
from .baselines.b_plan import BPlanPlanner, SearchConfig
from .facts import Truth

CARD = "CP-DISR-C1-MECH-CONFIRM-V1"
TRAIN_BINDING_PLACEMENTS = ("a:PLACE:target:container:v1", "a:PLACE_BUFFER:second_object:buffer:v1")
PLACEMENTS = {"a:PLACE:target:container:v1": "target->container", "a:PLACE_BUFFER:second_object:buffer:v1": "second_object->buffer",
              "a:PLACE:second_object:container:v1": "second_object->container", "a:PLACE_BUFFER:target:buffer:v1": "target->buffer"}
PLANNER_DEPTH_LIMIT = 8
PLANNER_MAX_NODES = 4096
PLANNER_CPU_SECONDS = 2.0
REFERENCE_SKILL_SECONDS = 4.2


def _facts_json(snapshot):
    return {k: (v.value if isinstance(v, Truth) else str(v)) for k, v in sorted(snapshot.facts.values.items())}


def _float(x):
    x = float(x)
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------------------- policy-like objects for the shared Collector
class TimedPolicy:
    """Read-only wrapper: delegates every attribute, records encoder forward calls and wall time of each forward call."""

    def __init__(self, policy):
        self._policy = policy
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def __call__(self, snapshot, hidden):
        enc = getattr(self._policy, "encoder", None)
        before = enc.forward_calls if enc is not None else 0
        t0 = time.perf_counter()
        out = self._policy(snapshot, hidden)
        dt = time.perf_counter() - t0
        self.calls.append({"decision_id": snapshot.decision_id, "episode_id": snapshot.episode_id, "encoder_forward_calls": (enc.forward_calls - before) if enc is not None else 0, "seconds": dt})
        return out


class PlannerPolicy:
    """B_PLAN as a Collector-compatible policy: plans from public facts / goal / registered contracts / remaining deadline only, returns its first action."""

    def __init__(self, bundle, config=None):
        import torch
        self.torch = torch
        self.bundle = bundle
        self.config = config or SearchConfig(depth_limit=PLANNER_DEPTH_LIMIT, max_nodes=PLANNER_MAX_NODES, cpu_time_limit_seconds=PLANNER_CPU_SECONDS,
                                             reference_skill_seconds=REFERENCE_SKILL_SECONDS)
        self.planner = BPlanPlanner(self.config)
        self.cache = {}
        self.unique_calls = []

    def initial_hidden(self):
        return self.torch.zeros(1)

    def advance_hidden(self, base_input, hidden):
        return hidden

    def eval(self):
        return self

    def remaining(self):
        return float(self.bundle.evaluator.deadline) - (float(self.bundle.clock.now_seconds()) - float(self.bundle.episode_start_seconds))

    def plan_for(self, snapshot):
        key = (snapshot.env_id, snapshot.episode_id, snapshot.decision_id)
        if key not in self.cache:
            result = self.planner.plan(snapshot.facts, snapshot.template, max(0.0, self.remaining()))
            self.cache[key] = result
            self.unique_calls.append({"key": list(key), **result.as_dict()})
        return self.cache[key]

    def __call__(self, snapshot, hidden):
        from .neural import PolicyOutput
        torch = self.torch
        result = self.plan_for(snapshot)
        n = len(snapshot.candidate_ids)
        mask = torch.tensor(snapshot.mask, dtype=torch.bool)
        logits = torch.full((n,), -torch.inf)
        planned = result.plan[0] if result.status == "PLAN_FOUND" and result.plan else None
        if planned is not None and planned in snapshot.candidate_ids and bool(snapshot.mask[snapshot.candidate_ids.index(planned)]):
            logits[snapshot.candidate_ids.index(planned)] = 0.0
        if not bool(torch.isfinite(logits).any()):
            return PolicyOutput(snapshot.candidate_ids, torch.zeros(n), mask, None, torch.zeros(()), torch.zeros(n), hidden, {"successor_used": False, "method": "B_PLAN"}, "NO_PLANNED_ACTION")
        return PolicyOutput(snapshot.candidate_ids, logits, mask, torch.distributions.Categorical(logits=logits), torch.zeros(()), torch.zeros(n), hidden,
                            {"successor_used": False, "method": "B_PLAN", "planner": result.as_dict()})


# ----------------------------------------------------------------------------- post-hoc references
def reference_first_actions(snapshot, goal_atoms, max_depth=PLANNER_DEPTH_LIMIT):
    """Shortest legal nominal plans from the CURRENT PUBLIC facts: (status, first actions, depth). Never reads hidden state."""
    facts0 = {k: v for k, v in snapshot.facts.values.items()}
    res = G.solve(list(snapshot.template.contracts), facts0, tuple(goal_atoms), max_depth)
    if res["depth"] is None:
        return "REFERENCE_UNAVAILABLE_PUBLIC_UNKNOWN", None, None
    if res["depth"] == 0:
        return "GOAL_ALREADY_TRUE_PUBLIC", [], 0
    start = G._key(facts0)
    return "AVAILABLE", sorted(res["optimal_actions"].get(start, [])), res["depth"]


def goal_atoms_true(bundle, goal_atoms):
    """Evaluator-level (hidden-truth) truth of each goal atom with the frozen atomic checks. Logging only."""
    ev = bundle.evaluator
    h = ev.env.hidden_truth()
    role = ev._second_role()
    out = {}
    for atom in goal_atoms:
        parts = atom.split(":")
        obj = "target" if parts[2] == "target" else role
        out[atom] = bool(ev._inside(h, obj) if parts[1] == "Inside" else ev._at_buffer(h, obj))
    return out


def divergence_type(selected, optimal, goal_atoms):
    if selected in TRAIN_BINDING_PLACEMENTS:
        return "PLACES_TRAIN_DEFAULT_BINDING"
    if selected in PLACEMENTS:
        return "PLACES_OTHER_BINDING_NOT_OPTIMAL"
    if ":PICK:" in selected:
        return "PICK_NOT_OPTIMAL"
    if ":OPEN:" in selected:
        return "OPEN_NOT_OPTIMAL"
    return "OTHER"


# ----------------------------------------------------------------------------- episode loop
def run_episode(root, v11, bundle, collector, policy, method, case_row, planner=False):
    goal_atoms = tuple(G.GOAL_SETS[case_row["goal_key"]])
    snap = bundle.start_case(case_row["case_id"])
    snap, prior, source_n = v11.apply_episode_prior("B2" if planner else method, bundle, snap, sampler=None, eval_original=True)
    collector.reset_episode(snap.env_id, snap.episode_id)
    done_goals = goal_atoms_true(bundle, goal_atoms)
    decisions, selected_seen = [], []
    G_ret, steps, success, reason, success_seconds = 0.0, 0, False, None, None
    t_start = float(bundle.clock.now_seconds())
    first_div = None
    while True:
        facts_before = _facts_json(snap)
        candidates, mask = list(snap.candidate_ids), [bool(m) for m in snap.mask]
        ref_status, ref_actions, ref_depth = reference_first_actions(snap, goal_atoms)
        completed_before = goal_atoms_true(bundle, goal_atoms)
        remaining_before = float(bundle.evaluator.deadline) - (float(bundle.clock.now_seconds()) - float(bundle.episode_start_seconds))
        collector.last_output = None
        collector.last_execution = None
        if planner:
            policy.plan_for(snap)
        ncalls = len(policy.calls) if isinstance(policy, TimedPolicy) else 0
        wall0 = time.perf_counter()
        t, result = collector.step(snap, deterministic=True)
        wall = time.perf_counter() - wall0
        out = collector.last_output
        rec = {"decision_index": len(decisions), "decision_id": snap.decision_id, "facts_before": facts_before, "goal_atoms": list(goal_atoms), "candidate_ids": candidates, "mask": mask,
               "reference_status": ref_status, "optimal_first_actions_public": ref_actions, "reference_depth_public": ref_depth, "completed_goals_before": completed_before,
               "remaining_seconds_before": remaining_before, "step_wall_seconds": wall}
        if out is not None and out.distribution is not None and not planner:
            rec["logits"] = [None if not math.isfinite(float(x)) else float(x) for x in out.logits.detach().cpu().tolist()]
            rec["probs"] = [float(x) for x in out.distribution.probs.detach().cpu().tolist()]
        if planner:
            pr = policy.plan_for(snap)
            rec["planner"] = pr.as_dict()
        if isinstance(policy, TimedPolicy) and len(policy.calls) > ncalls:
            call = policy.calls[ncalls]
            rec["encoder_forward_calls"] = call["encoder_forward_calls"]
            rec["forward_seconds"] = call["seconds"]
        if t is None:
            reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
            success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
            rec.update(selected=None, no_transition=True, termination_reason=reason, success=success, sim_seconds_after=float(bundle.clock.now_seconds()) - float(bundle.episode_start_seconds))
            decisions.append(rec)
            break
        selected = t.selected_candidate_id
        raw = dict(getattr(bundle.executor, "last", None) or {})
        done_after = goal_atoms_true(bundle, goal_atoms)
        destroyed = [a for a in goal_atoms if completed_before.get(a) and not done_after.get(a)]
        repeated = selected in selected_seen
        diverged = ref_status == "AVAILABLE" and bool(ref_actions) and selected not in ref_actions
        rec.update(selected=selected, selected_index=candidates.index(selected), controller_exit=raw.get("controller_exit"), sim_duration=raw.get("sim_duration"), duration=float(t.duration),
                   facts_after=_facts_json(t.next_snapshot), evaluator_reason=(result.reason if not isinstance(result, dict) else result.get("reason")),
                   evaluator_success=bool(result.success if not isinstance(result, dict) else result.get("success")), completed_goals_after=done_after, destroyed_completed_goals=destroyed,
                   repeated_action=repeated, diverged_from_public_optimal=bool(diverged), divergence_type=divergence_type(selected, ref_actions, goal_atoms) if diverged else None,
                   placement_binding=PLACEMENTS.get(selected), terminated=bool(t.terminated), truncated=bool(t.truncated), sim_seconds_after=float(bundle.clock.now_seconds()) - float(bundle.episode_start_seconds))
        if diverged and first_div is None:
            first_div = {"index": len(decisions), "selected": selected, "optimal_first_actions_public": ref_actions, "facts_before": facts_before, "completed_goals_before": completed_before,
                         "type": rec["divergence_type"], "placement_binding": PLACEMENTS.get(selected)}
        selected_seen.append(selected)
        decisions.append(rec)
        G_ret += float(t.reward) * float(t.weight)
        steps += 1
        if rec["evaluator_success"]:
            success = True
            success_seconds = float(t.duration)
        snap = t.next_snapshot
        if t.terminated or t.truncated:
            reason = rec["evaluator_reason"]
            break
    ended_reason = str(reason)
    return {"case_id": case_row["case_id"], "cell": case_row["cell"], "goal_key": case_row["goal_key"], "method": method, "success": bool(success), "reason": reason, "steps": steps, "G": G_ret,
            "success_seconds": success_seconds if success else None, "elapsed_seconds": float(bundle.clock.now_seconds()) - float(bundle.episode_start_seconds), "n_decisions": len(decisions),
            "first_divergence": first_div, "repeated_action_count": sum(1 for d in decisions if d.get("repeated_action")),
            "destroyed_completed_goal_count": sum(len(d.get("destroyed_completed_goals") or []) for d in decisions), "reference_unavailable_decisions": sum(1 for d in decisions if d["reference_status"] != "AVAILABLE"),
            "ended_with_no_candidate_safe_termination": ended_reason.startswith("NO_CANDIDATE_SAFE_TERMINATION"),
            "no_candidate_after_earlier_divergence": bool(ended_reason.startswith("NO_CANDIDATE_SAFE_TERMINATION") and first_div is not None),
            "decisions": decisions, "prior_mode": prior.audit_mode, "source_n": source_n}


def run_cases(root, v11, method, rows, device, out_jsonl, checkpoint=None, label="c1", model_seed=None, log=print):
    """One formal evaluation per (model, case): completed cases in out_jsonl are never re-run (resume-after-crash only)."""
    from .collector import Collector
    from .torch_rl import load_checkpoint
    c1_fresh_confirm.install_goal_sets()          # the BUF_T goal set must exist before the bundle wrapper builds one template per goal set
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["case_id"])
    task = "T_B"
    rng_before = v11.s1.capture_rng()
    split_rel = v11.ENABLED_SPLITS[task]
    bundle = v11.make_bundle(root, task, split_rel)
    prof = v11.resolve_runtime(root)
    split_index = {r["case_id"]: r for r in rows}
    v11.attach_split_cases(bundle, root, list(split_index.values()), task, prof["task_deadlines"][task])
    v11.s1.seed_all(0)
    planner = method == "B_PLAN"
    if planner:
        policy = PlannerPolicy(bundle)
    else:
        raw = v11.make_policy(bundle.template, method, device)
        load_checkpoint(checkpoint, raw)
        raw.eval()
        policy = TimedPolicy(raw)
    collector = Collector(bundle, policy)
    results = []
    try:
        for row in rows:
            if row["case_id"] in done:
                continue
            v11.require_case_cache(root, row)
            t0 = time.time()
            ep = run_episode(root, v11, bundle, collector, policy, method, row, planner=planner)
            ep.update(label=label, model_seed=model_seed, checkpoint=str(checkpoint) if checkpoint else None, wall_seconds=time.time() - t0)
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep, sort_keys=True, default=str) + "\n")
            results.append(ep)
            log("[c1-eval] %s %s %s success=%s reason=%s steps=%d" % (method, row["case_id"], row["cell"], ep["success"], ep["reason"], ep["steps"]))
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
        v11.s1.restore_rng(rng_before)
    return results

"""Label pool for the grounded-task experiments: exact optimal plans, optimal action sets and exact successor costs for small tasks (ExactOracle), fixed-rule training trajectories, and exact
'requires goal destruction' labels. Labels are used ONLY in losses and tables, never in a forward pass.

Trajectory rule (fixed, no model): the canonical optimal plan; plus, at 0 / 25 / 50 / 75 % of the plan, one deviation = the legal non-optimal action chosen by a hash of (case id, step) among the actions
sorted by id, followed by the canonical optimal continuation from the deviated state.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .task import ExactOracle, StripsTask, depots_task, step_cap_for

DEV_FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def rank_key(case_id, state_list):
    return "%s|%s" % (case_id, ",".join(map(str, state_list)))


def _label(task, oracle, s):
    d, succ = oracle.query(s)
    opt = [task.actions[a].aid for a, dd in succ if dd == d - 1]
    dist = {task.actions[a].aid: dd for a, dd in succ}
    return d, opt, dist, succ


def _follow_optimal(task, oracle, s):
    """Canonical optimal continuation (lowest action index among optimal actions) from state s: list of (state, action index)."""
    out = []
    while True:
        d, succ = oracle.query(s)
        if d <= 0:
            break
        a = min(a for a, dd in succ if dd == d - 1)
        out.append((s, a))
        s = task.apply(s, task.actions[a])
    return out, s


def build_case_data(domain_path, problem_path, case_id, max_states=30_000_000, type_kw=None, light=False):
    t0 = time.time()
    task = depots_task(domain_path, problem_path)
    oracle = ExactOracle(task, max_states)
    if not oracle.ok:
        oracle.close()
        return {"case_id": case_id, "status": "TOO_BIG", "states_seen": oracle.n_states}
    try:
        d0, succ0 = oracle.query(task.init_mask)
        plan = oracle.plan()
        cap = step_cap_for(d0)
        if light:                                                                   # evaluation-only problems: exact optimum, plan and destruction label, no training trajectories
            return {"case_id": case_id, "status": "OK", "optimal_length": d0, "plan": plan, "n_states": oracle.n_states, "n_goal_states": oracle.n_goal_states, "max_dist": oracle.max_dist, "n_actions": len(task.actions),
                    "n_dyn_atoms": len(task.dyn_atoms), "n_optimal_first_actions": sum(1 for a, dd in succ0 if dd == d0 - 1), "trajectories": [], "rank_labels": {}, "requires_goal_destruction": requires_destruction(task, oracle),
                    "seconds": round(time.time() - t0, 2), "light": True}
        labels, trajs = {}, []

        def traj_from(prefix_states, prefix_actions, source):
            """prefix_states: masks s_0..s_k, prefix_actions: indices a_0..a_{k-1}; the canonical optimal continuation is appended."""
            cont, _final = _follow_optimal(task, oracle, prefix_states[-1])
            states, actions = list(prefix_states), list(prefix_actions)
            for st_, a in cont:                                         # states[i] is the state before actions[i]; the last state is the goal
                actions.append(a)
                states.append(task.apply(st_, task.actions[a]))
            ids, astar, rem = [], [], []
            for st_, a in zip(states[:-1], actions):
                d, opt, dist, _s = _label(task, oracle, st_)
                labels[rank_key(case_id, task.state_list(st_))] = dist
                astar.append(opt)
                rem.append(d)
                ids.append(task.actions[a].aid)
            return {"tid": "%s:%s" % (source, case_id), "source": source, "case_id": case_id, "n_blocks": len(task.problem.objects), "step_cap": cap, "states": [task.state_list(x) for x in states],
                    "actions": ids, "astar": astar, "remaining": rem, "success": bool(task.goal_satisfied(states[-1]))}
        # the canonical optimal plan as a trajectory
        s = task.init_mask
        pre_states, pre_actions = [s], []
        for aid in plan:
            a = task.action_by_id[aid]
            pre_actions.append(a.index)
            s = task.apply(s, a)
            pre_states.append(s)
        trajs.append(traj_from(pre_states[:1], [], "expert"))
        # one deviation at each fixed fraction of the plan
        seen_t = set()
        for fr in DEV_FRACTIONS:
            t = int(fr * len(plan))
            if t in seen_t or t >= len(plan):
                continue
            seen_t.add(t)
            st_ = pre_states[t]
            d, succ = oracle.query(st_)
            cands = sorted((task.actions[a].aid, a) for a, dd in succ if dd != d - 1 and dd >= 0)
            if not cands:
                continue
            k = int(hashlib.sha1(("%s|%d" % (case_id, t)).encode()).hexdigest(), 16) % len(cands)
            a = cands[k][1]
            nxt = task.apply(st_, task.actions[a])
            trajs.append(traj_from(pre_states[:t + 1] + [nxt], pre_actions[:t] + [a], "dev%d" % t))
        res = {"case_id": case_id, "status": "OK", "optimal_length": d0, "plan": plan, "n_states": oracle.n_states, "n_goal_states": oracle.n_goal_states, "max_dist": oracle.max_dist,
               "n_actions": len(task.actions), "n_dyn_atoms": len(task.dyn_atoms), "n_optimal_first_actions": sum(1 for a, dd in succ0 if dd == d0 - 1), "trajectories": trajs, "rank_labels": labels,
               "requires_goal_destruction": requires_destruction(task, oracle), "seconds": round(time.time() - t0, 2)}
        return res
    finally:
        oracle.close()


def requires_destruction(task, oracle):
    """True when EVERY optimal plan, at some step, makes false a goal atom that was true before; False when some optimal plan never does."""
    gm = task.goal_mask
    memo = {}

    def ok(s):
        """exists an optimal continuation from s that never destroys a true goal atom"""
        if s in memo:
            return memo[s]
        d, succ = oracle.query(s)
        if d == 0:
            memo[s] = True
            return True
        res = False
        for a, dd in succ:
            if dd != d - 1:
                continue
            t = task.apply(s, task.actions[a])
            if (s & gm) & ~(t & gm):
                continue
            if ok(t):
                res = True
                break
        memo[s] = res
        return res
    return not ok(task.init_mask)


# ------------------------------------------------------------------------------------------------ Fast Downward reference
FD = Path.home() / "ext" / "downward" / "fast-downward.py"


def run_fd(domain_path, problem_path, workdir, alias, time_limit, memory_mb=8000, tag=None):
    """One Fast Downward run; returns dict(status, best_length, lengths_by_time, wall, search_time, peak_kb). LAMA aliases write plan.N files (anytime)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plan = workdir / ("%s.plan" % (tag or alias))
    t0 = time.time()
    cmd = ["python3", str(FD), "--alias", alias, "--plan-file", str(plan), "--overall-time-limit", str(time_limit), "--overall-memory-limit", "%dM" % memory_mb,
           "--sas-file", str(workdir / ("%s.sas" % (tag or alias))), str(domain_path), str(problem_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 120, cwd=str(workdir))
        out = r.stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    wall = time.time() - t0
    lengths = []
    for line in out.splitlines():
        if "Plan length:" in line:
            try:
                tt = float(line.split("t=")[1].split("s")[0])
                lengths.append((tt, int(line.split("Plan length:")[1].split()[0])))
            except (IndexError, ValueError):
                pass
    plans = sorted(workdir.glob("%s.plan*" % (tag or alias)))
    best = None
    best_file = None
    for pf in plans:
        n = sum(1 for l in pf.read_text().splitlines() if l.startswith("("))
        if best is None or n < best:
            best, best_file = n, pf
    solved = "Solution found." in out
    exhausted = "Search stopped without finding a solution." in out or "Completely explored state space" in out
    peak = 0
    for line in out.splitlines():
        if line.startswith("Peak memory:"):
            try:
                peak = max(peak, int(line.split()[2]))
            except (IndexError, ValueError):
                pass
    return {"alias": alias, "solved": bool(solved or best is not None), "best_length": best, "best_plan_file": str(best_file) if best_file else None, "lengths_by_time": lengths, "wall_seconds": round(wall, 2),
            "time_limit": time_limit, "peak_kb": peak, "unsolvable_proved": bool(exhausted)}


def plan_ids_from_file(path):
    return ["a:" + l.strip()[1:-1].split()[0] + ":" + ":".join(l.strip()[1:-1].split()[1:]) + ":v1" for l in Path(path).read_text().splitlines() if l.startswith("(")]

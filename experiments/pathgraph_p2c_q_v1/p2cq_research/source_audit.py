"""AST plus fresh-process runtime audits. String grep is not sufficient."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


FORBID_ENV = ("potentials", "graph_model", "oracle", "stable_baselines", "gymnasium", "mujoco")
FORBID_GRAPH = ("SkillEnv.step",)


def _parse(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def imported_names(path):
    tree = _parse(path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def scan_tree(root):
    root = Path(root)
    files = sorted(root.joinpath("p2cq_research").glob("*.py"))
    rows = []
    for f in files:
        imps = imported_names(f)
        rows.append({"file": str(f.relative_to(root)), "imports": imps})
    env_imps = imported_names(root / "p2cq_research/environment.py")
    graph_src = (root / "p2cq_research/graph_model.py").read_text(encoding="utf-8")
    env_ok = not any(any(bad in (m or "") for bad in FORBID_ENV) for m in env_imps)
    graph_ok = "from .environment" not in graph_src and "SkillEnv" not in graph_src
    return {
        "files": rows,
        "environment_imports_clean": env_ok,
        "graph_does_not_import_env": graph_ok,
        "env_imports": env_imps,
    }


def runtime_clone_and_parity(tmp):
    code = r'''
import json, hashlib, sys
sys.path.insert(0, sys.argv[1])
from p2cq_research.generator import pair_contracts
from p2cq_research.environment import SkillEnv
from p2cq_research.observations import observation_digest, public_input_digest
from p2cq_research.task_contract import action_index
left, right = pair_contracts("development", "PRECEDENCE", 0)
e1 = SkillEnv(left); e1.reset()
snap = e1.snapshot()
parent = e1.state.fingerprint()
e1.step(action_index(0, "ACQUIRE"))
child = e1.state.fingerprint()
e1.restore(snap)
assert e1.state.fingerprint() == parent, "parent mutated"
e2 = SkillEnv(left); e2.reset()
e2b = SkillEnv(left); e2b.reset()
assert observation_digest(left, e2) == observation_digest(left, e2b)
# reconstruct
e3 = SkillEnv(left); e3.reset()
d1 = observation_digest(left, e3)
e4 = SkillEnv(left); e4.reset()
assert d1 == observation_digest(left, e4)
# same encoder both methods: no method arg
assert public_input_digest(left, e3) == public_input_digest(left, e4)
# different tasks different public hash
er = SkillEnv(right); er.reset()
assert public_input_digest(left, e3) != public_input_digest(right, er)
print("RUNTIME_AUDIT_PASS")
'''
    root = Path(tmp)
    exp = str(root)
    r = subprocess.run(
        [sys.executable, "-B", "-c", code, exp],
        cwd=str(root),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    return {
        "exit": r.returncode,
        "stdout": r.stdout[-2000:],
        "stderr": r.stderr[-2000:],
        "passed": r.returncode == 0 and "RUNTIME_AUDIT_PASS" in r.stdout,
    }


def brute_oracle_small():
    """Independent path enumeration vs DP for n=2 PRECEDENCE."""
    from .environment import SkillEnv
    from .generator import pair_contracts
    from .enumerate_mdp import enumerate_mdp
    left, _ = pair_contracts("development", "PRECEDENCE", 0)
    # shrink horizon for brute
    d = left.public_dict()
    d["horizon"] = 8
    from .task_contract import TaskContract
    c = TaskContract(d)
    mdp, _ = enumerate_mdp(c)
    # brute: BFS of action sequences of length <=8 looking for success
    env = SkillEnv(c)
    best = {}

    def rec(depth):
        key = (env.state.fingerprint(), depth)
        if key in best:
            return best[key]
        if c.goal_satisfied(env.state.valid):
            best[key] = 1.0
            return 1.0
        if depth <= 0:
            best[key] = 0.0
            return 0.0
        snap = env.snapshot()
        v = 0.0
        for a in range(37):
            env.restore(snap)
            _, r, term, _, _ = env.step(a)
            if r:
                val = 1.0
            elif term:
                val = 0.0
            else:
                val = c.gamma * rec(depth - 1)
            if val > v:
                v = val
        env.restore(snap)
        best[key] = v
        return v

    env.reset()
    brute = rec(c.horizon)

    def dp(sid, h, memo):
        recs = mdp["states"][sid]
        if h <= 0 or recs["terminal"]:
            return 0.0
        key = (sid, h)
        if key in memo:
            return memo[key]
        best = 0.0
        gamma = mdp["gamma"]
        for outcomes in recs["outcomes"]:
            q = 0.0
            for o in outcomes:
                q += o["p"] * (o["task_reward"] + gamma * dp(o["next"], h - 1, memo))
            if q > best:
                best = q
        memo[key] = best
        return best

    dp_v = dp(mdp["initial_state"], c.horizon, {})
    ok = abs(brute - dp_v) <= 1e-9
    return {
        "brute_value": brute,
        "dp_value": dp_v,
        "horizon": c.horizon,
        "n_states": mdp["meta"]["n_states"],
        "initial": mdp["initial_state"],
        "matched": ok,
    }


def run_all(exp_root):
    ast_rep = scan_tree(exp_root)
    rt = runtime_clone_and_parity(exp_root)
    small = brute_oracle_small()
    return {
        "ast": ast_rep,
        "runtime": rt,
        "small_oracle": small,
        "passed": ast_rep["environment_imports_clean"] and ast_rep["graph_does_not_import_env"] and rt["passed"] and bool(small.get("matched")),
    }

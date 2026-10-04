"""CP-DISR-TP-BEHAVIOR-STATIC-1: static eligibility scan of BEHAVIOR-1K (BDDL3 + official symbolic primitives + transition rules).

Research-only. No simulator, physics, rendering, provider, VLM, LLM or RL. The pinned monorepo (v3.9.3-post1 / bd049de) is read as source. Stage A uses the OFFICIAL
BDDL3 parser, Conditions and recipe loaders; primitive and rule inventories are AST extractions of the official source. Anything that would need the official
primitives or rules to EXECUTE (Stages C-H) is only attempted if the official API can run without OmniGibson; otherwise the card stops with ENGINEERING_BLOCKED.
Nothing here re-implements primitive or transition semantics.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CARD = "CP-DISR-TP-BEHAVIOR-STATIC-1"
TAG = "v3.9.3-post1"
COMMIT_PREFIX = "bd049de"
COMMIT_FULL = "bd049de3119acdcdf2334fe9e1ebe060fa20c108"
REMOTE = "https://github.com/StanfordVL/BEHAVIOR-1K.git"
DOMAIN = "behavior-1k"
GO_THRESHOLDS = {"eligible_states": 20, "activity_families": 5, "discrete_consequence_states": 5,
                 "note": "project thresholds declared before the scan; they are not lowered after results"}
EXPECTED_PRIMITIVES = ["GRASP", "PLACE_ON_TOP", "PLACE_INSIDE", "OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF", "SOAK_UNDER", "SOAK_INSIDE", "WIPE", "CUT", "PLACE_NEAR_HEATING_ELEMENT", "NAVIGATE_TO", "RELEASE"]
LOGIC_HEADS = {"and", "or", "not", "imply", "forall", "exists", "forn", "forpairs", "fornpairs"}
PRIMITIVE_FILE = "OmniGibson/omnigibson/action_primitives/symbolic_semantic_action_primitives.py"
STARTER_FILE = "OmniGibson/omnigibson/action_primitives/starter_semantic_action_primitives.py"
RULE_FILE = "OmniGibson/omnigibson/transition_rules.py"
BDDL_UTILS_FILE = "OmniGibson/omnigibson/utils/bddl_utils.py"
RESOURCE_KEYS = ("simulator_launches", "physics_steps", "rendering", "provider_requests", "vlm_or_llm_calls", "rl_transitions", "optimizer_steps", "asset_downloads")
CACHE_CAPS = {"planner_nodes": 20000, "planner_depth": 4, "planner_seconds": 30}
STAGES = ["A1_bddl3_parse", "A2_primitive_inventory", "A3_transition_inventory", "A4_symbolic_coverage", "B_canonical_state", "C_legal_actions", "D_official_successor_and_closure", "E_remaining_plan_J", "F_pair_gates",
          "G_shortcut_audit", "H_full_vs_astat_identifiability"]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def digest(v):
    return sha_bytes(json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode())


def tree_hash(base, patterns=("*",)):
    base = Path(base)
    rows = sorted((p.relative_to(base).as_posix(), sha_file(p)) for pat in patterns for p in base.rglob(pat) if p.is_file() and "__pycache__" not in p.parts)
    return {"files": len(rows), "sha256": digest(rows)}


def git_out(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------------------------------------ source preflight
def source_manifest(repo):
    repo = Path(repo)
    head = git_out(["rev-parse", "HEAD"], repo)
    tag = git_out(["describe", "--tags", "--exact-match"], repo)
    m = {"remote_url": git_out(["remote", "get-url", "origin"], repo), "tag": tag, "commit": head, "expected_tag": TAG, "expected_commit_prefix": COMMIT_PREFIX, "tag_matches": tag == TAG,
         "commit_matches": head.startswith(COMMIT_PREFIX) and head == COMMIT_FULL, "dirty_tracked_files": len([l for l in git_out(["status", "--porcelain", "--untracked-files=no"], repo).splitlines() if l.strip()]),
         "commit_date": git_out(["log", "-1", "--format=%cI"], repo), "clone_mode": "shallow (--depth 1) of the tag, git-lfs smudge disabled",
         "hashes": {"bddl3_readme": sha_file(repo / "bddl3/README.md"), "primitive_source": sha_file(repo / PRIMITIVE_FILE), "starter_primitive_source": sha_file(repo / STARTER_FILE), "omnigibson_transition_rules": sha_file(repo / RULE_FILE),
                    "bddl_transition_rules": sha_file(repo / "bddl3/bddl/transition_rules.py"), "transition_rule_recipes_json": sha_file(repo / "bddl3/bddl/generated_data/transition_rule_recipes.json"),
                    "activity_manifest": sha_file(repo / "bddl3/bddl/activity_manifest.txt"), "license": sha_file(repo / "LICENSE"), "bddl_utils": sha_file(repo / BDDL_UTILS_FILE)},
         "trees": {"activity_definitions": tree_hash(repo / "bddl3/bddl/activity_definitions", ("*.bddl",)), "transition_map_jsons": tree_hash(repo / "bddl3/bddl/generated_data/transition_map/tm_jsons", ("*.json",))}}
    m["status"] = "PASS" if m["tag_matches"] and m["commit_matches"] and m["dirty_tracked_files"] == 0 else "VERSION_BASELINE_UNAVAILABLE"
    return m


def license_preflight(repo):
    repo = Path(repo)
    L = ["# Licence preflight (read-only)", "", "Code licence file(s) found at depth <= 3:", ""]
    for p in sorted(repo.rglob("LICENSE*")):
        if len(p.relative_to(repo).parts) <= 3 and p.is_file():
            L.append("- `%s` (%s): %s" % (p.relative_to(repo), sha_file(p)[:12], p.read_text(errors="ignore").splitlines()[0] if p.read_text(errors="ignore").strip() else "(empty)"))
    L += ["", "Mentions of dataset / asset terms in docs and READMEs (first lines only):", ""]
    hits = 0
    for p in list(repo.glob("README.md")) + list((repo / "docs").rglob("*.md")) + list((repo / "datasets").rglob("*.md")):
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if re.search(r"licen[sc]e|terms of use|CC[- ]BY|non-?commercial|dataset", line, re.I) and hits < 25:
                L.append("- `%s:%d`: %s" % (p.relative_to(repo), i, line.strip()[:160]))
                hits += 1
    L += ["", "No dataset or scene asset was downloaded: the static BDDL3 parse reads text files that ship in the repository, and git-lfs smudge was disabled for the clone.", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- A2 primitives (AST)
def _attr_chain(n):
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return list(reversed(parts))


def _functions(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            out.setdefault(node.name, node)
    return out


def function_facts(fn):
    reads, writes, raises, placement_predicates = set(), [], set(), set()
    counts = Counter()
    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "states":
            ch = _attr_chain(n.slice)
            if ch and ch[0] == "object_states":
                reads.add(ch[-1])
            counts["state_subscripts"] += 1
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "set_value":
            tgt = None
            v = n.func.value
            if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Attribute) and v.value.attr == "states":
                ch = _attr_chain(v.slice)
                tgt = ch[-1] if ch and ch[0] == "object_states" else None
            val = None
            if n.args:
                last = n.args[-1]
                val = last.value if isinstance(last, ast.Constant) else "param"
            if tgt:
                writes.append({"state": tgt, "value": val})
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call):
            for a in ast.walk(n.exc):
                ch = _attr_chain(a) if isinstance(a, ast.Attribute) else []
                if len(ch) >= 3 and ch[-3:-1] == ["ActionPrimitiveError", "Reason"]:
                    raises.add(ch[-1])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_place_with_predicate":
            for a in n.args:
                ch = _attr_chain(a)
                if ch and ch[0] == "object_states":
                    placement_predicates.add(ch[-1])
        if isinstance(n, ast.Attribute):
            if n.attr in ("env", "scene", "system_registry"):
                counts["sim_attribute_refs"] += 1
        if isinstance(n, ast.Name) and n.id in ("og", "th"):
            counts["engine_name_refs"] += 1
    return {"args": [a.arg for a in fn.args.args], "object_states_read": sorted(reads), "set_value_writes": writes, "error_reasons": sorted(raises), "placement_predicates": sorted(placement_predicates),
            "sim_object_references": dict(counts), "lineno": fn.lineno}


def merged_facts(fn, fnmap, depth=3):
    """function_facts of a primitive plus the facts of the self.* helpers it calls (the state writes live in helpers such as _open_or_close)."""
    base = function_facts(fn)
    seen = {fn.name}
    stack = [(fn, depth)]
    while stack:
        f, d = stack.pop()
        if d == 0:
            continue
        for n in ast.walk(f):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "self":
                name = n.func.attr
                if name in fnmap and name not in seen:
                    seen.add(name)
                    h = function_facts(fnmap[name])
                    for k in ("object_states_read", "error_reasons", "placement_predicates"):
                        base[k] = sorted(set(base[k]) | set(h[k]))
                    base["set_value_writes"] = base["set_value_writes"] + h["set_value_writes"]
                    for k, v in h["sim_object_references"].items():
                        base["sim_object_references"][k] = base["sim_object_references"].get(k, 0) + v
                    stack.append((fnmap[name], d - 1))
    base["helpers_followed"] = sorted(seen - {fn.name})
    return base


def extract_primitives(sym_src, starter_src):
    tree, st_tree = ast.parse(sym_src), ast.parse(starter_src)
    enum_members = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("PrimitiveSet"):
            for b in node.body:
                if isinstance(b, ast.Assign) and len(b.targets) == 1 and isinstance(b.targets[0], ast.Name) and b.targets[0].id.isupper():
                    doc = None
                    v = b.value
                    if isinstance(v, ast.Tuple) and len(v.elts) >= 2 and isinstance(v.elts[1], ast.Constant):
                        doc = v.elts[1].value
                    enum_members.append({"name": b.targets[0].id, "doc": doc})
            break
    mapping = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "controller_functions" for t in node.targets) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                kc, vc = _attr_chain(k), _attr_chain(v)
                if kc and vc:
                    mapping[kc[-1]] = vc[-1]
    fns, st_fns = _functions(tree), _functions(st_tree)
    prims = {}
    for m in enum_members:
        fname = mapping.get(m["name"])
        fn, where = (fns.get(fname), "symbolic") if fname in fns else (st_fns.get(fname), "starter_base_class") if fname in st_fns else (None, None)
        prims[m["name"]] = dict(m, implementation=fname, defined_in=where, facts=merged_facts(fn, {**st_fns, **fns}) if fn else None)
    names = [m["name"] for m in enum_members]
    diff = {"expected_not_in_source": sorted(set(EXPECTED_PRIMITIVES) - set(names)), "in_source_not_expected": sorted(set(names) - set(EXPECTED_PRIMITIVES)), "identical": names == EXPECTED_PRIMITIVES or set(names) == set(EXPECTED_PRIMITIVES)}
    return {"primitives": prims, "controller_function_mapping": mapping, "diff_against_card_expectation": diff}


def primitive_writes(inv):
    """(state, value) pairs the official primitives can set, from set_value calls and placement predicates (AST-derived)."""
    out = set()
    for p in inv["primitives"].values():
        f = p.get("facts")
        if not f:
            continue
        for w in f["set_value_writes"]:
            out.add((w["state"], w["value"]))
        for s in f["placement_predicates"]:
            out.add((s, True))
    # a helper that receives the state as a parameter (open/close, toggle) is expanded to both polarities by the AST facts above ("param")
    return out


# ------------------------------------------------------------------------------------------------ A3 transitions
def extract_rule_classes(src):
    tree = ast.parse(src)
    imports = sorted({a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module})
    classes = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name.endswith("Rule"):
            classes.append({"name": n.name, "bases": [_attr_chain(b)[-1] if _attr_chain(b) else "?" for b in n.bases], "lineno": n.lineno, "doc": (ast.get_docstring(n) or "").splitlines()[0] if ast.get_docstring(n) else None})
    return {"top_level_imports": imports, "imports_omnigibson": "omnigibson" in imports, "imports_torch": "torch" in imports, "rule_classes": classes}


def recipe_inventory():
    from bddl import transition_rules as TR

    def conv(v):
        if hasattr(v, "predicate") and hasattr(v, "value"):
            return [v.predicate.__name__, v.value]
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    out = {}
    out_states = defaultdict(set)
    for fn_name in ("load_cooking_recipes", "load_machine_recipes", "load_mixing_recipes", "load_substance_cooking_recipes", "load_washer_rule"):
        try:
            res = getattr(TR, fn_name)()
            items = res if isinstance(res, (list, tuple)) else (list(res.values()) if isinstance(res, dict) else [res])
            rows = []
            for r in items:
                d = {k: conv(v) for k, v in vars(r).items()}
                rows.append(d)
                os_ = d.get("output_states") or {}
                if isinstance(os_, dict):
                    for syn, conds in os_.items():
                        for c in (conds or []):
                            if isinstance(c, list) and len(c) == 2:
                                out_states[c[0]].add(bool(c[1]))
            out[fn_name] = {"count": len(rows), "fields": sorted({k for r in rows for k in r}), "sample": rows[:2], "status": "OK"}
        except Exception as e:  # noqa: BLE001
            out[fn_name] = {"status": "ERROR", "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    return out, {k: sorted(v) for k, v in out_states.items()}


# ------------------------------------------------------------------------------------------------ A1 tasks (official parser)
def flatten_goal(node, polarity=True, acc=None, ops=None, unknown=None, token_ok=None):
    acc = acc if acc is not None else []
    ops = ops if ops is not None else []
    unknown = unknown if unknown is not None else []
    if isinstance(node, list) and node and isinstance(node[0], str):
        head = node[0].lower()
        if head in LOGIC_HEADS:
            ops.append(head)
            pol = (not polarity) if head == "not" else polarity
            for c in node[1:]:
                if isinstance(c, list):
                    flatten_goal(c, pol, acc, ops, unknown, token_ok)
        elif token_ok(head):
            acc.append((head, polarity))
        elif head.startswith("?") or head.isdigit():
            pass
        else:
            unknown.append(head)
            for c in node[1:]:
                if isinstance(c, list):
                    flatten_goal(c, polarity, acc, ops, unknown, token_ok)
    elif isinstance(node, list):
        for c in node:
            flatten_goal(c, polarity, acc, ops, unknown, token_ok)
    return acc, ops, unknown


def scan_tasks(bddl_root):
    sys.path.insert(0, str(Path(bddl_root)))
    for m in [k for k in sys.modules if k == "bddl" or k.startswith("bddl.")]:
        del sys.modules[m]
    from bddl import activity as A
    from bddl import predicates as PR
    token_ok = lambda t: t in PR.TOKEN_TO_PREDICATE  # noqa: E731
    rows, errors = [], []
    for act in sorted(A.get_all_activities()):
        try:
            n = A.get_instance_count(act)
        except Exception as e:  # noqa: BLE001
            errors.append({"activity": act, "stage": "instance_count", "error": str(e)[:200]})
            continue
        for i in range(n):
            row = {"activity": act, "instance": i, "parse_ok": False}
            try:
                conds = A.Conditions(act, i, DOMAIN)
                row["parse_ok"] = True
                objs = conds.parsed_objects
                row["objects"] = {k: len(v) for k, v in sorted(objs.items())}
                row["n_objects"] = sum(len(v) for v in objs.values())
                row["synsets"] = sorted(objs)
                init = conds.parsed_initial_conditions
                row["init_tokens"] = dict(Counter(a[0] for a in init if isinstance(a, list) and a))
                row["n_init"] = len(init)
                leaves, ops, unknown = flatten_goal(conds.parsed_goal_conditions, True, token_ok=token_ok)
                row["goal_leaves"] = sorted({"%s%s" % ("" if pol else "not ", t) for t, pol in leaves})
                row["goal_tokens"] = sorted({t for t, _ in leaves})
                row["goal_ops"] = sorted(set(ops))
                row["goal_unknown_heads"] = sorted(set(unknown))
                row["has_agent"] = any(s.startswith("agent.") for s in objs)
                try:
                    scope = A.get_object_scope(conds)
                    A.get_initial_conditions(conds, scope, generate_ground_options=False)
                    A.get_goal_conditions(conds, scope, generate_ground_options=False)
                    row["compile_ok"] = True
                except Exception as e:  # noqa: BLE001
                    row["compile_ok"] = False
                    row["compile_error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
                row["_parsed"] = (objs, init, conds.parsed_goal_conditions)
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
                errors.append({"activity": act, "instance": i, "stage": "parse", "error": row["error"]})
            rows.append(row)
    return rows, errors, sorted(PR.TOKEN_TO_PREDICATE)


# ------------------------------------------------------------------------------------------------ A4 coverage screen
def predicate_to_state_map(src):
    """The official PREDICATE_TO_STATE dict (BDDL predicate class -> OmniGibson object state), read from source."""
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PREDICATE_TO_STATE" for t in n.targets) and isinstance(n.value, ast.Dict):
            return {_attr_chain(k)[-1]: _attr_chain(v)[-1] for k, v in zip(n.value.keys, n.value.values)}
    return {}


def token_class_name(token):
    return "".join(p.capitalize() for p in token.split("_")) if token not in ("on_fire", "toggled_on", "nextto", "insource", "inroom", "ontop") else {"on_fire": "OnFire", "toggled_on": "ToggledOn", "nextto": "NextTo", "insource": "InSource", "inroom": "InRoom", "ontop": "OnTop"}[token]


def static_paths(token_polarities, pred2state, writes, rule_states):
    """For each goal literal: PRIMITIVE_WRITABLE / RULE_OUTPUT_STATE / NO_STATIC_PATH / UNMAPPED. A necessary-condition screen, NOT an execution result."""
    out = {}
    for tok, pol in token_polarities:
        cls = token_class_name(tok)
        state = pred2state.get(cls)
        if state is None:
            out[(tok, pol)] = "UNMAPPED"
            continue
        if any(s == state and (v == pol or v == "param") for s, v in writes):
            out[(tok, pol)] = "PRIMITIVE_WRITABLE"
        elif pol in rule_states.get(cls, []):
            out[(tok, pol)] = "RULE_OUTPUT_STATE"
        else:
            out[(tok, pol)] = "NO_STATIC_PATH"
    return out


def coverage_for(row, pred2state, writes, rule_states):
    if not row["parse_ok"]:
        return "UNKNOWN", "parse failure", {}
    leaves = [(l.replace("not ", ""), not l.startswith("not ")) for l in row["goal_leaves"]]
    if not leaves or row.get("goal_unknown_heads"):
        return "UNKNOWN", "no recognised goal literal or an unknown goal head", {}
    sp = static_paths(leaves, pred2state, writes, rule_states)
    vals = set(sp.values())
    detail = {"%s%s" % ("" if p else "not ", t): v for (t, p), v in sp.items()}
    if "UNMAPPED" in vals:
        return "UNKNOWN", "a goal literal has no official predicate-to-state mapping", detail
    ok = {"PRIMITIVE_WRITABLE", "RULE_OUTPUT_STATE"}
    if vals <= ok:
        return "FULL", "every goal literal has an official primitive write or a recipe output state", detail
    if vals & ok:
        return "PARTIAL", "some goal literals have no static official path", detail
    return "UNSUPPORTED", "no goal literal has a static official path", detail


# ------------------------------------------------------------------------------------------------ B canonical state
def canon_goal(g):
    if isinstance(g, list):
        kids = [canon_goal(c) for c in g]
        if kids and isinstance(g[0], str) and g[0].lower() in ("and", "or"):
            return [g[0].lower()] + sorted(kids[1:], key=lambda x: json.dumps(x, sort_keys=True))
        return kids
    return g


def canonical_state(objects, init, goal):
    objs = sorted((inst, syn) for syn, insts in objects.items() for inst in insts)
    facts = sorted([list(map(str, a)) for a in init if isinstance(a, list) and a])
    g = canon_goal(goal)
    return {"objects": objs, "init_facts": facts, "goal": g}, digest({"objects": objs, "init_facts": facts}), digest(g)


def permutation_invariance(rows, seed=20261004):
    rng = random.Random(seed)
    bad = 0
    for r in rows:
        if "_parsed" not in r:
            continue
        objs, init, goal = r["_parsed"]
        _, h1, g1 = canonical_state(objs, init, goal)
        items = list(objs.items())
        rng.shuffle(items)
        objs2 = {k: rng.sample(v, len(v)) for k, v in items}
        init2 = rng.sample(init, len(init))

        def shuf(x):
            if isinstance(x, list) and x and isinstance(x[0], str) and x[0].lower() in ("and", "or"):
                kids = [shuf(c) for c in x[1:]]
                rng.shuffle(kids)
                return [x[0]] + kids
            return [shuf(c) for c in x] if isinstance(x, list) else x
        _, h2, g2 = canonical_state(objs2, init2, shuf(goal))
        if (h1, g1) != (h2, g2):
            bad += 1
    return {"instances_tested": sum(1 for r in rows if "_parsed" in r), "mismatches": bad, "seed": seed}


# ------------------------------------------------------------------------------------------------ official API probe
def official_api_probe(repo):
    repo = Path(repo)
    sym = (repo / PRIMITIVE_FILE).read_text(encoding="utf-8")
    starter = (repo / STARTER_FILE).read_text(encoding="utf-8")
    rules = extract_rule_classes((repo / RULE_FILE).read_text(encoding="utf-8"))
    sym_tree = ast.parse(sym)
    cls_init = [n for n in ast.walk(sym_tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    setup_src = (repo / "OmniGibson/setup.py").read_text(encoding="utf-8") if (repo / "OmniGibson/setup.py").exists() else ""
    reqs = re.findall(r"""["']([A-Za-z0-9_.\-]+(?:[<>=~!][^"']*)?)["']""", setup_src.split("install_requires", 1)[1].split("]", 1)[0]) if "install_requires" in setup_src else []
    inv = extract_primitives(sym, starter)
    states_touched = sorted({s for p in inv["primitives"].values() if p["facts"] for s in p["facts"]["object_states_read"]})
    probe = {
        "omnigibson_importable": importlib.util.find_spec("omnigibson") is not None,
        "isaacsim_importable": importlib.util.find_spec("isaacsim") is not None,
        "primitive_module_docstring": ast.get_docstring(sym_tree),
        "primitive_class_constructor_arguments": [a.arg for n in cls_init for a in n.args.args][:6],
        "primitive_module_imports_engine": [n for n in ("torch", "omnigibson") if re.search(r"^(import|from) %s\b" % n, sym, re.M)],
        "primitives_reading_or_writing_simulator_object_states": {k: bool(v["facts"] and (v["facts"]["object_states_read"] or v["facts"]["sim_object_references"])) for k, v in inv["primitives"].items()},
        "object_states_touched_by_primitives": states_touched,
        "transition_rule_module": {"imports_omnigibson": rules["imports_omnigibson"], "imports_torch": rules["imports_torch"], "top_level_imports": rules["top_level_imports"], "rule_class_count": len(rules["rule_classes"])},
        "omnigibson_install_requires": reqs, "install_requires_mentions_isaac": any("isaac" in r.lower() for r in reqs),
        "bddl3_provides_only": "parser, Conditions/compile with a backend evaluate_fn, predicate vocabulary and typed recipe data; it contains no primitive or transition-rule executor",
        "standalone_symbolic_engine_found": False,
    }
    probe["blocked"] = (not probe["omnigibson_importable"]) and (probe["transition_rule_module"]["imports_omnigibson"]) and bool(probe["primitive_module_imports_engine"])
    probe["blocker"] = "OFFICIAL_SYMBOLIC_API_UNAVAILABLE" if probe["blocked"] else None
    probe["reading"] = ("the official primitives teleport simulator objects into post-condition states by reading and writing OmniGibson object states, and the official transition rules are OmniGibson classes over simulator objects and particle systems; "
                        "applying them needs a live OmniGibson (Isaac Sim) environment, which this card does not authorise, and writing a substitute would be a re-implementation the card forbids")
    return probe, inv


# ------------------------------------------------------------------------------------------------ driver
def write_json(p, v):
    Path(p).write_text(json.dumps(v, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str, ensure_ascii=False) + "\n")


def run(repo, project_root, out):
    repo, project_root, out = Path(repo).resolve(), Path(project_root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = source_manifest(repo)
    man["method_boundary"] = {"authoritative_docs": {p: sha_file(project_root / p) for p in ("docs/authoritative/CP_DISR_Final_Research_Content_v5.md", "docs/authoritative/CP_DISR_Final_Experimental_Plan_v3.md")}, "default_method": "2.1.1",
                              "A_STAT_interface": "same K, DK, context and Actor/V/Q as Full; only the DP input is replaced by S_R(F); candidate context c_i and u_i^K remain candidate-specific, so A_STAT is not a fixed heuristic and is not assumed to fail"}
    man["project_head"] = git_out(["rev-parse", "HEAD"], project_root)
    write_json(out / "source_manifest.json", man)
    if man["status"] != "PASS":
        verdict = {"card": CARD, "engineering_status": "FAIL", "verdict": "ENGINEERING_BLOCKED", "blocker": "VERSION_BASELINE_UNAVAILABLE", "next_card": "NONE"}
        write_json(out / "verdict.json", verdict)
        return verdict
    (out / "license_preflight.md").write_text(license_preflight(repo), encoding="utf-8")
    probe, inv = official_api_probe(repo)
    inv["source_sha256"] = {"symbolic": man["hashes"]["primitive_source"], "starter": man["hashes"]["starter_primitive_source"]}
    write_json(out / "primitive_inventory.json", inv)
    rules = extract_rule_classes((repo / RULE_FILE).read_text(encoding="utf-8"))
    recipes, rule_states = recipe_inventory_from(repo)
    write_json(out / "transition_inventory.json", {"omnigibson_rule_classes": rules, "bddl_recipe_inventory": recipes, "recipe_output_states": rule_states, "source_sha256": {"omnigibson_transition_rules": man["hashes"]["omnigibson_transition_rules"],
                                                                                                                                 "bddl_transition_rules": man["hashes"]["bddl_transition_rules"], "recipes_json": man["hashes"]["transition_rule_recipes_json"]},
                                                  "note": "recipes come from the official typed loaders; the rule classes are an AST list only, because applying them needs OmniGibson"})
    rows, errors, tokens = scan_tasks(repo / "bddl3")
    pred2state = predicate_to_state_map((repo / BDDL_UTILS_FILE).read_text(encoding="utf-8"))
    writes = primitive_writes(inv)
    cov_counts, cov_rows = Counter(), []
    for r in rows:
        c, why, detail = coverage_for(r, pred2state, writes, rule_states)
        r["coverage"], r["coverage_reason"], r["coverage_detail"] = c, why, detail
        r["coverage_basis"] = "STATIC_TOKEN_PATH_SCREEN (necessary-condition screen from official source; not execution-verified)"
        cov_counts[c] += 1
    inst_public = [{k: v for k, v in r.items() if k != "_parsed"} for r in rows]
    write_jsonl(out / "task_inventory.jsonl", inst_public)
    sanity = {a: [r["parse_ok"] for r in rows if r["activity"] == a] for a in ("making_lemonade", "storing_the_groceries", "baking_sugar_cookies")}
    write_json(out / "coverage_report.json", {"basis": "STATIC_TOKEN_PATH_SCREEN", "activities": len({r["activity"] for r in rows}), "instances": len(rows), "parseable": sum(r["parse_ok"] for r in rows), "compile_ok": sum(bool(r.get("compile_ok")) for r in rows),
                                              "coverage_counts": dict(cov_counts), "parse_errors": errors[:50], "n_parse_errors": len(errors), "predicate_to_state_entries": len(pred2state), "primitive_writes": sorted(map(list, writes), key=str),
                                              "recipe_output_states": rule_states, "goal_token_frequency": dict(Counter(t for r in rows for t in r.get("goal_tokens", []))),
                                              "goal_op_frequency": dict(Counter(o for r in rows for o in r.get("goal_ops", []))), "init_token_frequency": dict(Counter(t for r in rows for t in r.get("init_tokens", {}))),
                                              "sanity_seed_parse_status": sanity, "token_vocabulary": tokens})
    inv_perm = permutation_invariance(rows)
    states = []
    for r in rows:
        if "_parsed" in r:
            st, sh, gh = canonical_state(*r["_parsed"])
            states.append({"activity": r["activity"], "instance": r["instance"], "state_hash": sh, "goal_hash": gh, "n_objects": len(st["objects"]), "n_init_facts": len(st["init_facts"]), "scope": "initial BDDL public facts only"})
    write_jsonl(out / "states.jsonl", states)
    write_json(out / "state_schema.json", {"fields": ["objects: sorted (instance, synset)", "init_facts: sorted public BDDL literals", "goal: canonical goal tree (and/or children sorted)"], "excluded": ["hidden pose or collision truth", "future outcomes", "relation truth", "Full/B2/A_STAT outputs", "winner labels"],
                                           "hash": "sha256 of canonical JSON", "permutation_invariance_test": inv_perm, "scope_limit": "only the initial symbolic state of each BDDL instance; successor states need the official primitives and rules"})
    write_json(out / "scanner_config.json", {"card": CARD, "go_thresholds": GO_THRESHOLDS, "planner_caps": CACHE_CAPS, "stages": STAGES, "domain": DOMAIN, "coverage_rule": "FULL: every goal literal has a primitive write (AST set_value / placement predicate) or a recipe output state; PARTIAL: some; UNSUPPORTED: none; UNKNOWN: parse failure, unknown goal head, or a literal without an official predicate-to-state mapping",
                                             "resource_limits": {k: 0 for k in RESOURCE_KEYS}, "declared_before_scan": True})
    write_json(out / "official_api_probe.json", probe)
    # ---- stages that need the official primitives / rules to execute
    reason = probe["blocker"] or "NOT_BLOCKED"
    stub = {"status": "NOT_RUN", "reason": reason, "detail": probe["reading"] if probe["blocked"] else ""}
    for name in ("action_legality.jsonl", "successors.jsonl", "planner_queries.jsonl", "eligible_pairs.jsonl", "rejected_pairs.jsonl", "matched_context_groups.jsonl"):
        write_jsonl(out / name, [stub])
    for name in ("shortcut_audit.json",):
        write_json(out / name, stub)
    (out / "candidate_summary.csv").write_text("status,reason\nNOT_RUN,%s\n" % reason, encoding="utf-8")
    (out / "family_summary.csv").write_text("status,reason\nNOT_RUN,%s\n" % reason, encoding="utf-8")
    denom = {"activities": len({r["activity"] for r in rows}), "problem_instances": len(rows), "parseable": sum(r["parse_ok"] for r in rows), "coverage_FULL": cov_counts["FULL"], "coverage_PARTIAL": cov_counts["PARTIAL"], "coverage_UNSUPPORTED": cov_counts["UNSUPPORTED"],
             "coverage_UNKNOWN": cov_counts["UNKNOWN"], "states_examined": len(states), "states_with_ge2_legal_actions": "NOT_REACHED", "pairs_examined": "NOT_REACHED", "dual_legal": "NOT_REACHED", "public_consequence": "NOT_REACHED",
             "both_side_recoverable": "NOT_REACHED", "discrete_utility": "NOT_REACHED", "non_geometric": "NOT_REACHED", "non_contract_redundant": "NOT_REACHED", "final_eligible": "NOT_REACHED", "matched_context_groups": "NOT_REACHED",
             "reversal_witnesses": "NOT_REACHED", "magnitude_only_witnesses": "NOT_REACHED"}
    ledger = {k: 0 for k in RESOURCE_KEYS}
    ledger["omnigibson_or_isaacsim_imported"] = ("omnigibson" in sys.modules) or ("isaacsim" in sys.modules)
    write_json(out / "budget_ledger.json", ledger)
    verdict = {"card": CARD, "behavior_ref": {"tag": man["tag"], "commit": man["commit"]}, "engineering_status": "FAIL" if probe["blocked"] else "PASS", "denominator": denom,
               "blocker": probe["blocker"], "stages_completed": STAGES[:5] if probe["blocked"] else STAGES, "stages_not_run": STAGES[5:] if probe["blocked"] else [],
               "state_canonicalization_test": inv_perm, "go_thresholds": GO_THRESHOLDS, "verdict": "ENGINEERING_BLOCKED" if probe["blocked"] else "UNDECIDED", "next_card": "NONE", "scientific_conclusion": "NONE: no scientific negative is claimed, the eligibility gates were never evaluated",
               "elapsed_s": round(time.time() - t0, 1)}
    write_json(out / "verdict.json", verdict)
    (out / "report.md").write_text(report_md(man, probe, inv, denom, verdict, recipes, rules, cov_counts), encoding="utf-8")
    return verdict


def recipe_inventory_from(repo):
    sys.path.insert(0, str(Path(repo) / "bddl3"))
    for m in [k for k in sys.modules if k == "bddl" or k.startswith("bddl.")]:
        del sys.modules[m]
    return recipe_inventory()


def report_md(man, probe, inv, denom, verdict, recipes, rules, cov):
    L = ["# %s - report" % CARD, "", "**Verdict: %s** (engineering status %s; blocker %s). No scientific negative is claimed: the eligibility gates F1-F7 were never evaluated." % (verdict["verdict"], verdict["engineering_status"], verdict["blocker"]), "",
         "## Source", "", "- remote %s, tag %s, commit %s, dirty tracked files %d; MIT licence; no dataset or scene asset downloaded." % (man["remote_url"], man["tag"], man["commit"], man["dirty_tracked_files"]), "",
         "## What ran (Stage A and B)", "", "- Official BDDL3 `Conditions` parsed %d of %d problem instances across %d activities." % (denom["parseable"], denom["problem_instances"], denom["activities"]),
         "- Primitive inventory (AST of the official source): %s" % ", ".join(inv["primitives"]), "- Diff against the card's expected primitive list: %s" % json.dumps(inv["diff_against_card_expectation"]),
         "- Official recipe loaders: %s" % json.dumps({k: v.get("count", v.get("status")) for k, v in recipes.items()}), "- OmniGibson rule classes (AST): %d" % len(rules["rule_classes"]),
         "- Static coverage screen (necessary-condition, not execution-verified): FULL %d, PARTIAL %d, UNSUPPORTED %d, UNKNOWN %d." % (cov["FULL"], cov["PARTIAL"], cov["UNSUPPORTED"], cov["UNKNOWN"]),
         "- Canonical state hash permutation test: %s" % json.dumps(verdict["state_canonicalization_test"]), "", "## Why the scan stops here", "", probe["reading"], "",
         "Evidence: OmniGibson importable = %s, Isaac Sim importable = %s; primitive module imports %s; transition-rule module imports omnigibson = %s and torch = %s; object states read or written by the primitives: %s." % (
             probe["omnigibson_importable"], probe["isaacsim_importable"], probe["primitive_module_imports_engine"], probe["transition_rule_module"]["imports_omnigibson"], probe["transition_rule_module"]["imports_torch"], probe["object_states_touched_by_primitives"]), "",
         "## Denominator", "", "| item | value |", "|---|---|"] + ["| %s | %s |" % (k, v) for k, v in denom.items()] + ["", "## Decision for the user", "",
         "Stages C-H need the official primitives and transition rules to execute. That requires a live OmniGibson (Isaac Sim) environment, which this card forbids (no simulator, no GPU, no physics). Writing a symbolic re-implementation is also forbidden. Options: authorise a headless OmniGibson symbolic-mode environment for a follow-up card, or stop.", ""]
    return "\n".join(L)

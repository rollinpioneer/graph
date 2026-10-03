"""CP-DISR-TP-LIBERO-OPPORTUNITY-AUDIT-1: zero-environment audit of the public LIBERO tasks.

Reads BDDL text, task map, asset XML and source code only. It never imports robosuite, mujoco or libero, never opens a
simulator, demonstration file, score or result, and never unpickles initial-state files (they are hashed as bytes only).
All rules, thresholds and tables below are declared in code and committed before the audit is run.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CARD = "CP-DISR-TP-LIBERO-OPPORTUNITY-AUDIT-1"
BASELINE_PREFIX = "73448db3d"
BASELINE_FULL = "73448db3dc86e59bd3be6bcee3499deadf7c8682"
UPSTREAM_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
UPSTREAM_SHA = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFERRED_SUITES = ("libero_90",)
ALLOWED_SUFFIX = {".bddl", ".py", ".xml", ".txt", ".json", ".md", ".yaml", ".yml", ".obj", ".stl", ".mtl", ".csv"}
FORBIDDEN = re.compile(r"\.hdf5|\.h5|demo|dataset|result|score|checkpoint|\.pt$|\.pth$|rollout|eval_out|wandb", re.I)

# ---- thresholds and tables declared before the audit ---------------------------------------------------
PINCH_LIMIT = 0.06            # horizontal extent a top-down pinch can take: open inner faces +-0.0399 m minus controller tolerance 0.008 (VIS card)
ROUND_RATIO = 1.25            # cross-section ratio below which yaw does not matter
ELEVATED_MAX = 0.15           # destinations / supports higher than this above the table are outside the validated executor envelope (hover z = table+0.22)
ONTABLE_BOUND = 0.075         # Verifier OnTable: blob top below table_top + 0.075
CAPACITY_MATERIAL = 0.5       # persistent occupancy effect is material only if summed footprints fill >= 50 % of the destination
FINGER_HALF_X = 0.010         # proxy finger half width used only for the neighbour-proximity flag
FINGER_OUTER_Y = 0.0664       # open finger outer face (BSI card gripper geometry)
PLATFORM_FLAG = "PLATFORM_BINDING_REQUIRED"
HIGH_COVERAGE = 0.80          # fraction of frozen tasks needed to use the UNMODIFIED_LIBERO_PUBLIC_VALIDATION label
CLASS_ORDER = ["CURRENT_SKILL_EXACT", "CURRENT_SKILL_WITH_NEW_BINDING", "NEW_SCRIPTED_SKILL_REQUIRED", "NEW_LOW_LEVEL_CONTROLLER_REQUIRED", "NOT_MAPPABLE"]
STRUCT_LABELS = ("SINGLE_ACTION_OR_SINGLE_GOAL", "HARD_PRECONDITION_DOMINATED", "COMMUTATIVE_MULTI_GOAL", "SUPPORT_OR_STACKING_CANDIDATE", "SHARED_WORKSPACE_CANDIDATE",
                 "SHARED_DESTINATION_CANDIDATE", "GOAL_CONDITIONED_RELEVANCE_CANDIDATE", "ARTICULATED_SOFT_EFFECT_CANDIDATE", "UNSUPPORTED_SKILL", "UNKNOWN_NEEDS_ASSET_AUDIT")
GATES = ["G%d" % i for i in range(1, 16)]
SURFACE_RE = re.compile(r"(^|_)(table|floor)$")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def digest(v):
    return sha_bytes(json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode())


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False, default=_default) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, sort_keys=True, default=_default) if isinstance(v, (list, dict, tuple, set)) else v) for k, v in r.items()})


class Guard:
    """Reader that only opens text/XML/mesh-geometry files and refuses anything that looks like a demo, score or result."""

    def __init__(self):
        self.opened, self.hashed_only, self.refused = 0, 0, []

    def check(self, path):
        s = str(path).replace("\\", "/")
        if FORBIDDEN.search(s):
            self.refused.append(s)
            raise PermissionError("FORBIDDEN_PATH:" + s)

    def read(self, path):
        self.check(path)
        if Path(path).suffix.lower() not in ALLOWED_SUFFIX:
            self.refused.append(str(path))
            raise PermissionError("FORBIDDEN_SUFFIX:" + str(path))
        self.opened += 1
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    def hash_only(self, path):
        self.check(path)
        self.hashed_only += 1
        return sha_file(path)


# ------------------------------------------------------------------------------------------ upstream identity
def tree_hash(base, guard, patterns=("*",), exclude=("__pycache__",)):
    base = Path(base)
    rows = []
    for pat in patterns:
        for p in sorted(base.rglob(pat)):
            if p.is_file() and not any(x in p.parts for x in exclude):
                rows.append((p.relative_to(base).as_posix(), guard.hash_only(p)))
    rows = sorted(set(rows))
    return {"files": len(rows), "sha256": digest(rows)}


def git_out(args, cwd):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()


def upstream_identity(libero_root, guard, pip_root=None):
    root = Path(libero_root).resolve()
    lib = root / "libero" / "libero"
    ident = {"repository": UPSTREAM_REPO, "expected_sha": UPSTREAM_SHA, "root": str(root)}
    ident["remote_url"] = git_out(["remote", "get-url", "origin"], root)
    ident["head_sha"] = git_out(["rev-parse", "HEAD"], root)
    ident["head_commit_date"] = git_out(["log", "-1", "--format=%cI"], root)
    ident["head_subject"] = git_out(["log", "-1", "--format=%s"], root)
    ident["dirty_tracked_files"] = len([l for l in git_out(["status", "--porcelain", "--untracked-files=no"], root).splitlines() if l.strip()])
    ident["sha_matches_expected"] = ident["head_sha"] == UPSTREAM_SHA
    ident["source_priority_used"] = "official repository (priority 2): no server copy carries a confirmable commit"
    ident["audit_date_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ident["task_map_file"] = "libero/libero/benchmark/libero_suite_task_map.py"
    ident["task_map_sha256"] = guard.hash_only(lib / "benchmark" / "libero_suite_task_map.py")
    ident["bddl_root_all_suites"] = tree_hash(lib / "bddl_files", guard)
    ident["bddl_per_suite"] = {s: tree_hash(lib / "bddl_files" / s, guard) for s in SUITES + DEFERRED_SUITES}
    ident["asset_definition_hash"] = {
        "assets_xml": tree_hash(lib / "assets", guard, ("*.xml",)),
        "object_definitions_py": tree_hash(lib / "envs" / "objects", guard, ("*.py",)),
        "predicates_py": tree_hash(lib / "envs" / "predicates", guard, ("*.py",)),
        "base_object_py": guard.hash_only(lib / "envs" / "base_object.py"),
    }
    ident["initial_state_files_bytes_only"] = {s: tree_hash(lib / "init_files" / s, guard) for s in SUITES}
    ident["initial_state_files_note"] = "pickled initial states are hashed and counted only; they are never unpickled in this card"
    if pip_root:
        pr = Path(pip_root)
        pip = {"root": str(pr), "package": "libero 0.1.1 (huggingface/lerobot-libero repackaging, no commit id)"}
        if (pr / "bddl_files").exists():
            pip["bddl_root_all_suites"] = tree_hash(pr / "bddl_files", guard)
            pip["task_map_sha256"] = guard.hash_only(pr / "benchmark" / "libero_suite_task_map.py")
            pip["bddl_identical_to_upstream"] = pip["bddl_root_all_suites"]["sha256"] == ident["bddl_root_all_suites"]["sha256"]
            pip["task_map_identical_to_upstream"] = pip["task_map_sha256"] == ident["task_map_sha256"]
        ident["server_pip_copy"] = pip
    ident["single_version_rule"] = "every table in this audit is computed from the upstream tree above; the pip copy is only compared"
    return ident


# ------------------------------------------------------------------------------------------------ BDDL parsing
def tokenize(text):
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexpr(tokens):
    pos = 0

    def rec():
        nonlocal pos
        out = []
        while pos < len(tokens):
            t = tokens[pos]
            pos += 1
            if t == "(":
                out.append(rec())
            elif t == ")":
                return out
            else:
                out.append(t)
        return out
    res = rec()
    return res[0] if len(res) == 1 and isinstance(res[0], list) else res


def _section(tree, head):
    for item in tree:
        if isinstance(item, list) and item and item[0] == head:
            return item
    return None


def _typed_pairs(items):
    """['a','b','-','t1','c','-','t2'] -> [('a','t1'),('b','t1'),('c','t2')]"""
    out, pending = [], []
    i = 0
    while i < len(items):
        if items[i] == "-":
            typ = items[i + 1]
            out += [(n, typ) for n in pending]
            pending = []
            i += 2
        else:
            pending.append(items[i])
            i += 1
    out += [(n, "UNTYPED") for n in pending]
    return out


def parse_regions(sec):
    regions = {}
    if not sec:
        return regions
    for r in sec[1:]:
        if not isinstance(r, list) or not r:
            continue
        name, rec = r[0], {"target": None, "ranges": [], "yaw": []}
        for part in r[1:]:
            if not isinstance(part, list) or not part:
                continue
            if part[0] == ":target":
                rec["target"] = part[1]
            elif part[0] == ":ranges":
                rec["ranges"] = [[float(x) for x in rng] for rng in part[1] if isinstance(rng, list)]
            elif part[0] == ":yaw_rotation":
                rec["yaw"] = [[float(x) for x in rng] for rng in part[1] if isinstance(rng, list)]
        key = (rec["target"], name)
        regions[key] = rec
    return regions


def flatten_goal(expr, ops=None):
    """Return atoms and the logical operators met on the way."""
    ops = ops if ops is not None else []
    if not isinstance(expr, list) or not expr:
        return [], ops
    head = expr[0].lower() if isinstance(expr[0], str) else ""
    if head in ("and", "or", "not"):
        ops.append(head)
        atoms = []
        for sub in expr[1:]:
            a, _ = flatten_goal(sub, ops)
            atoms += a
        return atoms, ops
    return [[expr[0]] + [str(x) for x in expr[1:]]], ops


def parse_bddl(text):
    tree = parse_sexpr(tokenize(text))
    lang = _section(tree, ":language")
    regions = parse_regions(_section(tree, ":regions"))
    fixtures = _typed_pairs(_section(tree, ":fixtures")[1:]) if _section(tree, ":fixtures") else []
    objects = _typed_pairs(_section(tree, ":objects")[1:]) if _section(tree, ":objects") else []
    ooi = (_section(tree, ":obj_of_interest") or [None])[1:]
    init = [[x[0]] + list(x[1:]) for x in (_section(tree, ":init") or [None])[1:] if isinstance(x, list)]
    goal_sec = _section(tree, ":goal")
    goal_atoms, goal_ops = flatten_goal(goal_sec[1], []) if goal_sec and len(goal_sec) > 1 else ([], [])
    domain = _section(tree, ":domain")
    prob = tree[1][1] if isinstance(tree[1], list) and len(tree[1]) > 1 else None
    return {"problem": prob, "domain": domain[1] if domain else None, "language": " ".join(lang[1:]) if lang else "", "regions": regions, "fixtures": fixtures,
            "objects": objects, "obj_of_interest": ooi, "init": init, "goal_atoms": goal_atoms, "goal_ops": goal_ops}


def load_task_map(path, guard):
    tree = ast.parse(guard.read(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "libero_task_map" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("libero_task_map not found")


# ------------------------------------------------------------------------------------- object / asset registry
def snake(cls_name):
    """Same key rule as upstream register_object: split before capitals AND digits (Chefmate8Frypan -> chefmate_8_frypan)."""
    return "_".join(re.sub(r"([A-Z0-9])", r" \1", cls_name).split()).lower()


def load_registry(lib, guard):
    """AST scan of envs/objects: class -> category, asset xml path, articulation ranges."""
    objdir = Path(lib) / "envs" / "objects"
    classes, bases = {}, {}
    for py in sorted(objdir.glob("*.py")):
        src = guard.read(py)
        tree = ast.parse(src)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            seg = ast.get_source_segment(src, node) or ""
            decos = [getattr(d, "id", getattr(d, "attr", "")) for d in node.decorator_list]
            defaults = {}
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    names = [a.arg for a in fn.args.args]
                    dfl = fn.args.defaults
                    for n, d in zip(names[len(names) - len(dfl):], dfl):
                        if isinstance(d, ast.Constant):
                            defaults[n] = d.value
            tmpl = re.search(r"assets/([A-Za-z0-9_]+)/(\{obj_name\}/)?\{obj_name\}\.xml", seg)
            ranges = {k: re.findall(r"\[[-0-9., ]+\]", v) for k, v in re.findall(r"\"(default_(?:open|close)_ranges)\"\]\s*=\s*(\[[-0-9., ]+\])", seg)} if seg else {}
            classes[node.name] = {"file": py.name, "bases": [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases], "registered": "register_object" in decos,
                                  "defaults": defaults, "asset_template": (tmpl.group(0) if tmpl else None), "asset_dir": (tmpl.group(1) if tmpl else None),
                                  "asset_nested": bool(tmpl and tmpl.group(2)), "ranges": {k: json.loads(v) for k, v in re.findall(r"\"(default_(?:open|close)_ranges)\"\]\s*=\s*(\[[-0-9., ]+\])", seg)},
                                  "overrides_rotation": bool(re.search(r"self\.rotation\b", seg)), "src_sha": sha_bytes(seg.encode())}
    out = {}
    for cname, c in classes.items():
        if not c["registered"]:
            continue
        # walk bases for the asset template
        cur, tmpl, adir, nested, art = c, c["asset_template"], c["asset_dir"], c["asset_nested"], False
        chain = [cname]
        seen = 0
        while cur and not tmpl and seen < 5:
            seen += 1
            nxt = next((b for b in cur["bases"] if b in classes), None)
            if not nxt:
                break
            chain.append(nxt)
            cur = classes[nxt]
            tmpl, adir, nested = cur["asset_template"], cur["asset_dir"], cur["asset_nested"]
        art = "ArticulatedObject" in chain
        obj_name = c["defaults"].get("obj_name") or snake(cname)
        if tmpl and adir:
            rel = "assets/%s/%s%s.xml" % (adir, (obj_name + "/") if nested else "", obj_name)
        else:
            rel = None
        out[snake(cname)] = {"class": cname, "file": c["file"], "chain": chain, "articulated": art, "xml": rel, "obj_name": obj_name, "ranges": c["ranges"], "src_sha": c["src_sha"]}
    return out


def _quat_to_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _floats(s, n, default):
    if s is None:
        return default
    v = [float(x) for x in s.split()]
    return v if len(v) >= n else default


def mesh_bounds(path, guard, scale):
    path = Path(path)
    try:
        if path.suffix.lower() == ".obj":
            pts = [[float(x) for x in l.split()[1:4]] for l in guard.read(path).splitlines() if l.startswith("v ")]
            arr = np.array(pts)
        else:
            return None
        if arr.size == 0:
            return None
        arr = arr * np.array(scale)
        return arr.min(0), arr.max(0)
    except Exception:
        return None


def asset_dims(xml_path, guard):
    """Extents of the collision geoms (group 0, contype != 0) plus LIBERO's own placement sites and joints."""
    xml_path = Path(xml_path)
    root = ET.fromstring(guard.read(xml_path))
    meshes = {m.get("name"): (m.get("file"), _floats(m.get("scale"), 3, [1, 1, 1])) for m in root.iter("mesh")}
    lo, hi, n_geom, mesh_unparsed = np.full(3, np.inf), np.full(3, -np.inf), 0, 0
    for g in root.iter("geom"):
        if g.get("group", "0") != "0":
            continue
        if g.get("contype") == "0" and g.get("conaffinity") == "0":
            continue
        typ = g.get("type", "sphere")
        pos = np.array(_floats(g.get("pos"), 3, [0, 0, 0]))
        quat = _floats(g.get("quat"), 4, [1, 0, 0, 0])
        R = _quat_to_mat(quat)
        half = None
        if typ == "box":
            s = np.array(_floats(g.get("size"), 3, [0, 0, 0]))
            half = np.abs(R) @ s
        elif typ in ("cylinder", "capsule"):
            s = _floats(g.get("size"), 2, [0, 0])
            half = np.abs(R) @ np.array([s[0], s[0], s[1] + (s[0] if typ == "capsule" else 0)])
        elif typ == "sphere":
            s = _floats(g.get("size"), 1, [0])
            half = np.array([s[0]] * 3)
        elif typ == "ellipsoid":
            s = np.array(_floats(g.get("size"), 3, [0, 0, 0]))
            half = np.abs(R) @ s
        elif typ == "mesh":
            fn, sc = meshes.get(g.get("mesh"), (None, [1, 1, 1]))
            b = mesh_bounds(xml_path.parent / fn, guard, sc) if fn else None
            if b is None:
                mesh_unparsed += 1
                continue
            c, h = (b[0] + b[1]) / 2, (b[1] - b[0]) / 2
            pos = pos + R @ c
            half = np.abs(R) @ h
        if half is None:
            continue
        n_geom += 1
        lo, hi = np.minimum(lo, pos - half), np.maximum(hi, pos + half)
    sites = {}
    for s in root.iter("site"):
        if s.get("name"):
            sites[s.get("name")] = {"pos": _floats(s.get("pos"), 3, [0, 0, 0]), "size": _floats(s.get("size"), 1, [0])}
    joints = [{"name": j.get("name"), "type": j.get("type", "hinge"), "range": _floats(j.get("range"), 2, None), "axis": _floats(j.get("axis"), 3, None)} for j in root.iter("joint")]
    ext = (hi - lo).tolist() if n_geom else None
    top, bot, hr = sites.get("top_site"), sites.get("bottom_site"), sites.get("horizontal_radius_site")
    site_dims = None
    if top and bot and hr:
        site_dims = {"height": top["pos"][2] - bot["pos"][2], "h_radius": math.hypot(hr["pos"][0], hr["pos"][1])}
    return {"xml": xml_path.as_posix(), "collision_extent_xyz": ext, "collision_geoms_used": n_geom, "mesh_collision_unparsed": mesh_unparsed,
            "site_dims": site_dims, "sites": sites, "joints": joints, "xml_sha256": sha_bytes(xml_path.read_bytes())}


def build_asset_table(lib, registry, types, guard):
    table = {}
    for t in sorted(types):
        rec = registry.get(t)
        if not rec or not rec["xml"]:
            table[t] = {"type": t, "resolved": False, "reason": "not in the registered object classes" if not rec else "no xml template found"}
            continue
        p = Path(lib) / rec["xml"]
        if not p.exists():
            table[t] = {"type": t, "resolved": False, "reason": "xml missing: %s" % rec["xml"], "class": rec["class"]}
            continue
        d = asset_dims(p, guard)
        table[t] = dict({"type": t, "resolved": True, "class": rec["class"], "articulated": rec["articulated"], "xml_rel": rec["xml"], "ranges": rec["ranges"]}, **d)
    return table


# ------------------------------------------------------------------------------------------- current CP-DISR interface
def current_skill_interface(root, guard):
    """What the current project can actually execute, read from source and contracts (not from LIBERO demonstrations)."""
    root = Path(root)
    ex = guard.read(root / "src/cp_disr/platforms/libero/skill_executor.py")
    d0 = guard.read(root / "src/cp_disr/platforms/libero/d0_env.py")
    ev = guard.read(root / "src/cp_disr/platforms/libero/task_evaluator.py")
    contracts = guard.read(root / "configs/contracts/d0_runtime_skills.yaml")
    skills = sorted(set(re.findall(r'skill == "([A-Z_]+)"', ex)) - {"MOVE"})
    predicates = re.findall(r"^  (\w+):", contracts.split("contracts:")[0], re.M)
    liberoref = [str(p.relative_to(root)) for p in (root / "src").rglob("*.py")
                 if not p.name.startswith("tp_libero_audit") and re.search(r"import libero|from libero|bddl|OffScreenRenderEnv", p.read_text(errors="ignore"))]
    return {
        "executable_skills": skills,
        "contract_predicates": [p for p in predicates if p != "predicate_types"],
        "orientation_actions_zeroed": bool(re.search(r"action\[3:6\]\s*=\s*0\.0", d0)),
        "action_space": "OSC_POSE with position deltas only; gripper open/close",
        "hover_height_above_table_m": 0.22,
        "grasp_site_is_table_relative": "table_top_z" in ex,
        "object_constants_are_cube_specific": bool(re.search(r"OBJECT_HALF\s*=\s*np\.array\(\[0\.020", d0)),
        "perception": "colour-threshold top-face blobs with named D0 colours (no texture/semantic model)",
        "environment_class": "project-owned robosuite D0ManipulationEnv (cubes, lid container, buffer pad)",
        "libero_references_in_src": liberoref,
        "evaluator": "hidden MuJoCo truth on D0 geometry (cp-disr-d0-task-evaluator)",
        "evaluator_is_d0_specific": "CONTAINER_INNER" in ev,
        "OPEN_meaning": "grasp a free lid and park it on the table (not a drawer, door or knob)",
        "snapshot_restore": "D0 snapshot module only",
    }


# --------------------------------------------------------------------------------------------- task facts / rules
def inst_maps(parsed):
    inst, sect = {}, {}
    for n, t in parsed["fixtures"]:
        inst[n], sect[n] = t, "fixture"
    for n, t in parsed["objects"]:
        inst[n], sect[n] = t, "object"
    return inst, sect


def is_surface(t, reg):
    return bool(SURFACE_RE.search(t or "")) and t not in reg


def resolve(ref, parsed, inst, sect):
    if ref in inst:
        return {"ref": ref, "instance": ref, "type": inst[ref], "region": None, "role": sect[ref]}
    for (tgt, rn) in parsed["regions"]:
        if tgt and ref == "%s_%s" % (tgt, rn):
            return {"ref": ref, "instance": tgt, "type": inst.get(tgt), "region": rn, "role": sect.get(tgt)}
    return {"ref": ref, "instance": None, "type": None, "region": None, "role": "UNRESOLVED"}


def z_extent(a):
    if a and a.get("resolved") and a.get("collision_extent_xyz"):
        return a["collision_extent_xyz"][2]
    if a and a.get("site_dims"):
        return a["site_dims"]["height"]
    return None


def sorted_extents(a):
    if a and a.get("resolved") and a.get("collision_extent_xyz"):
        return sorted(a["collision_extent_xyz"])
    return None


def _hyp_class(a, b, geoms):
    """Pinch class for one assumed pair of horizontal extents (a, b) under the fixed spread axis and unknown yaw."""
    rnd = abs(a - b) / max(a, b, 1e-9) <= 0.05 and geoms >= 8
    worst = max(a, b) if rnd else math.hypot(a, b)
    aligned = min(a, b)
    if worst <= PINCH_LIMIT:
        return "COMPACT", worst, aligned
    if aligned <= PINCH_LIMIT:
        return "YAW_DEPENDENT", worst, aligned
    return "NOT_PINCHABLE", worst, aligned


def grasp_class(a):
    """Pinch classification under the current controller (fixed spread axis, no wrist rotation, PINCH_LIMIT aperture).

    The spawn orientation cannot be verified without an environment, so two resting hypotheses are kept: 'lying' (the two
    largest extents horizontal; the pessimistic, strict one) and 'standing' (the two smallest horizontal; only for tall
    bodies with e3/e2 >= 1.5). Strict binding needs COMPACT under the lying hypothesis; the optimistic bound takes the best hypothesis."""
    e = sorted_extents(a)
    if e is None:
        return {"class": "UNKNOWN", "reason": "no collision extents", "strict_binding": False, "binding_if_orientation_verified": False, "optimistic_class": "UNKNOWN"}
    e1, e2, e3 = e
    geoms = a.get("collision_geoms_used", 0)
    hyps = {"lying": _hyp_class(e2, e3, geoms)}
    if e3 / max(e2, 1e-9) >= 1.5:
        hyps["standing"] = _hyp_class(e1, e2, geoms)
    order = ["COMPACT", "YAW_DEPENDENT", "NOT_PINCHABLE"]
    strict = hyps["lying"][0]
    best_name = min(hyps, key=lambda k: order.index(hyps[k][0]))
    optimistic = hyps[best_name][0]
    reason = "lying hypothesis horizontal extents (%.3f, %.3f): worst-case yaw width %.3f, aligned width %.3f vs limit %.2f -> %s" % (e2, e3, hyps["lying"][1], hyps["lying"][2], PINCH_LIMIT, strict)
    if "standing" in hyps:
        reason += "; standing hypothesis (%.3f, %.3f): worst %.3f -> %s" % (e1, e2, hyps["standing"][1], hyps["standing"][0])
    return {"class": strict, "reason": reason, "optimistic_class": optimistic, "optimistic_hypothesis": best_name, "sorted_extents_m": [round(x, 4) for x in e],
            "strict_binding": strict == "COMPACT", "binding_if_orientation_verified": optimistic == "COMPACT",
            "hypotheses": {k: {"class": v[0], "worst_yaw_width_m": round(v[1], 4), "aligned_width_m": round(v[2], 4)} for k, v in hyps.items()}}

def dest_kind(res, assets, reg):
    t = res["type"]
    if res["instance"] is None:
        return {"kind": "UNRESOLVED", "height": None}
    if is_surface(t, reg):
        return {"kind": "TABLE_FLAT", "height": 0.0}
    a = assets.get(t, {})
    art = bool(a.get("articulated"))
    rn = res["region"]
    h = z_extent(a)
    slide = any(j["type"] == "slide" for j in a.get("joints", []))
    if art:
        if rn in ("top_region", "middle_region", "bottom_region") and slide:
            return {"kind": "DRAWER_INTERIOR", "height": h}
        if rn and rn.endswith("_side"):
            return {"kind": "ELEVATED_FLAT_TOP", "height": h}
        if rn == "cook_region":
            return {"kind": "FLAT_APPLIANCE_TOP", "height": h}
        if rn == "heating_region":
            return {"kind": "ENCLOSED_APPLIANCE_INTERIOR", "height": h}
        if rn is None:
            return {"kind": "ARTICULATED_INSTANCE", "height": h}
        return {"kind": "UNKNOWN", "height": h}
    if rn and rn.endswith("contain_region"):
        return {"kind": "OPEN_CONTAINER", "height": h}
    if rn == "top_region" and t == "wine_rack":
        return {"kind": "SLOT_RACK", "height": h}
    if rn is None:
        return {"kind": "SUPPORT_OBJECT_TOP", "height": h}
    return {"kind": "UNKNOWN", "height": h}


def worst(classes):
    return max(classes, key=CLASS_ORDER.index) if classes else "NOT_MAPPABLE"


def derive_actions(task, assets, reg):
    """Required high-level actions from the goal atoms and the initial predicates (BDDL is the source of truth)."""
    parsed = task["parsed"]
    inst, sect = inst_maps(parsed)
    init_on = {}
    open_init = set()
    for a in parsed["init"]:
        p = a[0].lower()
        if p in ("on", "in") and len(a) >= 3:
            init_on[a[1]] = (p, resolve(a[2], parsed, inst, sect))
        if p == "open":
            open_init.add(a[1])
    actions = []

    def pick_action(obj):
        t = inst.get(obj)
        gc = grasp_class(assets.get(t, {}))
        src = init_on.get(obj)
        reasons, cls = [], None
        if t is None or sect.get(obj) != "object":
            return {"skill": "PICK", "object": obj, "class": "NOT_MAPPABLE", "reasons": ["moving a fixture or unresolved instance"], "grasp": gc, "source": None}
        if gc["class"] == "UNKNOWN":
            return {"skill": "PICK", "object": obj, "class": "NOT_MAPPABLE", "reasons": ["asset extents unavailable"], "grasp": gc, "source": None}
        cls = "CURRENT_SKILL_WITH_NEW_BINDING" if gc["strict_binding"] else "NEW_SCRIPTED_SKILL_REQUIRED"
        reasons.append("grasp class %s: %s" % (gc["class"], gc["reason"]))
        srcd = None
        if src:
            p, sres = src
            sk = dest_kind(sres, assets, reg)
            srcd = {"relation": p, "support_kind": sk["kind"], "support_height_m": sk["height"], "support_instance": sres["instance"], "support_type": sres["type"],
                    "container_open_initially": any(r.startswith(sres["ref"]) or sres["ref"].startswith(r) for r in open_init)}
            if sk["kind"] in ("DRAWER_INTERIOR", "ENCLOSED_APPLIANCE_INTERIOR"):
                cls, reasons = worst([cls, "NEW_SCRIPTED_SKILL_REQUIRED"]), reasons + ["initially inside an articulated container"]
            elif sk["kind"] in ("TABLE_FLAT",):
                pass
            else:
                zo = z_extent(assets.get(t, {})) or 0.0
                top = (sk["height"] or 0.0) + zo
                if (sk["height"] or 0.0) > ELEVATED_MAX or top > ONTABLE_BOUND:
                    cls = worst([cls, "NEW_SCRIPTED_SKILL_REQUIRED"])
                    reasons.append("starts on a support (%s %.3f m) so the picked top reaches %.3f m > Verifier OnTable bound %.3f m: contract PRE OnTable and the Verifier would change" % (sk["kind"], sk["height"] or 0.0, top, ONTABLE_BOUND))
                else:
                    reasons.append("starts on a low support (%s %.3f m): contract OnTable predicate needs re-binding" % (sk["kind"], sk["height"] or 0.0))
        return {"skill": "PICK", "object": obj, "type": t, "class": cls, "reasons": reasons, "grasp": gc, "source": srcd}

    for a in parsed["goal_atoms"]:
        p = a[0].lower()
        if p in ("on", "in") and len(a) >= 3:
            obj, dres = a[1], resolve(a[2], parsed, inst, sect)
            pk = pick_action(obj)
            dk = dest_kind(dres, assets, reg)
            kind = dk["kind"]
            if kind == "TABLE_FLAT":
                pc, sk, why = "CURRENT_SKILL_WITH_NEW_BINDING", "PLACE_BUFFER", "table region: PLACE_BUFFER with a new destination binding"
            elif kind == "OPEN_CONTAINER":
                pc, sk, why = "CURRENT_SKILL_WITH_NEW_BINDING", "PLACE", "open container: PLACE with container height and inner extents re-bound"
            elif kind in ("FLAT_APPLIANCE_TOP", "SUPPORT_OBJECT_TOP"):
                if (dk["height"] or 0) <= ELEVATED_MAX:
                    pc, sk, why = "CURRENT_SKILL_WITH_NEW_BINDING", "PLACE_BUFFER", "low flat support %.3f m: PLACE_BUFFER with new drop height" % (dk["height"] or 0)
                else:
                    pc, sk, why = "NEW_SCRIPTED_SKILL_REQUIRED", "PLACE_BUFFER", "support %.3f m is above the validated envelope" % dk["height"]
            elif kind == "ELEVATED_FLAT_TOP":
                pc, sk, why = "NEW_SCRIPTED_SKILL_REQUIRED", "PLACE_BUFFER", "destination %.3f m above the table exceeds the %.2f m envelope (hover z fixed)" % (dk["height"] or 0, ELEVATED_MAX)
            elif kind == "SLOT_RACK":
                pc, sk, why = "NEW_SCRIPTED_SKILL_REQUIRED", "PLACE", "rack slot insertion"
            elif kind in ("DRAWER_INTERIOR", "ENCLOSED_APPLIANCE_INTERIOR"):
                pc, sk, why = "NEW_SCRIPTED_SKILL_REQUIRED", "PLACE", "placement into an articulated container (also needs a prior opening)"
            else:
                pc, sk, why = "NOT_MAPPABLE", "PLACE", "unresolved destination kind %s" % kind
            actions += [pk, {"skill": sk, "object": obj, "dest": a[2], "dest_kind": kind, "dest_height_m": dk["height"], "class": pc, "reasons": [why]}]
            if kind in ("DRAWER_INTERIOR", "ENCLOSED_APPLIANCE_INTERIOR"):
                container_ref = a[2]
                if container_ref not in open_init:
                    actions.append(open_action(dres, assets, "prerequisite: container closed in the initial state"))
        elif p in ("open", "close"):
            res = resolve(a[1], parsed, inst, sect)
            actions.append(open_action(res, assets, "goal predicate %s" % a[0], close=(p == "close")))
        elif p in ("turnon", "turnoff"):
            res = resolve(a[1], parsed, inst, sect)
            at = assets.get(res["type"], {})
            hinge = any(j["type"] == "hinge" for j in at.get("joints", []))
            actions.append({"skill": "TURN_ON" if p == "turnon" else "TURN_OFF", "object": a[1], "class": "NEW_LOW_LEVEL_CONTROLLER_REQUIRED" if hinge else "NEW_SCRIPTED_SKILL_REQUIRED",
                            "reasons": ["rotary knob joint (hinge): needs wrist rotation or tangential contact, rotation actions are zeroed in the executor"] if hinge else ["articulated switch"]})
        elif p == "up":
            actions.append(dict(pick_action(a[1]), skill="PICK_LIFT"))
        else:
            actions.append({"skill": "UNKNOWN_PREDICATE_" + a[0], "object": a[1] if len(a) > 1 else None, "class": "NOT_MAPPABLE", "reasons": ["predicate outside the audited vocabulary"]})
    return actions


def open_action(res, assets, why, close=False):
    at = assets.get(res["type"], {})
    joints = at.get("joints", [])
    slide = any(j["type"] == "slide" for j in joints)
    hinge = any(j["type"] == "hinge" for j in joints)
    if slide:
        cls, how = "NEW_SCRIPTED_SKILL_REQUIRED", "sliding drawer: handle grasp and linear pull (the current OPEN is lid grasp-and-park)"
    elif hinge:
        cls, how = "NEW_LOW_LEVEL_CONTROLLER_REQUIRED", "hinged door: constrained arc motion needing rotation actions"
    else:
        cls, how = "NOT_MAPPABLE", "no articulation found for %s" % res["ref"]
    return {"skill": "CLOSE" if close else "OPEN", "object": res["ref"], "class": cls, "reasons": [how, why]}


def hard_precondition(actions, parsed, task_inst):
    """Hard preconditions: access to an articulated interior, or Close after In (container gating)."""
    flags = []
    for a in actions:
        if a["skill"] in ("OPEN", "CLOSE"):
            flags.append("articulated %s on %s" % (a["skill"], a["object"]))
        if a.get("dest_kind") in ("DRAWER_INTERIOR", "ENCLOSED_APPLIANCE_INTERIOR"):
            flags.append("destination inside a closed articulated container: %s" % a["dest"])
        src = a.get("source") or {}
        if a["skill"] == "PICK" and src.get("support_kind") in ("DRAWER_INTERIOR", "ENCLOSED_APPLIANCE_INTERIOR") and not src.get("container_open_initially"):
            flags.append("object starts inside a closed articulated container")
    return flags


def rect_gap(r1, r2):
    """min/max centre displacement components between two axis-aligned sampling rectangles [x1,y1,x2,y2]."""
    out = []
    for k in (0, 1):
        a_lo, a_hi, b_lo, b_hi = r1[k], r1[k + 2], r2[k], r2[k + 2]
        d_min = max(0.0, max(a_lo - b_hi, b_lo - a_hi))
        d_max = max(abs(a_hi - b_lo), abs(b_hi - a_lo))
        out.append((d_min, d_max))
    return out


def init_region_of(obj, parsed, inst, sect):
    for a in parsed["init"]:
        if a[0].lower() == "on" and a[1] == obj and len(a) >= 3:
            res = resolve(a[2], parsed, inst, sect)
            rec = parsed["regions"].get((res["instance"], res["region"]))
            return res, rec
    return None, None


def proximity_flags(task, assets):
    """Proxy flag: can another movable object sit inside the fixed-spread-axis finger envelope of a goal object (table-placed objects only)."""
    parsed = task["parsed"]
    inst, sect = inst_maps(parsed)
    goal_objs = sorted({a[1] for a in parsed["goal_atoms"] if len(a) > 1 and a[0].lower() in ("on", "in") and sect.get(a[1]) == "object"})
    movables = [n for n, s in sect.items() if s == "object" and not assets.get(inst[n], {}).get("articulated")]
    rows = []
    for g in goal_objs:
        rg, reg_g = init_region_of(g, parsed, inst, sect)
        eg = (assets.get(inst[g], {}).get("collision_extent_xyz") or [0.05, 0.05, 0.05])
        for o in movables:
            if o == g:
                continue
            ro, reg_o = init_region_of(o, parsed, inst, sect)
            if not (reg_g and reg_o and reg_g["ranges"] and reg_o["ranges"] and rg["type"] and ro["type"]):
                continue
            if not (is_surface(rg["type"], {}) and is_surface(ro["type"], {})):
                continue
            eo = assets.get(inst[o], {}).get("collision_extent_xyz") or [0.05, 0.05, 0.05]
            hx_o, hy_o = eo[0] / 2, eo[1] / 2
            poss = cert = False
            dmins, dmaxs = [], []
            for r1 in reg_g["ranges"]:
                for r2 in reg_o["ranges"]:
                    (dxm, dxM), (dym, dyM) = rect_gap(r1, r2)
                    dmins.append(math.hypot(dxm, dym))
                    dmaxs.append(math.hypot(dxM, dyM))
                    x_hit_poss, y_hit_poss = dxm < FINGER_HALF_X + hx_o, dym < FINGER_OUTER_Y + hy_o
                    x_hit_cert, y_hit_cert = dxM < FINGER_HALF_X + hx_o, dyM < FINGER_OUTER_Y + hy_o
                    poss = poss or (x_hit_poss and y_hit_poss)
                    cert = cert or (x_hit_cert and y_hit_cert and dym > 0)
            rows.append({"goal_object": g, "other": o, "other_type": inst[o], "min_center_distance_m": round(min(dmins), 4), "max_center_distance_m": round(max(dmaxs), 4),
                         "finger_envelope_overlap_possible": bool(poss), "finger_envelope_overlap_certain": bool(cert)})
    return rows


def scene_signature(parsed):
    inst, _ = inst_maps(parsed)
    return digest({"types": sorted(inst.values()), "init": sorted(" ".join(a) for a in parsed["init"]),
                   "regions": sorted("%s|%s|%s" % (k[0], k[1], v["ranges"]) for k, v in parsed["regions"].items())})


def task_row(suite, idx, name, bddl_rel, parsed, reg, assets):
    inst, sect = inst_maps(parsed)
    task = {"suite": suite, "index": idx, "name": name, "bddl": bddl_rel, "parsed": parsed}
    types_missing = sorted({t for t in inst.values() if not is_surface(t, reg) and not (assets.get(t) or {}).get("resolved")})
    task["actions"] = derive_actions(task, assets, reg)
    task["hard_flags"] = hard_precondition(task["actions"], parsed, inst)
    task["types_missing"] = types_missing
    task["coverage"] = worst([a["class"] for a in task["actions"]])
    task["coverage_if_orientation_verified"] = worst([("CURRENT_SKILL_WITH_NEW_BINDING" if (a["skill"] == "PICK" and a["class"] == "NEW_SCRIPTED_SKILL_REQUIRED" and a["grasp"].get("binding_if_orientation_verified")
                                                      and (a.get("source") is None or a["source"]["support_kind"] == "TABLE_FLAT")) else a["class"]) for a in task["actions"]])
    task["articulated_instances"] = sorted(n for n, ty in inst.items() if (assets.get(ty) or {}).get("articulated"))
    task["signature"] = scene_signature(parsed)
    task["proximity"] = proximity_flags(task, assets)
    task["dest_keys"] = [a["dest"] for a in task["actions"] if a.get("dest")]
    return task


def classify_structure(task, sig_groups, assets):
    parsed = task["parsed"]
    inst, sect = inst_maps(parsed)
    atoms = parsed["goal_atoms"]
    movers = sorted({a[1] for a in atoms if len(a) > 1 and a[0].lower() in ("on", "in", "up") and sect.get(a[1]) == "object"})
    labels = []
    if len(atoms) == 1:
        labels.append("SINGLE_ACTION_OR_SINGLE_GOAL")
    if task["hard_flags"]:
        labels.append("HARD_PRECONDITION_DOMINATED")
    arts = [a for a in atoms if a[0].lower() in ("open", "close", "turnon", "turnoff")]
    dests = [a[2] for a in atoms if a[0].lower() in ("on", "in") and len(a) > 2]
    shared = len(movers) >= 2 and len(dests) != len(set(dests))
    if len(movers) >= 2 and not task["hard_flags"]:
        labels.append("COMMUTATIVE_MULTI_GOAL")
    if shared:
        labels.append("SHARED_DESTINATION_CANDIDATE")
    support = []
    for a in parsed["init"]:
        if a[0].lower() == "on" and len(a) > 2:
            res = resolve(a[2], parsed, inst, sect)
            if res["instance"] and sect.get(res["instance"]) == "object" and sect.get(a[1]) == "object":
                support.append("%s on %s" % (a[1], res["instance"]))
    if support:
        labels.append("SUPPORT_OR_STACKING_CANDIDATE")
    if any(r["finger_envelope_overlap_possible"] for r in task["proximity"]):
        labels.append("SHARED_WORKSPACE_CANDIDATE")
    grp = sig_groups[task["signature"]]
    if len(grp["tasks"]) >= 2 and len({tuple(map(tuple, t)) for t in grp["goals"]}) >= 2:
        labels.append("GOAL_CONDITIONED_RELEVANCE_CANDIDATE")
    if arts and len(atoms) >= 2 and not task["hard_flags"]:
        labels.append("ARTICULATED_SOFT_EFFECT_CANDIDATE")
    if task["coverage"] in ("NEW_LOW_LEVEL_CONTROLLER_REQUIRED", "NOT_MAPPABLE"):
        labels.append("UNSUPPORTED_SKILL")
    if task["types_missing"]:
        labels.append("UNKNOWN_NEEDS_ASSET_AUDIT")
    return labels, {"movers": movers, "shared_destination": shared, "init_support": support, "scene_group_size": len(grp["tasks"])}


# ----------------------------------------------------------------------------------------------- mechanism cards
CURRENT_VOCAB = {"GripperEmpty", "Held", "OnTable", "Open", "Inside", "AtBuffer"}
LIBERO_VOCAB = {"On", "In", "Open", "Close", "Turnon", "Turnoff", "Up"}
RESOURCE_KEYS = ("environment_constructions", "env_reset", "start_case", "skill_calls", "physical_episodes", "provider_requests", "provider_retries", "rl_transitions",
                 "optimizer_steps", "elastic_attempts", "benchmark_score_reads", "test_result_reads", "demo_replay")


def capacity(dest_type, region, item_types, assets):
    """Fraction of the destination footprint the items can fill (min/max over unknown spawn orientation)."""
    site = (assets.get(dest_type, {}).get("sites") or {}).get(region)
    if not site or len(site["size"]) < 2:
        return None
    area = 4.0 * site["size"][0] * site["size"][1]
    lo = hi = 0.0
    for t in item_types:
        e = sorted_extents(assets.get(t, {}))
        if not e:
            return None
        lo += e[0] * e[1]
        hi += e[1] * e[2]
    return {"destination_area_m2": round(area, 5), "site_size": site["size"], "sum_footprint_min_m2": round(lo, 5), "sum_footprint_max_m2": round(hi, 5),
            "fill_ratio_min": round(lo / area, 3), "fill_ratio_max": round(hi / area, 3), "material_threshold": CAPACITY_MATERIAL}


def gate(status, reason):
    return {"status": status, "reason": reason}


def evaluate_gates(f):
    """Apply G1-G15 mechanically to a card's fact sheet. Statuses: PASS, FAIL, UNVERIFIABLE_STATICALLY."""
    g = {}
    g["G1"] = gate("PASS" if f["n_legal_first_actions"] >= 2 else "FAIL", "%d legal first high-level actions in the decision state: %s" % (f["n_legal_first_actions"], f["legal_first_actions_note"]))
    if f["hard_precondition"]:
        g["G2"] = gate("FAIL", "hard precondition decides the order (%s): belongs in PRE/ADD/DEL, the contract planner returns the winner" % f["hard_precondition_note"])
    elif f["contract_gives_winner"]:
        g["G2"] = gate("FAIL", f["contract_note"])
    else:
        g["G2"] = gate("PASS", f["contract_note"])
    pe = f["persistent_effect"]
    g["G3"] = gate({True: "PASS", False: "FAIL", None: "UNVERIFIABLE_STATICALLY"}[pe["material"]], pe["note"])
    dd = f["downstream_dependence"]
    g["G4"] = gate({True: "PASS", False: "FAIL", None: "UNVERIFIABLE_STATICALLY"}[dd["value"]], dd["note"])
    g["G5"] = gate("FAIL" if f["path_only"] else "PASS", "the difference between the orders is only low-level path" if f["path_only"] else "difference is not a pure path distance: " + f["not_path_note"])
    g["G6"] = gate("PASS" if f["relation_expressible"] else "FAIL", f["relation_note"])
    g["G7"] = gate("FAIL" if f["relation_is_contract_renaming"] else "PASS", f["renaming_note"])
    cov = f["covers"]
    g["G8"] = gate("PASS" if len(cov["classes"]) >= 2 and cov["established"] else ("UNVERIFIABLE_STATICALLY" if len(cov["classes"]) >= 2 else "FAIL"), cov["note"])
    g["G9"] = gate("FAIL" if f["fixed_rule_solves"] else "PASS", f["fixed_rule_note"])
    sk = f["skills"]
    ok = sk["worst_class"] in ("CURRENT_SKILL_EXACT", "CURRENT_SKILL_WITH_NEW_BINDING") and sk["platform_ok"]
    g["G10"] = gate("PASS" if ok else "FAIL", sk["note"])
    g["G11"] = gate(f["verifier"]["status"], f["verifier"]["note"])
    g["G12"] = gate(f["evaluator"]["status"], f["evaluator"]["note"])
    g["G13"] = gate(f["restore"]["status"], f["restore"]["note"])
    g["G14"] = gate("PASS" if f["canary_episodes"] <= 4 else "FAIL", "first mechanism canary needs about %d episodes" % f["canary_episodes"])
    g["G15"] = gate("PASS", "task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read")
    return g


def classes_str(rows):
    c = Counter(rows)
    return ", ".join("%s x%d" % (k.replace("CURRENT_SKILL_", "").replace("_REQUIRED", ""), c[k]) for k in CLASS_ORDER if k in c)


PLATFORM_NOTE = ("the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its "
                 "snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding")
D0_HOST_NOTE = ("hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIBERO objects are textured and non-cubic, so perception (colour override), per-object grasp heights and "
                "container binding are new")


def base_facts(tasks, assets, iface, host):
    cov = [t["coverage"] for t in tasks]
    return {"worst_class": worst(cov), "mix": classes_str(cov)}


def assign_mechanism(t, labels):
    if "HARD_PRECONDITION_DOMINATED" in labels:
        return "M4B_ARTICULATED_HARD_PRECONDITION"
    kinds = {a.get("dest_kind") for a in t["actions"] if a.get("dest_kind")}
    if "SHARED_DESTINATION_CANDIDATE" in labels and "OPEN_CONTAINER" in kinds:
        return "M1_BASKET_TWO_ITEMS"
    if "SHARED_DESTINATION_CANDIDATE" in labels and "FLAT_APPLIANCE_TOP" in kinds:
        return "M2_STOVE_TWO_POTS"
    if "ARTICULATED_SOFT_EFFECT_CANDIDATE" in labels:
        return "M4A_ARTICULATED_PLUS_PLACE"
    if "COMMUTATIVE_MULTI_GOAL" in labels:
        return "M3_TWO_TARGETS_DISTINCT_DESTINATIONS"
    return {"libero_spatial": "M5_SPATIAL_SINGLE_GOAL_DISTRACTORS", "libero_object": "M6_OBJECT_SINGLE_GOAL_SAME_SCENE"}.get(t["suite"], "M7_SINGLE_GOAL_OTHER")


def member_numbers(members, assets):
    out = {"proximity_possible": sum(1 for t in members if any(r["finger_envelope_overlap_possible"] for r in t["proximity"])),
           "proximity_certain": sum(1 for t in members if any(r["finger_envelope_overlap_certain"] for r in t["proximity"]))}
    return out


def build_public_cards(tasks, labels_by, assets, iface):
    groups = defaultdict(list)
    for t in tasks:
        groups[assign_mechanism(t, labels_by[(t["suite"], t["index"])])].append(t)
    cards = {}
    plat = PLATFORM_NOTE
    for mid, members in sorted(groups.items()):
        ids = ["%s#%d" % (t["suite"], t["index"]) for t in members]
        cov = base_facts(members, assets, iface, "LIBERO_ENV")
        nums = member_numbers(members, assets)
        skills = {"worst_class": cov["worst_class"], "platform_ok": False, "note": "task skill coverage: %s; %s" % (cov["mix"], plat)}
        verifier = {"status": "UNVERIFIABLE_STATICALLY", "note": "BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have"}
        evaluator = {"status": "PASS", "note": "BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator"}
        restore = {"status": "PASS", "note": "official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new"}
        f = {"card_id": mid, "origin": "UNMODIFIED_PUBLIC_TASK", "host": "LIBERO_ENV", "tasks": ids, "task_names": [t["name"] for t in members], "canary_episodes": 4,
             "skills": skills, "verifier": verifier, "evaluator": evaluator, "restore": restore}
        if mid == "M1_BASKET_TWO_ITEMS":
            cap = []
            for t in members:
                items = sorted({a["object"] for a in t["actions"] if a["skill"] == "PLACE"})
                types = [next(a for a in t["actions"] if a["skill"] == "PICK" and a["object"] == o)["type"] for o in items]
                cap.append({"task": t["name"], "items": types, "capacity": capacity("basket", "contain_region", types, assets)})
            f["capacity_by_task"] = cap
            worst_fill = max((c["capacity"]["fill_ratio_max"] for c in cap if c["capacity"]), default=None)
            f.update(n_legal_first_actions=2, legal_first_actions_note="PICK item one or PICK item two; both are on the table",
                     hard_precondition=False, hard_precondition_note="", contract_gives_winner=False,
                     contract_note="PICK/PLACE contracts have no PRE/ADD/DEL that distinguishes the two orders; B_PLAN ties and breaks by id",
                     persistent_effect={"material": (worst_fill is not None and worst_fill >= CAPACITY_MATERIAL), "note": "first item occupies the basket, but the items fill at most %s of the contain_region (threshold %.2f)" % (worst_fill, CAPACITY_MATERIAL)},
                     downstream_dependence={"value": False, "note": "both orders give the same end state and skill set; only the path differs"},
                     path_only=True, not_path_note="", relation_expressible=False, relation_note="no relation separates the two orders", relation_is_contract_renaming=False, renaming_note="no relation",
                     covers={"classes": ["neutral"], "established": True, "note": "only the neutral case exists in the public tasks"},
                     fixed_rule_solves=True, fixed_rule_note="any fixed order is optimal: COMMUTATIVE_MULTI_GOAL")
            f["template"] = {"state": "both items on the table, basket empty", "candidate_A": "PICK/PLACE first item into the basket", "candidate_B": "PICK/PLACE second item first", "persistent_effect": "first item inside the basket",
                             "downstream_effect": "none beyond path; basket capacity is not binding"}
        elif mid == "M2_STOVE_TWO_POTS":
            cap = []
            for t in members:
                types = sorted({next(a for a in t["actions"] if a["skill"] == "PICK" and a["object"] == o)["type"] for o in {a["object"] for a in t["actions"] if a["skill"] == "PLACE"}})
                cap.append({"task": t["name"], "items": types, "capacity": capacity("flat_stove", "cook_region", types, assets)})
            f["capacity_by_task"] = cap
            fill = max((c["capacity"]["fill_ratio_max"] for c in cap if c["capacity"]), default=None)
            f.update(n_legal_first_actions=2, legal_first_actions_note="place pot one or pot two first; the goal also contains Turnon, which is a separate action",
                     hard_precondition=False, hard_precondition_note="", contract_gives_winner=False,
                     contract_note="no contract relation orders the two pots; Turnon is outside the current predicate vocabulary",
                     persistent_effect={"material": None if fill is None else (fill >= CAPACITY_MATERIAL), "note": "cook_region fill ratio up to %s for two pots; whether a pot on the burner blocks knob access is a physical question" % fill},
                     downstream_dependence={"value": None, "note": "knob-access and pot-collision effects cannot be established without a simulator"},
                     path_only=False, not_path_note="Turnon and the two placements interact only through physical access", relation_expressible=False, relation_note="the two pots are identical in role; the only candidate relation is knob access, which is geometric",
                     relation_is_contract_renaming=False, renaming_note="would be a geometric access relation, i.e. a safety/contract precondition if it exists",
                     covers={"classes": ["neutral"], "established": True, "note": "symmetric pots give only the neutral case"},
                     fixed_rule_solves=True, fixed_rule_note="symmetric objects: any fixed order, with Turnon first, dominates")
            f["skills"]["note"] += "; Turnon needs a rotary knob skill (NEW_LOW_LEVEL_CONTROLLER_REQUIRED)"
        elif mid == "M3_TWO_TARGETS_DISTINCT_DESTINATIONS":
            f.update(n_legal_first_actions=2, legal_first_actions_note="either object can be picked first",
                     hard_precondition=False, hard_precondition_note="", contract_gives_winner=False, contract_note="independent PICK/PLACE pairs; B_PLAN ties",
                     persistent_effect={"material": False, "note": "the two destinations are different plates or regions; placing one object does not change the other's destination"},
                     downstream_dependence={"value": False, "note": "orders are interchangeable up to path"},
                     path_only=True, not_path_note="", relation_expressible=False, relation_note="no relation to express", relation_is_contract_renaming=False, renaming_note="no relation",
                     covers={"classes": ["neutral"], "established": True, "note": "independent subtasks: neutral only"}, fixed_rule_solves=True, fixed_rule_note="commutative: any fixed order")
        elif mid == "M4A_ARTICULATED_PLUS_PLACE":
            f.update(n_legal_first_actions=2, legal_first_actions_note="turn the stove on or place the pot first",
                     hard_precondition=False, hard_precondition_note="", contract_gives_winner=False, contract_note="Turnon has no contract; no current relation orders it against the placement",
                     persistent_effect={"material": None, "note": "stove state and pot position persist; their physical interaction (knob blocked by the pot) cannot be established statically"},
                     downstream_dependence={"value": None, "note": "needs physical measurement"},
                     path_only=False, not_path_note="knob access", relation_expressible=False, relation_note="at most a geometric access relation", relation_is_contract_renaming=False, renaming_note="geometric",
                     covers={"classes": ["neutral"], "established": True, "note": "no goal variation exists in the public tasks"}, fixed_rule_solves=True, fixed_rule_note="turn on first, then place: a single ordering rule")
            f["skills"]["note"] += "; the rotary knob needs rotation actions that the executor zeroes"
        elif mid == "M4B_ARTICULATED_HARD_PRECONDITION":
            f.update(n_legal_first_actions=1, legal_first_actions_note="the container must be opened before anything can go in; closing must follow insertion",
                     hard_precondition=True, hard_precondition_note="; ".join(sorted({x for t in members for x in t["hard_flags"]}))[:300],
                     contract_gives_winner=True, contract_note="OPEN/CLOSE with Open(container) as PRE of In/PLACE gives the winner",
                     persistent_effect={"material": True, "note": "container state persists, but it is a hard precondition, not a prior"},
                     downstream_dependence={"value": True, "note": "container state gates the next action"},
                     path_only=False, not_path_note="", relation_expressible=True, relation_note="container-open relation", relation_is_contract_renaming=True, renaming_note="Open(container) is already in the current contract vocabulary",
                     covers={"classes": ["helpful"], "established": True, "note": "always required"}, fixed_rule_solves=True, fixed_rule_note="open first, close last")
        else:
            single = len(members)
            f.update(n_legal_first_actions=1,
                     legal_first_actions_note="one required action chain; moving a distractor is legal but never needed (%d of %d tasks have a distractor that may enter the finger envelope, proxy)" % (nums["proximity_possible"], single),
                     hard_precondition=False, hard_precondition_note="", contract_gives_winner=False, contract_note="a single goal chain; B_PLAN returns it directly",
                     persistent_effect={"material": False, "note": "no second goal for the first action to affect"},
                     downstream_dependence={"value": False, "note": "single goal"}, path_only=True, not_path_note="",
                     relation_expressible=(mid in ("M5_SPATIAL_SINGLE_GOAL_DISTRACTORS", "M6_OBJECT_SINGLE_GOAL_SAME_SCENE")),
                     relation_note="the language relation identifies WHICH object to move (grounding), not WHICH order to use",
                     relation_is_contract_renaming=False, renaming_note="goal grounding",
                     covers={"classes": ["helpful", "neutral"] if mid != "M7_SINGLE_GOAL_OTHER" else ["neutral"], "established": True, "note": "same-scene goal switching only changes the target object"},
                     fixed_rule_solves=True, fixed_rule_note="act on the goal object directly: one rule solves every task")
        f["gates"] = evaluate_gates(f)
        f["failed_gates"] = [k for k, v in f["gates"].items() if v["status"] == "FAIL"]
        f["unverifiable_gates"] = [k for k, v in f["gates"].items() if v["status"] == "UNVERIFIABLE_STATICALLY"]
        f["skill_mix"] = cov["mix"]
        f["proximity"] = nums
        cards[mid] = f
    return cards


def build_derived_cards(assets, iface):
    """Controlled variants built from public LIBERO assets. Any of them would be LIBERO_DERIVED_DIAGNOSTIC_TASK."""
    gc = {t: grasp_class(a) for t, a in assets.items() if a.get("resolved")}
    grocery = ["alphabet_soup", "bbq_sauce", "butter", "chocolate_pudding", "cream_cheese", "ketchup", "milk", "orange_juice", "salad_dressing", "tomato_sauce"]
    strict_bind = [t for t in grocery if gc.get(t, {}).get("strict_binding")]
    orient_bind = [t for t in grocery if gc.get(t, {}).get("binding_if_orientation_verified")]
    basket = assets.get("basket", {})
    bsite = (basket.get("sites") or {}).get("contain_region")
    pile = {}
    for t in grocery:
        e = sorted_extents(assets.get(t, {}))
        if e:
            pile[t] = {"min_height": round(e[0], 3), "max_height": round(e[2], 3)}
    basket_depth = round(bsite["size"][2] * 2, 3) if bsite and len(bsite["size"]) >= 3 else None
    cards = {}

    def mk(cid, host, **kw):
        f = {"card_id": cid, "origin": "LIBERO_DERIVED_DIAGNOSTIC_TASK", "host": host, "canary_episodes": 4}
        f.update(kw)
        f["gates"] = evaluate_gates(f)
        f["failed_gates"] = [k for k, v in f["gates"].items() if v["status"] == "FAIL"]
        f["unverifiable_gates"] = [k for k, v in f["gates"].items() if v["status"] == "UNVERIFIABLE_STATICALLY"]
        return f

    common_ok = {"verifier": {"status": "UNVERIFIABLE_STATICALLY", "note": "postconditions (Held, In basket) need segmentation of the chosen LIBERO assets; colour override of materials is a design choice, not an existing component"},
                 "evaluator": {"status": "PASS", "note": "BDDL-style predicates on simulator state give an independent Evaluator"},
                 "restore": {"status": "UNVERIFIABLE_STATICALLY", "note": "the project's snapshot restore is D0-only; LIBERO assets inside D0 would need the restore contract re-validated in a physical card"}}
    cards["DV1_STACK_ORDER_IN_SHARED_BASKET"] = mk(
        "DV1_STACK_ORDER_IN_SHARED_BASKET", "D0_ENV_WITH_LIBERO_ASSETS", tasks=["derived: two grocery items, one public basket"],
        template={"state": "two items on the table, empty basket; PLACE drops at a fixed basket site", "candidate_A": "PLACE the sturdy/heavy item first", "candidate_B": "PLACE the fragile/light item first",
                  "persistent_effect": "pile order in the basket: the second item lands on the first", "downstream_effect": "tipping or ejection of the top item would force re-pick (extra skills)"},
        n_legal_first_actions=2, legal_first_actions_note="either item can be placed first",
        hard_precondition=False, hard_precondition_note="", contract_gives_winner=False, contract_note="no contract relation orders the two items (the drop site occupancy is not a predicate)",
        persistent_effect={"material": None, "note": "pile heights per item (asset z extents, orientation unverified): %s; basket contain_region depth %s m. Whether the second drop topples or ejects an item is a contact-dynamics question." % (json.dumps({k: pile[k] for k in list(pile)[:4]}), basket_depth)},
        downstream_dependence={"value": None, "note": "success of In(basket) and the rework rate need a physical probe; the static asset data cannot establish them"},
        path_only=False, not_path_note="an ejected item needs a re-pick, which is extra skills not path", relation_expressible=True, relation_note="sturdy/heavy/large first is a natural commonsense relation readable from the image and task language",
        relation_is_contract_renaming=False, renaming_note="weight/fragility are not in the contract vocabulary",
        covers={"classes": ["helpful", "neutral", "harmful"], "established": False, "note": "helpful/neutral/reversed can be designed by choosing item pairs, but the physical existence of the effect is not shown"},
        fixed_rule_solves=False, fixed_rule_note="a 'larger first' rule fails on pairs where size and fragility disagree; this can be designed in, but the real effect size is unmeasured",
        skills={"worst_class": "NEW_SCRIPTED_SKILL_REQUIRED" if len(strict_bind) < 4 else "CURRENT_SKILL_WITH_NEW_BINDING", "platform_ok": False,
                "note": "only %d of %d grocery assets are orientation-agnostic pinchable (%s); %d more would be pinchable if spawn orientation were verified (%s). %s" % (len(strict_bind), len(grocery), ",".join(strict_bind) or "none", len(orient_bind) - len(strict_bind), ",".join(sorted(set(orient_bind) - set(strict_bind))) or "none", D0_HOST_NOTE)},
        **common_ok)
    cards["DV2_SUPPORT_REMOVE_TOP_FIRST"] = mk(
        "DV2_SUPPORT_REMOVE_TOP_FIRST", "D0_ENV_WITH_LIBERO_ASSETS", tasks=["derived: item A resting on item B, goals vary over A, B or both"],
        template={"state": "A rests on B", "candidate_A": "move A first", "candidate_B": "pick B directly", "persistent_effect": "A removed from the support of B",
                  "downstream_effect": "B cannot be picked while A rests on it"},
        n_legal_first_actions=1, legal_first_actions_note="picking B is blocked while A rests on it (hard); the only legal first move for a B-goal is A",
        hard_precondition=True, hard_precondition_note="Clear(B) / not supported-by(A, B)", contract_gives_winner=True, contract_note="a support predicate in PRE of PICK(B) determines the order",
        persistent_effect={"material": True, "note": "support relation persists, but it is a hard precondition"}, downstream_dependence={"value": True, "note": "gates the second action"},
        path_only=False, not_path_note="", relation_expressible=True, relation_note="support relation is visible", relation_is_contract_renaming=True, renaming_note="Support is a precondition predicate, i.e. a contract fact",
        covers={"classes": ["helpful", "neutral"], "established": True, "note": "A-only goals are neutral, B goals are forced"}, fixed_rule_solves=True, fixed_rule_note="remove whatever rests on a needed object",
        skills={"worst_class": "NEW_SCRIPTED_SKILL_REQUIRED", "platform_ok": False, "note": "stacked starts violate the OnTable contract and the Verifier OnTable bound; " + D0_HOST_NOTE}, **common_ok)
    cards["DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE"] = mk(
        "DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE", "D0_ENV_WITH_LIBERO_ASSETS", tasks=["derived: a neighbour placed inside the pinch envelope of the goal object"],
        template={"state": "neighbour next to the goal object", "candidate_A": "clear the neighbour first", "candidate_B": "grasp directly", "persistent_effect": "neighbour moved away", "downstream_effect": "the grasp collides with the neighbour or not"},
        n_legal_first_actions=2, legal_first_actions_note="clearing is legal; grasping directly is legal only if the envelope is free (safety mask)",
        hard_precondition=True, hard_precondition_note="finger-envelope collision is a Safety/contract-level hard mask; the BSI, OCC and VIS preflights on the same Panda gripper found no safe soft band",
        contract_gives_winner=True, contract_note="Safety hard mask already removes the colliding candidate",
        persistent_effect={"material": True, "note": "neighbour position persists"}, downstream_dependence={"value": True, "note": "collision or not"},
        path_only=False, not_path_note="", relation_expressible=True, relation_note="adjacency is visible", relation_is_contract_renaming=True, renaming_note="adjacency/collision is a geometric precondition the Safety module encodes",
        covers={"classes": ["helpful", "neutral"], "established": True, "note": "far neighbours are neutral"}, fixed_rule_solves=True, fixed_rule_note="clear any neighbour within the envelope",
        skills={"worst_class": "CURRENT_SKILL_WITH_NEW_BINDING", "platform_ok": False, "note": "PICK and PLACE_BUFFER would suffice but " + D0_HOST_NOTE}, **common_ok)
    cards["DV4_SHARED_DESTINATION_CAPACITY"] = mk(
        "DV4_SHARED_DESTINATION_CAPACITY", "D0_ENV_WITH_LIBERO_ASSETS", tasks=["derived: a plate that takes one item, a goal-relevant item and a goal-irrelevant item"],
        template={"state": "goal item and distractor both on the table, small destination", "candidate_A": "place the goal item first", "candidate_B": "place the distractor first", "persistent_effect": "destination occupied", "downstream_effect": "the goal item no longer fits"},
        n_legal_first_actions=2, legal_first_actions_note="placing the distractor on the destination is legal", hard_precondition=False, hard_precondition_note="",
        contract_gives_winner=True, contract_note="the goal contract only wants the goal item at the destination; B_PLAN never plans the distractor",
        persistent_effect={"material": True, "note": "occupied destination"}, downstream_dependence={"value": True, "note": "blocks the goal item"}, path_only=False, not_path_note="",
        relation_expressible=True, relation_note="goal-relevance of an item is visible in language", relation_is_contract_renaming=True, renaming_note="relevance is the goal itself",
        covers={"classes": ["helpful", "harmful"], "established": True, "note": "distractor-first is only ever harmful"}, fixed_rule_solves=True, fixed_rule_note="never place a non-goal item on the destination",
        skills={"worst_class": "CURRENT_SKILL_WITH_NEW_BINDING", "platform_ok": False, "note": D0_HOST_NOTE}, **common_ok)
    cards["DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE"] = mk(
        "DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE", "LIBERO_ENV", tasks=["derived: official scene, goal switched between objects"],
        template={"state": "same scene, goal names one of several objects", "candidate_A": "act on the named object", "candidate_B": "act on another object", "persistent_effect": "none", "downstream_effect": "none beyond cost"},
        n_legal_first_actions=2, legal_first_actions_note="acting on a non-goal object is legal", hard_precondition=False, hard_precondition_note="", contract_gives_winner=True, contract_note="the goal contract names the object",
        persistent_effect={"material": False, "note": "no persistent effect on the goal chain"}, downstream_dependence={"value": False, "note": "only extra cost for the wrong object"}, path_only=False, not_path_note="",
        relation_expressible=True, relation_note="language names the object", relation_is_contract_renaming=True, renaming_note="goal grounding",
        covers={"classes": ["helpful", "harmful"], "established": True, "note": "wrong-object-first is only harmful"}, fixed_rule_solves=True, fixed_rule_note="act on the goal object",
        skills={"worst_class": "NEW_SCRIPTED_SKILL_REQUIRED", "platform_ok": False, "note": PLATFORM_NOTE}, **common_ok)
    cards["DV6_ARTICULATED_OPTIONAL_PREPARATION"] = mk(
        "DV6_ARTICULATED_OPTIONAL_PREPARATION", "LIBERO_ENV", tasks=["derived: drawer opened early to stage an item that a later goal may need"],
        template={"state": "closed drawer, goal may or may not need it", "candidate_A": "open the drawer early", "candidate_B": "wait", "persistent_effect": "drawer open", "downstream_effect": "saves or wastes a skill"},
        n_legal_first_actions=2, legal_first_actions_note="opening is legal at any time", hard_precondition=False, hard_precondition_note="",
        contract_gives_winner=False, contract_note="optional preparation has no contract winner",
        persistent_effect={"material": True, "note": "drawer state persists"}, downstream_dependence={"value": True, "note": "saves the later OPEN when a later goal needs the drawer"},
        path_only=False, not_path_note="", relation_expressible=True, relation_note="whether a later goal needs the drawer is expressible", relation_is_contract_renaming=True,
        renaming_note="a later In(drawer) goal makes Open(drawer) a hard PRE of that goal; the 'optional' reading disappears once the goal is stated",
        covers={"classes": ["helpful", "neutral"], "established": True, "note": "harmful when the drawer blocks nothing but costs a skill"}, fixed_rule_solves=True, fixed_rule_note="open iff a pending goal needs the drawer: a one-line rule on the goal text",
        skills={"worst_class": "NEW_SCRIPTED_SKILL_REQUIRED", "platform_ok": False, "note": "sliding drawer needs a handle-pull skill; " + PLATFORM_NOTE}, **common_ok)
    return cards, {"strict_binding_grocery": strict_bind, "orientation_binding_grocery": orient_bind, "grasp_classes": gc, "pile": pile, "basket_depth_m": basket_depth}


ATTACK_KEYS = ("A1_fixed_rule", "A2_contract_planner", "A3_static_prior", "A4_llm_vlm_direct", "A5_memorisation")


def attacks(card):
    f = card
    a = {}
    a["A1_fixed_rule"] = {"status": "FIXED_HEURISTIC_SOLVABLE" if f["fixed_rule_solves"] else "NOT_SOLVED_BY_ONE_RULE", "note": f["fixed_rule_note"]}
    a["A2_contract_planner"] = {"status": "CONTRACT_SOLVABLE" if (f["hard_precondition"] or f["contract_gives_winner"]) else "CONTRACT_TIE_NO_WINNER", "note": f["contract_note"]}
    a["A3_static_prior"] = {"status": "STATIC_PRIOR_SUFFICIENT" if (f["fixed_rule_solves"] and f["relation_expressible"]) else "NOT_APPLICABLE", "note": "relation without consequence information is enough when one rule decides every state"}
    a["A4_llm_vlm_direct"] = {"status": "KEEP_AS_LLM_VLM_BASELINE" if f["relation_expressible"] else "NOT_APPLICABLE", "note": "would require neutral, reversed, harmful and goal-dependent sub-conditions: %s" % ",".join(f["covers"]["classes"])}
    memorisable = f["origin"] == "UNMODIFIED_PUBLIC_TASK"
    a["A5_memorisation"] = {"status": "MEMORISABLE_FIXED_SCENES" if memorisable else "NEEDS_STATE_AND_GOAL_VARIATION", "note": "public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners" if memorisable else "a derived family must randomise layout and goal so labels do not leak"}
    return a


# ------------------------------------------------------------------------------------------------------- outputs
def protected_hashes(root):
    root = Path(root)
    out = {}
    for sub in ("runs/final_master/S4", "runs/final_master/2.1.1/T_B", "runs/stage_0a", "docs/authoritative", "status", "src", "configs"):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "tp_libero_audit" not in str(p) and "__pycache__" not in p.parts and not re.search(r"holdout|test30|test_id|final_test", str(p).replace("\\", "/"), re.I):
                out[str(p.relative_to(root))] = sha_file(p)
    return out


def prior_card_evidence(root):
    out = {}
    for card in ("tp_bsi_preflight", "tp_occ_preflight", "tp_vis_preflight"):
        for p in sorted((Path(root) / "runs/final_master/S4" / card).glob("*/preflight_verdict.json")):
            try:
                v = json.loads(p.read_text())
                out[card] = {"verdict": v.get("verdict"), "stop_phase": v.get("stop_phase"), "file": str(p.relative_to(root))}
            except Exception as exc:
                out[card] = {"error": str(exc)}
    return out


def task_inventory_row(t, labels, extra):
    p = t["parsed"]
    inst, sect = inst_maps(p)
    return {"suite": t["suite"], "task_index": t["index"], "task_name": t["name"], "bddl_path": t["bddl"], "language": p["language"],
            "fixtures": [n + ":" + ty for n, ty in p["fixtures"]], "movable_objects": [n + ":" + ty for n, ty in p["objects"]], "objects_of_interest": p["obj_of_interest"],
            "init_predicates": [" ".join(a) for a in p["init"]], "goal_predicates": [" ".join(a) for a in p["goal_atoms"]],
            "articulated_objects": t["articulated_instances"],
            "support_predicates": extra["init_support"], "region_count": len(p["regions"]), "initial_state_source": "init_files/%s/%s.pruned_init (official, hashed only)" % (t["suite"], t["name"]),
            "asset_dependencies": sorted({ty for ty in inst.values()}), "scene_signature": t["signature"][:16]}


def bddl_json(t):
    p = t["parsed"]
    return {"suite": t["suite"], "task_index": t["index"], "task_name": t["name"], "bddl": t["bddl"], "problem": p["problem"], "domain": p["domain"], "language": p["language"],
            "fixtures": p["fixtures"], "objects": p["objects"], "obj_of_interest": p["obj_of_interest"], "init": p["init"], "goal_atoms": p["goal_atoms"], "goal_ops": sorted(set(p["goal_ops"])),
            "regions": [{"target": k[0], "name": k[1], "ranges": v["ranges"], "yaw": v["yaw"]} for k, v in sorted(p["regions"].items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))]}


FILL_DEFAULT = {"expected_extra_skills": "not established statically", "expected_effect_on_J": "not established statically"}


def fill_template(c):
    tp = dict(c.get("template", {}))
    cls = c["covers"]["classes"]
    tp.setdefault("state", "; ".join(c.get("tasks", [])[:3]) if c["origin"] == "UNMODIFIED_PUBLIC_TASK" else "")
    tp.setdefault("candidate_A", "first required action chain")
    tp.setdefault("candidate_B", "alternative first action")
    tp["both_legal"] = "%d legal first actions: %s" % (c["n_legal_first_actions"], c["legal_first_actions_note"])
    tp.setdefault("persistent_effect", c["persistent_effect"]["note"])
    tp.setdefault("downstream_effect", c["downstream_dependence"]["note"])
    tp["hard_precondition_or_not"] = "HARD: " + c["hard_precondition_note"] if c["hard_precondition"] else "NOT_HARD"
    tp["candidate_consequence_role"] = "none: a fixed rule already decides" if c["fixed_rule_solves"] else "needed: no single rule decides every state"
    tp["semantic_relation"] = c["relation_note"]
    for k, lab in (("helpful_case", "helpful"), ("neutral_case", "neutral"), ("reversed_or_harmful_case", "harmful")):
        tp[k] = ("present: " if lab in cls else "absent: ") + c["covers"]["note"]
    if c["path_only"]:
        tp["expected_extra_skills"] = "0: the orders differ only by path"
        tp["expected_effect_on_J"] = "path time only"
    else:
        tp["expected_extra_skills"] = FILL_DEFAULT["expected_extra_skills"]
        tp["expected_effect_on_J"] = FILL_DEFAULT["expected_effect_on_J"]
    tp["simple_rule_attack"] = c["fixed_rule_note"]
    return tp


def cards_markdown(cards):
    L = ["# Candidate mechanism cards", "", "Statuses per hard gate: PASS, FAIL, UNVERIFIABLE_STATICALLY (needs a physical measurement that this card is forbidden to take). A candidate needs all G1-G15 = PASS.", ""]
    for cid, c in cards.items():
        tp = fill_template(c)
        L += ["## %s" % cid, "", "origin: %s, host: %s" % (c["origin"], c["host"]), "tasks: %s" % ("; ".join(c["tasks"]) if len(c["tasks"]) <= 12 else "%d tasks" % len(c["tasks"])), ""]
        for k in ("state", "candidate_A", "candidate_B", "both_legal", "persistent_effect", "downstream_effect", "hard_precondition_or_not", "candidate_consequence_role", "semantic_relation",
                  "helpful_case", "neutral_case", "reversed_or_harmful_case", "expected_extra_skills", "expected_effect_on_J", "simple_rule_attack"):
            L.append("- %s: %s" % (k, tp.get(k, "")))
        L += ["", "| gate | status | reason |", "|---|---|---|"]
        for g in GATES:
            r = c["gates"][g]
            L.append("| %s | %s | %s |" % (g, r["status"], r["reason"].replace("|", "/")))
        L += ["", "failed: %s; unverifiable: %s" % (", ".join(c["failed_gates"]) or "none", ", ".join(c["unverifiable_gates"]) or "none"), ""]
    return "\n".join(L)


def attack_markdown(cards, which, title, keys):
    L = ["# " + title, ""]
    L += ["| card | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for cid, c in cards.items():
        a = attacks(c)
        L.append("| %s | " % cid + " | ".join(a[k]["status"] for k in keys) + " |")
    L.append("")
    for cid, c in cards.items():
        a = attacks(c)
        L.append("### " + cid)
        for k in keys:
            L.append("- %s: %s. %s" % (k, a[k]["status"], a[k]["note"]))
        L.append("")
    return "\n".join(L)


def planner_markdown(cards, planner_info):
    L = ["# Planner attack (Attack 2)", "", "B_PLAN: %s" % planner_info, "",
         "A card survives this attack only when the contract gives NO winner (CONTRACT_TIE_NO_WINNER) and the order is not a hard precondition.", "",
         "| card | verdict | reason |", "|---|---|---|"]
    for cid, c in cards.items():
        a = attacks(c)["A2_contract_planner"]
        L.append("| %s | %s | %s |" % (cid, a["status"], a["note"].replace("|", "/")))
    return "\n".join(L) + "\n"


def objection_markdown(cards):
    L = ["# Reviewer objection matrix", "", "| card | objection | evidence in this audit | survives |", "|---|---|---|---|"]
    for cid, c in cards.items():
        ob = []
        if c["fixed_rule_solves"]:
            ob.append(("a one-line rule solves every state", c["fixed_rule_note"]))
        if c["contract_gives_winner"] or c["hard_precondition"]:
            ob.append(("the contract already decides it", c["contract_note"]))
        if c["path_only"]:
            ob.append(("the orders differ only by path", "commutative / path-only"))
        if c["unverifiable_gates"]:
            ob.append(("the effect size is unmeasured", "unverifiable statically: " + ", ".join(c["unverifiable_gates"])))
        if c["gates"]["G10"]["status"] == "FAIL":
            ob.append(("the current skills cannot execute it", c["gates"]["G10"]["reason"][:160]))
        if c["origin"] == "LIBERO_DERIVED_DIAGNOSTIC_TASK":
            ob.append(("a modified task is not the LIBERO benchmark", "must be labelled LIBERO_DERIVED_DIAGNOSTIC_TASK and kept out of the public-validation table"))
        survives = "yes" if not c["failed_gates"] and not c["unverifiable_gates"] else "no"
        for o, e in ob or [("none", "")]:
            L.append("| %s | %s | %s | %s |" % (cid, o, e.replace("|", "/"), survives))
    return "\n".join(L) + "\n"


def public_validation(tasks, iface):
    rows = []
    for t in tasks:
        rows.append({"suite": t["suite"], "task_index": t["index"], "task_name": t["name"], "coverage_class": t["coverage"], "coverage_if_orientation_verified": t["coverage_if_orientation_verified"],
                     "actions": ",".join(a["skill"] for a in t["actions"]), "blocking_reasons": " || ".join(sorted({r for a in t["actions"] if a["class"] == t["coverage"] for r in a["reasons"]}))[:400],
                     "needs_platform_binding": True, "official_goal_kept": True, "official_init_kept": True, "maps_to_candidate_selection": False})
    total = len(rows)
    ex = sum(r["coverage_class"] == "CURRENT_SKILL_EXACT" for r in rows)
    bi = sum(r["coverage_class"] == "CURRENT_SKILL_WITH_NEW_BINDING" for r in rows)
    ub = sum(r["coverage_if_orientation_verified"] in ("CURRENT_SKILL_EXACT", "CURRENT_SKILL_WITH_NEW_BINDING") for r in rows)
    reasons = Counter()
    for r in rows:
        if r["coverage_class"] not in ("CURRENT_SKILL_EXACT", "CURRENT_SKILL_WITH_NEW_BINDING"):
            reasons[r["coverage_class"]] += 1
    summary = {"total_tasks": total, "exactly_supported": ex, "binding_supported": bi, "unsupported": total - ex - bi, "coverage_rate": round((ex + bi) / total, 4) if total else 0.0,
               "upper_bound_if_orientation_verified": round(ub / total, 4) if total else 0.0, "reasons": dict(reasons),
               "per_suite": {s: {"total": sum(r["suite"] == s for r in rows), "exact_or_binding": sum(r["suite"] == s and r["coverage_class"] in ("CURRENT_SKILL_EXACT", "CURRENT_SKILL_WITH_NEW_BINDING") for r in rows),
                                 "upper_bound": sum(r["suite"] == s and r["coverage_if_orientation_verified"] in ("CURRENT_SKILL_EXACT", "CURRENT_SKILL_WITH_NEW_BINDING") for r in rows)} for s in SUITES}}
    rate = summary["coverage_rate"]
    summary["label_if_future_run"] = "UNMODIFIED_LIBERO_PUBLIC_VALIDATION" if rate >= HIGH_COVERAGE else "LIBERO_COMPATIBLE_SUBSET"
    summary["label_rule"] = "UNMODIFIED_LIBERO_PUBLIC_VALIDATION needs coverage >= %.2f of the frozen scope, official goals/inits/identities unchanged and a platform binding; otherwise LIBERO_COMPATIBLE_SUBSET; modified tasks are LIBERO_DERIVED_DIAGNOSTIC" % HIGH_COVERAGE
    return rows, summary


def public_recommendation_md(summ, iface):
    L = ["# Public validation recommendation", "",
         "Scope: LIBERO-Spatial, -Object, -Goal and -10 as frozen by the upstream task map (LIBERO-90 not audited).", "",
         "- total tasks: %d; exactly supported: %d; supported with new binding: %d; unsupported: %d" % (summ["total_tasks"], summ["exactly_supported"], summ["binding_supported"], summ["unsupported"]),
         "- coverage rate (strict): %.3f; upper bound if spawn orientation were verified: %.3f" % (summ["coverage_rate"], summ["upper_bound_if_orientation_verified"]),
         "- unsupported reasons: %s" % json.dumps(summ["reasons"]), "- per suite: %s" % json.dumps(summ["per_suite"]), "",
         "Label rule: " + summ["label_rule"], "", "Result: **%s** is the only label this scope could carry today, so the unmodified public tasks cannot be called a LIBERO benchmark result." % summ["label_if_future_run"], "",
         "Why the public tasks are poor mechanism carriers as well as poor coverage: most are single-goal chains, so Full, B2 and the contract planner see the same one-step plan and the comparison cannot separate priors.", "",
         "What a future public-validation run needs (not authorised here): a LIBERO environment binding (new), perception for textured objects, a BDDL-based Evaluator, handle/knob/rim skills, and an explicit statement of which tasks are covered. Report coverage per suite and never extrapolate to the whole benchmark.", ""]
    return "\n".join(L)


def rejection_reason(t, mech_cards, mech_of):
    c = mech_cards[mech_of[(t["suite"], t["index"])]]
    return "%s: failed %s (first: %s)" % (c["card_id"], ",".join(c["failed_gates"]), c["gates"][c["failed_gates"][0]]["reason"][:140]) if c["failed_gates"] else "none"


def write_not_requested(out):
    (Path(out) / "next_mechanism_canary_request.md").write_text("# next_mechanism_canary_request\n\nNOT_REQUESTED\n", encoding="utf-8")


def run(root, libero_root, out, pip_root=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard = Guard()
    events = [{"event": "registry_key_rule_aligned_with_upstream", "technical_reason": "upstream register_object splits class names at digits as well as capitals (Chefmate8Frypan -> chefmate_8_frypan); the first parser probe left that type unresolved",
               "science_unchanged": True, "regression_test": "test_registry_key_rule_matches_upstream"}]
    before = protected_hashes(root)
    write_json(out / "protected_before.json", before)
    head = git_out(["rev-parse", "HEAD"], root)
    dirty = git_out(["status", "--porcelain", "--untracked-files=no"], root)
    write_json(out / "authorization.json", {
        "card": CARD, "authorized_by": "user chat instruction approving CP-DISR-TP-LIBERO-OPPORTUNITY-AUDIT-1", "baseline_branch": "codex/cp-disr-tp-vis-preflight", "baseline_expected_prefix": BASELINE_PREFIX,
        "frozen_scope": {"suites": list(SUITES), "deferred_not_audited": list(DEFERRED_SUITES)},
        "limits": {k: 0 for k in RESOURCE_KEYS},
        "thresholds_declared_before_run": {"PINCH_LIMIT": PINCH_LIMIT, "ELEVATED_MAX": ELEVATED_MAX, "ONTABLE_BOUND": ONTABLE_BOUND, "CAPACITY_MATERIAL": CAPACITY_MATERIAL, "FINGER_HALF_X": FINGER_HALF_X,
                                           "FINGER_OUTER_Y": FINGER_OUTER_Y, "HIGH_COVERAGE": HIGH_COVERAGE, "ROUND_RATIO": ROUND_RATIO},
        "gate_rule": "all of G1-G15 must be PASS; FAIL and UNVERIFIABLE_STATICALLY both block selection",
        "g10_scope": "skill coverage AND platform binding: a LIBERO-hosted task needs a new environment/perception/Evaluator binding that is not limited"})
    write_json(out / "source_identity.json", {"project_head_full": head, "expected_prefix": BASELINE_PREFIX, "prefix_matches": head.startswith(BASELINE_PREFIX), "expected_full": BASELINE_FULL,
                                              "full_matches": head == BASELINE_FULL, "branch": git_out(["branch", "--show-current"], root), "dirty_tracked_files": len([l for l in dirty.splitlines() if l.strip()])})
    ident = upstream_identity(libero_root, guard, pip_root)
    write_json(out / "upstream_libero_identity.json", ident)
    if not ident["sha_matches_expected"]:
        raise RuntimeError("UPSTREAM_SHA_MISMATCH:%s" % ident["head_sha"])
    lib = Path(libero_root).resolve() / "libero" / "libero"
    tmap = load_task_map(lib / "benchmark" / "libero_suite_task_map.py", guard)
    reg = load_registry(lib, guard)
    parsed_all, errors = {}, []
    for s in SUITES:
        for i, name in enumerate(tmap[s]):
            rel = "libero/libero/bddl_files/%s/%s.bddl" % (s, name)
            try:
                parsed_all[(s, i)] = (name, rel, parse_bddl(guard.read(lib / "bddl_files" / s / (name + ".bddl"))))
            except Exception as exc:
                errors.append({"suite": s, "index": i, "name": name, "error": "%s: %s" % (type(exc).__name__, exc)})
    types = sorted({ty for (_, _, p) in parsed_all.values() for _, ty in p["fixtures"] + p["objects"] if ty in reg})
    assets = build_asset_table(lib, reg, types, guard)
    iface = current_skill_interface(root, guard)
    tasks = [task_row(s, i, name, rel, p, reg, assets) for (s, i), (name, rel, p) in sorted(parsed_all.items())]
    sig_groups = defaultdict(lambda: {"tasks": [], "goals": []})
    for t in tasks:
        sig_groups[t["signature"]]["tasks"].append((t["suite"], t["index"]))
        sig_groups[t["signature"]]["goals"].append([tuple(a) for a in t["parsed"]["goal_atoms"]])
    labels_by, extras = {}, {}
    for t in tasks:
        lab, ex = classify_structure(t, sig_groups, assets)
        labels_by[(t["suite"], t["index"])], extras[(t["suite"], t["index"])] = lab, ex
    # ---- inventories
    write_csv(out / "libero_task_inventory.csv", [task_inventory_row(t, labels_by[(t["suite"], t["index"])], extras[(t["suite"], t["index"])]) for t in tasks],
              ["suite", "task_index", "task_name", "bddl_path", "language", "fixtures", "movable_objects", "objects_of_interest", "init_predicates", "goal_predicates", "articulated_objects",
               "support_predicates", "region_count", "initial_state_source", "asset_dependencies", "scene_signature"])
    write_json(out / "libero_bddl_inventory.json", {"upstream_sha": ident["head_sha"], "task_map_sha256": ident["task_map_sha256"], "parse_errors": errors, "tasks": [bddl_json(t) for t in tasks]})
    unresolved = sorted({ty for p in (v[2] for v in parsed_all.values()) for _, ty in p["fixtures"] + p["objects"] if ty not in reg and not is_surface(ty, reg)})
    write_json(out / "fixture_and_asset_inventory.json", {"registered_object_classes": {k: {kk: vv for kk, vv in v.items() if kk != "chain"} for k, v in reg.items()}, "asset_table": assets,
                                                           "surface_types_without_asset": sorted({ty for p in (v[2] for v in parsed_all.values()) for _, ty in p["fixtures"] + p["objects"] if is_surface(ty, reg)}),
                                                           "unresolved_types": unresolved})
    preds = Counter()
    pred_by_suite = defaultdict(Counter)
    for (s, _), (_, _, p) in parsed_all.items():
        for a in p["init"]:
            preds["init:" + a[0]] += 1
            pred_by_suite[s]["init:" + a[0]] += 1
        for a in p["goal_atoms"]:
            preds["goal:" + a[0]] += 1
            pred_by_suite[s]["goal:" + a[0]] += 1
    write_json(out / "predicate_inventory.json", {"counts": dict(preds), "by_suite": {k: dict(v) for k, v in pred_by_suite.items()}, "libero_vocabulary_used": sorted({k.split(":")[1] for k in preds}),
                                                  "current_cp_disr_vocabulary": iface["contract_predicates"], "no_current_equivalent": sorted({"Close", "Turnon", "Up"} & {k.split(":")[1] for k in preds}),
                                                  "note": "On/In map to AtBuffer/Inside/OnTable only for table and open-container destinations; support relations have no current predicate"})
    write_json(out / "current_skill_interface.json", iface)
    # ---- skill coverage and structure
    srows = []
    for t in tasks:
        for a in t["actions"]:
            srows.append({"suite": t["suite"], "task_index": t["index"], "task_name": t["name"], "action": a["skill"], "object": a.get("object"), "class": a["class"],
                          "dest_kind": a.get("dest_kind", ""), "reasons": " || ".join(a["reasons"])[:500], "needs_platform_binding": True})
    write_csv(out / "skill_coverage_matrix.csv", srows, ["suite", "task_index", "task_name", "action", "object", "class", "dest_kind", "reasons", "needs_platform_binding"])
    write_csv(out / "task_structure_classification.csv", [{"suite": t["suite"], "task_index": t["index"], "task_name": t["name"], "labels": labels_by[(t["suite"], t["index"])], "task_coverage": t["coverage"],
                                                           "hard_flags": t["hard_flags"], "movers": extras[(t["suite"], t["index"])]["movers"], "init_support": extras[(t["suite"], t["index"])]["init_support"],
                                                           "scene_group_size": extras[(t["suite"], t["index"])]["scene_group_size"],
                                                           "finger_overlap_possible": sum(1 for r in t["proximity"] if r["finger_envelope_overlap_possible"]),
                                                           "finger_overlap_certain": sum(1 for r in t["proximity"] if r["finger_envelope_overlap_certain"])} for t in tasks],
              ["suite", "task_index", "task_name", "labels", "task_coverage", "hard_flags", "movers", "init_support", "scene_group_size", "finger_overlap_possible", "finger_overlap_certain"])
    # ---- mechanism cards
    pub = build_public_cards(tasks, labels_by, assets, iface)
    der, der_info = build_derived_cards(assets, iface)
    cards = {}
    cards.update(pub)
    cards.update(der)
    mech_of = {(t["suite"], t["index"]): assign_mechanism(t, labels_by[(t["suite"], t["index"])]) for t in tasks}
    write_json(out / "candidate_mechanism_cards.json", {"cards": cards, "derived_card_asset_facts": der_info, "prior_cards": prior_card_evidence(root)})
    (out / "candidate_mechanism_cards.md").write_text(cards_markdown(cards), encoding="utf-8")
    (out / "simple_rule_attack.md").write_text(attack_markdown(cards, None, "Attacks 1, 3, 4 and 5", ("A1_fixed_rule", "A3_static_prior", "A4_llm_vlm_direct", "A5_memorisation")), encoding="utf-8")
    planner = "uniform-cost search over the contract state space with a relaxed goal-distance heuristic; cost is the reference skill duration (src/cp_disr/baselines/b_plan.py); ties break by contract id"
    (out / "planner_attack.md").write_text(planner_markdown(cards, planner), encoding="utf-8")
    (out / "reviewer_objection_matrix.md").write_text(objection_markdown(cards), encoding="utf-8")
    write_json(out / "contract_redundancy_audit.json", {"current_vocabulary": sorted(CURRENT_VOCAB), "libero_vocabulary": sorted(LIBERO_VOCAB), "planner": planner,
                                                        "per_card": {cid: {"hard_precondition": c["hard_precondition"], "contract_gives_winner": c["contract_gives_winner"], "relation_is_contract_renaming": c["relation_is_contract_renaming"],
                                                                           "note": c["contract_note"], "renaming_note": c["renaming_note"]} for cid, c in cards.items()}})
    opp = []
    for t in tasks:
        c = cards[mech_of[(t["suite"], t["index"])]]
        opp.append({"suite": t["suite"], "task_index": t["index"], "task_name": t["name"], "mechanism": c["card_id"], "labels": labels_by[(t["suite"], t["index"])], "failed_gates": c["failed_gates"],
                    "unverifiable_gates": c["unverifiable_gates"], "rejection_reason": rejection_reason(t, cards, mech_of)})
    for cid, c in der.items():
        opp.append({"suite": "DERIVED", "task_index": "", "task_name": cid, "mechanism": cid, "labels": [], "failed_gates": c["failed_gates"], "unverifiable_gates": c["unverifiable_gates"],
                    "rejection_reason": "%s: failed %s" % (cid, ",".join(c["failed_gates"]))})
    write_csv(out / "prior_opportunity_matrix.csv", opp, ["suite", "task_index", "task_name", "mechanism", "labels", "failed_gates", "unverifiable_gates", "rejection_reason"])
    # ---- denominator
    den = []
    for s in SUITES:
        ts = [t for t in tasks if t["suite"] == s]
        rej = Counter(rejection_reason(t, cards, mech_of).split(":")[0] for t in ts)
        den.append({"suite": s, "task_count": len(tmap[s]), "parsed": len(ts), "unsupported": sum(1 for t in ts if t["coverage"] in ("NEW_LOW_LEVEL_CONTROLLER_REQUIRED", "NOT_MAPPABLE")),
                    "candidate": sum(1 for t in ts if not cards[mech_of[(t["suite"], t["index"])]]["failed_gates"]), "rejected": sum(1 for t in ts if cards[mech_of[(t["suite"], t["index"])]]["failed_gates"]),
                    "rejection_reason": dict(rej)})
    write_csv(out / "audit_denominator.csv", den, ["suite", "task_count", "parsed", "unsupported", "candidate", "rejected", "rejection_reason"])
    # ---- public validation
    prow, psum = public_validation(tasks, iface)
    write_csv(out / "public_validation_coverage.csv", prow + [{"suite": "ALL", "task_index": "", "task_name": json.dumps(psum), "coverage_class": "SUMMARY"}],
              ["suite", "task_index", "task_name", "coverage_class", "coverage_if_orientation_verified", "actions", "blocking_reasons", "needs_platform_binding", "official_goal_kept", "official_init_kept", "maps_to_candidate_selection"])
    write_json(out / "public_validation_summary.json", psum)
    (out / "public_validation_recommendation.md").write_text(public_recommendation_md(psum, iface), encoding="utf-8")
    # ---- selection
    passing = [cid for cid, c in cards.items() if all(c["gates"][g]["status"] == "PASS" for g in GATES)]
    near = sorted(cards.values(), key=lambda c: (len(c["failed_gates"]), len(c["unverifiable_gates"])))
    if passing:
        verdict = "CANDIDATE_PASSED_HARD_GATES_ENGINEERING_UNRESOLVED"
    else:
        verdict = "NO_LOW_COST_TP_CANDIDATE_FOUND"
    sel = {"status": verdict, "passing_cards": passing, "hard_gate_rule": "all G1-G15 PASS",
           "nearest_misses_not_selected": [{"card": c["card_id"], "failed": c["failed_gates"], "unverifiable": c["unverifiable_gates"]} for c in near[:3]],
           "ranking": "not applicable: no card passes the hard gates, so no ranking was produced",
           "decisions_that_would_change_the_outcome": [
               "authorise a physical probe (environment construction) so that the unverifiable persistent-effect gates of DV1 can be measured; this card is forbidden to do so",
               "authorise new scripted skills (yaw-aligned grasp, rim grasp, drawer pull, knob) and a LIBERO environment binding; without them G10 fails for every LIBERO-hosted task",
               "authorise hosting a derived family inside the D0 platform with LIBERO assets (still needs the physical probe)",
               "authorise LIBERO-90 (does not help: the failures are mechanism-level and skill-level, not coverage-level)"]}
    write_json(out / "selected_candidate.json", sel)
    (out / "selected_tp_design_draft.md").write_text("# selected_tp_design_draft\n\n%s\n\nNo design draft was produced because no candidate passed all hard gates.\n" % verdict, encoding="utf-8")
    write_not_requested(out)
    forbidden_loaded = sorted(m for m in ("robosuite", "mujoco", "libero", "torch", "gym", "gymnasium") if m in sys.modules)
    ledger = {k: 0 for k in RESOURCE_KEYS}
    ledger.update({"files_read_as_text": guard.opened, "files_hashed_only": guard.hashed_only, "guard_refusals": len(guard.refused), "forbidden_modules_loaded": forbidden_loaded, "lib_libero_wildcards": "none"})
    write_json(out / "budget_ledger.json", ledger)
    (out / "engineering_events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8")
    after = protected_hashes(root)
    write_json(out / "protected_after.json", after)
    vd = {"verdict": verdict, "cards": len(cards), "public_cards": len(pub), "derived_cards": len(der), "tasks": len(tasks), "parse_errors": len(errors), "upstream_sha": ident["head_sha"]}
    write_json(out / "audit_verdict.json", vd)
    return vd


def final_summary(out):
    out = Path(out)
    sel = json.loads((out / "selected_candidate.json").read_text())
    psum = json.loads((out / "public_validation_summary.json").read_text())
    ident = json.loads((out / "upstream_libero_identity.json").read_text())
    cards = json.loads((out / "candidate_mechanism_cards.json").read_text())["cards"]
    L = ["# %s - final audit summary" % CARD, "", "Verdict: **%s**" % sel["status"], "",
         "Upstream: %s @ %s (%s); task map sha256 %s; BDDL root (all suites) sha256 %s; audit date %s" % (ident["repository"], ident["head_sha"], ident["head_commit_date"], ident["task_map_sha256"][:16],
                                                                                                          ident["bddl_root_all_suites"]["sha256"][:16], ident["audit_date_utc"]), "",
         "## Mechanism cards", ""]
    for cid, c in cards.items():
        L.append("- %s (%s, %s): failed %s; unverifiable %s" % (cid, c["origin"], c["host"], ",".join(c["failed_gates"]) or "none", ",".join(c["unverifiable_gates"]) or "none"))
    L += ["", "## Public validation coverage", "", "- strict coverage %.3f (exact %d, binding %d of %d); upper bound %.3f; label %s" % (psum["coverage_rate"], psum["exactly_supported"], psum["binding_supported"], psum["total_tasks"],
                                                                                           psum["upper_bound_if_orientation_verified"], psum["label_if_future_run"]), "",
         "## Nearest misses", ""] + ["- %s: failed %s; unverifiable %s" % (n["card"], ",".join(n["failed"]) or "none", ",".join(n["unverifiable"]) or "none") for n in sel["nearest_misses_not_selected"]]
    L += ["", "## Decisions that would change the outcome", ""] + ["- " + d for d in sel["decisions_that_would_change_the_outcome"]]
    L += ["", "Budget: environment constructions 0, env.reset 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, score reads 0, demo replay 0.", ""]
    (out / "final_audit_summary.md").write_text("\n".join(L), encoding="utf-8")


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["authorization.json", "source_identity.json", "upstream_libero_identity.json", "protected_before.json", "protected_after.json", "libero_task_inventory.csv", "libero_bddl_inventory.json",
            "fixture_and_asset_inventory.json", "predicate_inventory.json", "skill_coverage_matrix.csv", "task_structure_classification.csv", "prior_opportunity_matrix.csv", "contract_redundancy_audit.json",
            "candidate_mechanism_cards.json", "candidate_mechanism_cards.md", "simple_rule_attack.md", "planner_attack.md", "reviewer_objection_matrix.md", "public_validation_coverage.csv",
            "public_validation_recommendation.md", "selected_candidate.json", "selected_tp_design_draft.md", "next_mechanism_canary_request.md", "engineering_events.jsonl", "budget_ledger.json",
            "final_audit_summary.md", "audit_denominator.csv"]
    led = json.loads((out / "budget_ledger.json").read_text())
    ident = json.loads((out / "upstream_libero_identity.json").read_text())
    src = json.loads((out / "source_identity.json").read_text())
    before, after = json.loads((out / "protected_before.json").read_text()), json.loads((out / "protected_after.json").read_text())
    sel = json.loads((out / "selected_candidate.json").read_text())
    inv = json.loads((out / "libero_bddl_inventory.json").read_text())
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".csv", ".jsonl"} and bool(re.search(r"sk-[A-Za-z0-9]{16,}|Authorization:|Bearer [A-Za-z0-9._-]{16,}|DASHSCOPE_API_KEY=", p.read_text(errors="ignore"))))
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "zero_resources": all(led[k] == 0 for k in RESOURCE_KEYS), "guard_refusals_zero": led["guard_refusals"] == 0,
              "no_sim_stack_imported": led["forbidden_modules_loaded"] == [], "upstream_sha_pinned": ident["sha_matches_expected"] and len(ident["head_sha"]) == 40,
              "project_full_head_recorded": len(src["project_head_full"]) == 40 and src["prefix_matches"], "protected_unchanged": before == after, "protected_files_hashed": len(before),
              "all_frozen_tasks_parsed": len(inv["tasks"]) == 40 and not inv["parse_errors"],
              "request_label_consistent": ("NOT_REQUESTED" in (out / "next_mechanism_canary_request.md").read_text()) == (sel["status"] == "NO_LOW_COST_TP_CANDIDATE_FOUND"),
              "no_secret_shaped_content": secrets == 0}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k not in ("outputs_present", "protected_files_hashed", "status")) and all(checks["outputs_present"].values()) else "FAIL"
    write_json(out / "verify.json", checks)
    return checks

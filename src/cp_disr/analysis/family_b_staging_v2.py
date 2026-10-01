"""CP-DISR-S4-FAMILY-B-STAGING-V2-CONTINUATION-1.

Completes the 24-branch Family B physical bank under the FROZEN FAMILY_B_PUBLIC_OBS_V2 profile:
Gate M (offline margin review) -> complementary technical wave (4 branches, strictly staged) ->
conditional release of the remaining 16 -> combined 24-branch analysis.  Provider, representation, RL,
optimizer, S2, S3 and formal test are never touched.  Only identity, scheduling, evaluation and analysis live
here; environment, observation, control, facts, planner, Evaluator and timing sources are the R2 sources.
"""
from __future__ import annotations

import csv
import fcntl
import itertools
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.analysis import family_b_obs_v2_r2 as r2
from cp_disr.platforms.libero.perception import THRESHOLDS

CARD = "CP-DISR-S4-FAMILY-B-STAGING-V2-CONTINUATION-1"
BASE = "48dc5e3b84f2ab587983dc95e54d27efc29c32e0"
R2_PREP = "d610cee7ca0608b2ca7383047e4ca15b83922ad3"
R2_DIR = "runs/final_master/S4/family_b_observation_v2_r2/20261001T045653Z_3c98a096"
CFG = "configs/final_master/family_b_staging_v2.yaml"
FROZEN_SHA = r2.FROZEN_SHA
FROZEN_FILES = r2.FROZEN_FILES
PY = r2.PY
LAYOUTS = ("layout_0", "layout_1")
CONTEXTS = ("B_PENDING", "C_PENDING", "BOTH_PENDING")
PADS = ("pad_u", "pad_v")
REPEATS = (0, 1)
TOL = 0.15
CAPS = {"attempts": 20, "constructions": 20, "resets": 20, "skills": 120, "complementary": 4, "remaining16": 16}
DONE = {"B_PENDING": ("obj_c",), "C_PENDING": ("obj_b",), "BOTH_PENDING": ()}
REMAINING_OBJ = {"B_PENDING": ("obj_b",), "C_PENDING": ("obj_c",), "BOTH_PENDING": ("obj_b", "obj_c")}
ALL24 = tuple(itertools.product(LAYOUTS, CONTEXTS, REPEATS, PADS))
R2_PAIRS = (("layout_0", "B_PENDING", 0), ("layout_1", "C_PENDING", 0))
R2_FOUR = tuple(k for k in ALL24 if k[:3] in R2_PAIRS)
NEW20 = tuple(k for k in ALL24 if k not in R2_FOUR)
COMPLEMENTARY = (("layout_0", "C_PENDING", 0, "pad_v"), ("layout_1", "B_PENDING", 0, "pad_v"),
                 ("layout_0", "C_PENDING", 0, "pad_u"), ("layout_1", "B_PENDING", 0, "pad_u"))
REMAINING16 = tuple(k for k in NEW20 if k not in COMPLEMENTARY)
COMP_PASS = "COMPLEMENTARY_CONTEXT_TECHNICAL_PASS"
COMP_FAIL = "COMPLEMENTARY_OBSERVATION_GATE_FAIL"
SOURCE_FILES = r2.SOURCE_FILES_R2 + (
    "src/cp_disr/analysis/family_b_staging_v2.py", "scripts/family_b_staging_v2.py", CFG)
read, write, sha = m.read, m.write, m.sha
WATCHDOG = 1800


def pair_key(k):
    return k[:3]


def attempt_identity(k):
    return f"FAMILY_B_STAGING_V2_CONT_ATTEMPT_1:{k[0]}:{k[1]}:r{k[2]}:{k[3]}"


def pair_list(keys):
    """Canonical order: layout, context order, repeat; U before V."""
    pairs = sorted({pair_key(k) for k in keys}, key=lambda p: (LAYOUTS.index(p[0]), CONTEXTS.index(p[1]), p[2]))
    return [[(p[0], p[1], p[2], pad) for pad in PADS if (p[0], p[1], p[2], pad) in keys] for p in pairs]


REMAINING16_PAIRS = pair_list(REMAINING16)


# ------------------------------------------------------------------- identity / registration
def source_hashes(root):
    return {p: sha(Path(root) / p) for p in SOURCE_FILES}


def build_branches(profile_sha, commit, source_hash, prefixes, old_ids, manifest_path, freeze_id):
    out = []
    for k in NEW20:
        layout, context, repeat, pad = k
        seed = int(m.hashlib.sha256(f"{freeze_id}:{layout}:{repeat}".encode()).hexdigest()[:8], 16)
        logical = {"layout": layout, "context": context, "repeat": repeat, "candidate": pad}
        ident = attempt_identity(k)
        bid = m.branch_identity({**logical, "card": CARD}, profile_sha, commit, source_hash, ident)
        phase = "complementary" if k in COMPLEMENTARY else "remaining16"
        out.append({
            **logical, "branch_id": bid, "case_id": layout, "restore_seed": seed,
            "candidate_id": f"a:PLACE_BUFFER:carrier:{pad}:v1", "prefix": prefixes[context],
            "wave": f"staging_v2_{phase}", "phase": phase, "manifest_path": str(manifest_path),
            "authorized": True, "execute_now": True, "plan_id": bid, "attempt_id": bid,
            "attempt_identity": ident, "card_id": CARD, "profile_version": m.PROFILE_VERSION,
            "observation_profile_sha256": profile_sha, "cameras": list(m.CAMERAS),
            "source_commit": commit, "source_hash": source_hash,
            "complementary_step": COMPLEMENTARY.index(k) if k in COMPLEMENTARY else None})
    ids = [b["branch_id"] for b in out]
    if len(out) != 20 or len(set(ids)) != 20 or set(ids) & set(old_ids):
        raise ValueError("STOPPED_SOURCE_IDENTITY:branch identities")
    if len({branch_key(b) for b in out}) != 20 or {branch_key(b) for b in out} & set(R2_FOUR):
        raise ValueError("STOPPED_SOURCE_IDENTITY:logical set")
    return out


def branch_key(b):
    return (b["layout"], b["context"], b["repeat"], b["candidate"])


def historical_ids(root):
    seen = set()
    for p in (Path(root) / "runs").rglob("registration*.json"):
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("branches"), list):
            seen |= {b.get("branch_id") for b in doc["branches"] if isinstance(b, dict)}
    return seen


def r2_branches(root):
    reg = read(Path(root) / R2_DIR / "registration_r2.json")
    return {branch_key(b): b for b in reg["branches"]}


# ---------------------------------------------------------------------- protection
def protected_paths(root):
    root = Path(root)
    paths = list(m.protected_paths())
    for rel in (m.OLD_EVIDENCE, m.OLD_REVIEW, r2.R1_DIR, r2.REPAIR_DIR, R2_DIR, m.CFG_DIR):
        if (root / rel).exists():
            paths.append(root / rel)
    return paths


def inventory_before(root, out):
    inv = m.inventory(protected_paths(root))
    write(Path(out) / "protected_before.json", {"sha256": inv, "files": len(inv),
                                                "capture_count": sum("/captures/" in k for k in inv)})
    return len(inv)


# ------------------------------------------------------------ compatibility with R2 execution
ALLOWED_NEW = ("src/cp_disr/analysis/family_b_staging_v2.py", "scripts/family_b_staging_v2.py", CFG,
               "tests/test_family_b_staging_v2.py")
COMPAT_COMPONENTS = {
    "env": "src/cp_disr/platforms/libero/family_b_env.py",
    "adapters_verifier_evaluator_perception": "src/cp_disr/platforms/libero/family_b_adapters.py",
    "obs_v2_env_and_perception": "src/cp_disr/platforms/libero/family_b_obs_v2.py",
    "runtime_v2_recorder": "src/cp_disr/platforms/libero/family_b_runtime_v2.py",
    "runtime_family_b": "src/cp_disr/platforms/libero/family_b_runtime.py",
    "skill_executor": "src/cp_disr/platforms/libero/skill_executor.py",
    "runtime_factory_clock_timing": "src/cp_disr/platforms/libero/runtime_factory.py",
    "recorder_instrumentation": "src/cp_disr/platforms/libero/tp_sr_instrumentation.py",
    "perception_thresholds": "src/cp_disr/platforms/libero/perception.py",
    "worker_execution_functions_pilot": "src/cp_disr/analysis/family_b_pilot.py",
    "worker_v2": "src/cp_disr/analysis/family_b_obs_v2.py",
    "attempt_registry": "src/cp_disr/analysis/s1_integration.py",
    "bplan_planner": "src/cp_disr/baselines/b_plan.py",
    "factstore": "src/cp_disr/facts.py",
    "contract_registry": "configs/runtime/tp_fb_contract_registry.yaml",
    "layouts": "configs/final_master/family_b_layouts.json",
    "v2_base_config": "configs/final_master/family_b_obs_v2/config.yaml",
    **{f"frozen_profile::{Path(p).name}": p for p in FROZEN_FILES},
}


def _git_blob(root, commit, rel):
    res = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{rel}"], capture_output=True)
    return res.stdout if res.returncode == 0 else None


def package_identity():
    import importlib.metadata as md
    out = {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve())}
    for name in ("robosuite", "mujoco", "numpy", "PyYAML", "pillow", "scipy", "torch"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def compatibility_manifest(root, out=None):
    """Production sources and frozen inputs: R2 executed identity vs this card's prep tree."""
    import hashlib
    root = Path(root)
    r2_ident = read(root / R2_DIR / "source_identity.json")
    rows, bad = {}, []
    for name, rel in COMPAT_COMPONENTS.items():
        old = _git_blob(root, R2_PREP, rel)
        new = (root / rel).read_bytes() if (root / rel).is_file() else None
        row = {"path": rel, "r2_prep_sha256": hashlib.sha256(old).hexdigest() if old is not None else None,
               "current_sha256": hashlib.sha256(new).hexdigest() if new is not None else None,
               "r2_recorded_sha256": r2_ident["sources"].get(rel)}
        row["identical"] = old is not None and old == new and (row["r2_recorded_sha256"] in (None, row["current_sha256"]))
        rows[name] = row
        if not row["identical"]:
            bad.append(name)
    changed = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", R2_PREP, "HEAD", "--", "src",
                                       "configs", "scripts"], text=True).split()
    unexpected = sorted(p for p in changed if p not in ALLOWED_NEW)
    if unexpected:
        bad.append("unexpected_source_changes")
    frozen = {p: sha(root / p) for p in FROZEN_FILES}
    frozen_ok = not r2.frozen_bytes_ok(root) and read(root / FROZEN_FILES[0])["profile_sha256"] == FROZEN_SHA
    if not frozen_ok:
        bad.append("frozen_profile")
    log_dir = root / R2_DIR / "logs"
    r2_started = (root / R2_DIR / "authorization_r2.json").stat().st_mtime
    pkgs = package_identity()
    pkg_mtime = {}
    try:
        import importlib.metadata as md
        for name in ("robosuite", "mujoco", "numpy"):
            dist = md.distribution(name)
            pkg_mtime[name] = Path(dist._path).stat().st_mtime  # dist-info directory
    except Exception:
        pass
    pkg_older = {n: t < r2_started for n, t in pkg_mtime.items()}
    manifest = {
        "card_id": CARD, "r2_prep_commit": R2_PREP, "r2_results_commit": BASE,
        "r2_executed_with": "R2 prep commit tree (source_identity.json of the R2 evidence)",
        "components": rows, "changed_paths_since_r2_prep": changed, "unexpected_changes": unexpected,
        "allowed_new_paths": list(ALLOWED_NEW), "frozen_files_sha256": frozen,
        "frozen_profile_unchanged": frozen_ok, "package_identity_current": pkgs,
        "package_dist_info_older_than_r2_run": pkg_older,
        "package_identity_note": "R2 did not record package versions; dist-info mtimes older than the R2 run and the "
                                 "identical interpreter path are the available evidence",
        "worker_note": "BOTH_PENDING needs context mapping {BOTH_PENDING: ()} which the R2 worker lacks; the new worker "
                       "is statement-identical to the R2 worker except that mapping (verified by an AST test).",
        "seed_note": "R2 branches use pair-derived restore seeds; new branches use layout+repeat seeds as in the frozen "
                     "v1 initial_seed_scope; seeds only select the restore RNG within an identical layout.",
        "incompatible_components": bad,
        "status": "COMPATIBLE" if not bad else "STOPPED_RUNTIME_PROFILE_INCOMPATIBLE"}
    if out is not None:
        write(Path(out) / "compatibility_manifest.json", manifest)
    return manifest


def r2_reference_manifest(root):
    """R2 four: read-only references (path/hash/branch id/source identity)."""
    root = Path(root)
    base = root / R2_DIR
    reg = read(base / "registration_r2.json")
    ident = read(base / "source_identity.json")
    items = []
    for b in reg["branches"]:
        res = base / "physical/branch_results" / f"{b['branch_id']}.json"
        items.append({"branch_id": b["branch_id"], "logical": list(branch_key(b)), "card_id": b["card_id"],
                      "attempt_identity": b["attempt_identity"], "result_path": str(res.relative_to(root)),
                      "result_sha256": sha(res), "captures_path": str((base / "captures" / b["branch_id"]).relative_to(root)),
                      "source_commit": b["source_commit"], "source_hash": b["source_hash"],
                      "observation_profile_sha256": b["observation_profile_sha256"],
                      "role": "V2_TECHNICAL_INPUT_FROZEN"})
    return {"evidence_dir": R2_DIR, "source_identity_sha256": sha(base / "source_identity.json"),
            "r2_commit": ident["commit"], "r2_source_hash": ident["source_hash"], "branches": items,
            "technical_gate_sha256": sha(base / "technical_gate_r2.json")}


def r2_phase_reconciliation(root, out):
    base = Path(root) / R2_DIR
    gate = read(base / "canary_gate.json")
    ledger = read(base / "budget_ledger_r2.json")
    events = [json.loads(x) for x in (base / "budget_events_r2.jsonl").read_text().splitlines()]
    charged = [e for e in events if e["event"] == "RESERVED_CHARGED"]
    rec = {
        "card_id": CARD, "r2_budget_ledger_sha256_unchanged_original": sha(base / "budget_ledger_r2.json"),
        "canary_gate": {"status": gate["status"], "remainder_released": gate["remainder_released"]},
        "budget_events_remainder_reserved_charged": sum(e.get("phase") == "remainder" for e in charged),
        "budget_events_canary_reserved_charged": sum(e.get("phase") == "canary" for e in charged),
        "phases_remainder_used": ledger["phases"]["remainder"]["used"],
        "stale_field": {"remainder_slots_status": ledger["remainder_slots_status"],
                        "interpretation": "not-updated derived field (coordinator rewrites it only on canary failure)"},
        "correct_derived_status": "RELEASED_AND_CONSUMED_AFTER_CANARY_PASS",
        "affects_r2_technical_gate_or_branch_results_or_budget_counts": False,
        "r2_attempts_used": ledger["attempts"]["used"]}
    ok = (rec["canary_gate"]["status"] == "CANARY_PASS" and rec["canary_gate"]["remainder_released"] is True
          and rec["budget_events_remainder_reserved_charged"] == 3 and rec["phases_remainder_used"] == 3)
    rec["consistent"] = ok
    if not ok:
        raise ValueError("STOPPED_R2_RECONCILIATION")
    write(Path(out) / "r2_phase_reconciliation.json", rec)
    return rec


# ================================================================================ Gate M (offline)
GATE_M_OK = "GATE_M_COMPLETE_REPLAY_REPRODUCES_R2"
GATE_M_STOP = "STOPPED_GATE_M"


def _png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _largest_component(mask):
    """4-connected largest component size of a boolean mask (pure numpy/python; images are 128x128)."""
    seen = np.zeros(mask.shape, dtype=bool)
    best = 0
    h, w = mask.shape
    for y0, x0 in zip(*np.where(mask)):
        if seen[y0, x0]:
            continue
        stack, size = [(y0, x0)], 0
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        best = max(best, size)
    return int(best)


def object_margin(rgb, depth, name):
    """Margins of the frozen production mask for one object (read-only analysis of saved RGB-D)."""
    from cp_disr.platforms.libero.family_b_adapters import _palette_mask
    from cp_disr.platforms.libero.family_b_env import PALETTE
    image = np.asarray(rgb, dtype=np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    own = np.linalg.norm(image - PALETTE[name][:3], axis=2)
    others = np.stack([np.linalg.norm(image - v[:3], axis=2) for k, v in PALETTE.items() if k != name])
    mask = _palette_mask(rgb, name)
    lo, hi = THRESHOLDS["depth_valid"]
    valid = np.isfinite(depth) & (depth > lo) & (depth < hi)
    sel = mask & valid
    row = {"mask_pixels": int(mask.sum()), "depth_supported_pixels": int(sel.sum()),
           "connected_support_size": _largest_component(sel),
           "min_pixels": THRESHOLDS["min_pixels"], "color_tol": THRESHOLDS["color_tol"],
           "support_minus_min_pixels": int(sel.sum()) - THRESHOLDS["min_pixels"]}
    if mask.any():
        row["min_color_margin_to_tol"] = float((THRESHOLDS["color_tol"] - own)[mask].min())
        row["min_margin_to_competing_palette"] = float((others.min(axis=0) - own)[mask].min())
        z = depth[mask]
        zf = z[np.isfinite(z)]
        row["min_depth_margin_to_valid_boundary_m"] = float(np.minimum(zf - lo, hi - zf).min()) if len(zf) else None
        row["pixels_depth_invalid"] = int((~valid[mask]).sum())
    else:
        row.update(min_color_margin_to_tol=None, min_margin_to_competing_palette=None,
                   min_depth_margin_to_valid_boundary_m=None, pixels_depth_invalid=0)
    return row


def _frame_sources(cap):
    """(stage, frame_id, images dir, saved frame record) for every saved per-view frame of a branch."""
    cap = Path(cap)
    for d in m.action_dirs(cap):
        vp = d / "views_perception.json"
        if vp.is_file():
            for frame in read(vp):
                yield d.name, frame["frame_id"], d, frame


def gate_m(root, out, r2_dir=R2_DIR):
    """Zero new samples: replay the frozen production perception over the saved R2 RGB-D and review margins."""
    from cp_disr.platforms.libero import family_b_obs_v2 as obs
    from cp_disr.platforms.libero.family_b_env import PALETTE
    root, out = Path(root), Path(out)
    base = root / r2_dir
    problems, rows, replay = [], [], {"frames": 0, "mismatches": []}
    reg = read(base / "registration_r2.json")
    if frozen_changed(root):
        problems.append("frozen profile files differ from the freeze commit")
    if read(root / FROZEN_FILES[0])["profile_sha256"] != FROZEN_SHA:
        problems.append("profile hash changed")
    scan = m._scan_forbidden()
    if scan:
        problems.append("perception module references hidden/QA state: " + ",".join(scan))
    timeline = base / "observation_support_timeline.csv"
    if not timeline.is_file():
        problems.append("observation_support_timeline.csv missing")
    for name in ("camera_coverage.json", "camera_selection_evidence.json"):
        if not (root / m.CFG_DIR / name).is_file():
            problems.append(f"{name} missing")
    pair_ev = base / "paired_restore_v2_r2.json"
    if not pair_ev.is_file():
        problems.append("R2 pair evidence missing")
    ref_by_branch = {}
    for b in reg["branches"]:
        cap = base / "captures" / b["branch_id"]
        bd = cap / "boundary" / "views_boundary.json"
        if not bd.is_file():
            problems.append(f"{b['branch_id']}: boundary views missing")
            continue
        vb = read(bd)
        calibs = vb["perception"]["calibrations"]
        n = len(b["prefix"])
        rem = DONE_REMAINING = REMAINING_OBJ[b["context"]]
        stages = []
        # boundary (pre-candidate) frame: saved per-view analysis vs replay
        bstage = {"stage": "boundary", "frame_id": "boundary", "dir": cap / "boundary", "saved": vb["perception"]["per_view"],
                  "calibs": calibs, "img": lambda c, d=cap / "boundary": (_png(d / f"{c}_rgb.png"), np.load(d / f"{c}_depth_metric.npy"))}
        stages.append(bstage)
        for stage, fid, d, frame in _frame_sources(cap):
            saved = {v: frame["cameras"][v]["objects"] for v in m.CAMERAS}
            cal = {v: frame["cameras"][v]["calibration"] for v in m.CAMERAS}
            stages.append({"stage": stage, "frame_id": fid, "dir": d, "saved": saved, "calibs": cal,
                           "img": lambda c, d=d, fid=fid: (_png(d / f"{fid}_{c}_rgb.png"),
                                                           np.load(d / f"{fid}_{c}_depth_metric.npy"))})
        ref_frame = None
        for st in stages:
            for view in m.CAMERAS:
                try:
                    rgb, depth = st["img"](view)
                except Exception as exc:
                    problems.append(f"{b['branch_id']}:{st['stage']}:{st['frame_id']}:{view}: images unreadable {exc}")
                    continue
                got = obs.analyze_view(view, rgb, depth, st["calibs"][view])
                for obj in PALETTE:
                    s, g = st["saved"][view][obj], got[obj]
                    same = all(s[k] == g[k] for k in ("mask_pixels", "depth_support", "blob_present"))
                    if same and g["xyz"] is not None:
                        same = bool(np.allclose(np.asarray(s["xyz"]), np.asarray(g["xyz"]), atol=1e-9, rtol=0))
                    replay["frames"] += 1
                    if not same:
                        replay["mismatches"].append({"branch_id": b["branch_id"], "stage": st["stage"],
                                                     "frame": st["frame_id"], "view": view, "object": obj})
                    mg = object_margin(rgb, depth, obj)
                    rows.append({"branch_id": b["branch_id"], "layout": b["layout"], "context": b["context"],
                                 "candidate": b["candidate"], "repeat": b["repeat"], "stage": st["stage"],
                                 "frame_id": st["frame_id"], "object": obj, "view": view,
                                 "is_remaining_target": obj in rem, "replay_matches_saved": same,
                                 "saved_depth_support": s["depth_support"], **mg,
                                 "evidence_source": "PUBLIC"})
        ref_by_branch[b["branch_id"]] = None
    if replay["mismatches"]:
        problems.append(f"{len(replay['mismatches'])} replay mismatches")
    if any("missing" in x or "unreadable" in x for x in problems):
        write(out / "observation_margin_review.json", {"card_id": CARD, "status": GATE_M_STOP, "problems": problems,
                                                       "new_samples": 0, "environments_constructed": 0})
        raise ValueError(GATE_M_STOP + ":" + ";".join(problems))
    return _gate_m_outputs(root, out, base, reg, rows, replay, problems)


def frozen_changed(root):
    return r2.frozen_bytes_ok(root)


def _gate_m_outputs(root, out, base, reg, rows, replay, problems):
    """Per layout/object/view aggregates, fusion reference/ratio, self-repeat flags, and the three outputs."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["layout"], r["context"], r["candidate"], r["object"], r["view"])].append(r)
    cells = []
    for key, items in sorted(groups.items()):
        sup = [i["saved_depth_support"] for i in items]
        post = [i for i in items if i["stage"] == f"action_{len(next(b for b in reg['branches'] if b['branch_id'] == items[0]['branch_id'])['prefix']):02d}"]
        last = post[-1] if post else None
        cells.append({
            "layout": key[0], "context": key[1], "candidate": key[2], "object": key[3], "view": key[4],
            "frames": len(items), "support_min": int(min(sup)), "support_max": int(max(sup)),
            "support_median": float(np.median(sup)),
            "support_min_minus_min_pixels": int(min(sup)) - THRESHOLDS["min_pixels"],
            "boundary_mask_pixels": next((i["mask_pixels"] for i in items if i["stage"] == "boundary"), None),
            "boundary_depth_support": next((i["saved_depth_support"] for i in items if i["stage"] == "boundary"), None),
            "post_candidate_last_support": None if last is None else last["saved_depth_support"],
            "post_candidate_last_margin": None if last is None else last["support_minus_min_pixels"],
            "min_connected_support": int(min(i["connected_support_size"] for i in items)),
            "min_color_margin_to_tol": min((i["min_color_margin_to_tol"] for i in items if i["min_color_margin_to_tol"] is not None), default=None),
            "min_depth_margin_m": min((i["min_depth_margin_to_valid_boundary_m"] for i in items if i["min_depth_margin_to_valid_boundary_m"] is not None), default=None),
            "is_remaining_target": items[0]["is_remaining_target"], "evidence_source": "PUBLIC"})
    # fusion reference/ratio and selected view, taken from the saved frames
    fusion = []
    for b in reg["branches"]:
        cap = base / "captures" / b["branch_id"]
        for stage, fid, d, frame in _frame_sources(cap):
            for obj, f in frame["fusion"].items():
                if obj in REMAINING_OBJ[b["context"]]:
                    fusion.append({"branch_id": b["branch_id"], "layout": b["layout"], "candidate": b["candidate"],
                                   "stage": stage, "frame_id": fid, "object": obj,
                                   "selected_view": f["selected_view"], "inputs": f["inputs"]})
    self_repeat, pairs = {}, {}
    for b in reg["branches"]:
        vb = read(base / "captures" / b["branch_id"] / "boundary" / "views_boundary.json")
        self_repeat[b["branch_id"]] = vb["self_repeat"]
        pairs.setdefault((b["layout"], b["context"], b["repeat"]), {})[b["candidate"]] = {
            v: {o: vb["perception"]["per_view"][v][o]["depth_support"] for o in vb["perception"]["per_view"][v]}
            for v in m.CAMERAS}
    pair_support = {"/".join(map(str, k)): {"pad_u_vs_pad_v_boundary_support_equal": v["pad_u"] == v["pad_v"], **v}
                    for k, v in pairs.items()}
    targets = [c for c in cells if c["is_remaining_target"] and c["post_candidate_last_support"] is not None]
    zero = [c for c in targets if c["post_candidate_last_margin"] == 0]
    view_dependence = {}
    for c in targets:
        k = f"{c['layout']}/{c['context']}/{c['candidate']}/{c['object']}"
        view_dependence.setdefault(k, {})[c["view"]] = c["post_candidate_last_support"]
    for k, v in view_dependence.items():
        v["single_view_only"] = sum(s >= THRESHOLDS["min_pixels"] for s in v.values()) == 1
    review = {
        "card_id": CARD, "status": GATE_M_OK if not problems else GATE_M_STOP, "problems": problems,
        "new_samples": 0, "environments_constructed": 0, "images_generated": 0,
        "profile_sha256": FROZEN_SHA, "min_pixels": THRESHOLDS["min_pixels"],
        "replay": {"object_view_cells_replayed": replay["frames"], "mismatches": replay["mismatches"],
                   "reproduces_saved_records": not replay["mismatches"]},
        "hidden_qa_in_formal_perception": bool(m._scan_forbidden()),
        "cells": cells, "fusion_remaining_targets": fusion, "self_repeat_boundary": self_repeat,
        "pair_boundary_support": pair_support, "remaining_target_view_dependence": view_dependence,
        "margin_zero_cells": zero,
        "evidence_sources": {"RGB-D replay, support, margins, self-repeat, pair": "PUBLIC",
                             "camera_coverage.json / camera_selection_evidence.json (geometry/heuristic)": "QA_HEURISTIC"},
        "statements": {
            "tested_pad_v_remaining_target_sideview_support": [
                {"layout": c["layout"], "context": c["context"], "object": c["object"],
                 "support": c["post_candidate_last_support"], "margin_to_min_pixels": c["post_candidate_last_margin"]}
                for c in targets if c["candidate"] == "pad_v" and c["view"] == "sideview"],
            "r2_pass_revoked": False,
            "untested_complementary_positions_inferred_from_r2": False,
            "min_pixels_lowered": False, "palette_changed": False, "carry_over": False,
            "margin_zero_alone_stops_the_run": False,
            "margin_zero_forces_complementary_pad_v_canaries_first": True}}
    if not problems:
        write(Path(out) / "observation_margin_review.json", review)
    else:
        write(Path(out) / "observation_margin_review.json", review)
    cov = json.loads((Path(root) / m.CFG_DIR / "camera_coverage.json").read_text())["rows"]
    tested = {(c["layout"], c["object"], c["view"]): c for c in cells if c["context"] in ("B_PENDING", "C_PENDING")}
    fields = ["layout", "object", "view", "evidence_source_geometry", "depth_m", "px_per_cube_edge",
              "inside_margin_8px", "tested_in_r2", "support_min", "support_max", "support_median",
              "support_min_minus_min_pixels", "post_candidate_last_support", "evidence_source_support", "note"]
    with (Path(out) / "observation_coverage_matrix.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in cov:
            t = tested.get((r["layout"], r["object"], r["camera"]))
            w.writerow({"layout": r["layout"], "object": r["object"], "view": r["camera"],
                        "evidence_source_geometry": "QA_HEURISTIC", "depth_m": r.get("depth_m"),
                        "px_per_cube_edge": r.get("px_per_cube_edge"), "inside_margin_8px": r.get("inside_margin_8px"),
                        "tested_in_r2": t is not None,
                        "support_min": None if t is None else t["support_min"],
                        "support_max": None if t is None else t["support_max"],
                        "support_median": None if t is None else t["support_median"],
                        "support_min_minus_min_pixels": None if t is None else t["support_min_minus_min_pixels"],
                        "post_candidate_last_support": None if t is None else t["post_candidate_last_support"],
                        "evidence_source_support": "PUBLIC" if t is not None else "NOT_MEASURED",
                        "note": "" if t is not None else "untested in R2; not inferred"})
    md = ["# Gate M — observation margin review (offline, zero new samples)", "",
          f"- Status: **{review['status']}**; environments constructed 0; images generated 0.",
          f"- Replay of the frozen production `analyze_view` over saved R2 RGB-D: {replay['frames']} object/view cells, "
          f"{len(replay['mismatches'])} mismatches.",
          "- Evidence: replay/support/margins/self-repeat/pair are PUBLIC; coverage geometry is QA_HEURISTIC.", ""]
    for s in review["statements"]["tested_pad_v_remaining_target_sideview_support"]:
        md.append(f"- Tested pad_v remaining-target sideview support ({s['layout']}/{s['context']}/{s['object']}): "
                  f"{s['support']} pixels, margin to min_pixels {s['margin_to_min_pixels']}.")
    md += ["- This does not revoke the R2 PASS.",
           "- Untested complementary positions (layout_0 C_PENDING obj_c, layout_1 B_PENDING obj_b) cannot be inferred from R2.",
           "- No lowering of min_pixels, no palette change, no carry-over.",
           "- A margin of 0 does not stop the run by itself; it forces the complementary pad_v canaries first.",
           f"- Problems: {'none' if not problems else '; '.join(problems)}"]
    (Path(out) / "observation_margin_review.md").write_text("\n".join(md) + "\n")
    if problems:
        raise ValueError(GATE_M_STOP + ":" + ";".join(problems))
    return review


# ================================================================================= registration
def check_uniform(registration):
    """One observation configuration for all 20; every U/V pair complete with one restore seed."""
    problems = []
    branches = registration["branches"]
    if len(branches) != 20:
        problems.append(f"{len(branches)} branches, expected 20")
    if {(b["observation_profile_sha256"], tuple(b["cameras"]), b["profile_version"]) for b in branches} != \
            {(FROZEN_SHA, tuple(m.CAMERAS), m.PROFILE_VERSION)}:
        problems.append("branches do not all use the frozen observation profile")
    if {branch_key(b) for b in branches} != set(NEW20):
        problems.append("logical branch set differs from the 20 specified")
    pairs = defaultdict(list)
    for b in branches:
        pairs[pair_key(branch_key(b))].append(b)
    if len(pairs) != 10:
        problems.append(f"{len(pairs)} pairs, expected 10")
    for key, members in pairs.items():
        if sorted(b["candidate"] for b in members) != sorted(PADS):
            problems.append(f"{key}: pad pair incomplete")
        if len({b["restore_seed"] for b in members}) != 1:
            problems.append(f"{key}: U/V restore seeds differ")
    return problems


def _contexts(cfg):
    return {k: v["prefix"] for k, v in cfg["contexts"].items()}


def init_state(out, branches):
    """Budget ledger, physical attempt cap and complementary state machine (shared with the tests)."""
    phys = Path(out) / "physical"
    write(phys / "budget_ledger.json", {"physical_witness_episodes": {"cap": CAPS["attempts"], "used": 0}})
    zero = {k: {"cap": 0, "used": 0} for k in ("skill_retries", "standalone_capture_resets", "provider_calls",
                                               "representation_forwards", "rl_transitions", "optimizer_steps", "elastic",
                                               "formal_test")}
    write(out / "budget_ledger.json", {
        "card_id": CARD,
        "history": {"v1": {"attempts": 4, "resets": 4, "skill_calls": 20, "status": "PERMANENTLY_CONSUMED"},
                    "v2_r1": {"attempt_slots_charged": 2, "environment_construction_attempts": 2,
                              "successful_environment_constructors": 0, "actual_explicit_reset_calls": "NOT_REACHED",
                              "unused_slots": 2, "unused_slots_status": "RETIRED_NOT_TRANSFERABLE"},
                    "v2_r2": {"attempts": 4, "constructions": 4, "resets": 4, "internal_resets": 8, "skill_calls": 24,
                              "task_success": "4/4"}},
        "attempts": {"cap": CAPS["attempts"], "used": 0},
        "environment_constructions": {"cap": CAPS["constructions"], "used": 0},
        "explicit_resets": {"cap": CAPS["resets"], "used": 0},
        "live_skill_calls": {"cap": CAPS["skills"], "used": 0},
        **zero,
        "phases": {"complementary": {"cap": CAPS["complementary"], "used": 0},
                   "remaining16": {"cap": CAPS["remaining16"], "used": 0}},
        "remaining16_released": False, "reserved_branches": [],
        "cumulative_family_b_physical_attempts_if_all_run": 4 + 2 + 4 + 20,
        "cumulative_note": "Family B physical-qualification/engineering count only; not the Experimental Plan v3 "
                           "'17+4' RL attempt count.",
        "actual": {"constructions_attempted": None, "constructor_successful": None, "explicit_resets": None,
                   "internal_resets": None, "skill_calls": None}})
    write(out / "complementary_state.json", {
        "status": "NOT_STARTED", "order": [list(k) for k in COMPLEMENTARY],
        "steps": {str(i): {"key": list(k), "branch_id": next(b["branch_id"] for b in branches if branch_key(b) == k),
                           "status": "NOT_RUN"} for i, k in enumerate(COMPLEMENTARY)}})


def register(root, out):
    import yaml
    from cp_disr.runtime import require_runtime
    root, out = Path(root).resolve(), Path(out)
    if m.git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree not clean")
    commit = m.git(root, "rev-parse", "HEAD")
    if m.git(root, "merge-base", "--is-ancestor", BASE, "HEAD") != "":
        raise ValueError("STOPPED_SOURCE_IDENTITY:base is not an ancestor")
    if r2.frozen_bytes_ok(root):
        raise ValueError("STOPPED_OBSERVATION_PROFILE:frozen file differs from freeze commit")
    profile = m.check_profile(root)
    if profile["profile_sha256"] != FROZEN_SHA:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:hash")
    cfg = yaml.safe_load((root / CFG).read_text())
    if (cfg["card_id"], cfg["observation_profile_sha256"], cfg["base_commit"]) != (CARD, FROZEN_SHA, BASE):
        raise ValueError("STOPPED_SOURCE_IDENTITY:config identity")
    if (out / "physical/registration.json").exists():
        raise ValueError("STOPPED_SOURCE_IDENTITY:already registered")
    out.mkdir(parents=True, exist_ok=True)
    compat = compatibility_manifest(root, out)
    if compat["status"] != "COMPATIBLE":
        raise ValueError("STOPPED_RUNTIME_PROFILE_INCOMPATIBLE:" + ",".join(compat["incompatible_components"]))
    recon = r2_phase_reconciliation(root, out)
    ref_manifest = r2_reference_manifest(root)
    write(out / "r2_reference_manifest.json", ref_manifest)
    hashes = source_hashes(root)
    source_hash = m.digest(hashes)
    manifest = yaml.safe_load((root / R2_DIR / "spec/runtime_manifest_T_P_FB_obs_v2_r2.yaml").read_text())
    src = root / "src/cp_disr/platforms/libero/family_b_runtime_v2.py"
    manifest["runtime"]["repository_path"] = str(root)
    manifest["runtime_factory"].update({"source_path": str(src), "sha256": sha(src)})
    require_runtime(manifest)
    manifest_path = out / "spec/runtime_manifest_T_P_FB_staging_v2.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    old_ids = historical_ids(root) | {b["branch_id"] for b in r2_branches(root).values()}
    freeze_id = m.digest({"card": CARD, "profile": FROZEN_SHA, "commit": commit})[:16]
    branches = build_branches(FROZEN_SHA, commit, source_hash, _contexts(cfg), old_ids, manifest_path, freeze_id)
    ref = float(read(root / R2_DIR / "bindings/reference_cost.json")["reference_skill_seconds"])
    registration = {"card_id": CARD, "branches": branches, "manifest_sha256": sha(manifest_path),
                    "reference_skill_seconds": ref}
    problems = check_uniform(registration)
    if problems:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems))
    phys = out / "physical"
    write(phys / "witnesses/e4_branch_registration.json", registration)
    write(phys / "registration.json", registration)
    write(out / "registration_v2_remaining20.json", registration)
    write(out / "bindings/reference_cost.json", {"reference_skill_seconds": ref})
    init_state(out, branches)
    _event(out, "REGISTERED", branches=[b["branch_id"] for b in branches], commit=commit)
    write(out / "authorization.json", {
        "card_id": CARD, "authorized_by": "user message approving CP-DISR-S4-FAMILY-B-STAGING-V2-CONTINUATION-1",
        "continuation_branch_attempts_max": 20, "environment_construction_attempts_max": 20,
        "successful_explicit_resets_max": 20, "live_skill_calls_max": 120, "skill_retries": 0,
        "standalone_capture_resets": 0, "provider": 0, "representation_forwards": 0, "rl_transitions": 0,
        "optimizer_steps": 0, "elastic": 0, "formal_test": 0, "s2": False, "s3": False, "tp_training": False,
        "complementary_wave": {"order": "A -> B -> (C parallel D)", "release_of_remaining_16": "only after pass"},
        "no_config_change_after_failure": True})
    write(out / "source_identity.json", {
        "card_id": CARD, "branch": m.git(root, "branch", "--show-current"), "commit": commit, "base_commit": BASE,
        "r2_prep_commit": R2_PREP, "source_hash": source_hash, "sources": hashes,
        "observation_profile_sha256": FROZEN_SHA, "manifest_sha256": sha(manifest_path),
        "frozen_file_sha256": {rel: sha(root / rel) for rel in FROZEN_FILES}})
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    for rel in FROZEN_FILES:
        write(out / Path(rel).name, read(root / rel))
    inventory_before(root, out)
    return {"status": "REGISTERED", "commit": commit, "branches": [b["branch_id"] for b in branches],
            "reconciliation_consistent": recon["consistent"]}


# ================================================================================ ledger / events
def _event(out, kind, **fields):
    path = Path(out) / "budget_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"event": kind, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields},
                                sort_keys=True) + "\n")
        handle.flush()


@contextmanager
def _lock(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    with (Path(out) / ".staging_ledger.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _comp_state(out):
    return read(Path(out) / "complementary_state.json")


def _step_gate(out, b):
    """Raises if the staged order or a recorded failure forbids dispatching this branch."""
    state = _comp_state(out)
    if b["phase"] == "remaining16":
        if state["status"] != COMP_PASS:
            raise ValueError("STOPPED_REMAINING16_NOT_RELEASED")
        return
    if state["status"] == COMP_FAIL:
        raise ValueError("STOPPED_COMPLEMENTARY_GATE_FAILED")
    step = b["complementary_step"]
    need = {0: [], 1: ["0"], 2: ["0", "1"], 3: ["0", "1"]}[step]
    if any(state["steps"][s]["status"] != "PASS" for s in need):
        raise ValueError(f"STOPPED_COMPLEMENTARY_ORDER:step {step} needs {need} PASS")
    if state["steps"][str(step)]["status"] != "NOT_RUN":
        raise ValueError("STOPPED_DUPLICATE_ATTEMPT")


def reserve_and_charge(out, b, phase):
    """Atomic: staged-order gate, then reserve the physical slot, then charge the ledger.  No change on violation."""
    from cp_disr.analysis.s1_integration import reserve_branch_attempt
    out = Path(out)
    with _lock(out):
        ledger = read(out / "budget_ledger.json")
        bid = b["branch_id"]
        if bid in ledger["reserved_branches"]:
            raise ValueError("STOPPED_DUPLICATE_ATTEMPT")
        if b["phase"] != phase:
            raise ValueError("STOPPED_PHASE_MISMATCH")
        _step_gate(out, b)
        if phase == "remaining16" and not ledger["remaining16_released"]:
            raise ValueError("STOPPED_REMAINING16_NOT_RELEASED")
        for key in ("attempts", "environment_constructions", "explicit_resets"):
            if ledger[key]["used"] + 1 > ledger[key]["cap"]:
                raise ValueError("STOPPED_BUDGET_EXHAUSTED:" + key)
        if ledger["phases"][phase]["used"] + 1 > ledger["phases"][phase]["cap"]:
            raise ValueError("STOPPED_BUDGET_EXHAUSTED:phase_" + phase)
        reserve_branch_attempt(out / "physical", bid)
        for key in ("attempts", "environment_constructions", "explicit_resets"):
            ledger[key]["used"] += 1
        ledger["phases"][phase]["used"] += 1
        ledger["reserved_branches"].append(bid)
        write(out / "budget_ledger.json", ledger)
        _event(out, "RESERVED_CHARGED", branch_id=bid, phase=phase)
    return ledger


def finalize_counts(out):
    out = Path(out)
    with _lock(out):
        ledger = read(out / "budget_ledger.json")
        res = [read(p) for p in sorted((out / "physical/branch_results").glob("*.json"))]
        counts = [r.get("env_counts", {}) for r in res]
        ledger["live_skill_calls"]["used"] = sum(len(r.get("actions", [])) for r in res)
        ledger["actual"] = {
            "constructions_attempted": len(res),
            "constructor_successful": sum(1 for c in counts if c.get("constructions") == 1 and c.get("reset_calls", -1) >= 1),
            "explicit_resets": sum(max(int(c.get("reset_calls", 0)), 0) for c in counts),
            "internal_resets": sum(max(int(c.get("internal_resets", 0)), 0) for c in counts),
            "skill_calls": ledger["live_skill_calls"]["used"]}
        write(out / "budget_ledger.json", ledger)
    return ledger


# ================================================================================ identity / worker
def verify_identity(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    ident = read(out / "source_identity.json")
    if m.git(root, "rev-parse", "HEAD") != ident["commit"]:
        raise ValueError("STOPPED_SOURCE_IDENTITY:HEAD moved since registration")
    if m.git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree dirty")
    if ident["sources"] != source_hashes(root):
        raise ValueError("STOPPED_SOURCE_IDENTITY:source drift")
    if r2.frozen_bytes_ok(root):
        raise ValueError("STOPPED_OBSERVATION_PROFILE:frozen file changed")
    reg = read(out / "physical/registration.json")
    problems = check_uniform(reg)
    if problems or reg["card_id"] != CARD:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems or ["registration"]))
    if Path(sys.executable).resolve() != Path(PY).resolve() or not str(m.__file__).startswith(str(root / "src")):
        raise ValueError("STOPPED_SOURCE_IDENTITY:interpreter or cp_disr import path")
    return ident


WORKER_OLD = '{"B_PENDING": ("obj_c",), "C_PENDING": ("obj_b",)}[b["context"]]'
WORKER_NEW = 'DONE_BY_CONTEXT[b["context"]]'


def worker_source():
    """The frozen v2 worker with exactly one change: the completed-target mapping also covers BOTH_PENDING."""
    import inspect
    src = inspect.getsource(m.worker)
    if src.count(WORKER_OLD) != 1:
        raise RuntimeError("STOPPED_SOURCE_IDENTITY:v2 worker no longer matches the expected mapping line")
    return src.replace(WORKER_OLD, WORKER_NEW).replace("def worker(", "def worker_with_both(", 1)


def build_worker():
    ns = dict(vars(m))
    ns["DONE_BY_CONTEXT"] = DONE
    exec(compile(worker_source(), "<family_b_staging_v2:worker>", "exec"), ns)
    return ns["worker_with_both"]


def worker(root, out, branch_id):
    """Staging worker: coordinator reservation required; frozen v2 worker body (plus BOTH mapping) otherwise."""
    from cp_disr.platforms.libero import family_b_obs_v2 as obs
    root, out = Path(root), Path(out)
    ledger = read(out / "budget_ledger.json")
    if branch_id not in ledger["reserved_branches"]:
        raise RuntimeError("STOPPED_BUDGET:branch was not reserved by the coordinator")
    m.SOURCE_FILES = SOURCE_FILES
    real = obs.verify_frozen_cameras
    cap = out / "captures" / branch_id
    cap.mkdir(parents=True, exist_ok=True)

    def recording_verify(env, profile):
        try:
            write(cap / "camera_live_vs_frozen.json", r2._camera_record(env, profile))
        except Exception as exc:
            write(cap / "camera_live_vs_frozen.json", {"status": "RECORD_ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return real(env, profile)

    obs.verify_frozen_cameras = recording_verify
    return build_worker()(root, out, branch_id)


def default_launcher(root, out, b, gpu):
    from cp_disr.analysis.family_b_pilot import _worker_env
    logs = Path(out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = (logs / f"{b['branch_id']}.log").open("ab")
    cmd = [sys.executable, str(Path(root) / "scripts/family_b_staging_v2.py"), "worker", "--root", str(root),
           "--output", str(out), "--branch-id", b["branch_id"]]
    proc = subprocess.Popen(cmd, cwd=root, env=_worker_env(gpu), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    proc.log_handle = log
    return proc


# ================================================================================ evaluation
GROUPS = {
    "A_source_config": ("registered_identity", "profile_identity", "camera_live_vs_frozen"),
    "B_environment_restore": ("no_exception", "constructor_and_single_reset", "restore_receipt", "recorder_errors_empty"),
    "C_decision_state": ("setup_complete", "held_carrier_true", "completed_target_confirmed", "both_pads_legal",
                         "candidate_ids_stable", "two_fixed_views_public", "hidden_flags_clean"),
    "D_observation_chain": ("chain_files_saved", "remaining_publicly_supported", "selected_view_present",
                            "on_table_true", "held_false", "pick_mask_true"),
    "E_execution": ("place_buffer_normal", "task_success_expected_sequence", "evaluator_terminal_success",
                    "planner_semantics", "planner_plan_found_after_candidate", "no_skill_retry",
                    "no_planner_after_success"),
}
BOTH_REPORT_ONLY = GROUPS["D_observation_chain"]  # for BOTH_PENDING the observation-chain checks are reported, not gated


def planner_semantics(cap, actions, evaluators, n):
    """CONTINUE needs a planner record; TASK_SUCCESS must not have a real planner call (setup actions < n skipped)."""
    problems = []
    last = len(actions) - 1
    for i, _ in enumerate(actions):
        d = Path(cap) / f"action_{i:02d}"
        ev = evaluators.get(i)
        has = (d / "planner.json").is_file()
        if i < n:
            continue
        if ev is None:
            problems.append(f"action_{i:02d}:evaluator missing")
        elif ev["terminated"] or ev["task_success"]:
            if has:
                problems.append(f"action_{i:02d}:planner called after TASK_SUCCESS")
        elif not has:
            problems.append(f"action_{i:02d}:CONTINUE without planner record")
        elif i < last:
            plan = read(d / "planner.json")
            if plan["status"] != "PLAN_FOUND" or plan["plan"][:1] != [actions[i + 1]["action_id"]]:
                problems.append(f"action_{i:02d}:planner does not match next action")
    return problems


def remaining_support(cap, n, objects, min_pixels):
    """Public chain after the candidate: pixels -> blob -> fact -> PICK mask, per remaining object."""
    d = Path(cap) / f"action_{n:02d}"
    frames = read(d / "views_perception.json") if (d / "views_perception.json").is_file() else []
    if not frames or not (d / "snapshot.json").is_file():
        return {obj: {"object": obj, "ok": False, "reason": "no views_perception/snapshot record"} for obj in objects}
    last, snapshot = frames[-1], read(d / "snapshot.json")
    facts, result = snapshot["fact_values"], {}
    for obj in objects:
        fusion = last["fusion"][obj]
        pick = f"a:PICK:{obj}:v1"
        sel = fusion["selected_view"]
        row = {"object": obj, "last_frame": last["frame_id"], "selected_view": sel, "inputs": fusion["inputs"],
               "on_table": facts.get(f"p:OnTable:{obj}"), "held": facts.get(f"p:Held:{obj}"),
               "pick_mask": bool(snapshot["candidate_mask"][snapshot["candidate_ids"].index(pick)]),
               "supported": fusion["blob"] is not None and fusion["inputs"][sel]["support"] >= min_pixels}
        row["ok"] = bool(row["supported"] and row["on_table"] == "TRUE" and row["held"] == "FALSE" and row["pick_mask"])
        result[obj] = row
    return result


def expected_tail(context):
    """Post-candidate sequences that count as the nominal completion of the context."""
    if context == "B_PENDING":
        return [["a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1"]]
    if context == "C_PENDING":
        return [["a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1"]]
    return [["a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1", "a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1"],
            ["a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1", "a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1"]]


def _chain_files(cap, last, n):
    d = Path(cap) / f"action_{n:02d}"
    need = [f"{last['frame_id']}_{c}_{k}" for c in m.CAMERAS for k in ("rgb.png", "depth_metric.npy")]
    return (all(any(d.rglob(x)) for x in need) and
            all(any(d.rglob(f"{s}_{c}_rgb.png")) for s in ("before", "after") for c in m.CAMERAS))


def evaluate_branch(out, b, attempts=None):
    """All per-branch checks for any of the 20 (or the same shape for R2 rows).  Never reads hidden truth for a verdict."""
    out = Path(out)
    bid, n, context = b["branch_id"], len(b["prefix"]), b["context"]
    cap = out / "captures" / bid
    attempts = attempts if attempts is not None else (
        read(out / "physical/attempt_registry.json") if (out / "physical/attempt_registry.json").is_file() else {})
    row = {"branch_id": bid, "layout": b["layout"], "context": context, "repeat": b["repeat"],
           "candidate": b["candidate"], "phase": b.get("phase")}
    checks, problems = {}, []

    def chk(name, ok, why=""):
        checks[name] = bool(ok)
        if not ok and not (context == "BOTH_PENDING" and name in BOTH_REPORT_ONLY):
            problems.append(f"{bid}:{name}" + (f":{why}" if why else ""))

    res_path = out / "physical/branch_results" / f"{bid}.json"
    if not res_path.is_file():
        chk("terminal_receipt", False, "no terminal result")
        return {**row, "checks": checks, "problems": problems, "terminal_receipt": False, "complete": False}
    r = read(res_path)
    acts = r.get("actions", [])
    row["status"] = r.get("status")
    chk("terminal_receipt", attempts.get(bid) in ("COMPLETED", "FAILED", "UNKNOWN"))
    chk("registered_identity", r.get("attempt_identity") == b["attempt_identity"] and r.get("branch_id") == bid and
        b.get("card_id") == CARD and "STAGING_V2_CONT" in b["attempt_identity"])
    chk("profile_identity", r.get("observation_profile_sha256") == FROZEN_SHA == b["observation_profile_sha256"])
    cam_path = cap / "camera_live_vs_frozen.json"
    cam = read(cam_path) if cam_path.is_file() else {}
    chk("camera_live_vs_frozen", cam.get("status") == "PASS" and cam.get("fallback_used") is False and
        list(cam.get("cameras", {})) == list(m.CAMERAS))
    chk("no_exception", r.get("status") != "EXCEPTION" and "traceback" not in r, str(r.get("detail", ""))[:200])
    counts = r.get("env_counts", {})
    chk("constructor_and_single_reset", counts.get("constructions") == 1 and counts.get("reset_calls") == 1 and
        counts.get("bootstrap_constructions", 0) == 0, str(counts))
    row["env_counts"] = counts
    chk("restore_receipt", bool(r.get("restore_receipt")))
    row["recorder_errors"] = r.get("recorder_errors", [])
    chk("recorder_errors_empty", not row["recorder_errors"])
    pub_path = cap / "boundary/public.json"
    chk("setup_complete", len(acts) >= n and [a["action_id"] for a in acts[:n]] == b["prefix"] and
        all(a["controller_exit"] == "NORMAL_TERMINATION" for a in acts[:n]) and pub_path.is_file())
    pub = read(pub_path) if pub_path.is_file() else None
    facts = pub["fact_values"] if pub else {}
    chk("held_carrier_true", facts.get("p:Held:carrier") == "TRUE")
    chk("completed_target_confirmed", all(facts.get(f"p:Inside:{o}:receiver") == "TRUE" for o in DONE[context]))
    ids = pub["candidate_ids"] if pub else []
    mask = pub["candidate_mask"] if pub else []
    chk("both_pads_legal", bool(pub) and all(c in ids and mask[ids.index(c)] for c in
                                             (f"a:PLACE_BUFFER:carrier:{p}:v1" for p in PADS)))
    snap_n = read(cap / f"action_{n:02d}/snapshot.json") if (cap / f"action_{n:02d}/snapshot.json").is_file() else None
    chk("candidate_ids_stable", bool(pub) and snap_n is not None and snap_n["candidate_ids"] == ids)
    vb_path = cap / "boundary/views_boundary.json"
    vb = read(vb_path) if vb_path.is_file() else {}
    chk("two_fixed_views_public", vb.get("cameras") == list(m.CAMERAS) and set(vb.get("views", {})) == set(m.CAMERAS))
    qa_flags, vflags = [], []
    for qa in cap.glob("action_*/qa_state.jsonl"):
        for line in qa.read_text().splitlines():
            rec = json.loads(line)
            qa_flags += [rec.get(k) for k in ("used_by_planner", "used_by_policy", "used_by_provider", "used_by_verifier")]
    for vp in cap.glob("action_*/views_perception.json"):
        vflags += [(f["hidden_truth_used"], f["qpos_qvel_used"]) for f in read(vp)]
    row["hidden_qa_isolation"] = {"qa_used_flags_any": any(qa_flags), "views_hidden_truth_or_qpos_used_any":
                                  any(any(f) for f in vflags)}
    chk("hidden_flags_clean", bool(qa_flags) and not any(qa_flags) and bool(vflags) and not any(any(f) for f in vflags))
    rem = REMAINING_OBJ[context]
    cand_vp = cap / f"action_{n:02d}/views_perception.json"
    if len(acts) > n and snap_n is not None and cand_vp.is_file():
        support = remaining_support(cap, n, rem, THRESHOLDS["min_pixels"])
        last = read(cand_vp)[-1]
        row["remaining_public_support"] = support
        row["per_view"] = {o: {v: {k: last["cameras"][v]["objects"][o][k] for k in
                                   ("mask_pixels", "depth_support", "blob_present")} for v in m.CAMERAS} for o in rem}
        chk("chain_files_saved", _chain_files(cap, last, n))
        chk("remaining_publicly_supported", all(s["supported"] for s in support.values()))
        chk("selected_view_present", all(s["selected_view"] in m.CAMERAS for s in support.values()))
        chk("on_table_true", all(s["on_table"] == "TRUE" for s in support.values()))
        chk("held_false", all(s["held"] == "FALSE" for s in support.values()))
        chk("pick_mask_true", all(s["pick_mask"] is True for s in support.values()))
    else:
        row["remaining_public_support"] = {"ok": False, "reason": "candidate action not executed or record missing"}
        for name in GROUPS["D_observation_chain"]:
            chk(name, False, "no post-candidate chain")
    seq = [a["action_id"] for a in acts]
    row["sequence"] = seq
    chk("place_buffer_normal", len(acts) > n and acts[n]["controller_exit"] == "NORMAL_TERMINATION")
    chk("task_success_expected_sequence", r.get("status") == "TASK_SUCCESS" and r.get("task_success") is True and
        seq[:n + 1] == b["prefix"] + [b["candidate_id"]] and seq[n + 1:] in expected_tail(context))
    evaluators = {i: read(cap / f"action_{i:02d}/evaluator.json") for i in range(len(acts))
                  if (cap / f"action_{i:02d}/evaluator.json").is_file()}
    chk("evaluator_terminal_success", bool(evaluators) and evaluators[max(evaluators)]["task_success"] is True and
        max(evaluators) == len(acts) - 1)
    psem = planner_semantics(cap, acts, evaluators, n)
    row["planner_semantics_problems"] = psem
    chk("planner_semantics", not psem, ";".join(psem))
    cp = read(cap / f"action_{n:02d}/planner.json") if (cap / f"action_{n:02d}/planner.json").is_file() else {}
    first_picks = [f"a:PICK:{o}:v1" for o in rem]
    chk("planner_plan_found_after_candidate", cp.get("status") == "PLAN_FOUND" and cp.get("plan", [None])[:1][0] in first_picks
        if cp.get("plan") else False)
    chk("no_skill_retry", len(seq) == len(set(seq)) and len(seq) == 6)
    chk("no_planner_after_success", bool(acts) and not (cap / f"action_{len(acts) - 1:02d}/planner.json").is_file())
    row.update(checks=checks, problems=problems, terminal_receipt=checks["terminal_receipt"],
               setup_complete=checks["setup_complete"], task_success=checks["task_success_expected_sequence"],
               complete=not problems, sideview_selected_any=r2._sideview_selected(cap),
               time_to_task_success=r.get("time_to_task_success"),
               candidate_start_sim_time=r.get("candidate_start_sim_time"),
               terminal_sim_time=r.get("terminal_sim_time"), worker_wall_seconds=r.get("worker_wall_seconds"),
               n_actions=len(acts), planner_reason=r.get("planner_reason"))
    return row


def branches_by_phase(out, phase=None):
    reg = read(Path(out) / "physical/registration.json")
    return [b for b in reg["branches"] if phase is None or b["phase"] == phase]


# ============================================================================= dispatch helpers
def _no_prior_results(out, branches):
    if any((Path(out) / "physical/branch_results" / f"{b['branch_id']}.json").exists() for b in branches):
        raise ValueError("STOPPED_DUPLICATE_ATTEMPT")


def _watchdog_finish(out, bid):
    from cp_disr.analysis.s1_integration import finish_branch_attempt
    finish_branch_attempt(Path(out) / "physical", bid, "UNKNOWN", execution_status="WATCHDOG_TIMEOUT")


def _result(out, bid):
    p = Path(out) / "physical/branch_results" / f"{bid}.json"
    return read(p) if p.is_file() else None


def run_branch(root, out, b, gpu, launcher=default_launcher, sleep=time.sleep):
    """Reserve+charge, launch, wait (watchdog), close.  Returns a dispatch record."""
    out = Path(out)
    reserve_and_charge(out, b, b["phase"])
    t0 = time.monotonic()
    proc = launcher(root, out, b, gpu)
    code = r2._wait(proc, timeout=WATCHDOG, sleep=sleep)
    if code == "WATCHDOG_TIMEOUT":
        r2._kill(proc)
        _watchdog_finish(out, b["branch_id"])
    r2._close(proc)
    return {"branch_id": b["branch_id"], "gpu": gpu, "exit_code": code, "wall_seconds": time.monotonic() - t0}


def step_verdict(out, b):
    row = evaluate_branch(out, b)
    ok = row["complete"] and row.get("task_success") is True and not row["problems"]
    return ok, row


def _set_step(out, step, status, **extra):
    with _lock(out):
        state = _comp_state(out)
        state["steps"][str(step)].update(status=status, **extra)
        write(Path(out) / "complementary_state.json", state)


def _fail_complementary(out, step, row, reason):
    with _lock(out):
        state = _comp_state(out)
        state["status"] = COMP_FAIL
        state["failed_step"] = step
        state["reason"] = reason
        write(Path(out) / "complementary_state.json", state)
        ledger = read(Path(out) / "budget_ledger.json")
        ledger["remaining16_released"] = False
        write(Path(out) / "budget_ledger.json", ledger)
        _event(out, "COMPLEMENTARY_OBSERVATION_GATE_FAIL", step=step, reason=reason)
    return complementary_gate(out)


def complementary_gate(out):
    """Recompute the complementary gate from the evidence; writes complementary_canary_gate.json."""
    out = Path(out)
    state = _comp_state(out)
    branches = {b["complementary_step"]: b for b in branches_by_phase(out, "complementary")}
    attempts = read(out / "physical/attempt_registry.json") if (out / "physical/attempt_registry.json").is_file() else {}
    rows, all_pass = {}, True
    for step in range(4):
        b = branches[step]
        row = evaluate_branch(out, b, attempts)
        ok = row["complete"] and row.get("task_success") is True and not row["problems"]
        rows[str(step)] = {"key": list(branch_key(b)), "branch_id": b["branch_id"], "pass": ok, "row": row,
                           "recorded_status": state["steps"][str(step)]["status"]}
        all_pass &= ok
    if all_pass and state["status"] != COMP_FAIL:
        status = COMP_PASS
    elif state["status"] == COMP_FAIL:
        status = COMP_FAIL
    else:
        status = "COMPLEMENTARY_IN_PROGRESS_OR_INCOMPLETE"
    gate = {"card_id": CARD, "status": status, "steps": rows, "failed_step": state.get("failed_step"),
            "reason": state.get("reason"), "remaining16_released": status == COMP_PASS,
            "technical_gate_only": True, "speed_criterion": None,
            "camera_threshold_or_config_changed": False, "fifth_complementary_attempt": False}
    write(out / "complementary_canary_gate.json", gate)
    return gate


def run_complementary(root, out, gpus, launcher=default_launcher, identity_check=verify_identity, sleep=time.sleep):
    """A -> B -> (C parallel D).  First failure writes COMPLEMENTARY_OBSERVATION_GATE_FAIL and stops."""
    root, out = Path(root).resolve(), Path(out).resolve()
    identity_check(root, out)
    state = _comp_state(out)
    if state["status"] != "NOT_STARTED":
        raise ValueError("STOPPED_COMPLEMENTARY_ALREADY_RUN")
    comp = {b["complementary_step"]: b for b in branches_by_phase(out, "complementary")}
    if len(comp) != 4 or [branch_key(comp[i]) for i in range(4)] != list(COMPLEMENTARY):
        raise ValueError("STOPPED_COMPLEMENTARY_CARDINALITY")
    _no_prior_results(out, comp.values())
    ledger = read(out / "budget_ledger.json")
    if ledger["attempts"]["used"] != 0 or ledger["reserved_branches"]:
        raise ValueError("STOPPED_COMPLEMENTARY_STATE")
    gpus = list(gpus)
    gpu_a = gpus[0]
    gpu_b = gpus[1] if len(gpus) > 1 else gpus[0]
    with _lock(out):
        st = _comp_state(out)
        st["status"] = "IN_PROGRESS"
        write(out / "complementary_state.json", st)
    started, dispatched = time.monotonic(), []

    def settle(step):
        ok, row = step_verdict(out, comp[step])
        _set_step(out, step, "PASS" if ok else "FAIL", problems=row["problems"][:20])
        return ok, row

    for step, gpu in ((0, gpu_a), (1, gpu_b)):
        dispatched.append({**run_branch(root, out, comp[step], gpu, launcher, sleep), "step": step})
        finalize_counts(out)
        ok, row = settle(step)
        if not ok:
            gate = _fail_complementary(out, step, row, f"step {step} {branch_key(comp[step])} failed its technical gate")
            _write_comp_throughput(out, dispatched, started, gpus)
            return {"gate": gate["status"], "dispatched": len(dispatched), "failed_step": step}
        state = _comp_state(out)
    # C and D (parallel when two GPUs exist)
    if gpu_a != gpu_b:
        procs = []
        for step, gpu in ((2, gpu_a), (3, gpu_b)):
            reserve_and_charge(out, comp[step], "complementary")
            procs.append((step, gpu, time.monotonic(), launcher(root, out, comp[step], gpu)))
        for step, gpu, t0, proc in procs:
            code = r2._wait(proc, timeout=WATCHDOG, sleep=sleep)
            if code == "WATCHDOG_TIMEOUT":
                r2._kill(proc)
                _watchdog_finish(out, comp[step]["branch_id"])
            r2._close(proc)
            dispatched.append({"branch_id": comp[step]["branch_id"], "gpu": gpu, "exit_code": code,
                               "wall_seconds": time.monotonic() - t0, "step": step, "parallel_with": "C/D"})
    else:
        for step in (2, 3):
            dispatched.append({**run_branch(root, out, comp[step], gpu_a, launcher, sleep), "step": step})
    finalize_counts(out)
    bad = [s for s in (2, 3) if not settle(s)[0]]
    if bad:
        gate = _fail_complementary(out, bad[0], None, f"steps {bad} failed their technical gate")
    else:
        with _lock(out):
            st = _comp_state(out)
            st["status"] = COMP_PASS
            write(out / "complementary_state.json", st)
            ledger = read(out / "budget_ledger.json")
            ledger["remaining16_released"] = True
            write(out / "budget_ledger.json", ledger)
            _event(out, COMP_PASS)
        gate = complementary_gate(out)
    _write_comp_throughput(out, dispatched, started, gpus)
    return {"gate": gate["status"], "dispatched": len(dispatched)}


def _write_comp_throughput(out, dispatched, started, gpus):
    write(Path(out) / "physical/throughput_complementary.json", {
        "phase": "complementary", "gpus": list(gpus), "dispatched": dispatched,
        "span_seconds": time.monotonic() - started, "single_worker_baseline": "NOT_MEASURED",
        "speedup_claim": "NONE (no matched single-worker baseline)"})


def _mark_end(dispatched, bid, t):
    for rec in reversed(dispatched):
        if rec["branch_id"] == bid:
            rec["t_end"] = t
            return


def engineering_exception(res):
    """Result missing, EXCEPTION or watchdog -> engineering; NO_PLAN/task failure are recorded separately."""
    if res is None:
        return "RESULT_MISSING"
    if res.get("status") == "EXCEPTION":
        return "EXCEPTION:" + str(res.get("detail", ""))[:60]
    if res.get("status") == "WATCHDOG_TIMEOUT":
        return "WATCHDOG_TIMEOUT"
    return None


def run_remaining16(root, out, gpus, workers=2, launcher=default_launcher, identity_check=verify_identity,
                    sleep=time.sleep):
    """Released only after the complementary pass.  One pair per worker at a time; U then V on the same GPU."""
    root, out = Path(root).resolve(), Path(out).resolve()
    gate = complementary_gate(out)
    if gate["status"] != COMP_PASS or _comp_state(out)["status"] != COMP_PASS:
        raise ValueError("STOPPED_REMAINING16_NOT_RELEASED")
    identity_check(root, out)
    ledger = read(out / "budget_ledger.json")
    if ledger["attempts"]["used"] != 4 or ledger["phases"]["remaining16"]["used"] != 0 or not ledger["remaining16_released"]:
        raise ValueError("STOPPED_REMAINING16_STATE")
    rest = branches_by_phase(out, "remaining16")
    if len(rest) != 16:
        raise ValueError("STOPPED_REMAINING16_CARDINALITY")
    _no_prior_results(out, rest)
    by_key = {branch_key(b): b for b in rest}
    queue = [[by_key[k] for k in pair] for pair in REMAINING16_PAIRS]
    gpus = list(gpus)[:max(1, min(int(workers), 4))]
    slots = {g: None for g in gpus}   # gpu -> {"pair": [...], "idx": int, "proc": ..., "t0": float}
    dispatched, faults, stop_reason, streak, last_sig = [], [], None, 0, None
    started = time.monotonic()
    while queue or any(slots.values()):
        for gpu, st in slots.items():
            if st is None and queue and stop_reason is None:
                slots[gpu] = st = {"pair": queue.pop(0), "idx": 0, "proc": None}
            if st is None:
                continue
            if st["proc"] is None:
                b = st["pair"][st["idx"]]
                reserve_and_charge(out, b, "remaining16")
                st["proc"], st["t0"] = launcher(root, out, b, gpu), time.monotonic()
                dispatched.append({"branch_id": b["branch_id"], "gpu": gpu, "pair": list(pair_key(branch_key(b))),
                                   "t_start": st["t0"] - started})
        if not any(slots.values()):
            break
        sleep(0.5)
        for gpu, st in slots.items():
            if st is None or st["proc"] is None:
                continue
            b = st["pair"][st["idx"]]
            code = st["proc"].poll()
            if code is None and time.monotonic() - st["t0"] > WATCHDOG:
                r2._kill(st["proc"])
                _watchdog_finish(out, b["branch_id"])
                code = "WATCHDOG_TIMEOUT"
            if code is None:
                continue
            r2._close(st["proc"])
            _mark_end(dispatched, b["branch_id"], time.monotonic() - started)
            sig = engineering_exception(_result(out, b["branch_id"]))
            if sig:
                faults.append({"branch_id": b["branch_id"], "exit_code": code, "signature": sig})
                streak = streak + 1 if sig == last_sig else 1
                last_sig = sig
                if streak >= 2 and stop_reason is None:
                    stop_reason = "TWO_CONSECUTIVE_COMMON_ENGINEERING_EXCEPTIONS"
            else:
                streak, last_sig = 0, None
            st["proc"] = None
            st["idx"] += 1
            if st["idx"] >= len(st["pair"]):
                slots[gpu] = None
    span = time.monotonic() - started
    finalize_counts(out)
    results = [r for r in (_result(out, b["branch_id"]) for b in rest) if r]
    skills = sum(len(r.get("actions", [])) for r in results)
    write(out / "physical/throughput_remaining16.json", {
        "phase": "remaining16", "workers": len(gpus), "gpus": gpus, "dispatched": dispatched, "faults": faults,
        "stop_reason": stop_reason, "not_dispatched": [b["branch_id"] for q in queue for b in q],
        "span_seconds": span, "executed_skills": skills,
        "aggregate_executed_skills_per_second": skills / span if span else None,
        "per_branch_wall_seconds": {r["branch_id"]: r.get("worker_wall_seconds") for r in results},
        "scaled_beyond_two_workers": len(gpus) > 2,
        "single_worker_baseline": "NOT_MEASURED", "speedup_claim": "NONE (no matched single-worker baseline)"})
    return {"dispatched": [d["branch_id"] for d in dispatched], "faults": faults, "stop_reason": stop_reason}


# =============================================================================== combined analysis
def combined_registry(root, out):
    """key -> (branch, evidence base dir, source label).  R2 four are read-only references."""
    items = {k: (b, Path(root) / R2_DIR, "R2_V2_TECHNICAL_INPUT_FROZEN") for k, b in r2_branches(root).items()}
    for b in branches_by_phase(out):
        items[branch_key(b)] = (b, Path(out), "STAGING_V2_CONTINUATION")
    return items


def _attempts(base):
    p = Path(base) / "physical/attempt_registry.json"
    return read(p) if p.is_file() else {}


def combined_rows(root, out):
    reg = combined_registry(root, out)
    rows = []
    for k in ALL24:
        if k not in reg:
            rows.append({"layout": k[0], "context": k[1], "repeat": k[2], "candidate": k[3], "present": False,
                         "complete": False, "task_success": False, "cost": None, "problems": ["branch not registered"]})
            continue
        b, base, src = reg[k]
        res = _result(base, b["branch_id"])
        ev = (r2.evaluate_branch(base, b, _attempts(base)) if src.startswith("R2") else
              evaluate_branch(base, b, _attempts(base)))
        seq = [a["action_id"] for a in (res or {}).get("actions", [])]
        success = bool(res and res.get("status") == "TASK_SUCCESS" and res.get("task_success") is True)
        rows.append({
            "layout": k[0], "context": k[1], "repeat": k[2], "candidate": k[3], "present": res is not None,
            "branch_id": b["branch_id"], "source": src, "card_id": b["card_id"], "status": (res or {}).get("status"),
            "task_success": success, "complete": bool(res) and not ev["problems"],
            "cost": (res or {}).get("time_to_task_success") if success else None,
            "candidate_start_sim_time": (res or {}).get("candidate_start_sim_time"),
            "terminal_sim_time": (res or {}).get("terminal_sim_time"),
            "worker_wall_seconds": (res or {}).get("worker_wall_seconds"), "sequence": seq,
            "n_actions": len(seq), "engineering_exception": engineering_exception(res) if res is not None else "RESULT_MISSING",
            "both_pads_legal": ev["checks"].get("both_pads_legal"), "problems": ev["problems"],
            "env_counts": (res or {}).get("env_counts")})
    return rows


def _skill_type(action_id):
    return action_id.split(":")[1]


def _tail(row, n_prefix):
    return row["sequence"][n_prefix + 1:]


PREFIX_LEN = {"B_PENDING": 3, "C_PENDING": 3, "BOTH_PENDING": 1}


def _sign(x):
    return 0 if x is None or x == 0 else (1 if x > 0 else -1)


def pairwise(rows):
    """U vs V per (layout, context, repeat).  Different post-candidate sequences are NOT cost evidence."""
    by = {(r["layout"], r["context"], r["repeat"], r["candidate"]): r for r in rows}
    out = []
    for layout, context, repeat in sorted({k[:3] for k in by}, key=lambda p: (LAYOUTS.index(p[0]), CONTEXTS.index(p[1]), p[2])):
        u, v = by.get((layout, context, repeat, "pad_u")), by.get((layout, context, repeat, "pad_v"))
        row = {"layout": layout, "context": context, "repeat": repeat, "cost_u": u and u.get("cost"),
               "cost_v": v and v.get("cost"), "delta_u_minus_v": None, "abs_delta": None, "sign": 0, "winner": None,
               "both_success": bool(u and v and u["task_success"] and v["task_success"]), "comparable": False,
               "tails_equal": None, "same_length": None, "same_skill_multiset": None}
        if u and v and u.get("sequence") and v.get("sequence"):
            n = PREFIX_LEN[context]
            row["tails_equal"] = _tail(u, n) == _tail(v, n)
            row["same_length"] = len(u["sequence"]) == len(v["sequence"])
            row["same_skill_multiset"] = (Counter(map(_skill_type, u["sequence"])) ==
                                          Counter(map(_skill_type, v["sequence"])))
            row["comparable"] = bool(row["tails_equal"] and row["same_length"] and row["same_skill_multiset"])
        if row["both_success"] and row["comparable"]:
            d = u["cost"] - v["cost"]
            row.update(delta_u_minus_v=d, abs_delta=abs(d), sign=_sign(d),
                       winner="pad_u" if d < 0 else ("pad_v" if d > 0 else None))
        out.append(row)
    return out


def direction_checks(cells):
    """cells: {(layout, context): {"deltas": {repeat: delta}, "mean": x}} for the gated B/C cells."""
    c5 = all(_sign(cells[(l, "B_PENDING")]["mean"]) != 0 and _sign(cells[(l, "C_PENDING")]["mean"]) != 0 and
             _sign(cells[(l, "B_PENDING")]["mean"]) != _sign(cells[(l, "C_PENDING")]["mean"]) for l in LAYOUTS)
    c7 = all(len({_sign(d) for d in cell["deltas"].values()}) == 1 and 0 not in {_sign(d) for d in cell["deltas"].values()}
             for cell in cells.values())
    winners = {"pad_u" if cell["mean"] < 0 else "pad_v" for cell in cells.values() if cell["mean"] not in (None, 0)}
    c8 = winners == set(PADS)
    return {"c5_sign_reversal_within_layout": bool(c5), "c7_repeats_agree": bool(c7),
            "c8_not_same_candidate_always_wins": bool(c8), "winner_set": sorted(winners)}


def mechanism_gate(rows, pair_restore_ok, tol=TOL):
    pairs = pairwise(rows)
    by = {(p["layout"], p["context"], p["repeat"]): p for p in pairs}
    gated = [(l, c) for l in LAYOUTS for c in ("B_PENDING", "C_PENDING")]
    structure = all((l, c, r) in by for l, c in gated for r in REPEATS)
    c1 = len(rows) == 24 and all(r["task_success"] for r in rows)
    c2 = len(rows) == 24 and all(r["complete"] for r in rows)
    engineering = [r.get("branch_id") or "/".join(map(str, (r["layout"], r["context"], r["repeat"], r["candidate"])))
                   for r in rows if r.get("engineering_exception") or not r.get("present", True)]
    cells, per_cell = {}, {}
    if structure:
        for l, c in gated:
            ds = {r: by[(l, c, r)]["delta_u_minus_v"] for r in REPEATS}
            valid = [d for d in ds.values() if d is not None]
            mean_d = float(np.mean([r_["cost_u"] for r_ in (by[(l, c, r)] for r in REPEATS)]) -
                           np.mean([r_["cost_v"] for r_ in (by[(l, c, r)] for r in REPEATS)])) \
                if len(valid) == len(REPEATS) else None
            cells[(l, c)] = {"deltas": ds, "mean": mean_d}
            per_cell[f"{l}/{c}"] = {"delta_by_repeat": ds, "mean_delta_u_minus_v": mean_d}
    complete_cells = structure and all(cell["mean"] is not None and None not in cell["deltas"].values() for cell in cells.values())
    dirs = direction_checks(cells) if complete_cells else {"c5_sign_reversal_within_layout": False,
                                                           "c7_repeats_agree": False,
                                                           "c8_not_same_candidate_always_wins": False, "winner_set": []}
    c6 = bool(complete_cells and all(abs(d) > tol for cell in cells.values() for d in cell["deltas"].values()) and
              all(abs(cell["mean"]) > tol for cell in cells.values()))
    gated_pairs = [by[(l, c, r)] for l, c in gated for r in REPEATS if (l, c, r) in by]
    c9 = len(rows) == 24 and all(r.get("both_pads_legal") for r in rows)
    c10 = bool(gated_pairs) and all(p["comparable"] for p in gated_pairs)
    both = [r for r in rows if r["context"] == "BOTH_PENDING"]
    c12 = len(both) == 8 and all(r["complete"] and r["task_success"] for r in both)
    c11 = bool(c1 and c2 and not engineering)
    conditions = {
        "1_24_of_24_task_success": c1, "2_24_of_24_complete_records": c2, "3_all_pair_restores_in_contract": bool(pair_restore_ok),
        "4_B_and_C_two_repeats_per_layout": structure, "5_sign_reversal_within_layout": dirs["c5_sign_reversal_within_layout"],
        "6_each_abs_delta_above_threshold": c6, "7_repeats_agree_in_direction": dirs["c7_repeats_agree"],
        "8_same_candidate_not_always_winner": dirs["c8_not_same_candidate_always_wins"],
        "9_both_candidates_contract_legal": c9, "10_plan_length_skill_multiset_and_tail_match": c10,
        "11_no_missing_precondition_no_plan_timeout_controller_or_unknown": c11,
        "12_both_pending_complete_and_reported": c12,
        "13_both_not_required_neutral": True, "14_no_significance_or_generalization_claim": True}
    unresolved = (not structure or not (c1 and c2 and pair_restore_ok) or bool(engineering) or not complete_cells)
    established = all(conditions.values())
    status = "UNRESOLVED_ENGINEERING" if unresolved else ("ESTABLISHED" if established else "NOT_ESTABLISHED")
    next_action = {"ESTABLISHED": "REQUEST_FAMILY_B_PROVIDER_AND_REPRESENTATION_AUTHORIZATION",
                   "NOT_ESTABLISHED": "S4_RESEARCH_DECISION",
                   "UNRESOLVED_ENGINEERING": "ENGINEERING_REVIEW_REQUIRED_USER_DECISION"}[status]
    return {"card_id": CARD, "status": status, "next_action": next_action, "tolerance_seconds": tol,
            "cost_definition": "C_s(a) = time_of_first_verified_task_success - candidate_start_sim_time",
            "delta_definition": "delta_s = mean C_s(pad_u) - mean C_s(pad_v)", "conditions": conditions,
            "winner_set": dirs["winner_set"], "cells": per_cell, "pairs": pairs,
            "both_pending_reported_only": [p for p in pairs if p["context"] == "BOTH_PENDING"],
            "engineering_incidents": engineering,
            "repeats_are_repeatability_checks_not_independent_geometries": True,
            "claims": {"significance": False, "generalization": False, "provider_or_representation_started": False},
            "unit": "2 layouts x 3 progress contexts x 2 repeats x 2 candidates = 24"}


def _csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items() if k in fields})


COST_FIELDS = ["layout", "context", "repeat", "candidate", "branch_id", "source", "status", "task_success", "complete",
               "cost", "candidate_start_sim_time", "terminal_sim_time", "worker_wall_seconds", "n_actions",
               "engineering_exception", "sequence"]
PAIR_FIELDS = ["layout", "context", "repeat", "cost_u", "cost_v", "delta_u_minus_v", "abs_delta", "sign", "winner",
               "both_success", "comparable", "tails_equal", "same_length", "same_skill_multiset"]


def support_combined(root, out):
    reg = combined_registry(root, out)
    rows = []
    for k in ALL24:
        if k not in reg:
            continue
        b, base, _ = reg[k]
        cap = base / "captures" / b["branch_id"]
        for stage, fid, d, frame in _frame_sources(cap):
            for obj, f in frame["fusion"].items():
                for view, inp in f["inputs"].items():
                    rows.append({"branch_id": b["branch_id"], "layout": k[0], "context": k[1], "repeat": k[2],
                                 "candidate": k[3], "action_dir": stage, "frame_id": fid, "object": obj, "view": view,
                                 "support": inp["support"], "reference": inp["reference"], "ratio": inp["ratio"],
                                 "valid": inp["valid"], "selected_view": f["selected_view"],
                                 "fused_present": f["blob"] is not None,
                                 "mask_pixels": frame["cameras"][view]["objects"][obj]["mask_pixels"]})
    _csv(Path(out) / "observation_support_combined.csv",
         ["branch_id", "layout", "context", "repeat", "candidate", "action_dir", "frame_id", "object", "view", "support",
          "reference", "ratio", "valid", "selected_view", "fused_present", "mask_pixels"], rows)
    return len(rows)


def build_paired_restore():
    """The frozen v2 pair evaluation, unchanged, applied to arbitrary (layout, context, repeat) keys."""
    import inspect
    src = inspect.getsource(m.paired_restore)
    edits = [("def paired_restore(out):", "def paired_restore_keys(out, KEYS, REG, CAPDIR):", 1),
             ('reg = read(out / "physical/registration.json")', "reg = REG", 1),
             ("for key in TECHNICAL:", "for key in KEYS:", 1),
             ('out / "captures" / b["branch_id"] / "boundary" for b in (u, v)', 'CAPDIR(b) / "boundary" for b in (u, v)', 1),
             ('out / "captures" / u["branch_id"]', "CAPDIR(u)", 2),
             ('out / "captures" / v["branch_id"]', "CAPDIR(v)", 2),
             ('write(out / "paired_restore_v2.json", result)', "pass", 1)]
    for old, new, count in edits:
        if src.count(old) != count:
            raise RuntimeError(f"STOPPED_SOURCE_IDENTITY:paired_restore changed ({old!r} x{src.count(old)})")
        src = src.replace(old, new)
    ns = dict(vars(m))
    exec(compile(src, "<family_b_staging_v2:paired_restore>", "exec"), ns)
    return ns["paired_restore_keys"]


def paired_restore_combined(root, out):
    reg = combined_registry(root, out)
    allb = {"branches": [v[0] for v in reg.values()]}
    base_of = {v[0]["branch_id"]: v[1] for v in reg.values()}
    keys = sorted({pair_key(k) for k in ALL24}, key=lambda p: (LAYOUTS.index(p[0]), CONTEXTS.index(p[1]), p[2]))
    result = build_paired_restore()(Path(out), keys, allb, lambda b: base_of[b["branch_id"]] / "captures" / b["branch_id"])
    result["card_id"] = CARD
    result["pairs_expected"] = len(keys)
    result["status"] = "PASS" if len(result["pairs"]) == 12 and all(p["status"] == "PASS" for p in result["pairs"].values()) else "FAIL"
    write(Path(out) / "paired_restore_combined.json", result)
    return result


def merge_throughput(out):
    out = Path(out)
    phases = {p.stem.split("_", 1)[1]: read(p) for p in sorted((out / "physical").glob("throughput_*.json"))}
    write(out / "throughput.json", {"phases": phases, "single_worker_baseline": "NOT_MEASURED",
                                    "speedup_claim": "NONE (no matched single-worker baseline)"})


def analyze(root, out):
    root, out = Path(root), Path(out)
    ledger = finalize_counts(out)
    rows = combined_rows(root, out)
    reg_new = {b["branch_id"] for b in branches_by_phase(out)}
    new_rows = [r for r in rows if r.get("branch_id") in reg_new]
    comp_rows = [r for r in new_rows if branch_key({"layout": r["layout"], "context": r["context"], "repeat": r["repeat"],
                                                    "candidate": r["candidate"]}) in COMPLEMENTARY]
    _csv(out / "branch_results_new20.csv", COST_FIELDS, new_rows)
    _csv(out / "complementary_branch_results.csv", COST_FIELDS, comp_rows)
    _csv(out / "combined_v2_24_results.csv", COST_FIELDS, rows)
    _csv(out / "physical_cost_by_branch.csv", COST_FIELDS, rows)
    support_combined(root, out)
    pr = paired_restore_combined(root, out)
    gate = mechanism_gate(rows, pr["status"] == "PASS")
    _csv(out / "physical_cost_pairwise.csv", PAIR_FIELDS, gate["pairs"])
    write(out / "physical_mechanism_gate.json", gate)
    merge_throughput(out)
    manifest = combined_manifest(root, out, rows)
    write(out / "combined_v2_24_manifest.json", manifest)
    return {"rows": len(rows), "mechanism": gate["status"], "paired_restore": pr["status"]}


def combined_manifest(root, out, rows):
    root = Path(root)
    reg = combined_registry(root, out)
    items = []
    for k in ALL24:
        b, base, src = reg[k]
        res = base / "physical/branch_results" / f"{b['branch_id']}.json"
        items.append({"logical": list(k), "branch_id": b["branch_id"], "source": src, "card_id": b["card_id"],
                      "attempt_identity": b["attempt_identity"],
                      "result_path": str(res.relative_to(root)) if res.is_file() else None,
                      "result_sha256": sha(res) if res.is_file() else None,
                      "captures_path": str((base / "captures" / b["branch_id"]).relative_to(root))
                      if str(base).startswith(str(root)) else str(base / "captures" / b["branch_id"]),
                      "source_commit": b["source_commit"], "source_hash": b["source_hash"],
                      "observation_profile_sha256": b["observation_profile_sha256"],
                      "runtime_manifest": b["manifest_path"]})
    compat = read(Path(out) / "compatibility_manifest.json")
    return {"card_id": CARD, "branches": items, "count": len(items),
            "r2_four_read_only_references": True, "r2_raw_evidence_copied_or_modified": False,
            "different_cards_and_attempt_ledgers": {"R2": R2_DIR, "staging": str(Path(out).relative_to(root))},
            "compatibility_status": compat["status"], "compatibility_rationale":
            "production sources, frozen profile, runtime/recorder/timing path, planner, FactStore and Verifier/Evaluator are "
            "byte-identical between the R2 prep commit and this card's prep commit apart from the registered additions",
            "unit": "2 layouts x 3 progress contexts x 2 repeats x 2 candidates = 24; repeats are repeatability checks, "
                    "not independent geometries",
            "family_b_physical_attempts_cumulative": 4 + 2 + 4 + sum(1 for r in rows if r.get("present") and r["source"].startswith("STAGING"))}


# ============================================================================= summary / verify
REQUIRED_OUTPUTS = (
    "authorization.json", "source_identity.json", "compatibility_manifest.json", "r2_phase_reconciliation.json",
    "protected_before.json", "protected_after.json", "observation_margin_review.json", "observation_margin_review.md",
    "observation_coverage_matrix.csv", "registration_v2_remaining20.json", "budget_ledger.json", "budget_events.jsonl",
    "complementary_canary_gate.json", "complementary_branch_results.csv", "branch_results_new20.csv",
    "combined_v2_24_manifest.json", "combined_v2_24_results.csv", "paired_restore_combined.json",
    "observation_support_combined.csv", "physical_cost_by_branch.csv", "physical_cost_pairwise.csv",
    "physical_mechanism_gate.json", "throughput.json", "final_summary.md", "verify.json")


def summarize(root, out):
    out = Path(out)
    ledger = read(out / "budget_ledger.json")
    act = ledger["actual"]
    comp = read(out / "complementary_canary_gate.json")
    gate = read(out / "physical_mechanism_gate.json")
    gm = read(out / "observation_margin_review.json")
    rows = [r for r in csv.DictReader((out / "combined_v2_24_results.csv").open())]
    used = ledger["attempts"]["used"]
    lines = [
        "# Family B v2 physical qualification (staging continuation)", "",
        f"- Card {CARD}; frozen profile `{m.PROFILE_VERSION}` {FROZEN_SHA} (unchanged).",
        f"- Gate M: **{gm['status']}** (offline, zero new samples). Tested pad_v remaining-target sideview support: "
        + "; ".join(f"{s['layout']}/{s['object']}={s['support']}px (margin {s['margin_to_min_pixels']})"
                    for s in gm["statements"]["tested_pad_v_remaining_target_sideview_support"]) + ".",
        f"- Complementary gate: **{comp['status']}**.",
        f"- Continuation attempts used {used}/20; constructions {act.get('constructions_attempted')}; constructor-successful "
        f"{act.get('constructor_successful')}; explicit resets {act.get('explicit_resets')}; internal resets "
        f"{act.get('internal_resets')}; live skill calls {act.get('skill_calls')}/120.",
        f"- Cumulative Family B physical attempts: {4 + 2 + 4 + used} (engineering count; not the Experimental Plan v3 '17+4').",
        "- Provider / representation / RL / optimizer / elastic / formal test = 0; S2/S3/TP training = false.",
        f"- Physical mechanism status: **{gate['status']}**; next action: `{gate['next_action']}`.", "",
        "| layout | context | repeat | C(pad_u) | C(pad_v) | delta u-v | comparable |", "|---|---|---|---|---|---|---|"]
    for p in gate["pairs"]:
        lines.append(f"| {p['layout']} | {p['context']} | {p['repeat']} | {p['cost_u']} | {p['cost_v']} | "
                     f"{p['delta_u_minus_v']} | {p['comparable']} |")
    lines += ["", "Conditions: " + ", ".join(f"{k}={v}" for k, v in gate["conditions"].items()),
              f"- 24-branch task success: {sum(r['task_success'] == 'True' for r in rows)}/24; complete records: "
              f"{sum(r['complete'] == 'True' for r in rows)}/24.",
              "- No significance or generalization claim; speedup NOT_MEASURED."]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n")


def verify(root, out):
    root, out = Path(root), Path(out)
    before = read(out / "protected_before.json")
    after_inv = m.inventory(protected_paths(root))
    write(out / "protected_after.json", {"sha256": after_inv, "files": len(after_inv),
                                         "capture_count": sum("/captures/" in k for k in after_inv)})
    diff = m.diff_inventory(before["sha256"], after_inv)
    ledger = read(out / "budget_ledger.json")
    reg = read(out / "physical/registration.json")
    results = list((out / "physical/branch_results").glob("*.json"))
    zero_keys = ("provider_calls", "representation_forwards", "rl_transitions", "optimizer_steps", "elastic",
                 "skill_retries", "standalone_capture_resets", "formal_test")
    missing = [n for n in REQUIRED_OUTPUTS if n != "verify.json" and not (out / n).is_file()]
    checks = {
        "protected_unchanged": not (diff["changed"] or diff["removed"] or diff["added"]),
        "frozen_profile_files_equal_freeze_commit": not r2.frozen_bytes_ok(root),
        "attempts_le_20": ledger["attempts"]["used"] <= 20 and len(results) <= 20,
        "constructions_le_20": ledger["environment_constructions"]["used"] <= 20,
        "resets_le_20": ledger["explicit_resets"]["used"] <= 20,
        "skills_le_120": ledger["live_skill_calls"]["used"] <= 120,
        "no_duplicate_reservation": len(set(ledger["reserved_branches"])) == len(ledger["reserved_branches"]),
        "registration_is_the_20": len(reg["branches"]) == 20,
        "provider_representation_rl_optimizer_zero": all(ledger[k]["used"] == 0 for k in zero_keys),
        "required_outputs_present": not missing}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "missing_outputs": missing,
              "protected_diff": diff, "protected_files": len(after_inv)}
    write(out / "verify.json", result)
    return result

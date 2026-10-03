"""CP-DISR-TP-CARRIER-SUPPORT-DESIGN-AUDIT-1: design audit of a carrier-support (B supports A) T_P family.

Zero environment: static BDDL/XML/mesh/source analysis, quasi-static mechanics and route/contract analysis only. It never constructs an
environment and never imports robosuite, mujoco or libero. Label: DESIGN_ONLY_NOT_AUTHORIZED. Stage 1 freezes the asset universes from
names and BDDL only (no geometry); stage 2 reads geometry and evaluates the gates. All rules and thresholds are declared in code before the run.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cp_disr.analysis import tp_libero_audit as la
from cp_disr.analysis import tp_dv1_binding_preflight as dv
from cp_disr.analysis import tp_occ_preflight as occ

CARD = "CP-DISR-TP-CARRIER-SUPPORT-DESIGN-AUDIT-1"
BASELINE_FULL = "eb262a8da1b66c4de1f1aef3dc8d909023f9b731"
UPSTREAM_SHA = la.UPSTREAM_SHA
LABEL = "DESIGN_ONLY_NOT_AUTHORIZED"
FAMILY = "T_P_CARRIER_SUPPORT_V1"
RESOURCE_KEYS = la.RESOURCE_KEYS
ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
CARRIER_NAME_RE = re.compile(r"tray|basket|caddy|plate|carrier")
PLACEMENT_REGION_RE = re.compile(r"contain|top|support|cook")      # contain / support / top regions
NEAR_DIST = 0.15                                                    # initial-region gap that counts as "near a carrier candidate"
FAIL_ORDER = ["TASK_FAIL_NO_GENERIC_CARRIER", "TASK_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR", "TASK_FAIL_NO_CARRIED_OBJECT", "TASK_FAIL_NO_STABLE_SUPPORT_PAIR", "TASK_FAIL_NO_STABILITY_CONTRAST",
              "TASK_FAIL_COMMUTATIVE", "TASK_FAIL_FIXED_HEURISTIC_SUFFICIENT", "TASK_FAIL_STATIC_PRIOR_SUFFICIENT", "TASK_FAIL_CONTRACT_REVEALED", "TASK_FAIL_REQUIRES_NEW_PERCEPTION_MODEL",
              "TASK_FAIL_EVALUATOR_SEMANTIC_CHANGE", "TASK_FAIL_SNAPSHOT_REWRITE", "TASK_FAIL_RELATION_NOT_EXPRESSIBLE", "TASK_FAIL_CANARY_NOT_BOUNDED", "ENGINEERING_UNRESOLVED"]
digest = la.digest
write_json = la.write_json
write_csv = la.write_csv
sha_file = la.sha_file


def git_out(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------------------------------------ stage 1: universes
def scan_official_tasks(lib, guard):
    """Parse every official BDDL of the pinned tree; return per-task facts needed for the universes (no geometry, no assets)."""
    tmap = la.load_task_map(Path(lib) / "benchmark" / "libero_suite_task_map.py", guard)
    reg = la.load_registry(lib, guard)
    tasks = []
    for s in ALL_SUITES:
        for i, name in enumerate(tmap[s]):
            p = la.parse_bddl(guard.read(Path(lib) / "bddl_files" / s / (name + ".bddl")))
            inst, sect = la.inst_maps(p)
            tasks.append({"suite": s, "index": i, "name": name, "parsed": p, "inst": inst, "sect": sect})
    return tasks, reg


def placements(task, reg):
    """Every On/In (init and goal) whose destination is an object/fixture instance (not a table surface)."""
    p, inst, sect = task["parsed"], task["inst"], task["sect"]
    out = []
    for where, atoms in (("init", p["init"]), ("goal", p["goal_atoms"])):
        for a in atoms:
            if a[0].lower() not in ("on", "in") or len(a) < 3:
                continue
            res = la.resolve(a[2], p, inst, sect)
            subj = inst.get(a[1])
            if res["instance"] is None or subj is None or sect.get(a[1]) != "object" or la.is_surface(res["type"], reg):
                continue
            if res["instance"] == a[1]:
                continue
            out.append({"where": where, "pred": a[0], "subject_type": subj, "subject": a[1], "dest_type": res["type"], "dest_instance": res["instance"], "dest_role": sect.get(res["instance"]), "region": res["region"],
                        "region_class": ("direct_on_instance" if res["region"] is None else ("qualifying" if PLACEMENT_REGION_RE.search(res["region"]) else "other"))})
    return out


def surface_ranges(task, instance):
    p, inst, sect = task["parsed"], task["inst"], task["sect"]
    for a in p["init"]:
        if a[0].lower() == "on" and a[1] == instance and len(a) >= 3:
            res = la.resolve(a[2], p, inst, sect)
            rec = p["regions"].get((res["instance"], res["region"]))
            if rec and rec["ranges"] and res["type"] and la.is_surface(res["type"], {}):
                return res["instance"], rec["ranges"]
    return None, None


def build_universes(tasks, reg):
    carrier_by_type = defaultdict(lambda: {"roles": set(), "evidence": [], "name_match": False, "task_count": 0})
    for t in tasks:
        for pl in placements(t, reg):
            if pl["region_class"] in ("direct_on_instance", "qualifying"):
                c = carrier_by_type[pl["dest_type"]]
                c["roles"].add(pl["dest_role"])
                c["task_count"] += 1
                if len(c["evidence"]) < 6:
                    c["evidence"].append({"suite": t["suite"], "task": t["name"], "pred": pl["pred"], "subject_type": pl["subject_type"], "region": pl["region"], "where": pl["where"]})
    for k in sorted(reg):
        if CARRIER_NAME_RE.search(k):
            carrier_by_type[k]["name_match"] = True
    for t in tasks:
        for n, ty in t["parsed"]["fixtures"] + t["parsed"]["objects"]:
            if ty in carrier_by_type:
                carrier_by_type[ty]["roles"].add("fixture" if (n, ty) in t["parsed"]["fixtures"] else "object")
    carriers = {}
    for ty, c in sorted(carrier_by_type.items()):
        if la.is_surface(ty, reg):
            continue
        roles = sorted(c["roles"])
        if "object" in roles:
            cls = "NATIVE_MOVABLE_OBJECT"
        elif "fixture" in roles:
            cls = "OFFICIAL_FIXTURE"
        elif ty in reg and not reg[ty]["articulated"]:
            cls = "NATIVE_MOVABLE_OBJECT"
        else:
            cls = "NOT_A_CARRIER"
        carriers[ty] = {"type": ty, "classification": cls, "roles_in_official_tasks": roles, "name_matches_rule": c["name_match"], "official_placement_count": c["task_count"], "used_in_official_tasks": c["task_count"] > 0 or bool(roles),
                        "registered_class": reg.get(ty, {}).get("class"), "articulated_class": bool(reg.get(ty, {}).get("articulated")), "evidence": c["evidence"],
                        "derived_free_body_possible": "decided in stage 2 (fixtures only)" if cls == "OFFICIAL_FIXTURE" else None}
    # carried objects: placed on/in a carrier type in an official task, or initially near a carrier-type instance
    carried = {}
    for t in tasks:
        for pl in placements(t, reg):
            if pl["dest_type"] in carriers and pl["region_class"] in ("direct_on_instance", "qualifying") and pl["subject_type"] in reg:
                c = carried.setdefault(pl["subject_type"], {"reasons": set(), "evidence": []})
                c["reasons"].add("PLACED_ON_OR_IN_CARRIER")
                if len(c["evidence"]) < 6:
                    c["evidence"].append({"suite": t["suite"], "task": t["name"], "carrier_type": pl["dest_type"], "pred": pl["pred"], "where": pl["where"]})
        movable = [n for n, s in t["sect"].items() if s == "object"]
        for cn in [n for n in movable if t["inst"][n] in carriers]:
            ctab, cr = surface_ranges(t, cn)
            if not cr:
                continue
            for on in movable:
                if on == cn or t["inst"][on] not in reg:
                    continue
                otab, orr = surface_ranges(t, on)
                if otab != ctab or not orr:
                    continue
                gap = min(max(la.rect_gap(r1, r2)[0][0], la.rect_gap(r1, r2)[1][0]) for r1 in cr for r2 in orr)
                if gap <= NEAR_DIST:
                    c = carried.setdefault(t["inst"][on], {"reasons": set(), "evidence": []})
                    c["reasons"].add("INITIALLY_NEAR_CARRIER")
                    if len(c["evidence"]) < 6:
                        c["evidence"].append({"suite": t["suite"], "task": t["name"], "carrier_type": t["inst"][cn], "gap_m": round(gap, 4)})
    carried = {k: {"type": k, "registered_class": reg[k]["class"], "reasons": sorted(v["reasons"]), "evidence": v["evidence"], "is_also_carrier_candidate": k in carriers} for k, v in sorted(carried.items())}
    return carriers, carried


def stage1(root, libero_root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard = la.Guard()
    head = git_out(["rev-parse", "HEAD"], root)
    ident = la.upstream_identity(libero_root, guard, None)
    write_json(out / "upstream_libero_identity.json", ident)
    if not ident["sha_matches_expected"]:
        raise RuntimeError("UPSTREAM_SHA_MISMATCH:%s" % ident["head_sha"])
    lib = Path(libero_root).resolve() / "libero" / "libero"
    tasks, reg = scan_official_tasks(lib, guard)
    carriers, carried = build_universes(tasks, reg)
    cu = {"card": CARD, "frozen_before_geometry": True, "rule": {"name_regex": CARRIER_NAME_RE.pattern, "placement_region_regex": PLACEMENT_REGION_RE.pattern, "near_distance_m": NEAR_DIST, "suites_scanned": list(ALL_SUITES),
                                                               "note": "all five suites of the pinned tree are scanned as TEXT (BDDL) to build the universe; LIBERO-90 tasks are not audited as tasks"},
          "carriers": carriers, "classification_counts": dict(Counter(c["classification"] for c in carriers.values()))}
    cu["universe_sha256"] = digest({k: (v["type"], v["classification"]) for k, v in carriers.items()})
    ou = {"card": CARD, "frozen_before_geometry": True, "rule": {"placed_on_or_in_a_carrier_type_in_an_official_task": True, "initially_within_near_distance_on_the_same_surface": NEAR_DIST}, "objects": carried}
    ou["universe_sha256"] = digest({k: v["reasons"] for k, v in carried.items()})
    write_json(out / "carrier_asset_universe.json", cu)
    write_json(out / "carried_object_universe.json", ou)
    write_json(out / "universe_manifest.json", {"card": CARD, "project_head_full": head, "upstream_sha": ident["head_sha"], "carrier_universe_sha256": cu["universe_sha256"], "carried_universe_sha256": ou["universe_sha256"],
                                                "carrier_types": sorted(carriers), "carried_types": sorted(carried), "files": {n: sha_file(out / n) for n in ("carrier_asset_universe.json", "carried_object_universe.json", "upstream_libero_identity.json")}})
    return {"carriers": sorted(carriers), "classes": cu["classification_counts"], "carried": sorted(carried)}

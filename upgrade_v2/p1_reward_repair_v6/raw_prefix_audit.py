"""Rebuild states from truncated original CSVs. Not a scored-state prefix check."""
from __future__ import annotations
import csv, json, shutil, tempfile
from copy import deepcopy
from pathlib import Path
from .util import require_new, sha256_file, write_csv, write_json

def _load_tools(pkg: Path):
    import sys
    sys.path.insert(0, str(Path(pkg)/"tools"))
    from normalize_v5 import normalize, load_upstream, read_raw
    from reward_v6 import score_episode
    from engine_parity import load_engine
    return normalize, load_upstream, read_raw, score_episode, load_engine


def _write_trunc(src_csv: Path, n: int, dest_csv: Path):
    with src_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("empty csv")
    header, body = rows[0], rows[1:]
    n = max(2, min(n, len(body)))
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with dest_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body[:n])
    return n


def _state_core(s: dict) -> dict:
    return {
        "state_index": s["state_index"],
        "available_at_ns": s["available_at_ns"],
        "physical_time_ns": s["physical_time_ns"],
        "capture_order": s["capture_order"],
        "task": s["task"],
        "success": s.get("success"),
        "gripper_closed": s.get("gripper_closed"),
        "objects": {k: {f: v[f] for f in ("held","valid","phase","pos","vel","target_xy") if f in v} for k,v in s["objects"].items()},
        "events": s.get("events") or [],
        "legacy": s.get("legacy"),
    }


def _event_cuts(seq: list[dict]) -> set[int]:
    cuts = set()
    n = len(seq)
    for frac in (0.25, 0.5, 0.75, 1.0):
        cuts.add(max(2, int(round(frac * n))))
    for i,s in enumerate(seq):
        if s.get("events"):
            cuts.add(max(2, i+1))
            if i >= 1:
                cuts.add(i)
            cuts.add(min(n, i+2))
    return {c for c in cuts if 2 <= c <= n}


def audit_one(dest: Path, mods, normalize, score_episode, Engine, tmp: Path) -> dict:
    full, prov = normalize(dest, mods)
    cuts = sorted(_event_cuts(full))
    rows = []
    # rebuild from truncated CSV
    for n in cuts:
        work = tmp/f"cut_{n}"
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(dest, work)
        kept = _write_trunc(dest/"timeseries.csv", n, work/"timeseries.csv")
        rebuilt, _ = normalize(work, mods)
        core_a = [_state_core(s) for s in full[:kept]]
        core_b = [_state_core(s) for s in rebuilt]
        match = core_a == core_b
        # rewards from truncated rebuild vs full prefix
        if Engine is not None and match and len(rebuilt) >= 2:
            d_full, _, _ = score_episode(full, None)
            d_cut, _, _ = score_episode(rebuilt, None)
            rf = [r for r in d_full if r["method"]=="V6_CAP_POTENTIAL" and r["step"] < kept-1]
            rc = [r for r in d_cut if r["method"]=="V6_CAP_POTENTIAL"]
            reward_match = [x["reward"] for x in rf] == [x["reward"] for x in rc]
        else:
            reward_match = match
        rows.append(dict(episode_id=full[0]["episode_id"], kind="CSV_REBUILD_PREFIX", cut=kept, n_full=len(full),
                         passed=bool(match and reward_match), state_match=match, reward_match=reward_match,
                         raw=str(dest)))
    # future geometry mutation on original CSV copy
    mid = max(2, len(full)//2)
    work = tmp/"future_mut"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(dest, work)
    with (dest/"timeseries.csv").open(encoding="utf-8", newline="") as f:
        rdr = list(csv.DictReader(f))
        fields = list(rdr[0].keys())
    for i,row in enumerate(rdr):
        if i >= mid:
            for key in row:
                if key.endswith("_x") or key == "x":
                    try:
                        row[key] = str(float(row[key]) + 0.007)
                    except (TypeError, ValueError):
                        pass
    with (work/"timeseries.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rdr)
    mut, _ = normalize(work, mods)
    a = [_state_core(s) for s in full[:mid]]
    b = [_state_core(s) for s in mut[:mid]]
    rows.append(dict(episode_id=full[0]["episode_id"], kind="CSV_FUTURE_GEOMETRY_MUTATION", cut=mid,
                     passed=a==b, raw=str(dest)))
    # case/id rename should not change numeric rewards
    work = tmp/"id_mut"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(dest, work)
    ident = json.loads((work/"identity.json").read_text(encoding="utf-8"))
    ident["episode_id"] = ident.get("episode_id","x") + "_RENAMED"
    ident["case_id"] = "mutated_case_name_must_not_score"
    (work/"identity.json").write_text(json.dumps(ident), encoding="utf-8")
    renamed, _ = normalize(work, mods)
    d0,_,_ = score_episode(full, None)
    d1,_,_ = score_episode(renamed, None)
    r0 = [r["reward"] for r in d0 if r["method"]=="V6_CAP_POTENTIAL"]
    r1 = [r["reward"] for r in d1 if r["method"]=="V6_CAP_POTENTIAL"]
    rows.append(dict(episode_id=full[0]["episode_id"], kind="IDENTITY_CASE_RENAME", cut=len(full),
                     passed=r0==r1, raw=str(dest)))
    return rows


def run_audit(repo: Path, pkg: Path, raw_root: Path, out: Path) -> dict:
    out = require_new(out)
    normalize, load_upstream, read_raw, score_episode, load_engine = _load_tools(pkg)
    mods = load_upstream(Path(repo))
    Engine = load_engine(Path(repo))
    dests = sorted({p.parent for p in Path(raw_root).rglob("timeseries.csv")})
    all_rows = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="p1v6_csv_prefix_") as td:
        tmp = Path(td)
        for dest in dests:
            try:
                all_rows.extend(audit_one(dest, mods, normalize, score_episode, Engine, tmp/dest.name))
            except Exception as exc:
                failures.append({"path": str(dest), "error": type(exc).__name__, "reason": str(exc)})
    write_csv(out/"csv_prefix_audit.csv", all_rows)
    summary = {
        "episodes": len(dests),
        "checks": len(all_rows),
        "failures": sum(not r["passed"] for r in all_rows),
        "rebuild_failures": sum(not r["passed"] for r in all_rows if r["kind"]=="CSV_REBUILD_PREFIX"),
        "mutation_failures": sum(not r["passed"] for r in all_rows if r["kind"]=="CSV_FUTURE_GEOMETRY_MUTATION"),
        "rename_failures": sum(not r["passed"] for r in all_rows if r["kind"]=="IDENTITY_CASE_RENAME"),
        "exceptions": failures,
        "passed": (not failures) and all(r["passed"] for r in all_rows) and len(dests)==112,
    }
    write_json(out/"csv_prefix_audit.json", summary)
    return summary

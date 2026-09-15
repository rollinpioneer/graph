"""Score frozen methods on already-generated holdout states. Do not regenerate."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path

V6_PKG = Path("/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0")
FOCUS = [
    "SPARSE_TERMINAL", "LINEAR_A_FIRST_R1", "LINEAR_B_FIRST_R1", "UNORDERED_VALID_COUNT",
    "VALID_COUNT_PLUS_MATCHED_EVENTS_V1", "GRAPH_COST_ONLY", "FULL_FROZEN", "V6_CAP_POTENTIAL",
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_holdout(repo: Path, holdout: Path, out: Path) -> dict:
    repo, holdout, out = Path(repo), Path(holdout), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(V6_PKG / "tools"))
    sys.path.insert(0, str(repo))
    from reward_v6 import score_episode
    from engine_parity import load_engine
    from upgrade_v2.p1_mainline_grasp_v6gm1.replay_bridge import extra_on_states

    engine = load_engine(repo)
    manifest = list(csv.DictReader((holdout / "holdout_manifest.csv").open(encoding="utf-8", newline="")))
    ledger = []
    export_states = []
    per_episode = []
    identity_rows = []
    for row in manifest:
        path = holdout / row["states_path"]
        digest = _sha(path)
        if digest != row["states_sha256"]:
            raise ValueError(f"hash mismatch {row['episode_id']}: frozen generator output changed")
        states = _load_jsonl(path)
        details, values, parity = score_episode(states, engine)
        extra = extra_on_states(states)
        extra_by = {r["from_state_index"]: r for r in extra}
        signed = defaultdict(float)
        pos = defaultdict(float)
        n = defaultdict(int)
        for d in details:
            rec = {
                "episode_id": d["episode_id"], "method": d["method"],
                "from_state_index": d["step"], "to_state_index": d["step"] + 1,
                "available_at_ns": d["t1_ns"], "reward": d["reward"],
                "score_status": d["score_status"],
                "psi_before": d["psi_before"], "psi_after": d["psi_after"],
            }
            ledger.append(rec)
            if d["reward"] is not None:
                signed[d["method"]] += float(d["reward"])
                pos[d["method"]] += max(0.0, float(d["reward"]))
                n[d["method"]] += 1
        for e in extra:
            for m in ("SPARSE_TERMINAL", "UNORDERED_VALID_COUNT", "VALID_COUNT_PLUS_MATCHED_EVENTS_V1"):
                r = e[m]
                ledger.append({
                    "episode_id": e["episode_id"], "method": m,
                    "from_state_index": e["from_state_index"], "to_state_index": e["to_state_index"],
                    "available_at_ns": e["available_at_ns"], "reward": r, "score_status": "SCORED",
                    "psi_before": values[e["from_state_index"]]["psi"],
                    "psi_after": values[e["to_state_index"]]["psi"],
                })
                signed[m] += float(r)
                pos[m] += max(0.0, float(r))
                n[m] += 1
        ep = {
            "episode_id": row["episode_id"], "family_id": row["family_id"], "case_id": row["case_id"],
            "task": row["task"], "n_states": len(states),
            "psi_start": values[0]["psi"], "psi_end": values[-1]["psi"],
            "endpoint_delta": values[-1]["psi"] - values[0]["psi"],
        }
        for m in FOCUS:
            ep[f"{m}_signed"] = signed.get(m)
            ep[f"{m}_pos"] = pos.get(m)
            ep[f"{m}_n"] = n.get(m)
        v6 = signed.get("V6_CAP_POTENTIAL", 0.0)
        ep["v6_minus_endpoint"] = v6 - ep["endpoint_delta"]
        ep["max_step_identity_error"] = max(
            abs((d["reward"] if d["reward"] is not None else 0.0) - (d["psi_after"] - d["psi_before"]))
            for d in details if d["method"] == "V6_CAP_POTENTIAL"
        )
        per_episode.append(ep)
        for i, s in enumerate(states):
            export_states.append({
                "episode_id": s["episode_id"], "state_index": i, "case_id": row["case_id"],
                "family_id": row["family_id"], "task": s["task"],
                "available_at_ns": s["available_at_ns"], "success": s["success"],
                "terminal_failure": s["terminal_failure"],
                "psi": values[i]["psi"], "node_key": values[i]["node_key"],
                "objects": {k: {"valid": o["valid"], "phase": o["phase"], "held": o["held"]}
                            for k, o in s["objects"].items()},
                "events": s.get("events") or [],
            })
        identity_rows.append({
            "episode_id": row["episode_id"], "case_id": row["case_id"],
            "n_states": len(states), "v6_signed": v6,
            "endpoint_delta": ep["endpoint_delta"],
            "abs_error": abs(ep["v6_minus_endpoint"]),
            "max_step_identity_error": ep["max_step_identity_error"],
            "passed": abs(ep["v6_minus_endpoint"]) <= 1e-12 and ep["max_step_identity_error"] <= 1e-12,
        })

    def dump_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    dump_csv(out / "per_transition_rewards.csv", ledger)
    dump_csv(out / "per_episode_returns.csv", per_episode)
    dump_csv(out / "potential_identity.csv", identity_rows)
    with (out / "states.jsonl").open("w", encoding="utf-8") as f:
        for s in export_states:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (out / "scored_states.sha256").write_text(_sha(out / "states.jsonl") + "  states.jsonl\n", encoding="utf-8")
    hashes = {
        "reward_v6.py": _sha(V6_PKG / "tools/reward_v6.py"),
        "reward_contract.json": _sha(V6_PKG / "contracts/reward_contract.json"),
        "extra_methods.py": _sha(repo / "upgrade_v2/p1_mainline_grasp_v6gm1/extra_methods.py"),
        "holdout_manifest.csv": _sha(holdout / "holdout_manifest.csv"),
        "states.sha256": _sha(holdout / "states.sha256"),
    }
    (out / "method_source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return {"episodes": len(manifest), "ledger": len(ledger), "identity_pass": sum(r["passed"] for r in identity_rows)}
#!/usr/bin/env python
"""CP-DISR-C1-MECH-CONFIRM-V1 Stage 1: offline recovery of old evidence (no environment, no model forward, no training).

    python scripts/c1_existing_evidence.py --root . --out runs/final_master/c1_route_b/mech_confirm_v1/prep

Reads the 12 CP-DISR-TB-STRUCT-GEN-V1 run directories (transition_log.jsonl, final episode pickle, eval_struct_test_final.json) and writes
existing_evidence_inventory.json, training_binding_exposure.json and old_test30_mechanism.json.
"""
import argparse
import collections
import glob
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CARD = "CP-DISR-C1-MECH-CONFIRM-V1"
BASE_COMMIT = "5a2b3d18d21cbab19b9d09c3c430b3203e873730"
# binding -> (place candidate, goal atom that the verified effect should make TRUE)
BINDINGS = {
    "target->container": ("a:PLACE:target:container:v1", "p:Inside:target:container"),
    "second_object->buffer": ("a:PLACE_BUFFER:second_object:buffer:v1", "p:AtBuffer:second_object:buffer"),
    "second_object->container": ("a:PLACE:second_object:container:v1", "p:Inside:second_object:container"),
    "target->buffer": ("a:PLACE_BUFFER:target:buffer:v1", "p:AtBuffer:target:buffer"),
}
TRAIN_BINDINGS = ("target->container", "second_object->buffer")
GOAL_BINDINGS = {"IN_T": ("target->container",), "BUF_S": ("second_object->buffer",), "IN_T+BUF_S": ("target->container", "second_object->buffer")}


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def run_key(path):
    return path.split("R-TB-SG-")[1].split("-2026")[0]


def exposure_for_run(run_dir, cell_of_case):
    """Per binding: goal episodes, decisions where the placement candidate was legal / selected / exited normally, and verified effects."""
    run_dir = Path(run_dir)
    per = {b: collections.Counter() for b in BINDINGS}
    episodes = {}
    decisions = collections.defaultdict(list)
    for line in open(run_dir / "transition_log.jsonl", encoding="utf-8"):
        r = json.loads(line)
        key = (r["env_id"], r["episode_id"])
        episodes[key] = r["case_id"]
        decisions[key].append(r)
    for key, case in episodes.items():
        cell = cell_of_case[case]
        for b in GOAL_BINDINGS.get(cell, ()):
            per[b]["goal_episodes"] += 1
    # public facts after each decision: the stored prefix of an episode holds the snapshot before every decision
    pkl = sorted((run_dir / "checkpoints").glob("final_n_*.episode.pkl"))
    prefixes = None
    if len(pkl) == 1:
        prefixes = pickle.load(open(pkl[0], "rb")).get("prefixes")
    verified_available = prefixes is not None
    for key, rows in decisions.items():
        rows.sort(key=lambda r: r["decision_id"])
        snaps = prefixes.get(key) if prefixes else None
        for i, r in enumerate(rows):
            cid = r["selected_candidate_id"]
            for b, (cand, atom) in BINDINGS.items():
                if cand in r["candidate_ids"] and r["mask"][r["candidate_ids"].index(cand)]:
                    per[b]["legal_decisions"] += 1
                if cid == cand:
                    per[b]["executed"] += 1
                    ok = r["controller_exit"] == "NORMAL_TERMINATION"
                    per[b]["normal_exit"] += int(ok)
                    if ok:
                        if snaps is not None and i + 1 < len(snaps):
                            after = snaps[i + 1].facts.values
                            per[b]["with_successor_snapshot"] += 1
                            per[b]["verified_effect_true"] += int(str(after[atom]).endswith("TRUE") and not str(after[atom]).endswith("FALSE"))
                        elif i + 1 == len(rows):
                            per[b]["last_decision_of_episode"] += 1
                            per[b]["last_decision_episode_success"] += int(r["verified_outcome"] == "TASK_SUCCESS")
    return {b: dict(c) for b, c in per.items()}, {"episodes": len(episodes), "decisions": sum(len(v) for v in decisions.values()), "verified_effect_recoverable": verified_available}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((root / "configs/splits/struct_gen_v1_train_dev.json").read_text())
    cell_of_case = {r["case_id"]: r["cell"] for r in split["train"] + split["dev"]}
    runs = sorted(glob.glob(str(root / "runs/final_master/2.1.1/T_B/*/seed_*/R-TB-SG-*")))
    inventory, exposure, test30 = {}, {}, {}
    for d in runs:
        run = run_key(d)
        dd = Path(d)
        man = json.loads((dd / "checkpoint_manifest.json").read_text())
        fin = [k for k in man["checkpoint_files"] if k.startswith("final_n_") and k.endswith(".pt")][0]
        now = sha256_file(dd / "checkpoints" / fin)
        files = {p.name: p.stat().st_size for p in dd.iterdir() if p.is_file()}
        tl = sum(1 for _ in open(dd / "transition_log.jsonl", encoding="utf-8"))
        t30 = json.loads((dd / "eval_struct_test_final.json").read_text())
        row_keys = sorted({k for r in t30["rows"] for k in r})
        inventory[run] = {"run_dir": str(dd.relative_to(root)), "final_checkpoint": fin, "sha256_recorded": man["checkpoint_files"][fin], "sha256_now": now, "sha256_match": now == man["checkpoint_files"][fin],
                          "files": files, "transition_log_lines": tl, "has_decision_level_training_trace": True, "test30_eval_row_keys": row_keys,
                          "test30_has_action_trace": any(k in row_keys for k in ("selected_candidate_id", "actions", "trace", "decisions"))}
        per, meta = exposure_for_run(dd, cell_of_case)
        exposure[run] = {"bindings": per, **meta}
        test30[run] = {"success_n": t30["success_n"], "n": t30["n"], "rows": [{k: r[k] for k in ("case_id", "success", "steps", "reason")} for r in t30["rows"]]}
    method_of = lambda r: r.rsplit("-", 1)[0]  # noqa: E731
    totals = {}
    for run, doc in exposure.items():
        t = totals.setdefault(method_of(run), {b: collections.Counter() for b in BINDINGS})
        for b, c in doc["bindings"].items():
            t[b].update(c)
    totals = {m: {b: dict(c) for b, c in v.items()} for m, v in totals.items()}
    unseen = [b for b in BINDINGS if b not in TRAIN_BINDINGS]
    any_unseen_executed = {run: {b: exposure[run]["bindings"][b].get("executed", 0) for b in unseen} for run in exposure}
    recoverable = all(doc["verified_effect_recoverable"] for doc in exposure.values())
    inv_doc = {"card": CARD, "base_commit": BASE_COMMIT, "runs": inventory, "all_final_checkpoints_sha256_match": all(v["sha256_match"] for v in inventory.values()),
               "raw_evidence": {"training_decision_trace": "transition_log.jsonl per run (case_id, candidate_ids, mask, selected candidate, controller_exit, verified_outcome)",
                                "training_public_facts": "checkpoints/final_n_*.episode.pkl prefixes (snapshot before every decision)",
                                "test30": "eval_struct_test_final.json rows carry only case_id/G/steps/reason/success/success_seconds/source_n/prior_mode: no selected actions"}}
    status = "RECOVERED" if recoverable else "RECOVERED_PARTIAL"
    exp_doc = {"card": CARD, "TRAINING_BINDING_EXECUTION_EXPOSURE": status, "definitions": {
        "goal_episodes": "training episodes whose goal contains the binding (case cell from the train split)", "legal_decisions": "decisions where the placement candidate was legal under the hard mask",
        "executed": "decisions where the policy selected the placement candidate", "normal_exit": "executed with controller exit NORMAL_TERMINATION",
        "verified_effect_true": "normal exits whose next public snapshot has the binding's atom TRUE (decisions with a stored successor snapshot only)",
        "last_decision_*": "executed as the last decision of an episode (no stored successor snapshot); episode_success = evaluator TASK_SUCCESS"},
        "train_bindings": list(TRAIN_BINDINGS), "unseen_test_bindings": unseen, "per_run": exposure, "per_method_totals": totals, "unseen_binding_executions_per_run": any_unseen_executed,
        "note": "Execution of an unseen binding during training is not goal exposure: those goals were never rewarded as goals; the placement was reached as an action of a different goal."}
    t30_doc = {"card": CARD, "OLD_TEST30_ACTION_MECHANISM": "NOT_RECOVERABLE" if not any(v["test30_has_action_trace"] for v in inventory.values()) else "RECOVERED",
               "reason": "old test30 evaluation stored episode outcomes only; the old test30 is not re-run (runbook 8.3)", "available_episode_summary": test30}
    for name, doc in (("existing_evidence_inventory.json", inv_doc), ("training_binding_exposure.json", exp_doc), ("old_test30_mechanism.json", t30_doc)):
        (out / name).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoints_match": inv_doc["all_final_checkpoints_sha256_match"], "exposure": status, "test30": t30_doc["OLD_TEST30_ACTION_MECHANISM"],
                      "unseen_executions": any_unseen_executed}, indent=1))


if __name__ == "__main__":
    main()

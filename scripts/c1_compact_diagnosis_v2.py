#!/usr/bin/env python
"""CP-DISR-C1-COMPACT-DIAGNOSIS-V2 driver. Stages are separate sub-commands so independent GPU / CPU tasks can run in parallel (the DAG of plan section 9 is realised by launching them as background processes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cp_disr.pddl import depots as DP                                               # noqa: E402
from cp_disr.pddl.compact import train as CT                                        # noqa: E402
from cp_disr.pddl.compact.models import C0_HIDDEN, D0_NEW_PARAMS, count_new, count_trainable, make_model_v2      # noqa: E402

PROC_START = time.time()
CARD = "CP-DISR-C1-COMPACT-DIAGNOSIS-V2"
CFG_REL = "configs/c1_compact_diagnosis_v2.yaml"


def load_cfg():
    import yaml
    return yaml.safe_load((ROOT / CFG_REL).read_text(encoding="utf-8"))


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def wj(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha_file(p):
    return CT.sha_file(p)


def rr_path():
    return Path((Path.home() / "work" / "rr_compact.txt").read_text().strip())


def ledger(rr, stage, **kv):
    p = Path(rr) / "receipts" / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kv), default=str) + "\n")


# ------------------------------------------------------------------------------------------------ init / prepare
def cmd_init(auth_text):
    cfg = load_cfg()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rr = ROOT / "runs" / "final_master" / "c1_route_b" / "compact_diagnosis_v2" / ts
    for d in ("plan", "assets", "registration", "results", "diagnostics", "training", "evaluation", "final", "receipts", "driver_logs", "checks"):
        (rr / d).mkdir(parents=True, exist_ok=True)
    (rr / "plan" / "runbook.md").write_text((ROOT / cfg["plan_doc"]).read_text(encoding="utf-8"), encoding="utf-8")
    (rr / "plan" / "config.yaml").write_text((ROOT / CFG_REL).read_text(encoding="utf-8"), encoding="utf-8")
    wj(rr / "plan" / "authorisation.json", {"card": CARD, "execution_authorized": True, "authorization_text": auth_text.strip(), "plan_doc_sha256": sha_file(ROOT / cfg["plan_doc"]),
                                            "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scope": "core D0 / C0 (neural) + W1 (one WL fit); T1 only if the pre-registered A/B/C thresholds are met; nothing else"})
    (Path.home() / "work" / "rr_compact.txt").write_text(str(rr) + "\n")
    print(rr)


def old_root(cfg):
    return ROOT / cfg["old_run_root"]


def cmd_prepare(rr):
    rr = Path(rr)
    cfg = load_cfg()
    root = old_root(cfg)
    t0 = time.time()
    man, exact, _ = CT.load_old(root)
    cases, trajs, labels = CT.train_inputs(root)
    lock = rj(root / "runs" / "lock.json")
    ident = {"old_run_root": cfg["old_run_root"], "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
             "exact_json_sha256": sha_file(root / "data" / "exact.json"), "manifest_sha256": sha_file(root / "data" / "manifest.json"), "lock_json_sha256": sha_file(root / "runs" / "lock.json"),
             "train_digest": CT.traj_digest(trajs), "train_cases": len(cases), "train_trajectories": len(trajs), "train_decisions": sum(len(t["actions"]) for t in trajs), "train_unique_parent_states": len({(t["case_id"], tuple(s)) for t in trajs for s in t["states"][:-1]}),
             "checkpoints": {}, "wl_models": {}}
    for mode in ("mg", "dense", "rel"):
        sel = rj(root / "runs" / mode / "selection.json")
        ident["checkpoints"][mode] = {"selected_update": sel["selected_update"], "path": sel["checkpoint"]["path"], "sha256": sel["checkpoint"]["sha256"], "file_sha256": sha_file(sel["checkpoint"]["path"])}
    assert ident["checkpoints"]["dense"]["file_sha256"] == cfg["frozen_dense"]["sha256"]
    for trk in ("typed", "ipc"):
        d = root / "goose" / ("train_" + trk)
        ident["wl_models"][trk] = {"model": str(d / "wl_goose.model"), "model_sha256": sha_file(d / "wl_goose.model"), "params_sha256": sha_file(d / "wl_goose.model.params"), "opts_sha256": sha_file(d / "wl_goose.model.opts")}
    wj(rr / "assets" / "asset_identity.json", ident)
    # panels (frozen before any new evaluation)
    joint = sorted(man["joint"], key=lambda c: c["sha256"])
    cells = sorted({c["cell"] for c in joint})
    joint32 = [c["case_id"] for cell in cells for c in [x for x in joint if x["cell"] == cell][:cfg["panels"]["joint32"]["per_cell"]]]
    panels = {"joint32": joint32, "joint32_cells": {cell: [c["case_id"] for c in [x for x in joint if x["cell"] == cell][:cfg["panels"]["joint32"]["per_cell"]]] for cell in cells}, "ipc22": [c["case_id"] for c in man["ipc"]], "dev24": [c["case_id"] for c in man["dev"]]}
    wj(rr / "registration" / "panels.json", panels)
    # deterministic schedule account shared by D0 / C0 / T1
    led = CT.schedule_ledger(trajs)
    CT.write_csv(rr / "training" / "schedule_ledger.csv", led)
    sched_sha = hashlib.sha256(json.dumps(led, sort_keys=True).encode()).hexdigest()
    # parameter counts (CPU, no training)
    import torch
    dev = torch.device("cpu")
    md, mc = make_model_v2("dense", dev), make_model_v2("c0", dev)
    counts = {"D0_new": count_new(md), "C0_new": count_new(mc), "D0_trainable": count_trainable(md), "C0_trainable": count_trainable(mc), "C0_hidden": C0_HIDDEN, "relative_difference_new": (count_new(mc) - count_new(md)) / count_new(md),
              "schedule_ledger_sha256": sched_sha, "updates": len(led), "decisions_total": sum(r["decisions"] for r in led), "batches_per_epoch": max(r["epoch"] for r in led) and len(CT.update_batches(trajs, 0))}
    wj(rr / "registration" / "arm_definitions.json", {"counts": counts, "arms": {"D0": "dense (original DENSE-G recipe, common init seed 20261008)", "C0": "c0 (pairwise MLP message mean, same pair features, zero-initialised output)",
                                                      "W1": "fixed L2 WL features (ILG, a-m pruning, set hash), rank-SVM (LinearSVC hinge, C=1), typed track only, pairs / weights from the D0 supervision", "T1": "conditional, S or W, section 6"}})
    ledger(rr, "prepare", seconds=round(time.time() - t0, 1), schedule_sha256=sched_sha)
    print(json.dumps(counts))


# ------------------------------------------------------------------------------------------------ fixtures (training arms)
def cmd_fixtures_train(rr, device_name):
    import torch
    rr = Path(rr)
    cfg = load_cfg()
    root = old_root(cfg)
    dev = torch.device(device_name)
    out = {}
    t0 = time.time()
    # 1: old assets read only, whitelist
    ident = rj(rr / "assets" / "asset_identity.json")
    out["F1_identity"] = {"dense_sha_before": ident["checkpoints"]["dense"]["file_sha256"], "dense_sha_now": sha_file(ident["checkpoints"]["dense"]["path"]), "exact_json_sha_now": sha_file(root / "data" / "exact.json"),
                          "whitelist": {"neural_arms": ["D0", "C0", "T1"], "wl_fits": ["W1"]}, "budget_updates": CT.BUDGET["updates"], "ok": ident["checkpoints"]["dense"]["file_sha256"] == sha_file(ident["checkpoints"]["dense"]["path"]) == cfg["frozen_dense"]["sha256"]}
    # 2: data isolation
    man, exact, _ = CT.load_old(root)
    cases, trajs, labels = CT.train_inputs(root)
    train_ids = {c.case_id for c in cases}
    others = {c["case_id"] for fam in ("dev", "struct", "joint", "ipc") for c in man[fam]}
    label_cases = {k.split("|")[0] for k in labels}
    out["F2_isolation"] = {"train_cases": len(train_ids), "only_train_ids": all(i.startswith("train_") for i in train_ids), "overlap_with_other_families": sorted(train_ids & others), "label_cases_subset_of_train": label_cases <= train_ids,
                           "traj_cases_subset_of_train": {t["case_id"] for t in trajs} <= train_ids, "dev_ids_disjoint": not (train_ids & {c["case_id"] for c in man["dev"]}), "train_digest": CT.traj_digest(trajs)}
    out["F2_isolation"]["ok"] = out["F2_isolation"]["only_train_ids"] and not out["F2_isolation"]["overlap_with_other_families"] and out["F2_isolation"]["label_cases_subset_of_train"] and out["F2_isolation"]["traj_cases_subset_of_train"]
    # 4: C0 definition
    md, mc = make_model_v2("dense", dev), make_model_v2("c0", dev)
    nd, nc = count_new(md), count_new(mc)
    base_equal = all(torch.equal(a, b) for (ka, a), (kb, b) in zip(sorted(md.mg.state_dict().items()), sorted(mc.mg.state_dict().items())) if ka == kb) and sorted(md.mg.state_dict()) == sorted(mc.mg.state_dict())
    from cp_disr.pddl.task import snapshot_at
    c0case = next(c for c in cases if c.case_id == "train_n3_000")
    snaps = [snapshot_at(c0case, s, t) for t, s in enumerate(next(t for t in trajs if t["case_id"] == "train_n3_000")["states"][:-1])][:6]
    md.eval(), mc.eval()
    with torch.no_grad():
        ld = [md.eval_logits(s) for s in snaps]
        lc = [mc.eval_logits(s) for s in snaps]
        gap0 = max(float((a - b).abs().masked_fill(~torch.isfinite(a), 0).max()) for a, b in zip(ld, lc))
        st, gprop = md.mg.goal_free_static(snaps[0].template)
        codes = md.mg.base._codes(st, snaps[0].facts.values).unsqueeze(0)
        gsd, gsc = md.attn._goal_static(st, gprop, snaps[0].template), mc.attn._goal_static(st, gprop, snaps[0].template)
        fd, ad = md.attn.pair_features(gsd, codes, gprop)
        fc, ac = mc.attn.pair_features(gsc, codes, gprop)
        same_inputs = bool(torch.equal(fd, fc) and torch.equal(ad, ac))
        idle = [n for n, p in mc.attn.named_parameters() if n.split(".")[0] in ("q", "k", "v", "bias", "o")]
    mc.train()
    from cp_disr.pddl import train as PT
    tr = PT.PddlTrainer(mc, cases, labels, dev, lr=CT.BUDGET["lr"], chunk=CT.BUDGET["chunk_decisions"])
    w0 = {n: p.detach().clone() for n, p in mc.attn.named_parameters()}
    batches = tr.update_batches(trajs, 0)
    tr.step(batches[0])
    w1 = {n: p.detach().clone() for n, p in mc.attn.named_parameters()}
    tr.step(batches[1])
    w2 = {n: p.detach().clone() for n, p in mc.attn.named_parameters()}
    out["F4_c0"] = {"new_params_D0": nd, "new_params_C0": nc, "relative_difference": (nc - nd) / nd, "within_2_percent": abs(nc - nd) / nd <= 0.02, "shared_base_parameters_equal": base_equal, "max_abs_logit_gap_at_init_vs_D0": gap0,
                    "same_pair_inputs_and_allowed": same_inputs, "idle_attention_parameters_in_C0": idle, "out_changes_after_step1": any(not torch.equal(w0[n], w1[n]) for n in w0 if n.startswith("out")),
                    "mlp_unchanged_after_step1": all(torch.equal(w0[n], w1[n]) for n in w0 if n.startswith("mlp")), "mlp_changes_after_step2": any(not torch.equal(w1[n], w2[n]) for n in w1 if n.startswith("mlp")),
                    "fixture_optimizer_steps_C0": 2}
    f4 = out["F4_c0"]
    f4["ok"] = bool(f4["within_2_percent"] and f4["shared_base_parameters_equal"] and f4["max_abs_logit_gap_at_init_vs_D0"] < 1e-5 and f4["same_pair_inputs_and_allowed"] and not f4["idle_attention_parameters_in_C0"] and f4["out_changes_after_step1"] and
                   f4["mlp_unchanged_after_step1"] and f4["mlp_changes_after_step2"])
    # 8 (training part): schedule account equals what the real trainer shows
    led = CT.schedule_ledger(trajs, 3)
    tr2 = PT.PddlTrainer(md, cases, labels, dev, lr=CT.BUDGET["lr"], chunk=CT.BUDGET["chunk_decisions"])
    md.train()
    nb = len(tr2.update_batches(trajs, 0))
    seen = []
    for u in range(3):
        ep, off = divmod(u, nb)
        seen.append(tr2.step(tr2.update_batches(trajs, ep)[off])["decisions"])
    out["F8_ledger"] = {"ledger_decisions_first3": [r["decisions"] for r in led], "trainer_decisions_first3": seen, "equal": [r["decisions"] for r in led] == seen, "fixture_optimizer_steps_D0": 3, "full_ledger_sha256": rj(rr / "registration" / "arm_definitions.json")["counts"]["schedule_ledger_sha256"]}
    out["all_passed"] = bool(out["F1_identity"]["ok"] and out["F2_isolation"]["ok"] and f4["ok"] and out["F8_ledger"]["equal"])
    out["fixture_seconds"] = round(time.time() - t0, 1)
    out["fixture_note"] = "fixture optimizer steps (5) are NOT accepted updates; they act on throw-away models built in the fixture process"
    p = rr / "checks" / "fixtures_training_arms.json"
    wj(p, out)
    ledger(rr, "fixtures_train", seconds=out["fixture_seconds"], passed=out["all_passed"], fixture_optimizer_steps=5, gpu_seconds=out["fixture_seconds"] if dev.type == "cuda" else 0)
    print("fixtures_train all_passed=%s" % out["all_passed"])
    return 0 if out["all_passed"] else 3


def cmd_freeze1(rr):
    rr = Path(rr)
    cfg = load_cfg()
    fx = rj(rr / "checks" / "fixtures_training_arms.json")
    if not fx["all_passed"]:
        raise SystemExit("fixtures failed")
    wj(rr / "registration" / "registration_1.json", {"card": CARD, "scope": "D0 / C0 training definition, schedule account, panels, thresholds; W1, state library, variants and evaluation code are registered later (registration_2) before their first output",
                                                     "base_commit": cfg["base_commit"], "plan_sha256": sha_file(ROOT / cfg["plan_doc"]), "config_sha256": sha_file(ROOT / CFG_REL), "fixtures": "checks/fixtures_training_arms.json",
                                                     "budget": cfg["budget"], "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("registration_1 written")


# ------------------------------------------------------------------------------------------------ training / selection
def cmd_train(rr, arm, device_name):
    import torch
    rr = Path(rr)
    cfg = load_cfg()
    mode = {"D0": "dense", "C0": "c0", "T1": None}[arm]
    if mode is None:
        raise SystemExit("T1 has its own stage")
    out = rr / "training" / arm
    out.mkdir(parents=True, exist_ok=True)
    acct_path = out / "training_accounting.json"
    if acct_path.is_file():
        print("already complete")
        return 0
    cases, trajs, labels = CT.train_inputs(old_root(cfg))
    dev = torch.device(device_name)
    t0 = time.time()
    acct, rows, m = CT.train_loop_v2(out, mode, dev, cases, trajs, labels, log=print)
    led = CT.schedule_ledger(trajs)
    acct["schedule_decisions_match_ledger"] = [r["decisions"] for r in rows] == [r["decisions"] for r in led]
    wj(acct_path, acct)
    CT.write_csv(out / "updates.csv", rows)
    ledger(rr, "train_" + arm, seconds=round(time.time() - t0, 1), gpu_device_hours=round((time.time() - PROC_START) / 3600.0, 3), accepted_updates=acct["optimizer_steps"], attempts=acct["attempts"], device=device_name)
    print("train %s done: %s" % (arm, json.dumps({k: acct[k] for k in ("optimizer_steps", "wall_seconds", "attempts", "decision_samples_shown", "nan_events")})))
    return 0


def cmd_devsel(rr, arm, device_name):
    import torch
    rr = Path(rr)
    cfg = load_cfg()
    mode = {"D0": "dense", "C0": "c0"}[arm]
    dev = torch.device(device_name)
    t0 = time.time()
    sel = CT.devsel(rr / "training" / arm, mode, dev, CT.dev_cases(old_root(cfg)))
    ledger(rr, "devsel_" + arm, seconds=round(time.time() - t0, 1), gpu_device_hours=round((time.time() - PROC_START) / 3600.0, 3), dev_policy_evaluations=5 * 24, selected=sel["selected_update"])
    print("devsel %s -> update %s" % (arm, sel["selected_update"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root")
    ap.add_argument("--authorization-text-file")
    ap.add_argument("--arm")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.cmd == "init":
        return cmd_init(Path(a.authorization_text_file).read_text(encoding="utf-8"))
    rr = a.run_root or rr_path()
    fn = {"prepare": lambda: cmd_prepare(rr), "fixtures-train": lambda: cmd_fixtures_train(rr, a.device), "freeze1": lambda: cmd_freeze1(rr), "train": lambda: cmd_train(rr, a.arm, a.device),
          "devsel": lambda: cmd_devsel(rr, a.arm, a.device)}
    from cp_disr.pddl.compact import cli_ext
    fn.update(cli_ext.commands(rr, a))
    r = fn[a.cmd]()
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())

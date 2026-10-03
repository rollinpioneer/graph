#!/usr/bin/env python
"""CP-DISR-TB-REP-CONTROLS-01: legacy B1-K / B1-K+E / B2 forward equivalence (old tree vs new tree).

`dump`    runs the (self-contained) forward on synthetic T_B states with whatever `cp_disr` PYTHONPATH points to and
          saves every output tensor, the fresh-init state_dict digest and the outputs after loading the real
          current-profile final checkpoints (strict load).
`compare` compares two dumps element by element (torch.equal) and writes a receipt.
No environment, no GPU, no optimizer step, no provider, no test split.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

LEGACY = ("B1-K", "B1-K+E", "B2")
CHECKPOINTS = {  # current-profile finals recorded in 03_frozen_model_list.md
    "B1-K": "runs/final_master/2.1.1/T_B/B1-K/seed_1/R-TB-K-1-20261001T115252Z-cac2fa5b/checkpoints/final_n_014531_u14.pt",
    "B1-K+E": "runs/final_master/2.1.1/T_B/B1-K+E/seed_0/R-TB-E-0-20261001T115252Z-cac2fa5b/checkpoints/final_n_014512_u14.pt",
    "B2": "runs/final_master/2.1.1/T_B/B2/seed_1/R-TB-DK-1-20261001T115252Z-cac2fa5b/checkpoints/final_n_014170_u13.pt",
}
OBS_DIM, CAND_DIM = 48, 8


def build(root):
    import yaml
    from cp_disr import contracts as C, graph as G
    from cp_disr.platforms.libero import runtime_factory as rf
    runtime = yaml.safe_load((Path(root) / "experiments/manifests/runtime_manifest_v211.yaml").read_text())["runtime"]
    objects = rf.TASK_OBJECTS["T_B"]
    contracts = rf._ground_contracts_for(Path(root) / "configs/runtime/stage_2a_contract_registry.yaml", runtime["skill_timeouts"]["T_B"], objects)
    goals = tuple(G.Goal(g, 1) for g in rf.TASK_GOALS["T_B"])
    return G.build_template(contracts, goals, rf.PREDICATES, objects), rf


def snapshot(template, seed):
    from cp_disr import contracts as C
    from cp_disr.common import digest
    from cp_disr.facts import FactRecord, FactStore, Truth
    from cp_disr.rl import Snapshot
    rng = random.Random(seed)
    ids = [n.id for n in template.nodes if n.kind == "PROPOSITION"]
    values = {fid: rng.choices((Truth.TRUE, Truth.FALSE, Truth.UNKNOWN), weights=(5, 3.5, 1.5))[0] for fid in ids}
    cids = tuple(c.id for c in template.contracts)
    mask = []
    for c in template.contracts:
        ok = C.precondition_value(c, values) == Truth.TRUE
        if ok:
            try:
                C.nominal_overlay(c, values, template.derived_rules, template.exclusive_groups)
            except Exception:
                ok = False
        mask.append(ok)
    r2 = random.Random(10_000 + seed)
    base = tuple(r2.uniform(-1, 1) for _ in range(OBS_DIM))
    feats = tuple(tuple(int.from_bytes(hashlib.sha256(("%s|%d" % (cid, j)).encode()).digest()[:4], "big") / 2 ** 32 for j in range(CAND_DIM)) for cid in cids)
    facts = FactStore(tuple(FactRecord(k, v, 0.0, 0.0, ("s",), Truth.TRUE, 0.0, "S", 0.9) for k, v in values.items()))
    return Snapshot("e", "p", 0, template, facts, cids, tuple(mask), (), digest(()), base, feats, "s", 0.0, synthetic_unit_fixture=True), sum(mask)


def policy_for(template, rf, method, seed):
    from cp_disr.neural import Policy
    types = set()
    for m in rf.TASK_OBJECTS.values():
        types.update(m.values())
    torch.manual_seed(seed)
    p = Policy(sorted({c.name for c in template.contracts}), sorted(rf.PREDICATES), sorted(types), OBS_DIM, CAND_DIM, method=method, B=0.5)
    p.eval()
    return p


def outputs(policy, snap):
    with torch.no_grad():
        o = policy(snap, policy.initial_hidden())
    return {"logits": o.logits.clone(), "value": o.value.clone(), "q": o.q.clone(), "hidden": o.hidden.clone(), "mask": o.mask.clone()}


def state_digest(policy):
    h = hashlib.sha256()
    for k, v in policy.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def dump(args):
    root = Path(args.root).resolve()
    template, rf = build(root)
    snaps = []
    seed = 0
    while len(snaps) < 4:
        snap, legal = snapshot(template, seed)
        if legal >= 3:
            snaps.append((seed, snap))
        seed += 1
    result = {"src": str(Path(sys.modules["cp_disr"].__file__).resolve().parent), "methods": {}}
    for method in LEGACY:
        entry = {"fresh": {}, "checkpoint": {}}
        for init_seed in (0, 1):
            p = policy_for(template, rf, method, init_seed)
            entry["fresh"][init_seed] = {"state_keys": list(p.state_dict().keys()), "state_digest": state_digest(p), "n_params": sum(x.numel() for x in p.parameters()),
                                         "outputs": {s: outputs(p, snap) for s, snap in snaps}}
        ck = Path(args.checkpoint_root).resolve() / CHECKPOINTS[method]
        if ck.is_file():
            p = policy_for(template, rf, method, 0)
            state = torch.load(ck, map_location="cpu", weights_only=False)
            p.load_state_dict(state["model"], strict=True)
            entry["checkpoint"] = {"path": str(ck), "sha256": hashlib.sha256(ck.read_bytes()).hexdigest(), "state_digest": state_digest(p),
                                   "outputs": {s: outputs(p, snap) for s, snap in snaps}}
        result["methods"][method] = entry
    torch.save(result, args.out)
    print(json.dumps({"saved": args.out, "src": result["src"], "checkpoints_loaded": [m for m, e in result["methods"].items() if e["checkpoint"]]}))


def compare(args):
    a, b = torch.load(args.a, weights_only=False), torch.load(args.b, weights_only=False)
    rows, ok = [], True
    for method in LEGACY:
        ea, eb = a["methods"][method], b["methods"][method]
        for init_seed in (0, 1):
            fa, fb = ea["fresh"][init_seed], eb["fresh"][init_seed]
            same = (fa["state_keys"] == fb["state_keys"] and fa["state_digest"] == fb["state_digest"] and fa["n_params"] == fb["n_params"]
                    and all(torch.equal(fa["outputs"][s][k], fb["outputs"][s][k]) for s in fa["outputs"] for k in fa["outputs"][s]))
            rows.append({"method": method, "kind": "fresh_init_seed_%d" % init_seed, "elementwise_equal": bool(same), "n_params": fa["n_params"]})
            ok &= same
        ca, cb = ea["checkpoint"], eb["checkpoint"]
        if ca and cb:
            same = (ca["sha256"] == cb["sha256"] and ca["state_digest"] == cb["state_digest"]
                    and all(torch.equal(ca["outputs"][s][k], cb["outputs"][s][k]) for s in ca["outputs"] for k in ca["outputs"][s]))
            rows.append({"method": method, "kind": "real_final_checkpoint_strict_load", "elementwise_equal": bool(same), "checkpoint": ca["path"], "sha256": ca["sha256"]})
            ok &= same
        else:
            rows.append({"method": method, "kind": "real_final_checkpoint_strict_load", "elementwise_equal": None, "note": "checkpoint missing in one dump"})
            ok = False
    receipt = {"old_src": a["src"], "new_src": b["src"], "all_elementwise_equal": bool(ok), "rows": rows}
    Path(args.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--root", required=True)
    d.add_argument("--checkpoint-root", required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--receipt", required=True)
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())

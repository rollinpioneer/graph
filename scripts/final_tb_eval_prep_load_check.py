#!/usr/bin/env python
"""Phase A worker (CP-DISR-TB-EVAL-PREP-LITE-01): zero-environment checkpoint compatibility check.

Per checkpoint: byte sha256 vs the frozen list and the sidecar, sidecar manifest identity,
key/shape/dtype listing, and a STRICT state_dict load into a Policy built by the source tree
found on PYTHONPATH (current tree or an archived tree).  No environment is constructed, no episode
is run, no optimizer is created or loaded, no weight is converted or re-saved.

Usage: PYTHONPATH=<src> python scripts/final_tb_eval_prep_load_check.py --models models.json --out out.json --label NAME
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_policy(method):
    """Same construction as stage2a_v11.make_policy, minus the environment template."""
    import cp_disr
    from cp_disr import stage2a_v11 as v11
    from cp_disr.neural import Policy
    from cp_disr.platforms.libero import runtime_factory as rf
    objects = rf.TASK_OBJECTS["T_B"]
    root = Path(cp_disr.__file__).resolve().parents[2]
    contract = root / "configs/runtime/stage_2a_contract_registry.yaml"
    names_probe = {}
    import yaml
    doc = yaml.safe_load(contract.read_text(encoding="utf-8"))
    timeouts = {c["name"]: 1.0 for c in doc["contracts"]}
    contracts = rf._ground_contracts_for(contract, timeouts, objects)
    actions = sorted({c.name for c in contracts})
    type_set = set()
    for m in rf.TASK_OBJECTS.values():
        type_set.update(m.values())
    model = Policy(actions=actions, predicates=sorted(rf.PREDICATES), types=sorted(type_set),
                   observation_dim=v11.OBS_DIM, candidate_dim=v11.CAND_DIM,
                   method=v11.policy_method(method), B=v11.B_PRIOR)
    return model, {"actions": actions, "contract_registry_sha256": sha256_file(contract),
                   "obs_dim": v11.OBS_DIM, "cand_dim": v11.CAND_DIM, "B_prior": v11.B_PRIOR,
                   "policy_method": v11.policy_method(method)}


def check_one(entry):
    import torch
    out = {"slot": entry["slot"], "plan_run": entry["plan_run"], "method": entry["method"], "seed": entry["seed"]}
    p = Path(entry["file"])
    side = p.with_suffix(".json")
    out["path"] = str(p)
    out["bytes"] = p.stat().st_size
    out["sha256"] = sha256_file(p)
    out["sha256_matches_frozen_list"] = out["sha256"] == entry["sha256"]
    out["bytes_match_frozen_list"] = out["bytes"] == entry["bytes"]
    meta = json.loads(side.read_text(encoding="utf-8"))
    out["sidecar_sha256_matches"] = meta.get("sha256") == out["sha256"]
    man = meta.get("manifest", {})
    out["manifest_identity"] = {
        "N": man.get("N"), "T": man.get("T"), "update_index": man.get("update_index", man.get("complete_updates")),
        "N_matches_list": man.get("N") == entry["N"],
        "T_matches_list": man.get("T") is not None and abs(float(man["T"]) - float(entry["T_s"])) < 1e-6,
        "method_in_manifest": man.get("method"), "attempt_id": man.get("attempt_id"),
    }
    state = torch.load(p, map_location="cpu", weights_only=False)
    out["state_top_keys"] = sorted(state.keys())
    sd = state["model"]
    out["n_param_tensors"] = len(sd)
    out["n_param_elements"] = int(sum(v.numel() for v in sd.values()))
    sig = [(k, list(v.shape), str(v.dtype)) for k, v in sd.items()]
    out["state_signature_sha256"] = hashlib.sha256(json.dumps(sig).encode()).hexdigest()
    out["dtypes"] = sorted({s[2] for s in sig})
    model, bind = build_policy(entry["method"])
    out["policy_binding"] = bind
    msd = model.state_dict()
    missing = sorted(set(msd) - set(sd))
    unexpected = sorted(set(sd) - set(msd))
    shape_mismatch = sorted(k for k in set(msd) & set(sd) if tuple(msd[k].shape) != tuple(sd[k].shape))
    dtype_mismatch = sorted(k for k in set(msd) & set(sd) if msd[k].dtype != sd[k].dtype)
    out["key_diff"] = {"model_keys": len(msd), "ckpt_keys": len(sd), "missing_in_ckpt": missing[:40], "n_missing_in_ckpt": len(missing),
                       "unexpected_in_ckpt": unexpected[:40], "n_unexpected_in_ckpt": len(unexpected),
                       "shape_mismatch": shape_mismatch[:40], "n_shape_mismatch": len(shape_mismatch),
                       "dtype_mismatch": dtype_mismatch[:40], "n_dtype_mismatch": len(dtype_mismatch)}
    try:
        model.load_state_dict(sd, strict=True)
        out["strict_load"] = True
        out["strict_load_error"] = None
    except Exception as exc:  # report, never repair
        out["strict_load"] = False
        out["strict_load_error"] = ("%s: %s" % (type(exc).__name__, str(exc)))[:1500]
    out["optimizer_created"] = False
    out["environment_constructed"] = False
    out["episodes_run"] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    import cp_disr
    import torch
    models = json.loads(Path(a.models).read_text(encoding="utf-8"))
    rows = []
    for m in models:
        if a.only and m["slot"] not in a.only:
            continue
        try:
            rows.append(check_one(m))
        except Exception as exc:
            rows.append({"slot": m["slot"], "plan_run": m["plan_run"], "error": "%s: %s" % (type(exc).__name__, str(exc))[:1500], "strict_load": False})
    doc = {"label": a.label, "cp_disr_package": str(Path(cp_disr.__file__).resolve()), "torch": torch.__version__,
           "tool": "final_tb_eval_prep_load_check.py", "tool_sha256": sha256_file(__file__), "rows": rows}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps([{k: r.get(k) for k in ("slot", "sha256_matches_frozen_list", "sidecar_sha256_matches", "strict_load", "error")} for r in rows], indent=1))


if __name__ == "__main__":
    sys.exit(main())

"""Problem-level runner of card C1-DEPOTS-SEARCH-MATCH-V1: asset resolution, scorer construction, one budgeted search run per (scorer, problem), shard loop with resumable per-case records."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from .. import depots as DP
from .. import task as T
from .budget import MILESTONES, Budget, HardTimeout, ResourceStop
from .engine import ModelNonFinite, SuccessorIndex, gbfs
from . import evaluators as EV

SCORERS = ("V_DENSE", "V_MG", "V_REL", "H_WL", "H_ADD", "H_COUNT")
NEURAL = {"V_DENSE": "dense", "V_MG": "mg", "V_REL": "rel"}
SETS = ("struct", "joint", "ipc")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rj(p):
    return json.loads(Path(p).read_text())


class OldRun:
    """Read-only view of the finished public-Depots run (frozen weights, WL models, problem manifest)."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = rj(self.root / "data" / "manifest.json")

    def checkpoint(self, mode):
        sel = rj(self.root / "runs" / mode / "selection.json")
        acct = rj(self.root / "runs" / mode / "training_accounting.json")
        u = str(sel["selected_update"])
        info = acct["checkpoints"][u]
        path = self.root / "runs" / mode / "checkpoints" / ("u%05d.pt" % int(u))
        return {"mode": mode, "update": int(u), "path": path, "sha256": info["sha256"], "init_seed": acct["budget"]["init_seed"]}

    def wl_params(self, set_name):
        track = "ipc" if set_name == "ipc" else "typed"
        d = self.root / "goose" / ("train_" + track)
        return {"track": track, "model": d / "wl_goose.model", "params": d / "wl_goose.model.params", "opts": d / "wl_goose.model.opts"}

    def cases(self, set_name):
        return self.manifest[set_name]

    def domain(self, set_name):
        return DP.DOMAIN_IPC if set_name == "ipc" else DP.DOMAIN_TYPED


def load_neural(old, scorer, device):
    import torch
    from ...torch_rl import load_checkpoint
    from .. import model as PM
    ck = old.checkpoint(NEURAL[scorer])
    if sha_file(ck["path"]) != ck["sha256"]:
        raise RuntimeError("checkpoint hash mismatch: %s" % ck["path"])
    m = PM.make_model(ck["mode"], device, seed=ck["init_seed"])
    load_checkpoint(ck["path"], m)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


def make_evaluator(old, scorer, set_name, device=None):
    if scorer in NEURAL:
        m, ck = load_neural(old, scorer, device)
        return EV.NeuralEval(scorer, m, device), {"checkpoint": str(ck["path"]), "sha256": ck["sha256"], "update": ck["update"]}
    if scorer == "H_WL":
        w = old.wl_params(set_name)
        return EV.WlEval(w["params"]), {"model": str(w["model"]), "model_sha256": sha_file(w["model"]), "params_sha256": sha_file(w["params"]), "track": w["track"]}
    if scorer == "H_ADD":
        return EV.HAddEval(), {"binary_sha256": sha_file(EV.HADD_BIN)}
    if scorer == "H_COUNT":
        return EV.CountEval(), {}
    raise ValueError(scorer)


def plan_text(task, action_indices):
    return "\n".join("(" + " ".join(task.actions[i].aid.split(":")[1:-1]) + ")" for i in action_indices) + "\n"


def run_case(evaluator, scorer, set_name, case, domain, cfg, plans_dir=None):
    """One budgeted search. Returns the per-case record (always; resource and technical failures are results)."""
    import torch
    cuda = torch.cuda.is_available() and getattr(getattr(evaluator, "device", None), "type", "") == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    budget = Budget(cfg["wall_seconds"], cfg["max_expansions"], cfg["max_rss_growth_bytes"], cfg.get("grace_seconds", 20.0))
    rec = {"scorer": scorer, "set": set_name, "case_id": case["case_id"], "status": None, "solved": False, "plan_valid": None, "plan_length": None, "plan_sha256": None}
    task = None
    t = {}
    try:
        budget.arm_hard_timer()
        t0 = time.perf_counter()
        task = T.depots_task(domain, case["file"])
        t["parse_ground"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        index = SuccessorIndex(task)
        t["index"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        evaluator.prepare(task, {"domain": str(domain), "problem": case["file"]}, budget)
        t["prepare"] = time.perf_counter() - t0
        res = gbfs(task, index, evaluator, budget, MILESTONES)
        d = res.as_dict()
        rec.update({k: d[k] for k in ("status", "expanded", "generated", "duplicates", "path_updates", "solved_at_expansion", "snapshots", "open_size", "closed_size", "best_h", "h_root", "search_seconds", "eval_seconds")})
        rec["eval_batches"] = d["evaluated_calls"]
        if res.status == "SOLVED":
            ids = [task.actions[i].aid for i in res.plan]
            fin = task.replay(ids)
            rec["plan_valid"] = bool(fin is not None and task.goal_satisfied(fin))
            rec["plan_length"] = len(ids)
            text = plan_text(task, res.plan)
            rec["plan_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            rec["solved"] = rec["plan_valid"]
            if not rec["plan_valid"]:
                rec["status"] = "INVALID_PLAN"
            elif plans_dir is not None:
                Path(plans_dir).mkdir(parents=True, exist_ok=True)
                (Path(plans_dir) / ("%s__%s__%s.plan" % (scorer, set_name, case["case_id"]))).write_text(text)
    except ResourceStop as e:
        rec["status"] = e.status
    except HardTimeout:
        rec["status"] = "TIMEOUT"
        rec["hard_timer"] = True
    except ModelNonFinite as e:
        rec["status"] = "MODEL_NONFINITE"
        rec["error"] = str(e)
    except Exception as e:                                                  # adapter / interface failure: technical, not a model result
        rec["status"] = "ADAPTER_ERROR"
        rec["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-400:]
    finally:
        Budget.disarm_hard_timer()
    rec["wall_total"] = budget.elapsed()
    rec["rss_peak_growth_bytes"] = budget.peak_rss_growth
    rec["timings"] = {k: round(v, 4) for k, v in t.items()}
    rec["timeout_before_node_limit"] = bool(rec["status"] == "TIMEOUT" and (rec.get("expanded") or 0) < cfg["max_expansions"])
    try:
        rec["evaluator_metrics"] = evaluator.metrics()
    except Exception:
        rec["evaluator_metrics"] = {}
    if task is not None:
        rec["task_size"] = {"actions": len(task.actions), "dyn_atoms": len(task.dyn_atoms)}
    if cuda:
        rec["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    try:
        evaluator.close()
    except Exception:
        pass
    del task
    gc.collect()
    return rec


def done_cases(path):
    p = Path(path)
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["case_id"]] = d
    return out


def run_shard(old, scorer, set_name, shard, n_shards, cfg, out_path, plans_dir, device=None, log=print):
    cases = sorted(old.cases(set_name), key=lambda c: c["case_id"])[shard::n_shards]
    domain = old.domain(set_name)
    t0 = time.perf_counter()
    evaluator, ident = make_evaluator(old, scorer, set_name, device)
    load_seconds = time.perf_counter() - t0
    if device is not None and device.type == "cuda":                       # warm-up on the first case's task would leak test data into timing: use a tiny synthetic forward instead
        _warmup(evaluator)
    done = done_cases(out_path)
    for c in cases:
        if c["case_id"] in done:
            continue
        if sha_file(c["file"]) != c["sha256"]:
            raise RuntimeError("problem file changed: %s" % c["file"])
        rec = run_case(evaluator, scorer, set_name, c, domain, cfg, plans_dir)
        rec["shard"] = [shard, n_shards]
        rec["model_load_seconds"] = round(load_seconds, 3)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        log("[search] %s %s %s -> %s expanded %s wall %.1fs" % (scorer, set_name, c["case_id"], rec["status"], rec.get("expanded"), rec["wall_total"]))
    return ident, load_seconds


def _warmup(evaluator):
    """CUDA / kernel warm-up on a tiny TRAINING-style task so the first real problem does not pay the one-off driver start (kept outside every problem's clock, reported once)."""
    import torch
    path = Path(os.environ.get("CPDISR_WARMUP_PROBLEM", ""))
    if not path.is_file():
        return
    task = T.depots_task(DP.DOMAIN_TYPED, path)
    b = Budget(60, 1000, 8 * 2 ** 30)
    evaluator.prepare(task, {"domain": str(DP.DOMAIN_TYPED), "problem": str(path)}, b)
    evaluator.evaluate([task.init_mask], b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    evaluator.close()

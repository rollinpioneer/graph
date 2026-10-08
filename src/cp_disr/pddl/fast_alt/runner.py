"""Problem-level runner of the FAST / ALT card: condition table, evaluators built once per worker, one budgeted search per (condition, problem), IPC timing blocks and resumable shard loops."""
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
from ..search_match import evaluators as EV
from ..search_match.budget import MILESTONES, Budget, HardTimeout, ResourceStop
from ..search_match.engine import ModelNonFinite, SuccessorIndex, gbfs
from ..search_match.runner import OldRun, load_neural, plan_text, sha_file
from .alt_engine import gbfs_alt
from .evaluators import FastNeuralEval, RefNeuralEval

SETS = ("struct", "joint", "ipc")
CONDITIONS = {
    "REF_DENSE": {"engine": "single", "main": "ref_dense"},
    "FAST_DENSE": {"engine": "single", "main": "fast_dense"},
    "EAGER_WL": {"engine": "single", "main": "wl"},
    "EAGER_HADD": {"engine": "single", "main": "hadd"},
    "ALT_DENSE_ADD": {"engine": "alt", "main": "dense", "add": "hadd"},
    "ALT_WL_ADD": {"engine": "alt", "main": "wl", "add": "hadd"},
}
NEURAL_CONDS = ("REF_DENSE", "FAST_DENSE", "ALT_DENSE_ADD")
MARKS = (1, 10, 100, 1000, 10000, 100000)


class Roll:
    """Rolling hash of the valid-expansion sequence (state ids in expansion order), recorded at fixed expansion counts."""

    def __init__(self, nbytes):
        self.h, self.nb, self.marks, self.n = hashlib.sha1(), max(1, nbytes), {}, 0

    def __call__(self, state, n):
        self.h.update(state.to_bytes(self.nb, "little"))
        self.n = n
        if n in MARKS:
            self.marks[str(n)] = self.h.copy().hexdigest()[:16]

    def end(self):
        self.marks["end"] = [self.n, self.h.hexdigest()[:16]]
        return self.marks


class Workers:
    """Evaluators of one worker process (weights loaded once; WL / h_add state is reset by ``prepare`` for every problem)."""

    def __init__(self, old, device, dense_backend="fast"):
        self.old, self.device, self.dense_backend = old, device, dense_backend
        self.model = None
        self.load_seconds = 0.0
        self._wl = {}
        self.hadd = EV.HAddEval()
        self.ident = {}

    def neural(self, kind):
        if self.model is None:
            t0 = time.perf_counter()
            self.model, ck = load_neural(self.old, "V_DENSE", self.device)
            self.load_seconds = time.perf_counter() - t0
            self.ident["V_DENSE"] = {"checkpoint": str(ck["path"]), "sha256": ck["sha256"], "update": ck["update"]}
            self._ref, self._fast = RefNeuralEval("REF", self.model, self.device), FastNeuralEval("FAST", self.model, self.device)
        return self._ref if kind == "ref" else self._fast

    def wl(self, set_name):
        track = "ipc" if set_name == "ipc" else "typed"
        if track not in self._wl:
            w = self.old.wl_params(set_name)
            self._wl[track] = EV.WlEval(w["params"])
            self.ident["WL_" + track] = {"model_sha256": sha_file(w["model"]), "params_sha256": sha_file(w["params"])}
        return self._wl[track]

    def pair(self, cond, set_name):
        spec = CONDITIONS[cond]
        main = {"ref_dense": lambda: self.neural("ref"), "fast_dense": lambda: self.neural("fast"), "dense": lambda: self.neural("fast" if self.dense_backend == "fast" else "ref"),
                "wl": lambda: self.wl(set_name), "hadd": lambda: self.hadd}[spec["main"]]()
        add = self.hadd if spec["engine"] == "alt" else None
        return main, add


def run_case(workers, cond, set_name, case, domain, cfg, plans_dir=None, order_tag=None):
    import torch
    spec = CONDITIONS[cond]
    main, add = workers.pair(cond, set_name)
    cuda = torch.cuda.is_available() and getattr(getattr(main, "device", None), "type", "") == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    budget = Budget(cfg["wall_seconds"], cfg["max_expansions"], cfg["max_rss_growth_bytes"], cfg.get("grace_seconds", 20.0))
    rec = {"condition": cond, "set": set_name, "case_id": case["case_id"], "status": None, "solved": False, "plan_valid": None, "plan_length": None, "plan_sha256": None, "order_tag": order_tag}
    t = {}
    task = None
    try:
        budget.arm_hard_timer()
        t0 = time.perf_counter()
        task = T.depots_task(domain, case["file"])
        t["parse_ground"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        index = SuccessorIndex(task)
        t["index"] = time.perf_counter() - t0
        ctx = {"domain": str(domain), "problem": case["file"]}
        t0 = time.perf_counter()
        main.prepare(task, ctx, budget)
        t["prepare_main"] = time.perf_counter() - t0
        if add is not None:
            t0 = time.perf_counter()
            add.prepare(task, ctx, budget)
            t["prepare_add"] = time.perf_counter() - t0
        roll = Roll((len(task.dyn_atoms) + 7) // 8)
        if spec["engine"] == "single":
            res = gbfs(task, index, main, budget, MILESTONES, observer=roll)
        else:
            res = gbfs_alt(task, index, main, add, budget, MILESTONES, observer=roll)
            rec["alt"] = res.extra
        d = res.as_dict()
        rec.update({k: d[k] for k in ("status", "expanded", "generated", "duplicates", "path_updates", "solved_at_expansion", "snapshots", "open_size", "closed_size", "best_h", "h_root", "search_seconds", "eval_seconds")})
        rec["eval_batches"] = d["evaluated_calls"]
        rec["prefix_hashes"] = roll.end()
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
                (Path(plans_dir) / ("%s__%s__%s.plan" % (cond, set_name, case["case_id"]))).write_text(text)
    except ResourceStop as e:
        rec["status"] = e.status
    except HardTimeout:
        rec["status"] = "TIMEOUT"
        rec["hard_timer"] = True
    except ModelNonFinite as e:
        rec["status"] = "MODEL_NONFINITE"
        rec["error"] = str(e)
    except Exception as e:
        rec["status"] = "ADAPTER_ERROR"
        rec["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-400:]
    finally:
        Budget.disarm_hard_timer()
    rec["wall_total"] = budget.elapsed()
    rec["rss_peak_growth_bytes"] = budget.peak_rss_growth
    rec["timings"] = {k: round(v, 4) for k, v in t.items()}
    rec["timeout_before_node_limit"] = bool(rec["status"] == "TIMEOUT" and (rec.get("expanded") or 0) < cfg["max_expansions"])
    try:
        rec["metrics_main"] = main.metrics()
        rec["metrics_add"] = add.metrics() if add is not None else None
    except Exception:
        rec["metrics_main"], rec["metrics_add"] = {}, None
    if task is not None:
        rec["task_size"] = {"actions": len(task.actions), "dyn_atoms": len(task.dyn_atoms), "template_edges": len(task.template().edges) if cond in NEURAL_CONDS else None}
    if cuda:
        rec["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    for ev in (main, add):
        try:
            if ev is not None:
                ev.close()
        except Exception:
            pass
    del task
    gc.collect()
    return rec


def block_order(case_id):
    """REF before FAST on even hash, FAST before REF on odd hash; ALT_DENSE_ADD always last (same GPU, same time block)."""
    even = int(hashlib.sha1(case_id.encode()).hexdigest(), 16) % 2 == 0
    return (["REF_DENSE", "FAST_DENSE"] if even else ["FAST_DENSE", "REF_DENSE"]) + ["ALT_DENSE_ADD"]


def done_keys(path):
    p = Path(path)
    out = set()
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.add((d["condition"], d["case_id"]))
    return out


def run_shard(old, conds, set_name, shard, nshards, cfg, out_path, plans_dir, device=None, dense_backend="fast", block=False, log=print):
    """Resumable loop over the cases of one shard. ``block``: IPC timing block -- for every case the neural conditions run back to back in the order given by ``block_order``."""
    cases = sorted(old.cases(set_name), key=lambda c: c["case_id"])[shard::nshards]
    domain = old.domain(set_name)
    workers = Workers(old, device, dense_backend)
    if device is not None:
        workers.neural("fast")                                              # weights are loaded (and the load time recorded) once, outside every problem's clock
        _warmup(workers)
    done = done_keys(out_path)
    for c in cases:
        if sha_file(c["file"]) != c["sha256"]:
            raise RuntimeError("problem file changed: %s" % c["file"])
        order = [x for x in block_order(c["case_id"]) if x in conds] if block else list(conds)
        for cond in order:
            if (cond, c["case_id"]) in done:
                continue
            rec = run_case(workers, cond, set_name, c, domain, cfg, plans_dir, order_tag="".join(x[0] for x in order) if block else None)
            rec["shard"] = [shard, nshards]
            rec["model_load_seconds"] = round(workers.load_seconds, 3)
            rec["cpu_count"] = os.cpu_count()
            try:
                rec["load_avg_1m"] = os.getloadavg()[0]
            except OSError:
                pass
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            log("[run] %s %s %s -> %s expanded %s wall %.1fs" % (cond, set_name, c["case_id"], rec["status"], rec.get("expanded"), rec["wall_total"]))
    return workers.ident, workers.load_seconds


def _warmup(workers):
    """CUDA / kernel warm-up on a Train96 problem (outside every problem's clock)."""
    import torch
    path = Path(os.environ.get("CPDISR_WARMUP_PROBLEM", ""))
    if not path.is_file():
        return
    task = T.depots_task(DP.DOMAIN_TYPED, path)
    b = Budget(60, 1000, 8 * 2 ** 30)
    for kind in ("ref", "fast"):
        ev = workers.neural(kind)
        ev.prepare(task, {}, b)
        ev.evaluate([task.init_mask], b)
        ev.close()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

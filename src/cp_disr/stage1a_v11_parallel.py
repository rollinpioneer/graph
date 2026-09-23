"""Resource-chunking parallel D0 collectors for Stage 1A.

num_envs is collection parallelism only. It does not change H, d_ref, PPO,
deadline, prior protocol, or the frozen method profile.
"""
from __future__ import annotations

import os
import traceback
from multiprocessing import get_context
from pathlib import Path


def _worker_main(conn, root, method, worker_id, run_id):
    os.chdir(root)
    os.environ["CP_DISR_ENV_ID"] = "d0-env-%s" % worker_id
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from cp_disr.stage1a_v11 import (
            apply_episode_prior,
            bind_profile,
            make_bundle,
            make_policy,
            require_case_cache,
            s1,
        )
        from cp_disr.collector import Collector
        from cp_disr.prior import PriorSampler

        root = str(Path(root))
        bind_profile(root)
        bundle = make_bundle(root)
        device = __import__("torch").device("cpu")
        policy = make_policy(bundle.template, method, device)
        policy.eval()
        collector = Collector(bundle, policy)
        sampler = None if method == "B2" else PriorSampler(0)
        split = __import__("json").loads((Path(root) / "configs/splits/D0_stage_1a.json").read_text(encoding="utf-8"))
        split_index = {r["case_id"]: r for r in (split.get("train") or []) + (split.get("dev") or [])}
        conn.send(("ready", {"worker_id": worker_id}))
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                try:
                    bundle.environment.close()
                except Exception:
                    pass
                conn.send(("closed", {"worker_id": worker_id}))
                break
            if cmd == "set_weights":
                policy.load_state_dict(payload)
                policy.eval()
                conn.send(("weights_ok", {"worker_id": worker_id}))
                continue
            if cmd != "run_episode":
                conn.send(("err", {"error": "unknown cmd %s" % cmd}))
                continue
            try:
                result = _run_episode(
                    root, method, worker_id, run_id, bundle, policy, collector, sampler, split_index, payload
                )
                conn.send(("ok", result))
            except Exception as exc:
                conn.send((
                    "err",
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "worker_id": worker_id,
                        "case_id": (payload or {}).get("case_id"),
                    },
                ))
    except Exception as exc:
        try:
            conn.send(("boot_err", {"error": str(exc), "traceback": traceback.format_exc(), "worker_id": worker_id}))
        except Exception:
            pass


def _run_episode(root, method, worker_id, run_id, bundle, policy, collector, sampler, split_index, payload):
    from cp_disr.rl import suite_half_life
    from cp_disr.stage1a_v11 import apply_episode_prior, require_case_cache, s1
    import math

    root = Path(root)
    case = payload["case_id"]
    rec = split_index[case]
    require_case_cache(root, rec)
    snap = bundle.start_case(case)
    env_id = os.environ.get("CP_DISR_ENV_ID", "d0-env-%s" % worker_id)
    snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler, eval_original=False)
    bundle.current_snapshot = snap
    collector.reset_episode(snap.env_id, snap.episode_id)
    cache_key = str(Path(root) / rec["cache_dir"])
    items = []
    ep_reward = 0.0
    success = False
    reason = None
    empty = False
    while True:
        t, result = collector.step(snap, deterministic=False)
        if t is None:
            reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
            success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
            empty = True
            break
        rec_t = s1.compact_transition(
            method, 0, case, t, result, collector.last_output, collector.last_execution,
            prior.audit_mode, prior.original_hash, source_n, cache_key, run_id,
        )
        rec_t["H"] = suite_half_life()
        rec_t["actor_episode_discount_weight"] = False
        rec_t["worker_id"] = worker_id
        if not rec_t["logits_finite"] or not rec_t["value_finite"] or not math.isfinite(rec_t["old_logp"]):
            raise RuntimeError("NaN/Inf in transition")
        items.append({"transition": t, "rec": rec_t, "success": bool(result.success) if not isinstance(result, dict) else bool(result.get("success"))})
        ep_reward += t.reward
        snap = t.next_snapshot
        bundle.current_snapshot = snap
        ended = t.terminated or t.truncated
        success = bool(result.success) if not isinstance(result, dict) else bool(result.get("success"))
        reason = result.reason if not isinstance(result, dict) else result.get("reason")
        if ended:
            break
    bundle.current_snapshot = None
    return {
        "worker_id": worker_id,
        "case_id": case,
        "items": items,
        "empty": empty,
        "success": success,
        "reason": reason,
        "ep_reward": ep_reward,
        "prior_mode": prior.audit_mode,
        "prior": {
            "env_id": prior.env_id,
            "episode_id": prior.episode_id,
            "audit_mode": prior.audit_mode,
            "original_hash": prior.original_hash,
            "effective_hash": prior.hash,
            "effective_relation_count": len(prior.edges),
            "source_cache_relation_count": source_n,
            "case_id": case,
            "method": method,
            "eval_original": False,
            "worker_id": worker_id,
        },
        "source_n": source_n,
        "cache_key": cache_key,
    }


class EnvPool:
    def __init__(self, root, method, num_envs, run_id):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.method = method
        ctx = get_context("spawn")
        self.conns = []
        self.procs = []
        try:
            for i in range(self.num_envs):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker_main, args=(child, str(root), method, i, run_id), daemon=True)
                proc.start()
                child.close()
                status, payload = parent.recv()
                if status != "ready":
                    raise RuntimeError("worker %s failed to start: %s" % (i, payload))
                self.conns.append(parent)
                self.procs.append(proc)
        except Exception:
            self.close()
            raise
        self.idle = list(range(self.num_envs))
        self.busy = {}
        self.case_i = 0
        self.weights_version = -1

    def close(self):
        for i, conn in enumerate(self.conns):
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=20)
            if proc.is_alive():
                proc.terminate()

    def set_weights(self, state_dict):
        cpu_sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        leftover = []
        while self.busy:
            leftover.append(self._recv_one())
        for i, conn in enumerate(self.conns):
            conn.send(("set_weights", cpu_sd))
        for conn in self.conns:
            status, payload = conn.recv()
            if status != "weights_ok":
                raise RuntimeError("weight sync failed: %s %s" % (status, payload))
        self.idle = list(range(self.num_envs))
        self.busy = {}
        return leftover

    def _dispatch(self, cases, stop):
        while self.idle and not stop():
            i = self.idle.pop(0)
            case = cases[self.case_i % len(cases)]
            self.case_i += 1
            self.conns[i].send(("run_episode", {"case_id": case}))
            self.busy[i] = case

    def _recv_one(self):
        while True:
            for i in list(self.busy):
                if self.conns[i].poll(0.05):
                    status, payload = self.conns[i].recv()
                    del self.busy[i]
                    self.idle.append(i)
                    if status != "ok":
                        raise RuntimeError("worker %s failed: %s" % (i, payload))
                    return payload
            if not self.busy:
                raise RuntimeError("no in-flight episodes")

    def next_episode(self, cases, stop=lambda: False):
        self._dispatch(cases, stop)
        if not self.busy:
            return None
        payload = self._recv_one()
        self._dispatch(cases, stop)
        return payload

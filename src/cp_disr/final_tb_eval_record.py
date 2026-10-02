"""Evaluation-only recorder for the unified T_B evaluation (schema ``eval_row_v2``).

CP-DISR-TB-EVAL-PREP-LITE-01.  This module OBSERVES; it never changes behaviour.

* It makes the same calls, in the same order, as ``stage2a_v11.eval_episodes`` (bind_H, make_bundle,
  resolve_runtime, attach_split_cases, seed_all(0), make_policy, load_checkpoint, start_case,
  apply_episode_prior, reset_episode, ``Collector.step(..., deterministic=True)``) and only adds
  recording.  The old row fields (success, G, steps, reason, success_seconds, source_n, prior_mode)
  are computed by the old code path, byte for byte.
* It imports no optimizer / trainer, creates no optimizer, writes no checkpoint or training artifact,
  and edits no production module (stage2a_v11, collector, neural, torch_rl, runtime_factory,
  controller, Verifier, Evaluator, reward, deadline, termination, checkpoint stay untouched).
* The model file is hashed before it is loaded and again after the last episode; a mismatch aborts.
* An exception that escapes an episode is an INFRASTRUCTURE_FAILURE: the partial payload is written,
  the evaluation stops, and nothing is retried or replaced.

The module is importable stand-alone (no relative imports) so that it can also be loaded by file path
under an archived source tree without being copied into it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SCHEMA_VERSION = "eval_row_v2"
SUCCESS_SECONDS_LABEL = "FINAL_SKILL_DURATION_ONLY"
TIME_TO_SUCCESS_LABEL = "EPISODE_START_TO_FIRST_INDEPENDENT_CONFIRMED_SUCCESS_SIM_SECONDS"
OLD_ROW_FIELDS = ("case_id", "success", "G", "steps", "reason", "success_seconds", "source_n", "prior_mode")
NEW_ROW_FIELDS = (
    "episode_start_clock_s", "time_to_first_confirmed_success_s", "time_to_first_confirmed_success_label",
    "first_confirmed_success_decision_index", "episode_end_elapsed_s", "actions", "inter_step_clock_gap_s",
    "no_transition_exit", "success_seconds_label",
)
ACTION_FIELDS = ("decision_index", "candidate_id", "clock_start", "clock_end", "duration", "reward", "weight",
                 "controller_exit", "terminated", "truncated", "reason", "success_confirmed")
UNVERIFIED = "UNVERIFIED"
MAX_DECISIONS_GUARD = 1000  # engineering guard only (a T_B episode has a 60 s simulated deadline)

NORMAL_POLICY_OUTCOME = "NORMAL_POLICY_OUTCOME"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
# Reasons that are legitimate policy / task outcomes: counted as failures, never re-run.
NORMAL_REASONS = (
    "TASK_SUCCESS", "DEADLINE", "INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE", "TASK_DEADLINE",
)


class EvalRecorderError(RuntimeError):
    pass


class ModelIdentityError(EvalRecorderError):
    """Checkpoint bytes changed (or differ from the registered identity): fail closed."""


class InfrastructureFailure(EvalRecorderError):
    """CUDA / EGL / process / filesystem / evaluation-code / environment failure: stop, report, no rerun."""

    def __init__(self, message, partial_path=None):
        super().__init__(message)
        self.partial_path = partial_path


# ----------------------------------------------------------------------------- identity helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def verify_model_identity(identity, expected_sha256=None):
    """Re-hash the checkpoint; raise ModelIdentityError unless it equals ``identity`` (and ``expected_sha256``)."""
    now = file_identity(identity["path"])
    if now["sha256"] != identity["sha256"] or now["bytes"] != identity["bytes"]:
        raise ModelIdentityError("checkpoint bytes changed: %s (%s -> %s)" % (identity["path"], identity["sha256"], now["sha256"]))
    if expected_sha256 is not None and now["sha256"] != expected_sha256:
        raise ModelIdentityError("checkpoint sha256 differs from the registered identity: %s" % identity["path"])
    return now


def generation_identity(ckpt_path):
    """N / T / update of the evaluated generation, read from the checkpoint sidecar manifest (no model load)."""
    p = Path(ckpt_path)
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    man = meta.get("manifest", {})
    return {
        "generation": p.stem, "N": man.get("N"), "T": man.get("T"),
        "update": man.get("update_index", man.get("complete_updates")),
        "complete_updates": man.get("complete_updates"), "fragment_updates": man.get("fragment_updates"),
        "final": bool(man.get("final")), "sidecar_sha256": meta.get("sha256"),
    }


def source_identity(files):
    """sha256 of each named source file (name -> path); a missing file is recorded as None."""
    return {k: (sha256_file(v) if Path(v).is_file() else None) for k, v in files.items()}


def no_prior_case_gate(root, rec):
    """Gate for a no-prior split (empty-R methods): the case row must carry NO cache pointer, and nothing is opened.

    Same intent as ``stage2a_v11.require_case_cache`` under a run context (prior_mode == absent), usable under an
    archived source tree that has no run-context support.  Never reads a cache, VLM output or relation truth.
    """
    if rec.get("cache_dir") or rec.get("cache_key") or rec.get("cache_status"):
        raise EvalRecorderError("no-prior split must not carry cache pointers: %s" % rec.get("case_id"))
    return None


# ----------------------------------------------------------------------------- per-episode recording
def _controller_exit(execution, raw):
    """Same rule as collector._exit_reason (re-stated here so the production module stays untouched)."""
    data = raw if isinstance(raw, dict) else (execution if isinstance(execution, dict) else {})
    if data.get("controller_exit"):
        return str(data.get("controller_exit"))
    if isinstance(execution, dict):
        return str(execution.get("controller_exit") or "")
    return str(getattr(execution, "controller_exit", "") or "")


def _result_get(result, key):
    return result.get(key) if isinstance(result, dict) else getattr(result, key, None)


def classify_outcome(reason, success=False):
    """NORMAL_POLICY_OUTCOME for every legal episode ending (failure counted, never re-run)."""
    return NORMAL_POLICY_OUTCOME


def record_episode(bundle, collector, snap, case_id, source_n, prior_mode):
    """The old eval loop plus recording.  ``snap`` is the post-prior start snapshot."""
    start_clock = float(bundle.episode_start_seconds)
    G, steps, success, reason, success_seconds = 0.0, 0, False, None, None
    actions = []
    first_success_elapsed = None
    first_success_index = None
    no_transition = None
    guard = 0
    while True:
        guard += 1
        if guard > MAX_DECISIONS_GUARD:
            raise InfrastructureFailure("decision guard exceeded in case %s" % case_id)
        t, result = collector.step(snap, deterministic=True)
        if t is None:
            reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
            success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
            no_transition = {
                "reason": reason, "success": success,
                "elapsed_s": _result_get(result, "elapsed"),
                "controller_exit": _result_get(result, "controller_exit"),
                "remaining_s": _result_get(result, "remaining"),
                "result_kind": "dict" if isinstance(result, dict) else type(result).__name__,
            }
            break
        G += float(t.reward) * float(t.weight)
        steps += 1
        confirmed = bool(result.success if not isinstance(result, dict) else result.get("success"))
        clock_start = float(t.snapshot.clock_seconds)
        clock_end = float(t.next_snapshot.clock_seconds)
        actions.append({
            "decision_index": int(t.snapshot.decision_id),
            "candidate_id": t.selected_candidate_id,
            "clock_start": clock_start, "clock_end": clock_end,
            "duration": float(t.duration), "reward": float(t.reward), "weight": float(t.weight),
            "controller_exit": _controller_exit(getattr(collector, "last_execution", None),
                                                getattr(getattr(bundle, "executor", None), "last", None)),
            "terminated": bool(t.terminated), "truncated": bool(t.truncated),
            "reason": getattr(t, "reason", None), "success_confirmed": confirmed,
        })
        if confirmed:
            success = True
            success_seconds = float(t.snapshot.elapsed_seconds + t.duration) if hasattr(t.snapshot, "elapsed_seconds") else t.duration
            if first_success_elapsed is None:  # only the FIRST independently confirmed success is recorded
                first_success_elapsed = clock_end - start_clock
                first_success_index = int(t.snapshot.decision_id)
        snap = t.next_snapshot
        if t.terminated or t.truncated:
            reason = result.reason if not isinstance(result, dict) else result.get("reason")
            break
    row = {
        "case_id": case_id, "success": success, "G": G, "steps": steps, "reason": reason,
        "success_seconds": success_seconds if success else None,
        "source_n": source_n, "prior_mode": prior_mode,
    }
    gap = None
    if first_success_elapsed is not None:
        covered = sum(a["duration"] for a in actions if a["decision_index"] <= first_success_index)
        gap = first_success_elapsed - covered  # never silently assumed zero
    end_elapsed = (actions[-1]["clock_end"] - start_clock) if actions else None
    if end_elapsed is None and no_transition and no_transition.get("elapsed_s") is not None:
        end_elapsed = float(no_transition["elapsed_s"])
    row.update({
        "success_seconds_label": SUCCESS_SECONDS_LABEL,
        "episode_start_clock_s": start_clock,
        "time_to_first_confirmed_success_s": first_success_elapsed,
        "time_to_first_confirmed_success_label": TIME_TO_SUCCESS_LABEL,
        "first_confirmed_success_decision_index": first_success_index,
        "episode_end_elapsed_s": end_elapsed,
        "actions": actions,
        "inter_step_clock_gap_s": gap,
        "no_transition_exit": no_transition,
        "outcome_class": classify_outcome(reason, success),
    })
    return row


def normalize_old_row(row):
    """Read an old-schema row: keep every old field, label the old time, and never invent a total time."""
    out = dict(row)
    if "time_to_first_confirmed_success_s" not in out:
        out["success_seconds_label"] = SUCCESS_SECONDS_LABEL
        out["time_to_first_confirmed_success_s"] = None
        out["episode_time_status"] = UNVERIFIED
    return out


def time_to_success_column(rows):
    """Episode-level success time column: only rows that carry a verified new value contribute."""
    col = []
    for r in rows:
        v = r.get("time_to_first_confirmed_success_s") if "time_to_first_confirmed_success_label" in r else None
        col.append(v)
    return col


def aggregate(rows):
    n = len(rows)
    success_n = sum(1 for r in rows if r["success"])
    returns = [r["G"] for r in rows]
    return {
        "success_n": success_n, "n": n, "success_rate": (success_n / max(1, n)) if rows else None,
        "mean_discounted_return": float(sum(returns) / len(returns)) if returns else None,
    }


# ----------------------------------------------------------------------------- orchestration
def _write_json_atomic(path, payload):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def evaluate_checkpoint_recorded(root, task_id, method, ckpt_path, cases, split_index, device, hashes_doc, out_eval,
                                 *, label, n_episodes=None, v11=None, collector_cls=None, load_checkpoint_fn=None,
                                 expected_sha256=None, case_gate=None, extra_identity=None):
    """Recorded twin of ``stage2a_v11.eval_episodes`` (deterministic argmax, isolated RNG, optimizer_steps 0).

    ``v11`` / ``collector_cls`` / ``load_checkpoint_fn`` default to the production objects of the source tree on
    ``sys.path``; they are injectable so the offline tests can run with scripted fakes (0 environment).
    ``case_gate(root, rec)`` defaults to ``v11.require_case_cache`` (historical semantics).
    """
    if v11 is None:
        from cp_disr import stage2a_v11 as v11  # noqa: PLC0415 - lazy: importing this module must stay light
    if collector_cls is None:
        from cp_disr.collector import Collector as collector_cls  # noqa: PLC0415
    if load_checkpoint_fn is None:
        from cp_disr.torch_rl import load_checkpoint as load_checkpoint_fn  # noqa: PLC0415
    gate = case_gate or v11.require_case_cache
    s1 = v11.s1
    ckpt_path = Path(ckpt_path)
    identity = file_identity(ckpt_path)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ModelIdentityError("checkpoint sha256 differs from the registered identity before load: %s" % ckpt_path)
    generation = generation_identity(ckpt_path)
    v11.bind_H(root)
    rng_before = s1.capture_rng()
    split_rel = v11.ENABLED_SPLITS[task_id]
    bundle = v11.make_bundle(root, task_id, split_rel)
    prof = v11.resolve_runtime(root)
    v11.attach_split_cases(bundle, root, list(split_index.values()), task_id, prof["task_deadlines"][task_id])
    s1.seed_all(0)
    policy = v11.make_policy(bundle.template, method, device)
    load_checkpoint_fn(ckpt_path, policy)
    policy.eval()
    collector = collector_cls(bundle, policy)
    rows = []
    selected = list(cases[:(n_episodes or len(cases))])
    infra = None
    try:
        for case in selected:
            rec = split_index[case]
            try:
                gate(root, rec)
                snap = bundle.start_case(case)
                snap, prior, source_n = v11.apply_episode_prior(method, bundle, snap, sampler=None, eval_original=True)
                collector.reset_episode(snap.env_id, snap.episode_id)
                rows.append(record_episode(bundle, collector, snap, case, source_n, prior.audit_mode))
            except InfrastructureFailure as exc:
                infra = {"case_id": case, "error": str(exc)[:1500], "kind": "decision_guard"}
                break
            except Exception as exc:  # any escaping exception stops the release: no rerun, no scene swap
                infra = {"case_id": case, "error": ("%s: %s" % (type(exc).__name__, exc))[:1500], "kind": INFRASTRUCTURE_FAILURE}
                break
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
        s1.restore_rng(rng_before)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id, "method": method, "alias": v11.method_dir_name(method), "checkpoint": str(ckpt_path),
        **aggregate(rows), "rows": rows, "hashes": hashes_doc,
        "isolated_rng": True, "H": v11.suite_half_life(), "eval_action": "deterministic_argmax", "label": label,
        "optimizer_steps": 0,
        "success_seconds_label": SUCCESS_SECONDS_LABEL, "time_to_first_confirmed_success_label": TIME_TO_SUCCESS_LABEL,
        "evaluated_model": identity, "evaluated_generation": generation,
        "case_order": selected, "n_cases_requested": len(selected),
        "eval_rng_isolation": {"rng_captured_before": True, "rng_restored_after": True, "seed_all": 0},
        "extra_identity": extra_identity or {},
    }
    if infra is not None:
        payload["status"] = INFRASTRUCTURE_FAILURE + "_STOPPED"
        payload["infrastructure_failure"] = infra
        _write_json_atomic(out_eval, payload)
        raise InfrastructureFailure("%s in %s: %s" % (INFRASTRUCTURE_FAILURE, infra["case_id"], infra["error"]), partial_path=str(out_eval))
    # model bytes must be unchanged after the last episode
    try:
        verify_model_identity(identity)
    except ModelIdentityError:
        payload["status"] = "ABORTED_MODEL_HASH_CHANGED"
        _write_json_atomic(out_eval, payload)
        raise
    payload["status"] = "COMPLETE"
    payload["model_sha256_after_last_episode"] = identity["sha256"]
    _write_json_atomic(out_eval, payload)
    return payload

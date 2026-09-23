"""Plan v1.1 / Method 2.1.1 Stage 1A one-task smoke. Historical stage1a_smoke.py is unchanged."""
from __future__ import annotations

import hashlib, json, math, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import BindingError, DataIntegrityError, digest
from .rl import gamma, set_suite_half_life, suite_half_life
from .platforms.libero.snapshot import load_cache_edges
class _S1:
    def __getattr__(self, name):
        from . import stage1a_smoke as mod
        return getattr(mod, name)

s1 = _S1()

STAGE_DIR = Path("runs/stage_1a")
STATUS_PATH = Path("status/stage_1a.json")
REPORT_PATH = Path("reports/stage_1a_summary.md")
RUNTIME_REL = Path("experiments/manifests/runtime_manifest_v211.yaml")
REFERENCE_REL = Path("runs/stage_0a/reference_execution_manifest.json")
SPLIT_REL = Path("configs/splits/D0_stage_1a.json")
STOP_REL = Path("experiments/part_2_exploration/stage_2a/orchestrator/STOP_SUPERSEDED_BY_PLAN_V1_1.json")
OLD_STAGE_DIR = Path("experiments/part_1_smoke/stage_1a")
BASE_COMMIT = "4323f4d6d14e1bca671e47695be9bffc5b0c59ac"
N_CAP = 16384
ROLLOUT_N = 1024
MAX_UPDATES = 16
EVAL_EVERY_N = 4096
DEV_EPISODES = 10
TEST_EPISODES = 30
OBS_DIM = 48
CAND_DIM = 8
B_PRIOR = 0.5
MAX_EMPTY = 8
PLANNED = {"B2": "v11_1A_D0_B2_s0", "Full": "v11_1A_D0_Full_s0"}
GATE_CASE = "D0_dev_00"


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg):
    print("[stage1a_v11] %s %s" % (utc_now(), msg), flush=True)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=s1._json_default) + "\n", encoding="utf-8")


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_profile(root):
    root = Path(root)
    ref = json.loads((root / REFERENCE_REL).read_text(encoding="utf-8"))
    runtime = load_yaml(root / RUNTIME_REL)
    H = float(ref["H"])
    d_ref = float(ref["families"]["D0"]["d_ref_median"])
    rt_h = float(runtime["runtime"]["suite_H_seconds"])
    rt_dref = float(runtime["runtime"]["reference_skill_seconds_by_task"]["D0"])
    if H != rt_h:
        raise BindingError("reference H %s != runtime suite_H_seconds %s" % (H, rt_h))
    if d_ref != rt_dref:
        raise BindingError("reference D0 d_ref %s != runtime %s" % (d_ref, rt_dref))
    deadline = float(runtime["runtime"]["task_deadlines"]["D0"])
    split = json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))
    train = [r["case_id"] for r in split["train"]]
    dev = [r["case_id"] for r in split["dev"]]
    test = [r["case_id"] for r in (split.get("test") or [])]
    return {
        "H": H,
        "d_ref": d_ref,
        "Tcap": float(N_CAP) * float(d_ref),
        "Ncap": N_CAP,
        "rollout": ROLLOUT_N,
        "max_updates": MAX_UPDATES,
        "actor_episode_discount_weight": False,
        "task_deadline_seconds": deadline,
        "runtime_path": str(RUNTIME_REL).replace("\\", "/"),
        "split_path": str(SPLIT_REL).replace("\\", "/"),
        "train_ids": train,
        "dev_ids": dev,
        "test_ids": test,
        "train_n": len(train),
        "dev_n": len(dev),
        "test_n": len(test),
        "split": split,
    }


def verify_gamma_half_life(H):
    if suite_half_life() != H:
        raise BindingError("suite H not bound to calibrated value")
    g = gamma(H)
    if abs(g - 0.5) > 1e-12:
        raise BindingError("gamma(H) is %s, expected 0.5" % g)
    return g


def bind_profile(root):
    prof = resolve_profile(root)
    set_suite_half_life(prof["H"])
    verify_gamma_half_life(prof["H"])
    return prof


def make_bundle(root):
    from .platforms.libero.runtime_factory import create_runtime
    manifest = load_yaml(root / RUNTIME_REL)
    runtime = dict(manifest["runtime"])
    runtime["active_task_id"] = "D0"
    manifest = dict(manifest)
    manifest["runtime"] = runtime
    return create_runtime(manifest)


def make_policy(template, method, device):
    from .neural import Policy
    from .platforms.libero.runtime_factory import PREDICATES, OBJECTS
    actions = sorted({c.name for c in template.contracts})
    model = Policy(actions=actions, predicates=sorted(PREDICATES), types=sorted(set(OBJECTS.values())), observation_dim=OBS_DIM, candidate_dim=CAND_DIM, method=method, B=B_PRIOR)
    return model.to(device)


def frozen_configsha8(prof, git_hash):
    payload = {
        "H": prof["H"],
        "d_ref": prof["d_ref"],
        "Tcap": prof["Tcap"],
        "Ncap": prof["Ncap"],
        "actor_episode_discount_weight": False,
        "runtime_path": prof["runtime_path"],
        "split_path": prof["split_path"],
        "task_deadline_seconds": prof["task_deadline_seconds"],
        "git_hash": git_hash,
        "base_commit": BASE_COMMIT,
        "profile": "method-2.1.1",
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=s1._json_default))[:8]


def audit_caches(root, split_rows):
    rows = []
    issues = []
    nonempty_source = []
    for rec in split_rows:
        case_id = rec["case_id"]
        cache_dir = rec.get("cache_dir")
        status = {
            "case_id": case_id,
            "split": rec.get("split"),
            "cache_dir": cache_dir,
            "source_cache_present": False,
            "complete": False,
            "source_relation_count": None,
            "legal_empty_cache": False,
            "missing_cache": False,
        }
        if not cache_dir:
            status["missing_cache"] = True
            issues.append("%s has no cache_dir" % case_id)
            rows.append(status)
            continue
        folder = root / cache_dir
        complete = (folder / "COMPLETE").is_file()
        edges_path = folder / "final_edges.json"
        if not edges_path.is_file():
            edges_path = folder / "accepted_relations.json"
        present = folder.is_dir() and edges_path.is_file()
        status["source_cache_present"] = bool(present)
        status["complete"] = bool(complete)
        if not present or not complete:
            status["missing_cache"] = True
            issues.append("%s missing source cache or COMPLETE (present=%s complete=%s)" % (case_id, present, complete))
            rows.append(status)
            continue
        edges = load_cache_edges(folder)
        status["source_relation_count"] = len(edges)
        status["legal_empty_cache"] = len(edges) == 0
        if edges:
            nonempty_source.append(case_id)
        rows.append(status)
    return {"rows": rows, "issues": issues, "nonempty_source_case_ids": nonempty_source, "n": len(rows), "n_missing": sum(1 for r in rows if r["missing_cache"]), "n_legal_empty": sum(1 for r in rows if r["legal_empty_cache"]), "n_nonempty": len(nonempty_source)}


def require_case_cache(root, rec):
    case_id = rec["case_id"]
    cache_dir = rec.get("cache_dir")
    if not cache_dir:
        raise BindingError("missing cache for enabled case %s; not a legal empty prior" % case_id)
    folder = root / cache_dir
    if not (folder / "COMPLETE").is_file():
        raise BindingError("missing COMPLETE cache for enabled case %s; not a legal empty prior" % case_id)
    if not ((folder / "final_edges.json").is_file() or (folder / "accepted_relations.json").is_file()):
        raise BindingError("missing cache edges for enabled case %s; not a legal empty prior" % case_id)
    return folder


def apply_episode_prior(method, bundle, snapshot, sampler, eval_original=False):
    from .prior import EpisodePrior
    source_n = len(bundle.original_prior_edges)
    original = tuple(tuple(e) for e in bundle.original_prior_edges)
    original_hash = digest(original)
    if method == "B2":
        snap = s1.empty_prior(snapshot)
        prior = EpisodePrior(snap.env_id, snap.episode_id, (), original_hash, digest(()), "absent")
        return snap, prior, source_n
    if eval_original:
        snap = s1.set_prior(snapshot, original)
        prior = EpisodePrior(snap.env_id, snap.episode_id, original, original_hash, digest(original), "original")
        return snap, prior, source_n
    prior = sampler.start(snapshot.env_id, snapshot.episode_id, original)
    snap = s1.set_prior(snapshot, prior.edges)
    return snap, prior, source_n


def startup_hard_checks(root, out, device):
    root = Path(root)
    out = Path(out)
    gates = []
    issues = []
    prof = bind_profile(root)
    split = prof["split"]
    enabled = list(split["train"]) + list(split["dev"])
    cache_audit = audit_caches(root, enabled)
    write_json(out / "startup" / "cache_audit.json", cache_audit)
    write_json(out / "startup" / "enabled_split.json", {"train": prof["train_ids"], "dev": prof["dev_ids"], "test": prof["test_ids"], "test_isolated": split.get("test_isolated")})
    stop_ok = (root / STOP_REL).is_file()
    gates.append({"gate": "OLD_STAGE2A_STOP_GUARD", "passed": stop_ok, "issues": [] if stop_ok else ["STOP_SUPERSEDED_BY_PLAN_V1_1.json missing"]})
    old_dref_issues = []
    if abs(prof["d_ref"] - 3.55) < 1e-12:
        old_dref_issues.append("D0 d_ref collapsed to old 3.55")
    if abs(prof["Tcap"] - (16384 * 3.55)) < 1e-8:
        old_dref_issues.append("Tcap used old 3.55")
    gates.append({"gate": "CALIBRATED_H_DREF_TCAP", "passed": not old_dref_issues, "issues": old_dref_issues, "H": prof["H"], "d_ref": prof["d_ref"], "Tcap": prof["Tcap"], "actor_episode_discount_weight": False})
    g = verify_gamma_half_life(prof["H"])
    gates.append({"gate": "GAMMA_HALF_LIFE", "passed": True, "gamma_H": g})
    cache_issues = list(cache_audit["issues"])
    gates.append({"gate": "CACHE_AUDIT", "passed": not cache_issues, "issues": cache_issues, "n_missing": cache_audit["n_missing"], "n_legal_empty": cache_audit["n_legal_empty"], "n_nonempty": cache_audit["n_nonempty"]})
    erratum = root / "runs/stage_0d/empty_patch_erratum.json"
    gates.append({"gate": "STAGE0D_ERRATUM", "passed": erratum.is_file(), "issues": [] if erratum.is_file() else ["0D empty-patch erratum missing"]})
    bundle = make_bundle(root)
    try:
        b2 = s1.run_corrected_dry_run(bundle, "B2", device, GATE_CASE, empty=True)
        write_json(out / "startup" / "b2_empty_prior.json", b2)
        b2_issues = []
        if b2["effective_prior_relation_count"] != 0:
            b2_issues.append("B2 effective prior not empty")
        if any((s.get("dp_rms") or 0) != 0 or (s.get("residual_abs") or 0) != 0 for s in b2["forward_structs"]):
            b2_issues.append("B2 DP/Delta not strictly 0")
        if not b2["logits_finite"]:
            b2_issues.append("B2 logits nonfinite")
        gates.append({"gate": "B2_EMPTY_PRIOR", "passed": not b2_issues, "issues": b2_issues, "source_relation_count": b2["source_cache_relation_count"]})
        issues.extend(b2_issues)
        full = s1.run_corrected_dry_run(bundle, "Full", device, GATE_CASE, force_original=True)
        write_json(out / "startup" / "full_original_prior.json", full)
        full_issues = []
        src_n = full["source_cache_relation_count"]
        eff_n = full["effective_prior_relation_count"]
        qualifying = [s for s in full["forward_structs"] if (not s["masked"]) and s["nominal_patch_nonempty"] and eff_n > 0]
        dp_hit = any(s["dp_nonzero"] for s in qualifying)
        delta_hit = any(s["residual_nonzero"] for s in qualifying)
        full_note = None
        if src_n <= 0:
            full_note = "N/A: source cache legally empty"
        elif not qualifying:
            full_note = "N/A: no qualifying nonempty-prior changed-patch candidate"
        elif not dp_hit or not delta_hit:
            full_issues.append("Full has opportunity but DP/Delta witness missing")
        if not full["logits_finite"]:
            full_issues.append("Full logits nonfinite")
        gates.append({"gate": "FULL_NATURAL_PRIOR", "passed": not full_issues, "issues": full_issues, "note": full_note, "source_relation_count": src_n, "effective_relation_count": eff_n, "qualifying_n": len(qualifying), "dp_nonzero_any": dp_hit, "delta_nonzero_any": delta_hit})
        issues.extend(full_issues)
        s1.seed_all(0)
        p1 = make_policy(bundle.template, "B2", device)
        s1.seed_all(0)
        p2 = make_policy(bundle.template, "Full", device)
        shared = {id(p) for p in p1.parameters()} & {id(p) for p in p2.parameters()}
        init_issues = ["shared B2/Full parameters"] if shared else []
        gates.append({"gate": "INDEPENDENT_INIT", "passed": not init_issues, "issues": init_issues})
        issues.extend(init_issues)
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
    blocked = any(not g["passed"] for g in gates)
    write_json(out / "startup" / "summary.json", {"gates": gates, "blocked": blocked, "time": utc_now(), "profile": {k: prof[k] for k in ("H", "d_ref", "Tcap", "Ncap", "actor_episode_discount_weight")}})
    return gates, blocked, prof, cache_audit


def save_ckpt(path, policy, optimizer, extra, rng_payload, prior_sampler, collector):
    from .torch_rl import save_checkpoint
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    extra = dict(extra)
    extra["H"] = suite_half_life()
    extra["rng_sidecar"] = str(path.with_suffix(".rng.json"))
    extra["collector"] = collector.state_dict()
    extra["actor_episode_discount_weight"] = False
    save_checkpoint(path, policy, optimizer, extra)
    sidecar = {"python_rng": s1.rng_sidecar(rng_payload)["python"], "numpy_rng": s1.rng_sidecar(rng_payload)["numpy"], "torch_rng": s1.rng_sidecar(rng_payload)["torch"], "cuda_rng": s1.rng_sidecar(rng_payload)["cuda"], "prior_sampler": s1.prior_state(prior_sampler), "extra": extra}
    write_json(path.with_suffix(".rng.json"), sidecar)
    return path


def select_checkpoint(eval_rows):
    usable = [r for r in eval_rows if r.get("mean_discounted_return") is not None]
    if not usable:
        return None, {"reason": "no eval windows"}
    annotated = []
    for i, row in enumerate(usable):
        window = usable[max(0, i - 2): i + 1]
        mean3 = float(sum(w["mean_discounted_return"] for w in window) / len(window))
        worst = float(min(w["mean_discounted_return"] for w in window))
        annotated.append({"index": i, "n_used": len(window), "mean3": mean3, "worst": worst, "transitions": row.get("skill_transitions"), "checkpoint": row.get("checkpoint"), "row": row})
    annotated.sort(key=lambda r: (-r["mean3"], r["worst"], r["transitions"] if r["transitions"] is not None else 10 ** 9))
    best = annotated[0]
    return best["checkpoint"], {"ranked": annotated, "selected": best, "fewer_than_3_windows": best["n_used"] < 3}


def smoke_gate(b2, full):
    reasons = []
    notes = []
    if (b2.get("complete_updates") or 0) < 8 or (full.get("complete_updates") or 0) < 8:
        reasons.append("fewer than 8 complete updates")
    def any_learned(job):
        later = [r for r in (job.get("eval") or []) if int(r.get("skill_transitions") or 0) > 0]
        return (job.get("train_success_episodes") or 0) > 0 or any((r.get("success_n") or 0) > 0 for r in later)
    if not any_learned(b2):
        reasons.append("B2 has no real policy success after training")
    if not any_learned(full):
        reasons.append("Full has no real policy success after training")
    if b2.get("NaN_n") or full.get("NaN_n") or b2.get("hard_fail") or full.get("hard_fail"):
        reasons.append("NaN/Inf or hard failure")
    if any((r.get("dp_rms_all") or 0) != 0 or (r.get("residual_rms_all") or 0) != 0 for r in (b2.get("train") or [])):
        reasons.append("B2 DP/Delta not strictly 0")
    full_opp = sum(int(r.get("dp_opportunity_n") or 0) for r in (full.get("train") or []))
    full_dp = any(int(r.get("dp_nonzero_n") or 0) > 0 for r in (full.get("train") or []))
    full_delta = any(int(r.get("delta_nonzero_n") or 0) > 0 for r in (full.get("train") or []))
    if full_opp == 0:
        notes.append("Full DP/Delta N/A: zero nonempty-prior changed-patch opportunities")
        reasons.append("Full DP/Delta N/A: zero nonempty-prior changed-patch opportunities")
    else:
        if not full_dp:
            reasons.append("Full DP has opportunity but no witness")
        if not full_delta:
            reasons.append("Full Delta has opportunity but no witness")
    return (not reasons), reasons, {"full_opportunity_n": full_opp, "notes": notes}


def start_episode(root, method, bundle, cases, split_index, sampler, collector, job_dir, eval_original=False):
    case = bundle.next_case(cases, 0)
    rec = split_index[case]
    require_case_cache(root, rec)
    snap = bundle.start_case(case)
    snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler, eval_original=eval_original)
    bundle.current_snapshot = snap
    collector.reset_episode(snap.env_id, snap.episode_id)
    cache_key = str(root / rec["cache_dir"])
    s1.append_jsonl(job_dir / "episode_priors.jsonl", {
        "env_id": prior.env_id,
        "episode_id": prior.episode_id,
        "audit_mode": prior.audit_mode,
        "original_hash": prior.original_hash,
        "effective_hash": prior.hash,
        "effective_relation_count": len(prior.edges),
        "source_cache_relation_count": source_n,
        "case_id": case,
        "method": method,
        "eval_original": eval_original,
    })
    return case, prior, source_n, cache_key


def eval_episodes(root, method, ckpt_path, cases, split_index, device, hashes_doc, out_eval, n_episodes=None, label="dev"):
    from .collector import Collector
    from .torch_rl import load_checkpoint
    bind_profile(root)
    rng_before = s1.capture_rng()
    bundle = make_bundle(root)
    s1.seed_all(0)
    policy = make_policy(bundle.template, method, device)
    load_checkpoint(ckpt_path, policy)
    policy.eval()
    collector = Collector(bundle, policy)
    rows = []
    success_n = 0
    returns = []
    try:
        for case in cases[:(n_episodes or len(cases))]:
            rec = split_index[case]
            require_case_cache(root, rec)
            snap = bundle.start_case(case)
            snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler=None, eval_original=True)
            collector.reset_episode(snap.env_id, snap.episode_id)
            G = 0.0
            steps = 0
            success = False
            reason = None
            success_seconds = None
            while True:
                t, result = collector.step(snap, deterministic=True)
                if t is None:
                    reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                    success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
                    break
                G += float(t.reward) * float(t.weight)
                steps += 1
                snap = t.next_snapshot
                ended = t.terminated or t.truncated
                success = bool(result.success) if not isinstance(result, dict) else bool(result.get("success"))
                reason = result.reason if not isinstance(result, dict) else result.get("reason")
                if success and success_seconds is None:
                    success_seconds = float(bundle.clock.now_seconds() - bundle.episode_start_seconds)
                if ended:
                    break
            if not success:
                success_seconds = None
            success_n += int(success)
            returns.append(G)
            rows.append({"case": case, "success": success, "reason": reason, "skills": steps, "discounted_return": G, "success_seconds": success_seconds, "prior_mode": prior.audit_mode, "source_cache_relation_count": source_n, "effective_prior_relation_count": len(prior.edges), "split": label})
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
        s1.restore_rng(rng_before)
    mean_G = float(sum(returns) / len(returns)) if returns else None
    payload = {
        "method": method,
        "checkpoint": str(ckpt_path),
        "success_n": success_n,
        "n": len(rows),
        "success_rate": (success_n / max(1, len(rows))) if rows else None,
        "mean_discounted_return": mean_G,
        "rows": rows,
        "hashes": hashes_doc,
        "isolated_rng": True,
        "H": suite_half_life(),
        "eval_action": "deterministic_argmax",
        "label": label,
    }
    write_json(out_eval, payload)
    return payload


def aggregate_train_buffer(method, trans_rows):
    agg = s1.aggregate_struct(trans_rows)
    qual = [r for r in trans_rows if r.get("qualifying_for_dp")]
    agg["dp_opportunity_n"] = len(qual)
    agg["dp_nonzero_n"] = sum(1 for r in qual if r["struct"]["dp_nonzero"])
    agg["delta_nonzero_n"] = sum(1 for r in qual if r["struct"]["residual_nonzero"])
    if method == "B2":
        agg["dp_opportunity_n"] = 0
    return agg


def maybe_eval_at_n(root, method, policy, trainer, collector, sampler, count, interaction_seconds, hashes_doc, job_dir, eval_cases, split_index, device, eval_rows, extra_base, done_ns):
    if count in done_ns:
        return
    if count != 0 and (count % EVAL_EVERY_N) != 0:
        return
    extra = dict(extra_base)
    extra.update({"update_count": extra_base.get("complete_updates"), "interaction_count": count, "interaction_seconds": interaction_seconds, "N": count, "T": interaction_seconds})
    ckpt = job_dir / "checkpoints" / ("n_%06d.pt" % count)
    save_ckpt(ckpt, policy, trainer.optimizer, extra, s1.capture_rng(), sampler, collector)
    log("%s eval N=%s" % (method, count))
    ev = eval_episodes(root, method, ckpt, eval_cases, split_index, device, hashes_doc, job_dir / ("eval_n_%06d.json" % count), n_episodes=DEV_EPISODES, label="dev")
    eval_rows.append({
        "update": extra_base.get("complete_updates"),
        "skill_transitions": count,
        "success_rate": ev["success_rate"],
        "success_n": ev["success_n"],
        "mean_discounted_return": ev["mean_discounted_return"],
        "checkpoint": str(ckpt),
    })
    s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
    done_ns.add(count)


def train_job(root, method, device, device_name, prof, hashes_doc, stamp, configsha8, max_updates, stop_after_updates=None, resume=False, num_envs=1):
    bind_profile(root)
    root = Path(root)
    job_dir = root / STAGE_DIR / "D0" / method / "seed_0" / ("%s_%s" % (stamp, configsha8))
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "checkpoints").mkdir(exist_ok=True)
    (job_dir / "plots").mkdir(exist_ok=True)
    split = prof["split"]
    cases = [r["case_id"] for r in split["train"]]
    eval_cases = [r["case_id"] for r in split["dev"]]
    split_index = {r["case_id"]: r for r in (split["train"] + split["dev"] + list(split.get("test") or []))}
    run_id = "v11_1A_D0_%s_s0_%s_%s" % (method, stamp, configsha8)
    cfg = {
        "planned_id": PLANNED[method],
        "run_id": run_id,
        "method": method,
        "seed": 0,
        "task": "D0",
        "H": prof["H"],
        "d_ref": prof["d_ref"],
        "Tcap": prof["Tcap"],
        "Ncap": prof["Ncap"],
        "actor_episode_discount_weight": False,
        "gamma_rule": "2**(-duration_seconds/H)",
        "runtime_path": prof["runtime_path"],
        "split_path": prof["split_path"],
        "task_deadline_seconds": prof["task_deadline_seconds"],
        "train_ids": cases,
        "dev_ids": eval_cases,
        "test_ids": prof["test_ids"],
        "hashes": hashes_doc,
        "old_output_dir_unused": str(OLD_STAGE_DIR),
        "old_D_REF_unused": 3.55,
        "resource_chunking": {
            "num_envs": int(num_envs),
            "note": "collection parallelism only; not a method/profile change",
        },
    }
    write_json(job_dir / "resolved_config.json", cfg)
    try:
        (job_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        (job_dir / "resolved_config.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_json(job_dir / "environment_snapshot.json", s1.environment_snapshot(device, device_name, 0))
    write_json(job_dir / "software_snapshot.json", s1.software_snapshot(root))
    write_json(job_dir / "manifest.json", {"run_id": run_id, "planned_id": PLANNED[method], "method": method, "seed": 0, "task": "D0", "hashes": hashes_doc, "H": prof["H"], "d_ref": prof["d_ref"], "Tcap": prof["Tcap"]})
    (job_dir / "tuning_history.jsonl").write_text("", encoding="utf-8")
    from .collector import Collector
    from .torch_rl import PPO, load_checkpoint
    from .prior import PriorSampler
    from .rl import Rollout
    s1.seed_all(0)
    bundle = make_bundle(root)
    policy = make_policy(bundle.template, method, device)
    trainer = PPO(policy)
    collector = Collector(bundle, policy)
    sampler = None if method == "B2" else PriorSampler(0)
    eval_rows = []
    train_rows = []
    resume_doc = {}
    if resume and (job_dir / "resume.json").is_file():
        resume_doc = json.loads((job_dir / "resume.json").read_text(encoding="utf-8"))
        ckpt_load = Path(resume_doc["checkpoint"])
        load_checkpoint(ckpt_load, policy, trainer.optimizer)
        rngp = json.loads(ckpt_load.with_suffix(".rng.json").read_text(encoding="utf-8"))
        s1.restore_rng({"python": rngp["python_rng"], "numpy": rngp["numpy_rng"], "torch": rngp["torch_rng"], "cuda": rngp["cuda_rng"]})
        if sampler is not None and rngp.get("prior_sampler"):
            st = rngp["prior_sampler"]
            try:
                sampler.load_state_dict({"rng": {k: (v[0], tuple(v[1]), v[2]) for k, v in (st.get("rng") or {}).items()}, "active": {}, "draw_count": st.get("draw_count") or 0})
            except Exception:
                pass
        if (job_dir / "eval_metrics.csv").exists():
            import csv
            with (job_dir / "eval_metrics.csv").open(encoding="utf-8") as f:
                eval_rows = list(csv.DictReader(f))
                for r in eval_rows:
                    for k in ("update", "skill_transitions", "success_n"):
                        if r.get(k) not in (None, ""):
                            r[k] = int(float(r[k]))
                    for k in ("success_rate", "mean_discounted_return"):
                        if r.get(k) not in (None, ""):
                            r[k] = float(r[k])
        if (job_dir / "train_metrics.csv").exists():
            import csv
            with (job_dir / "train_metrics.csv").open(encoding="utf-8") as f:
                train_rows = list(csv.DictReader(f))
    count = int(resume_doc.get("count") or 0)
    complete_updates = int(resume_doc.get("complete_updates") or 0)
    fragment_updates = int(resume_doc.get("fragment_updates") or 0)
    interaction_seconds = float(resume_doc.get("interaction_seconds") or 0.0)
    empty_episodes = 0
    train_success_episodes = int(resume_doc.get("train_success_episodes") or 0)
    zero_reward_episodes = int(resume_doc.get("zero_reward_episodes") or 0)
    original_ep = int(resume_doc.get("original_episode_n") or 0)
    absent_ep = int(resume_doc.get("absent_episode_n") or 0)
    nan_n = int(resume_doc.get("NaN_n") or 0)
    mask_error_n = 0
    ep_reward = 0.0
    case = prior = source_n = cache_key = None
    trans_buffer = []
    hard_fail = None
    resource_cap = None
    done_ns = set(int(r.get("skill_transitions")) for r in eval_rows if r.get("skill_transitions") not in (None, ""))
    target_updates = max_updates if stop_after_updates is None else min(max_updates, stop_after_updates)
    extra_base = {"method": method, "complete_updates": complete_updates, "fragment_updates": fragment_updates, **hashes_doc, "device": str(device), "H": prof["H"]}
    log("%s training start dir=%s target_complete_updates=%s num_envs=%s" % (method, job_dir, target_updates, int(num_envs)))
    rollout = Rollout()
    pool = None

    try:
        if 0 not in done_ns:
            extra0 = dict(extra_base)
            extra0.update({"update_count": 0, "interaction_count": 0, "interaction_seconds": 0.0, "N": 0, "T": 0.0})
            ckpt0 = job_dir / "checkpoints" / "n_000000.pt"
            save_ckpt(ckpt0, policy, trainer.optimizer, extra0, s1.capture_rng(), sampler, collector)
            ev0 = eval_episodes(root, method, ckpt0, eval_cases, split_index, device, hashes_doc, job_dir / "eval_n_000000.json", n_episodes=DEV_EPISODES, label="dev")
            eval_rows.append({"update": 0, "skill_transitions": 0, "success_rate": ev0["success_rate"], "success_n": ev0["success_n"], "mean_discounted_return": ev0["mean_discounted_return"], "checkpoint": str(ckpt0)})
            s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
            done_ns.add(0)
        if int(num_envs) > 1:
            from .stage1a_v11_parallel import EnvPool
            try:
                bundle.environment.close()
            except Exception:
                pass
            pool = EnvPool(root, method, int(num_envs), run_id)
            pool.set_weights(policy.state_dict())
            log("%s parallel collectors ready n=%s" % (method, int(num_envs)))

        def consume_parallel_episode(ep, do_update=True):
            nonlocal empty_episodes, original_ep, absent_ep, train_success_episodes, zero_reward_episodes
            case_id = ep["case_id"]
            s1.append_jsonl(job_dir / "episode_priors.jsonl", ep["prior"])
            if ep["prior_mode"] == "original":
                original_ep += 1
            else:
                absent_ep += 1
            if ep["empty"]:
                empty_episodes += 1
                s1.append_jsonl(job_dir / "decision_log.jsonl", {"method": method, "no_transition": True, "reason": ep.get("reason"), "complete_updates": complete_updates, "case_id": case_id, "empty_streak": empty_episodes, "worker_id": ep.get("worker_id")})
                if empty_episodes >= MAX_EMPTY:
                    raise BindingError("Runtime repeatedly exposes no executable skill")
            else:
                empty_episodes = 0
                for item in ep["items"]:
                    rec = item["rec"]
                    tstep = item["transition"]
                    s1.append_jsonl(job_dir / "transition_log.jsonl", rec)
                    s1.append_jsonl(job_dir / "decision_log.jsonl", {"decision_id": rec["decision_id"], "selected_candidate_id": rec["selected_candidate_id"], "mask_hash": rec["mask_hash"], "prior_mode": rec["prior_mode"], "H": rec["H"], "worker_id": rec.get("worker_id")})
                    rollout.append(tstep)
                    trans_buffer.append(rec)
                    nonlocal_count_add(tstep, rec)
                    extra_base["complete_updates"] = complete_updates
                    extra_base["fragment_updates"] = fragment_updates
                    maybe_eval_at_n(root, method, policy, trainer, collector, sampler, count, interaction_seconds, hashes_doc, job_dir, eval_cases, split_index, device, eval_rows, extra_base, done_ns)
            if ep["success"]:
                train_success_episodes += 1
            if ep["ep_reward"] == 0:
                zero_reward_episodes += 1
            s1.append_jsonl(job_dir / "episode_log.jsonl", {"method": method, "case_id": case_id, "success": ep["success"], "reward": ep["ep_reward"], "prior_mode": ep.get("prior_mode"), "transitions_so_far": count, "T": interaction_seconds, "worker_id": ep.get("worker_id")})
            if do_update:
                return maybe_update()
            return count >= prof["Ncap"] or interaction_seconds >= prof["Tcap"]

        def nonlocal_count_add(tstep, rec):
            nonlocal count, interaction_seconds, nan_n, hard_fail
            count += 1
            interaction_seconds += tstep.duration
            if not rec["logits_finite"] or not rec["value_finite"] or not math.isfinite(rec["old_logp"]):
                nan_n += 1
                hard_fail = {"error": "NaN/Inf in transition", "rec": rec}
                write_json(job_dir / "failure.json", hard_fail)
                raise DataIntegrityError("NaN/Inf in transition")

        def maybe_update():
            nonlocal complete_updates, fragment_updates, nan_n, resource_cap, trans_buffer
            extra_base["complete_updates"] = complete_updates
            extra_base["fragment_updates"] = fragment_updates
            maybe_eval_at_n(root, method, policy, trainer, collector, sampler, count, interaction_seconds, hashes_doc, job_dir, eval_cases, split_index, device, eval_rows, extra_base, done_ns)
            hit_cap = count >= prof["Ncap"] or interaction_seconds >= prof["Tcap"]
            if pool is not None and (len(rollout.transitions) >= ROLLOUT_N or hit_cap):
                while pool.busy:
                    consume_parallel_episode(pool._recv_one(), do_update=False)
                pool.idle = list(range(pool.num_envs))
                hit_cap = count >= prof["Ncap"] or interaction_seconds >= prof["Tcap"]
            n_roll = len(rollout.transitions)
            if n_roll >= ROLLOUT_N or (hit_cap and n_roll > 0):
                complete = n_roll >= ROLLOUT_N
                use_n = ROLLOUT_N if complete else n_roll
                extra_ts = list(rollout.transitions[use_n:])
                extra_buf = list(trans_buffer[use_n:])
                rollout.transitions = list(rollout.transitions[:use_n])
                used_buf = list(trans_buffer[:use_n])
                kind = "complete" if complete else "fragment"
                log("%s PPO %s update transitions=%s N=%s T=%s num_envs=%s" % (method, kind, use_n, count, interaction_seconds, int(num_envs)))
                logs = trainer.update(rollout)
                if complete:
                    complete_updates += 1
                else:
                    fragment_updates += 1
                agg = aggregate_train_buffer(method, used_buf)
                tot = [row.get("total") for row in logs]
                grad = [row.get("grad_norm") for row in logs]
                if any((x is None) or (not math.isfinite(x)) for x in tot + grad):
                    nan_n += 1
                    raise DataIntegrityError("Nonfinite PPO log")
                row = {
                    "update": complete_updates,
                    "fragment_updates": fragment_updates,
                    "complete": complete,
                    "valid_transitions": count,
                    "rollout_n": use_n,
                    "success_episodes": train_success_episodes,
                    "zero_reward_episodes": zero_reward_episodes,
                    "policy_loss": float(sum(r["actor"] for r in logs) / len(logs)),
                    "value_loss": float(sum(r["v"] for r in logs) / len(logs)),
                    "q_loss": float(sum(r["q"] for r in logs) / len(logs)),
                    "total_loss": float(sum(r["total"] for r in logs) / len(logs)),
                    "grad_norm": float(sum(r["grad_norm"] for r in logs) / len(logs)),
                    "NaN_n": nan_n,
                    "original_episode_n": original_ep,
                    "absent_episode_n": absent_ep,
                    "interaction_seconds": interaction_seconds,
                    "H": suite_half_life(),
                    "actor_episode_discount_weight": False,
                    "num_envs": int(num_envs),
                    **agg,
                }
                train_rows.append(row)
                s1.csv_write(job_dir / "train_metrics.csv", train_rows)
                extra = dict(extra_base)
                extra.update({"complete_updates": complete_updates, "fragment_updates": fragment_updates, "interaction_count": count, "interaction_seconds": interaction_seconds, "N": count, "T": interaction_seconds, "num_envs": int(num_envs)})
                ckpt_u = job_dir / "checkpoints" / (("update_complete_%s.pt" % complete_updates) if complete else ("update_fragment_%s.pt" % fragment_updates))
                save_ckpt(ckpt_u, policy, trainer.optimizer, extra, s1.capture_rng(), sampler, collector)
                trans_buffer = extra_buf
                for t2 in extra_ts:
                    rollout.append(t2)
                write_json(job_dir / "resume.json", {"complete_updates": complete_updates, "fragment_updates": fragment_updates, "count": count, "interaction_seconds": interaction_seconds, "train_success_episodes": train_success_episodes, "zero_reward_episodes": zero_reward_episodes, "original_episode_n": original_ep, "absent_episode_n": absent_ep, "NaN_n": nan_n, "checkpoint": str(ckpt_u), "num_envs": int(num_envs)})
                if pool is not None:
                    pool.set_weights(policy.state_dict())
            if hit_cap:
                if complete_updates < 8:
                    resource_cap = "RESOURCE_CAP_BEFORE_SMOKE_GATE"
            return hit_cap

        while complete_updates < target_updates and count < prof["Ncap"] and interaction_seconds < prof["Tcap"]:
            if pool is not None:
                try:
                    ep = pool.next_episode(cases, stop=lambda: complete_updates >= target_updates or count >= prof["Ncap"] or interaction_seconds >= prof["Tcap"])
                except Exception as exc:
                    hard_fail = {"error": str(exc), "traceback": traceback.format_exc(), "complete_updates": complete_updates, "count": count}
                    write_json(job_dir / "failure.json", hard_fail)
                    s1.append_jsonl(job_dir / "failures.jsonl", hard_fail)
                    raise
                if ep is None:
                    break
                if consume_parallel_episode(ep, do_update=True):
                    if complete_updates < 8 and (count >= prof["Ncap"] or interaction_seconds >= prof["Tcap"]):
                        resource_cap = resource_cap or "RESOURCE_CAP_BEFORE_SMOKE_GATE"
                    if complete_updates >= target_updates or count >= prof["Ncap"] or interaction_seconds >= prof["Tcap"]:
                        break
                continue
            if bundle.current_snapshot is None:
                case, prior, source_n, cache_key = start_episode(root, method, bundle, cases, split_index, sampler, collector, job_dir)
                if prior.audit_mode == "original":
                    original_ep += 1
                else:
                    absent_ep += 1
                ep_reward = 0.0
            snap = bundle.current_snapshot
            try:
                tstep, result = collector.step(snap, deterministic=False)
            except Exception as exc:
                hard_fail = {"error": str(exc), "traceback": traceback.format_exc(), "complete_updates": complete_updates, "count": count, "case": case}
                write_json(job_dir / "failure.json", hard_fail)
                s1.append_jsonl(job_dir / "failures.jsonl", hard_fail)
                raise
            ended = False
            success = False
            if tstep is None:
                empty_episodes += 1
                reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                s1.append_jsonl(job_dir / "decision_log.jsonl", {"method": method, "no_transition": True, "reason": reason, "complete_updates": complete_updates, "case_id": case, "empty_streak": empty_episodes})
                ended = True
                if empty_episodes >= MAX_EMPTY:
                    raise BindingError("Runtime repeatedly exposes no executable skill")
            else:
                empty_episodes = 0
                rec = s1.compact_transition(method, 0, case, tstep, result, collector.last_output, collector.last_execution, prior.audit_mode, prior.original_hash, source_n, cache_key, run_id)
                rec["H"] = suite_half_life()
                rec["actor_episode_discount_weight"] = False
                s1.append_jsonl(job_dir / "transition_log.jsonl", rec)
                s1.append_jsonl(job_dir / "decision_log.jsonl", {"decision_id": rec["decision_id"], "selected_candidate_id": rec["selected_candidate_id"], "mask_hash": rec["mask_hash"], "prior_mode": rec["prior_mode"], "H": rec["H"]})
                rollout.append(tstep)
                trans_buffer.append(rec)
                nonlocal_count_add(tstep, rec)
                bundle.current_snapshot = tstep.next_snapshot
                ended = tstep.terminated or tstep.truncated
                success = bool(result.success) if not isinstance(result, dict) else bool(result.get("success"))
                ep_reward += tstep.reward
            if ended:
                if success:
                    train_success_episodes += 1
                if ep_reward == 0:
                    zero_reward_episodes += 1
                s1.append_jsonl(job_dir / "episode_log.jsonl", {"method": method, "case_id": case, "success": success, "reward": ep_reward, "prior_mode": None if prior is None else prior.audit_mode, "transitions_so_far": count, "T": interaction_seconds})
                bundle.current_snapshot = None
            if maybe_update():
                break

        extra_final = dict(extra_base)
        extra_final.update({"complete_updates": complete_updates, "fragment_updates": fragment_updates, "interaction_count": count, "interaction_seconds": interaction_seconds, "N": count, "T": interaction_seconds, "final": True})
        ckpt_final = job_dir / "checkpoints" / "final.pt"
        save_ckpt(ckpt_final, policy, trainer.optimizer, extra_final, s1.capture_rng(), sampler, collector)
        if count not in done_ns:
            evf = eval_episodes(root, method, ckpt_final, eval_cases, split_index, device, hashes_doc, job_dir / "eval_final.json", n_episodes=DEV_EPISODES, label="dev")
            eval_rows.append({"update": complete_updates, "skill_transitions": count, "success_rate": evf["success_rate"], "success_n": evf["success_n"], "mean_discounted_return": evf["mean_discounted_return"], "checkpoint": str(ckpt_final)})
            s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
        selected, sel_doc = select_checkpoint(eval_rows)
        write_json(job_dir / "checkpoint_selection.json", sel_doc)
        summary = {
            "method": method,
            "planned_id": PLANNED[method],
            "run_id": run_id,
            "job_dir": str(job_dir),
            "complete_updates": complete_updates,
            "fragment_updates": fragment_updates,
            "valid_transitions": count,
            "interaction_seconds": interaction_seconds,
            "train_success_episodes": train_success_episodes,
            "step0_dev_success": None if not eval_rows else eval_rows[0].get("success_rate"),
            "final_dev_success": None if not eval_rows else eval_rows[-1].get("success_rate"),
            "NaN_n": nan_n,
            "mask_error_n": mask_error_n,
            "original_episode_n": original_ep,
            "absent_episode_n": absent_ep,
            "hard_fail": hard_fail,
            "resource_cap": resource_cap,
            "selected_checkpoint": selected,
            "H": suite_half_life(),
            "d_ref": prof["d_ref"],
            "Tcap": prof["Tcap"],
            "eval": eval_rows,
            "train": train_rows,
        }
        write_json(job_dir / "job_summary.json", summary)
        (job_dir / "run_summary.md").write_text("\n".join([
            "# %s seed=0" % method, "",
            "run_id: %s" % run_id,
            "complete_updates: %s fragment_updates: %s" % (complete_updates, fragment_updates),
            "N/T: %s / %s" % (count, interaction_seconds),
            "train_success_episodes: %s" % train_success_episodes,
            "resource_cap: %s" % resource_cap,
            "selected_checkpoint: %s" % selected,
            "",
        ]), encoding="utf-8")
        return summary
    finally:
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
        try:
            bundle.environment.close()
        except Exception:
            pass


def run_test_ids(root, method, ckpt, prof, split_index, device, hashes_doc, job_dir):
    test_ids = list(prof["test_ids"] or [])
    if not test_ids:
        doc = {"status": "NOT_CREATED", "reason": "D0_stage_1a.json has no registered test-ID list (test_isolated=true); dev cannot substitute", "n_requested": TEST_EPISODES, "n_available": 0}
        write_json(job_dir / "test_id_eval.json", doc)
        return doc
    if len(test_ids) < TEST_EPISODES:
        doc = {"status": "NOT_CREATED", "reason": "registered test-IDs %s < required %s" % (len(test_ids), TEST_EPISODES), "n_requested": TEST_EPISODES, "n_available": len(test_ids), "test_ids": test_ids}
        write_json(job_dir / "test_id_eval.json", doc)
        return doc
    ev = eval_episodes(root, method, ckpt, test_ids[:TEST_EPISODES], split_index, device, hashes_doc, job_dir / "test_id_eval.json", n_episodes=TEST_EPISODES, label="test")
    ev["status"] = "PASS"
    write_json(job_dir / "test_id_eval.json", ev)
    return ev


def write_stage_report(root, payload):
    root = Path(root)
    lines = [
        "# Stage 1A — One-task Smoke (Plan v1.1 / Method 2.1.1)",
        "",
        "Status: `" + str(payload.get("status")) + "`.",
        "",
        "This report is the new-profile ledger. It does not overwrite `experiments/part_1_smoke/stage_1a/`.",
        "",
        "## Startup repairs",
        "",
        "- 0D empty-patch identity: " + str(payload.get("stage0d_erratum")),
        "- deadline production tests: " + str(payload.get("deadline_tests")),
        "- runner: `python -m cp_disr.cli stage-1a-v11-run` using `runtime_manifest_v211.yaml`",
        "",
        "## Frozen profile",
        "",
        "- H = " + str(payload.get("H")),
        "- D0 d_ref = " + str(payload.get("d_ref")),
        "- Tcap = Ncap * d_ref = " + str(payload.get("Tcap")),
        "- actor_episode_discount_weight = false",
        "- Gamma = 2**(-duration_seconds/H)",
        "",
        "## Jobs",
        "",
    ]
    for method in ("B2", "Full"):
        job = (payload.get("jobs") or {}).get(method) or {}
        lines.extend([
            "### " + method,
            "",
            "- planned_id: " + str(job.get("planned_id")),
            "- run_id: " + str(job.get("run_id")),
            "- N/T: %s / %s" % (job.get("valid_transitions"), job.get("interaction_seconds")),
            "- complete/fragment updates: %s / %s" % (job.get("complete_updates"), job.get("fragment_updates")),
            "- train_success_episodes: %s" % job.get("train_success_episodes"),
            "- step0_dev_success: %s" % job.get("step0_dev_success"),
            "- selected_checkpoint: %s" % job.get("selected_checkpoint"),
            "- test-ID: %s" % ((job.get("test") or {}).get("status")),
            "",
        ])
    lines.extend([
        "## Smoke gate",
        "",
        "- passed: " + str(payload.get("smoke_passed")),
        "- reasons: " + str(payload.get("smoke_reasons")),
        "- resource_cap: " + str(payload.get("resource_cap")),
        "",
        "## Cost",
        "",
        json.dumps(payload.get("cost") or {}, ensure_ascii=False, indent=2),
        "",
        "Next stage suggestion only: 1B if smoke incomplete; otherwise wait for explicit Stage 1B/2A authorization.",
        "",
    ])
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")


def cmd_stage_1a_v11_run(root, gpu=0, only_startup_gates=False, resume=False, max_updates=16, stop_after_updates=None, method=None, stamp=None, configsha=None, skip_startup_gates=False, num_envs=1):
    root = Path(root)
    os.chdir(root)
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        device = torch.device("cuda", int(gpu))
        device_name = torch.cuda.get_device_name(int(gpu))
    else:
        device = torch.device("cpu")
        device_name = "cpu"
    prof = bind_profile(root)
    git_hash = git_commit(root)
    stamp = stamp or utc_stamp()
    configsha8 = configsha or frozen_configsha8(prof, git_hash)
    stage_out = root / STAGE_DIR / "D0"
    stage_out.mkdir(parents=True, exist_ok=True)
    caches = s1.cache_fingerprint(root)
    hashes_doc = s1.hashes(root, caches)
    hashes_doc.update({"git_commit": git_hash, "base_commit": BASE_COMMIT, "runtime_manifest": str(RUNTIME_REL), "H": prof["H"], "d_ref": prof["d_ref"], "Tcap": prof["Tcap"]})
    status = {
        "stage": "1A",
        "plan_version": "1.1",
        "method_version": "2.1.1",
        "document_version": "3.1",
        "status": "RUNNING",
        "started_at": utc_now(),
        "planned_ids": PLANNED,
        "stamp": stamp,
        "configsha8": configsha8,
        "H": prof["H"],
        "d_ref": prof["d_ref"],
        "Tcap": prof["Tcap"],
        "git_hash": git_hash,
        "base_commit": BASE_COMMIT,
        "num_envs": int(num_envs or 1),
        "resource_chunking": True,
    }
    if method is None:
        write_json(root / STATUS_PATH, status)
    log("startup hard checks")
    if skip_startup_gates:
        gates, blocked, cache_audit = [], False, {"n": None, "n_missing": None, "n_legal_empty": None, "n_nonempty": None}
        log("skip_startup_gates; using frozen profile H=%s d_ref=%s Tcap=%s" % (prof["H"], prof["d_ref"], prof["Tcap"]))
    else:
        gates, blocked, prof, cache_audit = startup_hard_checks(root, stage_out, device)
    if not skip_startup_gates:
        status["startup_gates"] = gates
        status["cache_audit"] = {"n": cache_audit["n"], "n_missing": cache_audit["n_missing"], "n_legal_empty": cache_audit["n_legal_empty"], "n_nonempty": cache_audit["n_nonempty"]}
    if blocked:
        status.update({"status": "BLOCKED", "completed_at": utc_now(), "execution_reason": "STARTUP_HARD_CHECK_FAILED"})
        write_json(root / STATUS_PATH, status)
        write_stage_report(root, {"status": "BLOCKED", "H": prof["H"], "d_ref": prof["d_ref"], "Tcap": prof["Tcap"], "jobs": {}, "smoke_passed": False, "smoke_reasons": ["startup blocked"], "stage0d_erratum": (root / "runs/stage_0d/empty_patch_erratum.json").is_file(), "deadline_tests": "see pytest output", "cost": {"new_vlm_calls": 0}})
        log("startup hard checks FAILED; no PPO")
        return {"status": "BLOCKED", "gates": gates}
    if only_startup_gates:
        status.update({"status": "RUNNING", "execution_reason": "STARTUP_GATES_PASS_WAITING_TRAIN"})
        write_json(root / STATUS_PATH, status)
        log("startup hard checks PASS; only_startup_gates")
        return {"status": "STARTUP_GATES_PASS", "gates": gates, "stamp": stamp, "configsha8": configsha8, "profile": {k: prof[k] for k in ("H", "d_ref", "Tcap", "Ncap")}}
    methods = [method] if method in ("B2", "Full") else ["B2", "Full"]
    jobs = {}
    reasons = []
    extra = {}
    final_status = "NEEDS_RERUN"
    resource_cap = None
    try:
        for target in (8, 12, 16):
            if target > int(max_updates):
                break
            if stop_after_updates is not None and target > int(stop_after_updates):
                break
            for m in methods:
                done = 0 if m not in jobs else (jobs[m].get("complete_updates") or 0)
                if done < target:
                    jobs[m] = train_job(root, m, device, device_name, prof, hashes_doc, stamp, configsha8, int(max_updates), stop_after_updates=target, resume=resume or done > 0, num_envs=int(num_envs or 1))
                    resource_cap = resource_cap or jobs[m].get("resource_cap")
            if all(m in jobs for m in ("B2", "Full")):
                ok, reasons, extra = smoke_gate(jobs["B2"], jobs["Full"])
                log("smoke at %s complete updates: %s %s" % (target, ok, reasons))
                if ok:
                    final_status = "PASS"
                    break
            if resource_cap == "RESOURCE_CAP_BEFORE_SMOKE_GATE":
                final_status = "NEEDS_RERUN"
                reasons.append("RESOURCE_CAP_BEFORE_SMOKE_GATE")
                break
        else:
            if all(m in jobs for m in ("B2", "Full")):
                ok, reasons, extra = smoke_gate(jobs["B2"], jobs["Full"])
                final_status = "PASS" if ok else "NEEDS_RERUN"
            else:
                reasons = ["incomplete authorized jobs"]
                extra = {}
                final_status = "NEEDS_RERUN"
    except Exception as exc:
        reasons = [str(exc)]
        extra = {}
        write_json(stage_out / "training_exception.json", {"error": str(exc), "traceback": traceback.format_exc()})
        final_status = "NEEDS_RERUN"
        log("training exception: %s" % exc)
    split_index = {r["case_id"]: r for r in (prof["split"]["train"] + prof["split"]["dev"] + list(prof["split"].get("test") or []))}
    if final_status == "PASS":
        for m, job in jobs.items():
            ckpt = job.get("selected_checkpoint")
            if ckpt:
                job["test"] = run_test_ids(root, m, ckpt, prof, split_index, device, hashes_doc, Path(job["job_dir"]))
            else:
                job["test"] = {"status": "NOT_CREATED", "reason": "no selected checkpoint"}
    else:
        for job in jobs.values():
            job.setdefault("test", {"status": "NOT_CREATED", "reason": "smoke not passed; test-ID not opened"})
    payload = {
        "status": final_status,
        "execution_reason": None if final_status == "PASS" else "; ".join(reasons),
        "H": prof["H"],
        "d_ref": prof["d_ref"],
        "Tcap": prof["Tcap"],
        "jobs": jobs,
        "smoke_passed": final_status == "PASS",
        "smoke_reasons": reasons,
        "resource_cap": resource_cap,
        "stage0d_erratum": str(root / "runs/stage_0d/empty_patch_erratum.json"),
        "deadline_tests": "tests/test_deadline_semantics.py",
        "cost": {"new_vlm_calls": 0, "new_rl_jobs": len(jobs), "startup_diagnostics": True},
        "stamp": stamp,
        "configsha8": configsha8,
        "git_hash": git_hash,
        "next_stage_suggestion": "1B" if final_status != "PASS" else "wait for explicit 1B/2A authorization",
    }
    write_json(stage_out / "combined_result.json", payload)
    write_stage_report(root, payload)
    status.update({"status": final_status, "completed_at": utc_now(), "jobs": {m: {"planned_id": PLANNED[m], "run_id": (jobs.get(m) or {}).get("run_id"), "complete_updates": (jobs.get(m) or {}).get("complete_updates"), "N": (jobs.get(m) or {}).get("valid_transitions"), "T": (jobs.get(m) or {}).get("interaction_seconds")} for m in methods}, "issues": reasons, "next_stage": payload["next_stage_suggestion"]})
    if method is None:
        write_json(root / STATUS_PATH, status)
    else:
        write_json(stage_out / ("job_%s_status.json" % method), status)
    log("Stage 1A v11 finished status=%s" % final_status)
    return payload

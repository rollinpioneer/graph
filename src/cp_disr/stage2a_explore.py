"""Stage 2A real PPO exploration runner. Reuses Stage 1A production Policy/Collector/PPO."""
from __future__ import annotations

import csv, hashlib, json, math, os, pickle, random, sys, time, traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import BindingError, DataIntegrityError, canonical, digest
from .graph import four_views
from .neural import Policy
from .collector import Collector
from .rl import Rollout, gamma
from .torch_rl import PPO, load_checkpoint, save_checkpoint
from .persistence import GenerationStore
from .prior import PriorSampler, EpisodePrior
from .platforms.libero.runtime_factory import create_task_runtime, PREDICATES, TASK_OBJECTS
from .stage2a_runner import planned_jobs, TRANSITIONS, UPDATES, UPDATE_EVERY, CHECKPOINTS
from . import stage1a_smoke as s1

STAGE_DIR = Path("experiments/part_2_exploration/stage_2a")
STATUS_PATH = Path("experiments/stage_status/stage_2a.json")
GATE_DIR = STAGE_DIR / "startup_gate"
MANIFEST_REL = Path("experiments/manifests/stage_2a_runtime_manifest.yaml")
EVAL_EVERY = 8
MAX_EMPTY = 8
OBS_DIM = 48
CAND_DIM = 8
B_PRIOR = 0.5
BASE_COMMIT = "9b23fdf468ea3f9701720a44e022635f9e16385b"

def _cache_rows(root):
    rows = []
    base = root / "experiments/vlm_cache/stage_2a"
    for split in ("train", "dev"):
        d = base / split
        if not d.is_dir():
            continue
        for key in sorted(pp.name for pp in d.iterdir() if pp.is_dir()):
            folder = d / key
            rec = {"split": split, "cache_key": key}
            for name in ("COMPLETE", "manifest.json", "final_edges.json", "accepted_relations.json"):
                fp = folder / name
                rec[name] = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else None
            rows.append(rec)
    return rows



def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print("[stage2a] " + str(msg), flush=True)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(obj) + "\n", encoding="utf-8")


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical(obj) + "\n")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(root):
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None


def load_status(root):
    path = root / STATUS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def save_status(root, payload):
    path = root / STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(payload) + "\n", encoding="utf-8")


def exact_d_ref(manifest, task_id):
    return float(manifest["runtime"]["reference_skill_seconds_by_task"][task_id])


def physical_cap(manifest, task_id):
    return 1.0 * float(TRANSITIONS) * exact_d_ref(manifest, task_id)


def model_kwargs(template, method, task_id):
    objects = TASK_OBJECTS[task_id]
    actions = sorted({c.name for c in template.contracts})
    predicates = sorted(PREDICATES)
    types = sorted(set(objects.values()))
    return dict(actions=actions, predicates=predicates, types=types, observation_dim=OBS_DIM, candidate_dim=CAND_DIM, method=method, B=B_PRIOR)


def make_bundle(root, task_id):
    manifest = load_yaml(root / MANIFEST_REL)
    manifest = dict(manifest)
    runtime = dict(manifest["runtime"])
    runtime["active_task_id"] = task_id
    manifest["runtime"] = runtime
    return create_task_runtime(manifest, task_id)


def make_policy(template, method, task_id, device):
    return Policy(**model_kwargs(template, method, task_id)).to(device)


def apply_episode_prior(method, bundle, snapshot, sampler, eval_mode=False):
    source_n = len(bundle.original_prior_edges)
    original = tuple(tuple(e) for e in bundle.original_prior_edges)
    original_hash = digest(original)
    if method in ("B0", "B2"):
        snap = s1.empty_prior(snapshot)
        prior = EpisodePrior(snap.env_id, snap.episode_id, (), original_hash, digest(()), "absent")
        return snap, prior, source_n
    if eval_mode:
        snap = s1.set_prior(snapshot, original)
        prior = EpisodePrior(snap.env_id, snap.episode_id, original, original_hash, digest(original), "original")
        return snap, prior, source_n
    prior = sampler.start(snapshot.env_id, snapshot.episode_id, original)
    snap = s1.set_prior(snapshot, prior.edges)
    return snap, prior, source_n



def _latest_ckpt(job_dir):
    ckpts = sorted((job_dir / "checkpoints").glob("update_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
    latest = job_dir / "checkpoints" / "latest.pt"
    if latest.exists():
        return latest
    return ckpts[-1] if ckpts else None

def load_resume_if_any(job_dir, policy, trainer, sampler, collector, bundle):
    resume_path = job_dir / "resume.json"
    if not resume_path.is_file():
        return 0, 0, 0, 0.0, 0
    doc = json.loads(resume_path.read_text(encoding="utf-8"))
    ckpt = _latest_ckpt(job_dir)
    if ckpt is None:
        return 0, 0, 0, 0.0, 0
    load_checkpoint(ckpt, policy, trainer.optimizer)
    rngp_path = ckpt.with_suffix(".rng.json")
    if rngp_path.is_file():
        rngp = json.loads(rngp_path.read_text(encoding="utf-8"))
        s1.restore_rng({"python": rngp["python_rng"], "numpy": rngp["numpy_rng"], "torch": rngp["torch_rng"], "cuda": rngp.get("cuda_rng")})
        if sampler is not None:
            _load_prior(sampler, rngp.get("prior_sampler"))
        if collector is not None and rngp.get("collector"):
            collector.load_discount_success(rngp["collector"])
    ep = ckpt.with_suffix(".episode.pkl")
    if ep.is_file():
        with ep.open("rb") as f:
            blob = pickle.load(f)
        if blob.get("sim") is not None and bundle is not None:
            try:
                restore_sim(bundle, blob["sim"])
                bundle.current_snapshot = blob.get("snapshot")
                if blob.get("prefixes") is not None:
                    collector.prefixes = blob["prefixes"]
                if blob.get("weights") is not None:
                    collector.weights = blob["weights"]
                if blob.get("success_seen") is not None:
                    collector.success_seen = blob["success_seen"]
                if blob.get("original_prior_edges") is not None:
                    bundle.original_prior_edges = blob["original_prior_edges"]
            except Exception as exc:
                boundary = {"resume_boundary": "sim_restore_failed", "error": str(exc), "discarded": "in_flight_episode", "extra_interaction": True}
                s1.write_json(job_dir / "resume_boundary.json", boundary)
                bundle.current_snapshot = None
    return int(doc.get("updates") or 0), int(doc.get("count") or 0), int(doc.get("optimizer_steps") or 0), float(doc.get("interaction_seconds") or 0.0), int(doc.get("train_success_episodes") or 0)

def hashes(root, task_id, method, seed, caches):
    split = root / f"configs/splits/{task_id}_stage_2a.json"
    contract = root / "configs/runtime/stage_2a_contract_registry.yaml"
    runtime = root / MANIFEST_REL
    return {
        "config_hash": digest({"budget": TRANSITIONS, "rollout": UPDATE_EVERY, "seed": seed, "method": method, "task": task_id}),
        "runtime_hash": file_sha(runtime),
        "contract_hash": file_sha(contract),
        "split_hash": file_sha(split),
        "cache_hash": s1.fingerprint_digest(_cache_rows(root)),
        "git_commit": git_commit(root),
        "source_commit": BASE_COMMIT,
    }


def capture_sim(bundle):
    env = bundle.environment
    sim = env.sim
    payload = {
        "qpos": np.array(sim.data.qpos, copy=True),
        "qvel": np.array(sim.data.qvel, copy=True),
        "time": float(sim.data.time),
        "clock_origin": float(bundle.clock._origin),
        "episode_start_seconds": float(bundle.episode_start_seconds),
        "n": int(bundle._n),
        "episode_id": bundle.snapshot_builder.episode_id,
        "evaluator_rewarded": bool(bundle.evaluator._rewarded),
        "evaluator_success_time": bundle.evaluator._success_time,
        "deadline": float(bundle.evaluator.deadline),
        "task_id": bundle.task_id,
    }
    try:
        payload["ctrl"] = np.array(sim.data.ctrl, copy=True)
    except Exception:
        payload["ctrl"] = None
    return payload


def restore_sim(bundle, payload):
    env = bundle.environment
    sim = env.sim
    sim.data.qpos[:] = payload["qpos"]
    sim.data.qvel[:] = payload["qvel"]
    sim.data.time = float(payload["time"])
    if payload.get("ctrl") is not None:
        try:
            sim.data.ctrl[:] = payload["ctrl"]
        except Exception:
            pass
    sim.forward()
    bundle.clock._origin = float(payload["clock_origin"])
    bundle.episode_start_seconds = float(payload["episode_start_seconds"])
    bundle._n = int(payload["n"])
    bundle.snapshot_builder.episode_id = payload["episode_id"]
    bundle.evaluator._rewarded = bool(payload["evaluator_rewarded"])
    bundle.evaluator._success_time = payload["evaluator_success_time"]
    bundle.evaluator.deadline = float(payload["deadline"])


def save_full_checkpoint(path, policy, optimizer, extra, rng_payload, prior_sampler, collector, bundle, case_id, snapshot, mid_episode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    manifest = dict(extra)
    manifest["rng_sidecar"] = str(path.with_suffix(".rng.json"))
    manifest["episode_sidecar"] = str(path.with_suffix(".episode.pkl"))
    save_checkpoint(path, policy, optimizer, manifest)
    sidecar = {
        "python_rng": s1.rng_sidecar(rng_payload)["python"],
        "numpy_rng": s1.rng_sidecar(rng_payload)["numpy"],
        "torch_rng": s1.rng_sidecar(rng_payload)["torch"],
        "cuda_rng": s1.rng_sidecar(rng_payload)["cuda"],
        "prior_sampler": None if prior_sampler is None else _prior_state(prior_sampler),
        "collector": None if collector is None else collector.state_dict(),
        "case_id": case_id,
        "case_scheduler_n": None if bundle is None else bundle._n,
        "mid_episode": bool(mid_episode),
        "extra": extra,
    }
    s1.write_json(path.with_suffix(".rng.json"), sidecar)
    blob = {
        "snapshot": snapshot,
        "sim": None if bundle is None else capture_sim(bundle),
        "prefixes": None if collector is None else collector.prefixes,
        "weights": None if collector is None else collector.weights,
        "success_seen": None if collector is None else collector.success_seen,
        "original_prior_edges": None if bundle is None else bundle.original_prior_edges,
    }
    with path.with_suffix(".episode.pkl").open("wb") as f:
        pickle.dump(blob, f, protocol=4)
    # Phase A may opt into immutable generations without changing legacy eval aliases.
    store_root = os.environ.get("CP_DISR_GENERATION_STORE")
    if store_root:
        store = GenerationStore(store_root)
        store.publish({
            "model.pt": path,
            "manifest.json": manifest,
            "rng.json": sidecar,
            "episode.pkl": path.with_suffix(".episode.pkl"),
        }, manifest)
    return path


def _prior_state(sampler):
    raw = sampler.state_dict()
    rng = {}
    for k, st in raw["rng"].items():
        rng[k] = [st[0], list(st[1]), st[2]]
    active = {}
    for k, prior in raw["active"].items():
        active[k] = {
            "env_id": prior.env_id,
            "episode_id": prior.episode_id,
            "edges": [list(e) for e in prior.edges],
            "original_hash": prior.original_hash,
            "hash": prior.hash,
            "audit_mode": prior.audit_mode,
        }
    return {"rng": rng, "active": active, "draw_count": raw["draw_count"]}


def _load_prior(sampler, payload):
    if not payload:
        return
    streams = {}
    for k, st in (payload.get("rng") or {}).items():
        r = random.Random()
        r.setstate((st[0], tuple(st[1]), st[2]))
        streams[k] = r
    active = {}
    for k, rec in (payload.get("active") or {}).items():
        active[k] = EpisodePrior(
            rec["env_id"], rec["episode_id"], tuple(tuple(e) for e in rec["edges"]),
            rec["original_hash"], rec["hash"], rec["audit_mode"],
        )
    sampler.streams = streams
    sampler.active = active
    sampler.draw_count = int(payload.get("draw_count") or 0)


def param_digest(policy):
    acc = hashlib.sha256()
    with torch.no_grad():
        for name, p in policy.named_parameters():
            acc.update(name.encode())
            acc.update(p.detach().cpu().numpy().tobytes())
    return acc.hexdigest()


def optimizer_digest(optimizer):
    acc = hashlib.sha256()
    for i, st in optimizer.state.items():
        for k, v in st.items():
            acc.update(str(k).encode())
            if torch.is_tensor(v):
                acc.update(v.detach().cpu().numpy().tobytes())
            else:
                acc.update(str(v).encode())
    return acc.hexdigest()


def start_episode(method, bundle, cases, seed, sampler, collector, job_dir, eval_mode=False):
    case = bundle.next_case(cases, seed)
    snap = bundle.start_case(case)
    snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler, eval_mode=eval_mode)
    bundle.current_snapshot = snap
    collector.reset_episode(snap.env_id, snap.episode_id)
    cache_key = str(bundle.caches[case]) if case in bundle.caches else None
    append_jsonl(job_dir / "episode_priors.jsonl", {
        "env_id": prior.env_id,
        "episode_id": prior.episode_id,
        "audit_mode": prior.audit_mode,
        "original_hash": prior.original_hash,
        "effective_hash": prior.hash,
        "effective_relation_count": len(prior.edges),
        "source_cache_relation_count": source_n,
        "case_id": case,
        "method": method,
        "eval_mode": bool(eval_mode),
    })
    return case, prior, source_n, cache_key


def eval_dev20(root, task_id, method, ckpt_path, cases, device, hashes_doc, out_eval):
    rng_before = s1.capture_rng()
    bundle = make_bundle(root, task_id)
    s1.seed_all(0)
    policy = make_policy(bundle.template, method, task_id, device)
    load_checkpoint(ckpt_path, policy)
    policy.eval()
    rows = []
    success_n = 0
    try:
        for case in cases:
            snap = bundle.start_case(case)
            snap, prior, source_n = apply_episode_prior(method, bundle, snap, None, eval_mode=True)
            collector = Collector(bundle, policy)
            collector.reset_episode(snap.env_id, snap.episode_id)
            ended = False
            success = False
            steps = 0
            while not ended and steps < 64:
                t, result = collector.step(snap)
                steps += 1
                if t is None:
                    actual = result if not isinstance(result, dict) else None
                    reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                    success = bool(getattr(actual, "success", False))
                    ended = True
                    rows.append({"case_id": case, "success": success, "reason": reason, "steps": steps, "no_transition": True})
                    break
                snap = t.next_snapshot
                bundle.current_snapshot = snap
                success = bool(result.success)
                ended = bool(result.terminated or result.truncated)
            if success:
                success_n += 1
            if not rows or rows[-1].get("case_id") != case:
                rows.append({"case_id": case, "success": success, "reason": getattr(result, "reason", None), "steps": steps})
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
        s1.restore_rng(rng_before)
    rate = success_n / max(1, len(cases))
    payload = {"method": method, "task_id": task_id, "checkpoint": str(ckpt_path), "success_n": success_n, "n": len(cases), "success_rate": rate, "rows": rows, "hashes": hashes_doc, "isolated_rng": True}
    s1.write_json(out_eval, payload)
    return payload


def train_job(root, task_id, method, seed, device, device_name, max_updates=None, stop_after_updates=None, resume=False):
    manifest = load_yaml(root / MANIFEST_REL)
    d_ref = exact_d_ref(manifest, task_id)
    job_cap = physical_cap(manifest, task_id)
    run_id = f"stage2a_{task_id}_{method}_seed{seed}"
    out_root = root / STAGE_DIR / run_id
    attempt = 0
    while (out_root / "attempts" / f"attempt_{attempt:03d}").exists():
        if resume:
            break
        attempt += 1
    if resume and (out_root / "attempts" / f"attempt_{attempt:03d}").exists():
        job_dir = out_root / "attempts" / f"attempt_{attempt:03d}"
    else:
        job_dir = out_root / "attempts" / f"attempt_{attempt:03d}"
        job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "checkpoints").mkdir(exist_ok=True)
    (job_dir / "plots").mkdir(exist_ok=True)
    split = json.loads((root / f"configs/splits/{task_id}_stage_2a.json").read_text(encoding="utf-8"))
    train_ids = [row["case_id"] for row in split["train"]]
    dev_ids = [row["case_id"] for row in split["dev"][:20]]
    s1.seed_all(int(seed))
    bundle = make_bundle(root, task_id)
    caches = {k: v for k, v in bundle.caches.items()}
    hashes_doc = hashes(root, task_id, method, seed, caches)
    policy = make_policy(bundle.template, method, task_id, device)
    trainer = PPO(policy)
    collector = Collector(bundle, policy)
    sampler = None if method in ("B0", "B2") else PriorSampler(int(seed))
    cfg = {
        "run_id": run_id,
        "task_id": task_id,
        "method": method,
        "seed": int(seed),
        "init": "from_scratch",
        "planned_transitions": TRANSITIONS,
        "planned_updates": UPDATES,
        "d_ref": d_ref,
        "physical_cap_seconds": job_cap,
        "eval_split": "dev20",
        "hashes": hashes_doc,
        "no_pretrained_checkpoint": True,
        "learning_enabled": True,
    }
    if not (job_dir / "config.json").exists():
        s1.write_json(job_dir / "config.json", cfg)
    if not (job_dir / "environment_snapshot.json").exists():
        s1.write_json(job_dir / "environment_snapshot.json", s1.environment_snapshot(device, device_name, seed))
    if not (job_dir / "software_snapshot.json").exists():
        s1.write_json(job_dir / "software_snapshot.json", {"git_commit": git_commit(root), "base_commit": BASE_COMMIT, "python": sys.executable, "torch": torch.__version__})
    target_updates = UPDATES if max_updates is None else int(max_updates)
    if stop_after_updates is not None:
        target_updates = min(target_updates, int(stop_after_updates))
    extra0 = {"run_id": run_id, "update_count": 0, "interaction_count": 0, "interaction_seconds": 0.0, **hashes_doc, "device": str(device)}
    ckpt0 = job_dir / "checkpoints" / "update_0.pt"
    if not ckpt0.exists() and not (job_dir / "checkpoints" / "step_0.pt").exists():
        save_full_checkpoint(ckpt0, policy, trainer.optimizer, extra0, s1.capture_rng(), sampler, collector, bundle, None, None, False)
    eval0_path = job_dir / "eval_update_0.json"
    if eval0_path.exists():
        ev0 = json.loads(eval0_path.read_text(encoding="utf-8"))
    else:
        log("%s eval update0" % run_id)
        ev0 = eval_dev20(root, task_id, method, ckpt0, dev_ids, device, hashes_doc, eval0_path)
    if resume:
        loaded = load_resume_if_any(job_dir, policy, trainer, sampler, collector, bundle)
        ppo_updates_l, count_l, optimizer_steps_l, interaction_seconds_l, train_success_episodes_l = loaded
    else:
        ppo_updates_l = count_l = optimizer_steps_l = train_success_episodes_l = 0
        interaction_seconds_l = 0.0
    rollout = Rollout()
    count = count_l
    updates = ppo_updates_l
    ppo_updates = ppo_updates_l
    optimizer_steps = optimizer_steps_l
    interaction_seconds = interaction_seconds_l
    skill_count = int(count_l)
    train_success_episodes = train_success_episodes_l
    zero_reward_episodes = 0
    empty_episodes = 0
    original_ep = 0
    absent_ep = 0
    nan_n = 0
    timeout_n = 0
    deadline_n = 0
    live_episodes = 0
    closed_episodes = 0
    started_episodes = 0
    trans_buffer = []
    train_rows = []
    eval_rows = [{"update": 0, "skill_transitions": 0, "success_rate": ev0["success_rate"], "success_n": ev0["success_n"]}]
    case = prior = source_n = cache_key = None
    ep_reward = 0.0
    first_update_evidence = None
    param0 = param_digest(policy)
    opt0 = optimizer_digest(trainer.optimizer)
    if resume:
        summ_path = job_dir / "summary.json"
        if summ_path.is_file():
            summ = json.loads(summ_path.read_text(encoding="utf-8"))
            skill_count = int(summ.get("skill_count") or skill_count)
            deadline_n = int(summ.get("deadline_n") or 0)
            started_episodes = int(summ.get("started_episodes") or 0)
            closed_episodes = int(summ.get("closed_episodes") or 0)
            live_episodes = int(summ.get("live_episodes") or 0)
            if summ.get("param0"):
                param0 = summ["param0"]
            if summ.get("opt0"):
                opt0 = summ["opt0"]
        live_episodes = 1 if bundle.current_snapshot is not None else 0
        evd_path = job_dir / "first_update_evidence.json"
        if evd_path.is_file():
            first_update_evidence = json.loads(evd_path.read_text(encoding="utf-8"))
        tm = job_dir / "train_metrics.csv"
        if tm.is_file() and tm.stat().st_size:
            with tm.open(encoding="utf-8", newline="") as f:
                train_rows = list(csv.DictReader(f))
        em = job_dir / "eval_metrics.csv"
        if em.is_file() and em.stat().st_size:
            with em.open(encoding="utf-8", newline="") as f:
                eval_rows = list(csv.DictReader(f))
        else:
            s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
    else:
        s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
    hard_fail = None
    wall_start = time.time()
    log("%s training start target_updates=%s d_ref=%s cap=%s" % (run_id, target_updates, d_ref, job_cap))
    try:
        while ppo_updates < target_updates and count < TRANSITIONS and interaction_seconds < job_cap:
            if bundle.current_snapshot is None:
                case, prior, source_n, cache_key = start_episode(method, bundle, train_ids, seed, sampler, collector, job_dir)
                started_episodes += 1
                live_episodes += 1
                if prior.audit_mode == "original":
                    original_ep += 1
                else:
                    absent_ep += 1
                ep_reward = 0.0
            snap = bundle.current_snapshot
            t, result = collector.step(snap)
            ended = False
            success = False
            if t is None:
                reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                if reason == "DEADLINE":
                    deadline_n += 1
                    ended = True
                    closed_episodes += 1
                    live_episodes = max(0, live_episodes - 1)
                    if ep_reward == 0:
                        zero_reward_episodes += 1
                else:
                    empty_episodes += 1
                    ended = True
                    closed_episodes += 1
                    live_episodes = max(0, live_episodes - 1)
                    if empty_episodes >= MAX_EMPTY:
                        raise BindingError("Runtime repeatedly exposes no executable skill")
                append_jsonl(job_dir / "decision_log.jsonl", {
                    "run_id": run_id, "no_transition": True, "reason": reason,
                    "update": ppo_updates, "case_id": case,
                    "decision_id": getattr(snap, "decision_id", None),
                    "episode_id": getattr(snap, "episode_id", None),
                })
                bundle.current_snapshot = None
            else:
                empty_episodes = 0
                rec = s1.compact_transition(method, seed, case, t, result, collector.last_output, collector.last_execution, prior.audit_mode, prior.original_hash, source_n, cache_key, run_id)
                rec["task"] = task_id
                rec["d_ref"] = d_ref
                rec["ppo_updates"] = ppo_updates
                rec["optimizer_steps"] = optimizer_steps
                append_jsonl(job_dir / "transition_log.jsonl", rec)
                append_jsonl(job_dir / "decision_log.jsonl", {
                    "decision_id": rec["decision_id"],
                    "selected_candidate_id": rec["selected_candidate_id"],
                    "mask_hash": rec["mask_hash"],
                    "prior_mode": rec["prior_mode"],
                    "reason": rec["reason"],
                    "duration_seconds": rec["duration_seconds"],
                })
                rollout.append(t)
                trans_buffer.append(rec)
                count += 1
                skill_count += 1
                interaction_seconds += t.duration
                ep_reward += t.reward
                bundle.current_snapshot = t.next_snapshot
                ended = bool(t.terminated)
                success = bool(result.success)
                if t.reason == "DEADLINE":
                    deadline_n += 1
                exit_code = rec.get("controller_exit") or ""
                if "timeout" in str(exit_code).lower() or str(exit_code) == "TASK_DEADLINE":
                    timeout_n += 1
                if not rec["logits_finite"] or not rec["value_finite"] or not math.isfinite(rec["old_logp"]):
                    nan_n += 1
                    raise DataIntegrityError("NaN/Inf in transition")
                if ended:
                    if success:
                        train_success_episodes += 1
                    if ep_reward == 0:
                        zero_reward_episodes += 1
                    closed_episodes += 1
                    live_episodes = max(0, live_episodes - 1)
                    append_jsonl(job_dir / "episode_log.jsonl", {
                        "run_id": run_id, "case_id": case, "success": success, "reward": ep_reward,
                        "reason": t.reason, "transitions_so_far": count,
                    })
                    bundle.current_snapshot = None
            if len(rollout.transitions) >= UPDATE_EVERY or (count >= TRANSITIONS or interaction_seconds >= job_cap):
                if not rollout.transitions:
                    break
                n_roll = len(rollout.transitions)
                if n_roll < UPDATE_EVERY and ppo_updates + 1 < target_updates and count < TRANSITIONS and interaction_seconds < job_cap:
                    continue
                before = param_digest(policy)
                before_opt = optimizer_digest(trainer.optimizer)
                logs = trainer.update(rollout)
                ppo_updates += 1
                optimizer_steps += len(logs)
                after = param_digest(policy)
                after_opt = optimizer_digest(trainer.optimizer)
                changed = before != after
                finite_logs = all(math.isfinite(float(r.get("total", 0))) and math.isfinite(float(r.get("grad_norm", 0))) for r in logs)
                if not finite_logs:
                    raise DataIntegrityError("Nonfinite PPO log")
                evd = {
                    "run_id": run_id,
                    "ppo_update": ppo_updates,
                    "optimizer_steps": optimizer_steps,
                    "rollout_transitions": n_roll,
                    "param_changed": changed,
                    "optimizer_changed": before_opt != after_opt,
                    "mean_total_loss": float(sum(r["total"] for r in logs) / len(logs)),
                    "mean_grad_norm": float(sum(r["grad_norm"] for r in logs) / len(logs)),
                    "finite": finite_logs,
                    "interaction_seconds": interaction_seconds,
                    "skill_count": skill_count,
                }
                if first_update_evidence is None:
                    first_update_evidence = evd
                evd_path = job_dir / "first_update_evidence.json"
                if not evd_path.exists():
                    s1.write_json(evd_path, first_update_evidence)
                row = {
                    "update": ppo_updates,
                    "valid_transitions": count,
                    "rollout_n": n_roll,
                    "optimizer_steps": optimizer_steps,
                    "success_episodes": train_success_episodes,
                    "zero_reward_episodes": zero_reward_episodes,
                    "policy_loss": float(sum(r["actor"] for r in logs) / len(logs)),
                    "value_loss": float(sum(r["v"] for r in logs) / len(logs)),
                    "q_loss": float(sum(r["q"] for r in logs) / len(logs)),
                    "total_loss": evd["mean_total_loss"],
                    "grad_norm": evd["mean_grad_norm"],
                    "param_changed": changed,
                    "deadline_n": deadline_n,
                    "interaction_seconds": interaction_seconds,
                    "skill_count": skill_count,
                    "started_episodes": started_episodes,
                    "closed_episodes": closed_episodes,
                    "live_episodes": live_episodes,
                }
                train_rows.append(row)
                s1.csv_write(job_dir / "train_metrics.csv", train_rows)
                extra = {"run_id": run_id, "update_count": ppo_updates, "interaction_count": count, "interaction_seconds": interaction_seconds, "optimizer_steps": optimizer_steps, **hashes_doc, "device": str(device)}
                if ppo_updates in CHECKPOINTS:
                    ckpt = job_dir / "checkpoints" / ("update_%s.pt" % ppo_updates)
                    if not ckpt.exists():
                        mid = bundle.current_snapshot is not None
                        save_full_checkpoint(ckpt, policy, trainer.optimizer, extra, s1.capture_rng(), sampler, collector, bundle, case, bundle.current_snapshot, mid)
                    if ppo_updates % EVAL_EVERY == 0:
                        log("%s eval update %s" % (run_id, ppo_updates))
                        ev = eval_dev20(root, task_id, method, ckpt, dev_ids, device, hashes_doc, job_dir / ("eval_update_%s.json" % ppo_updates))
                        eval_rows.append({"update": ppo_updates, "skill_transitions": count, "success_rate": ev["success_rate"], "success_n": ev["success_n"]})
                        s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
                lp = job_dir / "checkpoints" / "latest.pt"
                for extra_path in (lp, Path(str(lp) + ".json"), lp.with_suffix(".json"), lp.with_suffix(".rng.json"), lp.with_suffix(".episode.pkl")):
                    if extra_path.exists():
                        extra_path.unlink()
                save_full_checkpoint(lp, policy, trainer.optimizer, extra, s1.capture_rng(), sampler, collector, bundle, case, bundle.current_snapshot, bundle.current_snapshot is not None)
                s1.write_json(job_dir / "resume.json", {
                    "updates": ppo_updates, "count": count, "interaction_seconds": interaction_seconds,
                    "optimizer_steps": optimizer_steps, "train_success_episodes": train_success_episodes,
                    "skill_count": skill_count, "deadline_n": deadline_n,
                    "started_episodes": started_episodes, "closed_episodes": closed_episodes,
                    "live_episodes": live_episodes,
                })
                trans_buffer = []
                log("%s PPO update %s transitions=%s opt_steps=%s changed=%s" % (run_id, ppo_updates, n_roll, optimizer_steps, changed))
                if stop_after_updates is not None and ppo_updates >= int(stop_after_updates):
                    break
        status = "COMPLETED" if ppo_updates >= target_updates or count >= TRANSITIONS or interaction_seconds >= job_cap else "STOPPED"
        summary = {
            "run_id": run_id,
            "status": status,
            "completed_updates": ppo_updates,
            "optimizer_steps": optimizer_steps,
            "valid_transitions": count,
            "interaction_seconds": interaction_seconds,
            "skill_count": skill_count,
            "d_ref": d_ref,
            "physical_cap_seconds": job_cap,
            "train_success_episodes": train_success_episodes,
            "deadline_n": deadline_n,
            "step0_dev_success": ev0["success_rate"],
            "first_update_evidence": first_update_evidence,
            "param0": param0,
            "opt0": opt0,
            "wall_seconds": time.time() - wall_start,
            "started_episodes": started_episodes,
            "closed_episodes": closed_episodes,
            "live_episodes": live_episodes,
        }
        s1.write_json(job_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        hard_fail = {"error": str(exc), "traceback": traceback.format_exc(), "updates": ppo_updates, "count": count}
        s1.write_json(job_dir / "failure.json", hard_fail)
        raise
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass


def cmd_stage_2a_run(root, task_id, method, seed, gpu=0, max_updates=None, stop_after_updates=None, resume=False):
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
    os.environ.pop("DASHSCOPE_API_KEY", None)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu) if str(gpu).isdigit() else 0)
        device = torch.device("cuda", torch.cuda.current_device())
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        device_name = "cpu"
    return train_job(Path(root), task_id, method, int(seed), device, device_name, max_updates=max_updates, stop_after_updates=stop_after_updates, resume=resume)

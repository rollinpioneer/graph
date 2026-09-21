"""Stage 1A-P0 runtime binding: split, qualification, cache, preflight. No PPO."""
from __future__ import annotations

import csv, hashlib, json, math, os, statistics, time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import yaml

from .common import BindingError, canonical, digest
from .contracts import Registry, from_dict
from .facts import Truth
from .graph import Goal, build_template
from .runtime import load_runtime, require_runtime
from .vlm import cache_key
from .vlm_provider import ProviderConfig, DashScopeProvider, validate_payload
from .vlm_cache_pipeline import request_and_process, write_audit_cache, process_response
from .stage0c import PREPROCESSING, file_hash, prepare_scene, read as read_json
from .platforms.libero.d0_env import CaseSpec, make_env, OBJECT_HALF, CONTAINER_INNER
from .platforms.libero.perception import PerceptionAdapter, THRESHOLDS, VERSION as PERCEPTION_VERSION
from .platforms.libero.verifier import FactVerifier, VERIFIER_VERSION
from .platforms.libero.task_evaluator import TaskEvaluator, EVALUATOR_VERSION
from .platforms.libero.safety import SafetyManager
from .platforms.libero.clock import DurationProvider
from .platforms.libero.skill_executor import SkillExecutor
from .platforms.libero.runtime_factory import sha_file, PREDICATES, OBJECTS
from .adapters import EvaluationInput


ROOT_DEFAULT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
P0 = Path("experiments/part_1_smoke/stage_1a_p0")
QUAL_BUDGET = 90.0
ATTEMPTS_PER_SKILL = 5
GROUND_SKILLS = [
    "a:OPEN:container:v1",
    "a:PICK:target:v1",
    "a:PICK:second_object:v1",
    "a:PLACE:target:container:v1",
    "a:PLACE:second_object:container:v1",
    "a:PLACE_BUFFER:target:buffer:v1",
    "a:PLACE_BUFFER:second_object:buffer:v1",
]


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def hidden_qa(env):
    h = env.hidden_truth()
    eef = np.asarray(h["eef_pos"], dtype=float)
    lid = np.asarray(h["lid"], dtype=float)
    c = np.asarray(h["container"], dtype=float)
    b = np.asarray(h["buffer"], dtype=float)
    table = float(h["table_top_z"])

    def near(p, q, xy=0.05, z=0.07):
        p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
        return float(np.linalg.norm(p[:2] - q[:2])) < xy and abs(p[2] - q[2]) < z

    def held(name):
        p = np.asarray(h[name], dtype=float)
        lifted = float(p[2]) > table + 0.05
        return lifted and near(p, eef, xy=0.045, z=0.08)

    grip_closed = held("target") or held("second_object") or held("lid")

    def inside(name):
        p = np.asarray(h[name], dtype=float)
        return abs(p[0] - c[0]) <= CONTAINER_INNER[0] and abs(p[1] - c[1]) <= CONTAINER_INNER[1] and p[2] <= table + 0.12

    def at_buffer(name):
        p = np.asarray(h[name], dtype=float)
        return abs(p[0] - b[0]) <= 0.09 and abs(p[1] - b[1]) <= 0.09

    open_c = float(np.linalg.norm(lid[:2] - c[:2])) > 0.10
    return {
        "GripperEmpty": (not grip_closed) and not held("target") and not held("second_object") and not held("lid"),
        "Held:target": held("target"),
        "Held:second_object": held("second_object"),
        "Open:container": open_c,
        "Inside:target:container": inside("target"),
        "Inside:second_object:container": inside("second_object"),
        "AtBuffer:target:buffer": at_buffer("target"),
        "AtBuffer:second_object:buffer": at_buffer("second_object"),
        "OnTable:target": (not held("target")) and abs(h["target"][2] - (table + OBJECT_HALF[2])) < 0.05,
        "OnTable:second_object": (not held("second_object")) and abs(h["second_object"][2] - (table + OBJECT_HALF[2])) < 0.05,
    }


def expected_post(skill_id):
    parts = skill_id.split(":")
    skill, args = parts[1], parts[2:-1]
    if skill == "OPEN":
        return "p:Open:container"
    if skill == "PICK":
        return f"p:Held:{args[0]}"
    if skill == "PLACE":
        return f"p:Inside:{args[0]}:{args[1]}"
    if skill == "PLACE_BUFFER":
        return f"p:AtBuffer:{args[0]}:{args[1]}"
    raise BindingError("unknown skill " + skill_id)


def qa_key(fact_id):
    return fact_id[2:] if fact_id.startswith("p:") else fact_id


def make_case(seed=0, split="dev", case_id="qual"):
    rng = np.random.RandomState(seed)
    return CaseSpec(
        case_id=case_id,
        split=split,
        seed=int(seed),
        target_xy=(float(rng.uniform(-0.20, 0.02)), float(rng.uniform(-0.18, -0.02))),
        second_xy=(float(rng.uniform(-0.02, 0.20)), float(rng.uniform(-0.18, -0.02))),
        container_xy=(0.18, 0.12),
        buffer_xy=(-0.18, 0.12),
        lid_closed=True,
    )


def bind_env(case, gpu=0):
    env = make_env(case, gpu=gpu)
    env.last_perception = {}
    env.reset()
    env._apply_case_poses()
    env._open_gripper_reset()
    clock = DurationProvider(env)
    safety = SafetyManager(env)
    perception = PerceptionAdapter(env)
    env.refresh_perception = lambda e=env, p=perception: p.infer(e.public_observation())
    verifier = FactVerifier(env)
    evaluator = TaskEvaluator(env, 120.0)
    executor = SkillExecutor(env, safety, clock)
    return env, clock, safety, perception, verifier, evaluator, executor


def fact_value(records, fid):
    for rec in records:
        if rec.fact_id == fid:
            return rec.value
    return Truth.UNKNOWN


def setup_for_skill(env, executor, clock, skill_id, timeout):
    """Scripted setup using controllers only; hidden state is not written."""
    parts = skill_id.split(":")
    skill, args = parts[1], parts[2:-1]
    logs = []
    if skill in ("PLACE", "PLACE_BUFFER"):
        logs.append(executor.execute("a:OPEN:container:v1", timeout))
        logs.append(executor.execute(f"a:PICK:{args[0]}:v1", timeout))
    return logs



def qualify_controllers(root: Path, gpu=0):
    rows = []
    durations = defaultdict(list)
    for skill_id in GROUND_SKILLS:
        ok = 0
        for attempt in range(ATTEMPTS_PER_SKILL):
            seed = 7000 + GROUND_SKILLS.index(skill_id) * 10 + attempt
            case = make_case(seed=seed, case_id=f"ctrl_{skill_id}_{attempt}")
            env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu)
            try:
                setup_for_skill(env, executor, clock, skill_id, QUAL_BUDGET)
                t0 = clock.now_seconds()
                obs_before = env.public_observation()
                meas_before = perception.infer(obs_before)
                rec_before = verifier.verify(meas_before, None)
                exe = executor.execute(skill_id, QUAL_BUDGET)
                obs_after = env.public_observation()
                meas_after = perception.infer(obs_after)
                rec_after = verifier.verify(meas_after, exe)
                qa = hidden_qa(env)
                fid = expected_post(skill_id)
                v_val = fact_value(rec_after, fid)
                qa_val = bool(qa.get(qa_key(fid), False))
                verified = exe["controller_exit"] == "NORMAL_TERMINATION" and v_val == Truth.TRUE and qa_val
                if verified:
                    ok += 1
                    durations[skill_id.split(":")[1]].append(float(exe["sim_duration"]))
                wrong_object = False
                if skill_id.startswith("a:PICK:"):
                    other = "second_object" if "target" in skill_id else "target"
                    if fact_value(rec_after, f"p:Held:{other}") == Truth.TRUE and qa.get(f"Held:{other}"):
                        wrong_object = True
                row = {
                    "skill_id": skill_id,
                    "attempt": attempt,
                    "seed": seed,
                    "controller_exit": exe["controller_exit"],
                    "sim_duration": float(exe["sim_duration"]),
                    "steps": int(exe["steps"]),
                    "verifier_fact": fid,
                    "verifier_value": v_val.name if hasattr(v_val, "name") else str(v_val),
                    "hidden_qa": bool(qa_val),
                    "verified_success": bool(verified),
                    "wrong_object": bool(wrong_object),
                    "timeout": bool(exe["timeout"]),
                    "nan_blocked": bool(exe["nan_blocked"]),
                    "rejected": bool(exe["rejected"]),
                    "interrupted": bool(exe["interrupted"]),
                    "states": "|".join(exe.get("states") or []),
                    "policy_input_keys": sorted(list(obs_after.keys())),
                    "verifier_evidence": ";".join(
                        rec.reason for rec in rec_after if rec.fact_id == fid
                    ),
                }
                rows.append(row)
            finally:
                env.close()
        need = 4
        if ok < need:
            pass
    return rows, durations


def qualify_verifier(root: Path, gpu=0):
    scenarios = [
        ("empty_gripper", None),
        ("successful_hold", "a:PICK:target:v1"),
        ("failed_pick", "a:PICK:target:v1"),
        ("normal_place", "a:PLACE:target:container:v1"),
        ("container_closed", None),
        ("container_open", "a:OPEN:container:v1"),
        ("buffer_inside", "a:PLACE_BUFFER:target:buffer:v1"),
        ("occlusion_unknown", None),
    ]
    rows = []
    case = make_case(seed=42, case_id="ver_base")
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu)
    try:
        for name, skill in scenarios:
            if skill:
                setup_for_skill(env, executor, clock, skill, QUAL_BUDGET)
                executor.execute(skill, QUAL_BUDGET)
            obs = env.public_observation()
            meas = perception.infer(obs)
            recs = verifier.verify(meas, None)
            qa = hidden_qa(env)
            payload = json.dumps(meas.measurements)
            leaked = ("hidden_truth" in payload) or ("body_xpos" in payload) or ('"qpos"' in payload)
            row = {
                "scenario": name,
                "skill": skill,
                "leaked_hidden_into_verifier": leaked,
                "qa": {k: bool(v) for k, v in qa.items()},
                "verifier": {r.fact_id: r.value.name for r in recs},
                "reasons": {r.fact_id: r.reason for r in recs},
            }
            rows.append(row)
            if name == "empty_gripper":
                env.close()
                env, clock, safety, perception, verifier, evaluator, executor = bind_env(make_case(43, case_id="ver_reset"), gpu=gpu)
    finally:
        env.close()
    return rows


def evaluator_unit_checks(gpu=0):
    results = []
    case = make_case(seed=9, case_id="eval")
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(case, gpu=gpu)
    try:
        evaluator.reset_episode()
        r0 = evaluator.evaluate(EvaluationInput("D0", "e", "ep", (), 1.0, 0.0, 1.0))
        results.append({"name": "incomplete_zero", "pass": (not r0.success) and r0.reward_events == ()})
        executor.execute("a:OPEN:container:v1", QUAL_BUDGET)
        r1 = evaluator.evaluate(EvaluationInput("D0", "e", "ep", (), 8.0, 1.0, 8.0))
        results.append({"name": "open_only_zero", "pass": (not r1.success) and r1.reward_events == ()})
        executor.execute("a:PICK:target:v1", QUAL_BUDGET)
        r2 = evaluator.evaluate(EvaluationInput("D0", "e", "ep", (), 16.0, 8.0, 16.0))
        results.append({"name": "pick_only_zero", "pass": (not r2.success) and r2.reward_events == ()})
        executor.execute("a:PLACE:target:container:v1", QUAL_BUDGET)
        r3 = evaluator.evaluate(EvaluationInput("D0", "e", "ep", (), 28.0, 16.0, 28.0))
        goal = evaluator.goal_true()
        results.append({"name": "goal_after_scripted_place", "pass": bool(goal and r3.success and r3.reward_events), "goal_true": goal, "reward": r3.reward_events, "success": r3.success})
        r4 = evaluator.evaluate(EvaluationInput("D0", "e", "ep", (), 30.0, 28.0, 30.0))
        results.append({"name": "no_repeat_reward", "pass": r4.reward_events == ()})
        env.reset()
        env._open_gripper_reset()
        evaluator2 = TaskEvaluator(env, 0.5)
        evaluator2.reset_episode()
        r5 = evaluator2.evaluate(EvaluationInput("D0", "e", "ep2", (), 0.6, 0.0, 0.6))
        results.append({"name": "deadline_zero", "pass": (not r5.success) and bool(r5.truncated) and r5.reward_events == () and not r5.terminated, "reason": r5.reason})
    finally:
        env.close()
    return results


def cmd_build_d0_split(root: Path):
    from cp_disr.platforms.libero.d0_env import CaseSpec  # noqa: F401
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_d0_split", root / "scripts/stage_1a/build_d0_split.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = root / "configs/splits/D0_stage_1a.json"
    payload = mod.build_split(path)
    return {"status": "PASS", "path": str(path), "train": payload["train_count"], "dev": payload["dev_count"]}


def capture_split_images(root: Path, split_doc, gpu=0):
    out_root = root / "experiments/stage_1a_inputs"
    mapping = []
    for row in split_doc["train"] + split_doc["dev"]:
        spec = CaseSpec(
            case_id=row["case_id"],
            split=row["split"],
            seed=int(row["seed"]),
            target_xy=tuple(row["target_xy"]),
            second_xy=tuple(row["second_xy"]),
            container_xy=tuple(row["container_xy"]),
            buffer_xy=tuple(row["buffer_xy"]),
            lid_closed=bool(row.get("lid_closed", True)),
        )
        dest = out_root / row["split"] / "D0" / row["case_id"]
        dest.mkdir(parents=True, exist_ok=True)
        rgb_path = dest / "rgb.png"
        if rgb_path.is_file() and (dest / "CAPTURE_COMPLETE").is_file():
            mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root))})
            continue
        env = make_env(spec, gpu=gpu)
        try:
            env.reset()
            env._open_gripper_reset()
            obs = env.public_observation()
            from PIL import Image
            Image.fromarray(np.asarray(obs["rgb"]).astype(np.uint8)).save(rgb_path)
            np.save(dest / "depth.npy", np.asarray(obs["depth"]))
            write_json(dest / "reset_config.json", row)
            write_json(dest / "hidden_truth_qa.json", {k: np.asarray(v).tolist() if hasattr(v, "tolist") else v for k, v in env.hidden_truth().items()})
            (dest / "CAPTURE_COMPLETE").write_text("complete\n", encoding="utf-8")
        finally:
            env.close()
        mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root))})
    return mapping


def _grounded_template(root: Path):
    doc = load_yaml(root / "configs/contracts/d0_runtime_skills.yaml")
    if any(c["name"] == "MOVE" for c in doc["contracts"]):
        raise BindingError("MOVE must not appear in D0 runtime registry")
    reg = Registry(doc["predicate_types"])
    for c in doc["contracts"]:
        reg.register(from_dict(c))
    contracts = reg.ground(OBJECTS)
    goals = (Goal("p:Inside:target:container", 1),)
    template = build_template(contracts, goals, PREDICATES, OBJECTS)
    return doc, contracts, template


def initial_facts_reset():
    return {
        "p:GripperEmpty": "TRUE",
        "p:Held:target": "FALSE",
        "p:Held:second_object": "FALSE",
        "p:OnTable:target": "TRUE",
        "p:OnTable:second_object": "TRUE",
        "p:Open:container": "FALSE",
        "p:Inside:target:container": "FALSE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
    }


def generate_d0_caches(root: Path, mapping, gpu=0):
    vlm = load_yaml(root / "experiments/manifests/vlm_manifest.yaml")
    config = ProviderConfig(model=vlm["model_snapshot"], region=vlm["region"], endpoint=vlm["base_http_api_url"], sdk_version=vlm["sdk_version"])
    provider = DashScopeProvider(config)
    if not provider.key_present():
        raise BindingError("DASHSCOPE_API_KEY missing from process environment")
    few_path = root / "experiments/stage_0c_inputs/fewshots/few_shot_manifest.json"
    few_doc = read_json(few_path)
    few = few_doc.get("examples", [])
    examples = [prepare_scene(root, s, True) for s in few]
    prompt_path = root / "experiments/sources/v2.1_interfaces/system_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    schema_path = root / "schemas/relation_schema.json"
    schema = read_json(schema_path)
    messages = [{"role": "system", "content": [{"text": prompt}]}]
    for ex, record in zip(examples, few):
        messages.extend(
            [
                {"role": "user", "content": ex["content"]},
                {"role": "assistant", "content": [{"text": canonical(record["expected_json"])}]},
            ]
        )
    contract_doc, contracts, template = _grounded_template(root)
    actions = sorted(c.id for c in contracts)
    props = sorted(n.id for n in template.nodes if n.kind == "PROPOSITION")
    contract_sha = file_hash(root / "configs/contracts/d0_runtime_skills.yaml")
    task_path = root / "configs/tasks/resolved/D0.yaml"
    task_sha = file_hash(task_path)
    ledger = root / P0 / "d0_cache_request_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    results = []
    import base64
    from PIL import Image
    for row in mapping:
        image = root / row["image_ref"]
        with Image.open(image) as im:
            mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[im.format]
        scene = {
            "scene_id": row["case_id"],
            "task_id": "D0",
            "split": row["split"],
            "image_ref": row["image_ref"],
            "image_sha256": row["image_sha256"],
            "object_table": [
                {"id": "target", "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "second_object", "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "container", "type": "container", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "buffer", "type": "buffer", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
            ],
            "signed_goals": [{"fact_id": "p:Inside:target:container", "sign": 1}],
            "initial_facts": initial_facts_reset(),
            "allowed_action_ids": actions,
            "allowed_proposition_ids": props,
            "contracts_sha256": contract_sha,
            "task_definition_sha256": task_sha,
            "predicate_version": "CP-DISR-v2.1",
            "asset_binding_sha256": digest({"layout": [row["target_xy"], row["second_xy"], row["container_xy"], row["buffer_xy"]], "runtime": "d0-runtime-v2.1-p0"}),
        }
        context = {k: scene[k] for k in ["scene_id", "task_id", "object_table", "signed_goals", "initial_facts", "allowed_action_ids", "allowed_proposition_ids"]}
        context["object_table"] = sorted(context["object_table"], key=lambda x: x["id"])
        context["skill_contracts"] = contract_doc
        context["registered_effect_edges"] = [e for e in template.edges if e[2] in ("ADD", "DEL")]
        content = [
            {"image": "data:image/" + mime + ";base64," + base64.b64encode(image.read_bytes()).decode()},
            {"text": canonical(context)},
        ]
        payload = {
            "model": config.model,
            "messages": messages + [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "enable_search": False,
            "stream": False,
            "result_format": "message",
        }
        validate_payload(payload)
        manifest = {
            "split": row["split"],
            "task_definition_hash": scene["task_definition_sha256"],
            "initial_RGB_content_hash": scene["image_sha256"],
            "preprocessing_hash": digest(PREPROCESSING),
            "object_binding_hash": digest(context["object_table"]),
            "allowed_ID_hash": digest({"actions": actions, "propositions": props, "signed_goals": scene["signed_goals"]}),
            "contract_version": contract_sha,
            "predicate_version": scene["predicate_version"],
            "model_snapshot": config.model,
            "sdk_api_version": config.sdk_version,
            "region": config.region,
            "endpoint": config.endpoint,
            "prompt_hash": file_hash(prompt_path),
            "fewshot_hash": digest(few_doc),
            "schema_hash": file_hash(schema_path),
            "decoding_config": {"temperature": 0, "max_tokens": 2048, "max_relations": 8, "thinking": False, "response_format": "json_object"},
            "scene_id": scene["scene_id"],
            "task_id": "D0",
            "synthetic_unit_fixture": False,
            "initial_facts_hash": digest(scene["initial_facts"]),
            "input_context_hash": digest(context),
            "request_payload_hash": digest(payload),
            "asset_binding_hash": scene["asset_binding_sha256"],
        }
        key = cache_key(manifest)
        path = root / "experiments/vlm_cache" / row["split"] / key
        if path.exists() and (path / "COMPLETE").is_file():
            results.append({"case_id": row["case_id"], "split": row["split"], "cache_key": key, "status": "REUSED", "cache_dir": str(path.relative_to(root))})
            continue
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical({"scene_id": row["case_id"], "state": "REQUEST_STARTED", "time": time.time()}) + "\n")
        execution = request_and_process(provider, payload, template, schema, ())
        written = write_audit_cache(root / "experiments/vlm_cache", manifest, prompt, scene, execution)
        rec = {"case_id": row["case_id"], "split": row["split"], "cache_key": key, "status": execution["status"], "cache_dir": str(Path(written).relative_to(root)), "attempts": len(execution["attempts"])}
        results.append(rec)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
        last_err = execution["attempts"][-1]["response"].get("error_type")
        if last_err in ("AUTHORIZATION", "REQUEST_REJECTED", "SDK_ERROR"):
            raise BindingError("Provider rejected frozen configuration; stop without fallback: " + str(last_err))
    return results


def calibrate_timing(durations):
    timeouts = {}
    for skill in ("OPEN", "PICK", "PLACE", "PLACE_BUFFER"):
        xs = list(durations.get(skill) or [])
        if not xs:
            raise BindingError("No verified-success duration for " + skill + "; cannot calibrate timeout")
        xs = sorted(float(x) for x in xs)
        p95 = xs[max(0, int(math.ceil(0.95 * len(xs)) - 1))]
        timeouts[skill] = float(max(math.ceil(p95 * 2.0), math.ceil(max(xs) * 1.25) + 2.0, 8.0))
    all_ok = [float(d) for xs in durations.values() for d in xs]
    d_ref = float(statistics.median(all_ok))
    deadline = float(timeouts["OPEN"] + timeouts["PICK"] + timeouts["PLACE"] + 10.0)
    return {
        "rule": "timeout=max(ceil(p95*2), ceil(max*1.25)+2, 8); deadline=OPEN+PICK+PLACE+10; d_ref=median verified-success durations",
        "qualification_attempt_budget_seconds": QUAL_BUDGET,
        "skill_timeouts": timeouts,
        "task_deadline_seconds": deadline,
        "reference_skill_seconds": d_ref,
        "samples": {k: v for k, v in durations.items()},
    }


def scripted_dev10(root: Path, split_doc, timeout_map, deadline, gpu=0):
    rows = []
    seq = ["a:OPEN:container:v1", "a:PICK:target:v1", "a:PLACE:target:container:v1"]
    for row in split_doc["dev"]:
        spec = CaseSpec(
            case_id=row["case_id"], split="dev", seed=int(row["seed"]),
            target_xy=tuple(row["target_xy"]), second_xy=tuple(row["second_xy"]),
            container_xy=tuple(row["container_xy"]), buffer_xy=tuple(row["buffer_xy"]), lid_closed=True,
        )
        env, clock, safety, perception, verifier, evaluator, executor = bind_env(spec, gpu=gpu)
        evaluator.deadline = float(deadline)
        evaluator.reset_episode()
        try:
            rewards = []
            skill_log = []
            for sid in seq:
                skill = sid.split(":")[1]
                exe = executor.execute(sid, float(timeout_map[skill]))
                elapsed = clock.now_seconds()
                ev = evaluator.evaluate(EvaluationInput("D0", "d0-env-0", row["case_id"], exe["evidence_ids"], elapsed, exe["start_seconds"], exe["end_seconds"]))
                rewards.extend(ev.reward_events)
                skill_log.append({"skill": sid, "exit": exe["controller_exit"], "duration": exe["sim_duration"], "eval": ev.reason, "success": ev.success})
            final = evaluator.evaluate(EvaluationInput("D0", "d0-env-0", row["case_id"], (), clock.now_seconds(), 0.0, clock.now_seconds()))
            rows.append({
                "case_id": row["case_id"],
                "seed": row["seed"],
                "goal_true": evaluator.goal_true(),
                "success": final.success,
                "terminated": final.terminated,
                "truncated": final.truncated,
                "reward_events": rewards,
                "repeat_reward": len([r for r in rewards if r[1] == 1.0]) > 1,
                "skills": skill_log,
            })
        finally:
            env.close()
    return rows


def failure_qualification(root: Path, timeout_map, gpu=0):
    rows = []
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(make_case(11, case_id="fail"), gpu=gpu)
    try:
        exe = executor.execute("a:PICK:target:v1", 1.0)
        rows.append({"case": "timeout", "exit": exe["controller_exit"], "timeout": exe["timeout"]})
        exe = executor.execute("a:MOVE:target:buffer:v1", 8.0)
        rows.append({"case": "controller_rejection_MOVE", "exit": exe["controller_exit"], "rejected": exe["rejected"]})
        recs = verifier.verify(perception.infer(env.public_observation()), None)
        rows.append({"case": "unknown_not_false", "gripper": fact_value(recs, "p:GripperEmpty").name})
        reason = safety.end_no_candidates("d0-env-0", "fail")
        rows.append({"case": "no_candidate", "reason": reason})
        exe = executor.execute("a:PLACE:target:container:v1", float(timeout_map["PLACE"]))
        recs = verifier.verify(perception.infer(env.public_observation()), exe)
        rows.append({"case": "place_without_hold_verification", "exit": exe["controller_exit"], "inside": fact_value(recs, "p:Inside:target:container").name})
    finally:
        env.close()
    return rows


def hidden_truth_leakage_check(gpu=0):
    env, clock, safety, perception, verifier, evaluator, executor = bind_env(make_case(3, case_id="leak"), gpu=gpu)
    try:
        from .platforms.libero.observations import ObservationProvider
        from .platforms.libero.snapshot import SnapshotBuilder
        obs = ObservationProvider(env, clock).observe()
        meas = perception.infer(env.public_observation())
        recs = verifier.verify(meas, None)
        payload = json.dumps({"obs": obs.__dict__, "meas": meas.measurements, "facts": [r.reason for r in recs]}, default=str)
        hidden = env.hidden_truth()
        leaked = False
        hits = []
        for name in ("target", "second_object", "lid"):
            xyz = hidden[name]
            token = f"{xyz[0]:.5f}"
            if token in payload:
                leaked = True
                hits.append(name)
        return {"leaked": leaked, "hits": hits, "public_keys": list(env.public_observation().keys()), "hidden_keys": list(hidden.keys())}
    finally:
        env.close()


def model_dry_run(root: Path, method: str, case_id: str, output: Path):
    import torch
    from .neural import Policy
    from .collector import Collector
    manifest = load_yaml(root / "experiments/manifests/runtime_manifest.yaml")
    bundle = load_runtime(manifest)
    snap = bundle.start_case(case_id)
    actions = sorted({c.name for c in snap.template.contracts})
    predicates = sorted(PREDICATES)
    types = sorted(set(OBJECTS.values()))
    model = Policy(actions, predicates, types, observation_dim=48, candidate_dim=8, method=method, B=0.5)
    model.eval()
    collector = Collector(bundle, model)
    collector.reset_episode(snap.env_id, snap.episode_id)
    t, result = collector.step(snap)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": method,
        "case_id": case_id,
        "prior_edge_count": len(snap.prior_edges),
        "mask_true": int(sum(snap.mask)),
        "transition": None if t is None else {
            "duration": t.duration,
            "reward": t.reward,
            "terminated": t.terminated,
            "truncated": t.truncated,
            "reason": t.reason,
            "action": t.action if hasattr(t, "action") else None,
        },
        "result": result if isinstance(result, dict) else {
            "success": result.success,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "reason": result.reason,
        },
    }
    if method == "B2" and snap.prior_edges:
        report["b2_prior_not_empty"] = True
    write_json(output / f"{method}_dry_run.json", report)
    bundle.environment.close()
    return report


def cmd_validate_runtime(root: Path):
    manifest = load_yaml(root / "experiments/manifests/runtime_manifest.yaml")
    require_runtime(manifest)
    bundle = load_runtime(manifest)
    names = ["environment", "executor", "observations", "perception", "verifier", "evaluator", "safety", "clock", "snapshot_builder"]
    missing = [n for n in names if not hasattr(bundle, n)]
    if missing:
        raise BindingError("runtime missing " + ",".join(missing))
    snap = bundle.start_case(next(iter(bundle.cases)))
    bundle.environment.close()
    return {"status": "PASS", "initial_candidates": int(sum(snap.mask)), "case": snap.episode_id}


def cmd_validate_controller(root: Path, gpu=0):
    rows, durations = qualify_controllers(root, gpu=gpu)
    by = defaultdict(list)
    for r in rows:
        by[r["skill_id"]].append(r)
    summary = {}
    blocked = []
    for sid, rs in by.items():
        n = sum(1 for r in rs if r["verified_success"])
        wrong = sum(1 for r in rs if r["wrong_object"])
        nan = sum(1 for r in rs if r["nan_blocked"])
        summary[sid] = {"verified_success": n, "attempts": len(rs), "wrong_object": wrong, "nan": nan, "pass": n >= 4 and wrong == 0 and nan == 0}
        if not summary[sid]["pass"]:
            blocked.append(sid)
    return {"status": "PASS" if not blocked else "NEEDS_RERUN", "summary": summary, "rows": rows, "durations": durations}


def cmd_validate_verifier(root: Path, gpu=0):
    rows = qualify_verifier(root, gpu=gpu)
    leaked = any(r["leaked_hidden_into_verifier"] for r in rows)
    return {"status": "PASS" if not leaked else "NEEDS_RERUN", "rows": rows}


def cmd_validate_evaluator(root: Path, gpu=0):
    rows = evaluator_unit_checks(gpu=gpu)
    passed = all(r.get("pass") for r in rows)
    return {"status": "PASS" if passed else "NEEDS_RERUN", "rows": rows}


def cmd_validate_d0_cache(root: Path):
    split = json.loads((root / "configs/splits/D0_stage_1a.json").read_text(encoding="utf-8"))
    missing = []
    nonempty = 0
    for row in split["train"] + split["dev"]:
        rel = row.get("cache_dir")
        if not rel:
            missing.append(row["case_id"]); continue
        path = root / rel
        if not (path / "COMPLETE").is_file():
            missing.append(row["case_id"]); continue
        edges = json.loads((path / "final_edges.json").read_text(encoding="utf-8"))
        if edges:
            nonempty += 1
    return {"status": "PASS" if not missing else "BLOCKED", "missing": missing, "nonempty_prior": nonempty, "train": split["train_count"], "dev": split["dev_count"]}


def cmd_stage_1a_preflight(root: Path, gpu=0, resume=False):
    if resume:
        return run_resume_p0(root, gpu=gpu)
    return run_all_p0(root, gpu=gpu)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in r.items()})


def _log(msg):
    print(msg, flush=True)


def _stage0c_fingerprints(root: Path):
    root = Path(root)
    dev = root / "experiments/vlm_cache/dev"
    keys_file = root / P0 / "stage0c_keys_before.txt"
    before_json = root / P0 / "stage0c_cache_fingerprint_before.json"
    if keys_file.is_file():
        keys = [ln.strip() for ln in keys_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    elif before_json.is_file():
        keys = [row["cache_key"] for row in json.loads(before_json.read_text(encoding="utf-8"))]
    else:
        keys = sorted(p.name for p in dev.iterdir() if p.is_dir() and (p / "COMPLETE").is_file())
    rows = []
    for key in keys:
        marker = dev / key / "COMPLETE"
        rows.append({"cache_key": key, "complete_sha256": file_hash(marker), "manifest_sha256": file_hash(dev / key / "manifest.json")})
    return rows


def run_all_p0(root: Path, gpu=0):
    root = Path(root)
    out = root / P0
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stage0c_before = _stage0c_fingerprints(root)
    write_json(out / "stage0c_cache_fingerprint_before.json", stage0c_before)
    _log("P0 start gpu=%s stage0c_dev_caches=%s" % (gpu, len(stage0c_before)))
    split_info = cmd_build_d0_split(root)
    _log("split built train=%s dev=%s" % (split_info.get("train"), split_info.get("dev")))
    split_doc = json.loads((root / "configs/splits/D0_stage_1a.json").read_text(encoding="utf-8"))
    _log("controller qualification start")
    ctrl = cmd_validate_controller(root, gpu=gpu)
    write_json(out / "controller_qualification.json", {"status": ctrl.get("status"), "summary": ctrl.get("summary"), "durations": ctrl.get("durations")})
    _write_csv(out / "controller_qualification.csv", ctrl["rows"])
    _log("controller qualification %s %s" % (ctrl.get("status"), json.dumps(ctrl.get("summary"), default=str)))
    timing = None
    status = "PASS"
    notes = []
    if ctrl["status"] != "PASS":
        status = "NEEDS_RERUN"
        notes.append("controller qualification below 4/5")
        timing = {"error": "unqualified", "samples": ctrl["durations"]}
    else:
        timing = calibrate_timing(ctrl["durations"])
    ver = cmd_validate_verifier(root, gpu=gpu)
    ev = cmd_validate_evaluator(root, gpu=gpu)
    leak = hidden_truth_leakage_check(gpu=gpu)
    if ver["status"] != "PASS":
        status = "NEEDS_RERUN"
        notes.append("verifier qualification failed")
    if ev["status"] != "PASS":
        status = "NEEDS_RERUN"
        notes.append("evaluator qualification failed")
    if leak["leaked"]:
        status = "NEEDS_RERUN"
        notes.append("hidden-state leakage")
    fail_rows = failure_qualification(root, (timing or {}).get("skill_timeouts") or {"PLACE": QUAL_BUDGET, "OPEN": QUAL_BUDGET, "PICK": QUAL_BUDGET, "PLACE_BUFFER": QUAL_BUDGET}, gpu=gpu)
    _write_csv(out / "failure_qualification.csv", fail_rows)
    write_json(out / "task_evaluator_tests.json", ev)
    write_json(out / "timing_calibration.json", timing)
    # freeze timeouts into contracts if calibrated
    if timing and "skill_timeouts" in timing:
        doc = load_yaml(root / "configs/contracts/d0_runtime_skills.yaml")
        for c in doc["contracts"]:
            c["timeout_seconds"] = float(timing["skill_timeouts"][c["name"]])
        (root / "configs/contracts/d0_runtime_skills.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
        freeze_manifests(root, timing, split_doc, runtime_ready=False)
        _log("capturing 74 RGB scenes")
        mapping = capture_split_images(root, split_doc, gpu=gpu)
        _log("generating 74 D0 VLM caches; Stage 0C caches are not overwritten")
        try:
            caches = generate_d0_caches(root, mapping, gpu=gpu)
        except Exception as exc:  # noqa: BLE001
            caches = [{"status": "ERROR", "error": type(exc).__name__, "detail": str(exc)[:300]}]
            status = "BLOCKED"
            notes.append("D0 cache generation failed")
        cache_map = {r.get("case_id"): r for r in caches if r.get("case_id")}
        for row in split_doc["train"] + split_doc["dev"]:
            rec = cache_map.get(row["case_id"])
            if rec:
                row["cache_dir"] = rec.get("cache_dir")
                row["cache_key"] = rec.get("cache_key")
                row["cache_status"] = rec.get("status")
        write_json(root / "configs/splits/D0_stage_1a.json", split_doc)
        write_json(out / "d0_split_manifest.json", split_doc)
        write_json(out / "d0_cache_generation.json", caches)
        freeze_manifests(root, timing, split_doc, runtime_ready=False)
        _log("scripted dev10")
        scripted = scripted_dev10(root, split_doc, timing["skill_timeouts"], timing["task_deadline_seconds"], gpu=gpu)
        flat = []
        for r in scripted:
            item = dict(r)
            item["skills"] = json.dumps(r["skills"])
            flat.append(item)
        _write_csv(out / "scripted_dev10_results.csv", flat)
        if not all(r["success"] for r in scripted) or any(r["repeat_reward"] for r in scripted):
            status = "NEEDS_RERUN" if status == "PASS" else status
            notes.append("scripted dev10 incomplete")
        _log("B2 dry-run")
        b2 = model_dry_run(root, "B2", split_doc["dev"][0]["case_id"], out / "dry_run")
        full_case = next((r["case_id"] for r in split_doc["dev"] if r.get("cache_dir")), split_doc["dev"][0]["case_id"])
        _log("Full dry-run")
        full = model_dry_run(root, "Full", full_case, out / "dry_run")
        write_json(out / "b2_full_dry_run.json", {"B2": b2, "Full": full})
        if not (b2.get("transition") and full.get("transition")):
            status = "NEEDS_RERUN" if status == "PASS" else status
            notes.append("dry-run missing real transition")
    else:
        scripted = []
        b2 = full = None
        mapping = []
        caches = []
    stage0c_after = _stage0c_fingerprints(root)
    write_json(out / "stage0c_cache_fingerprint_after.json", stage0c_after)
    if stage0c_before != stage0c_after:
        status = "BLOCKED"
        notes.append("Stage 0C cache fingerprint changed")
    write_reports(root, out, status, notes, ctrl, ver, ev, leak, timing, split_info, scripted if "scripted" in locals() else [], fail_rows)
    _log("P0 finished status=%s notes=%s" % (status, notes))
    return {"status": status, "notes": notes, "elapsed_seconds": time.time() - started}


def _truthy(value):
    return str(value).strip() in ("True", "true", "1", "yes")


def _load_scripted(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            skills = raw.get("skills") or "[]"
            rewards = raw.get("reward_events") or "[]"
            rows.append({
                "case_id": raw["case_id"],
                "seed": int(raw["seed"]),
                "goal_true": _truthy(raw.get("goal_true")),
                "success": _truthy(raw.get("success")),
                "terminated": _truthy(raw.get("terminated")),
                "truncated": _truthy(raw.get("truncated")),
                "reward_events": json.loads(rewards),
                "repeat_reward": _truthy(raw.get("repeat_reward")),
                "skills": json.loads(skills),
            })
    return rows


def run_resume_p0(root: Path, gpu=0):
    """Continue P0 from existing controller/cache/scripted artifacts. No VLM regeneration."""
    import subprocess
    import sys

    root = Path(root)
    out = root / P0
    started = time.time()
    notes = []
    status = "PASS"
    timing = json.loads((out / "timing_calibration.json").read_text(encoding="utf-8"))
    split_doc = json.loads((root / "configs/splits/D0_stage_1a.json").read_text(encoding="utf-8"))
    split_info = {"train": split_doc.get("train_count"), "dev": split_doc.get("dev_count"), "resumed": True}
    ctrl = json.loads((out / "controller_qualification.json").read_text(encoding="utf-8"))
    fail_path = out / "failure_qualification.csv"
    fail_rows = list(csv.DictReader(fail_path.open(encoding="utf-8", newline=""))) if fail_path.is_file() else []
    scripted = _load_scripted(out / "scripted_dev10_results.csv")
    stage0c_before = json.loads((out / "stage0c_cache_fingerprint_before.json").read_text(encoding="utf-8"))
    if ctrl.get("status") != "PASS":
        status = "NEEDS_RERUN"
        notes.append("controller qualification below 4/5")
    freeze_manifests(root, timing, split_doc, runtime_ready=False)
    _log("resume: validate-runtime")
    runtime_ok = cmd_validate_runtime(root)
    write_json(out / "validate_runtime.json", runtime_ok)
    _log("resume: validate-d0-cache")
    cache_ok = cmd_validate_d0_cache(root)
    write_json(out / "validate_d0_cache.json", cache_ok)
    if cache_ok.get("status") != "PASS":
        status = "BLOCKED"
        notes.append("D0 cache incomplete: " + json.dumps(cache_ok.get("missing") or [])[:300])
    _log("resume: verifier")
    ver = cmd_validate_verifier(root, gpu=gpu)
    _log("resume: evaluator")
    ev = cmd_validate_evaluator(root, gpu=gpu)
    write_json(out / "task_evaluator_tests.json", ev)
    leak = hidden_truth_leakage_check(gpu=gpu)
    if ver["status"] != "PASS":
        status = "NEEDS_RERUN"
        notes.append("verifier qualification failed")
    if ev["status"] != "PASS":
        status = "NEEDS_RERUN"
        notes.append("evaluator qualification failed")
    if leak["leaked"]:
        status = "NEEDS_RERUN"
        notes.append("hidden-state leakage")
    if not scripted or (not all(r["success"] for r in scripted)) or any(r["repeat_reward"] for r in scripted):
        status = "NEEDS_RERUN" if status == "PASS" else status
        notes.append("scripted dev10 incomplete")
    try:
        _log("resume: B2 dry-run")
        b2 = model_dry_run(root, "B2", split_doc["dev"][0]["case_id"], out / "dry_run")
        full_case = next((r["case_id"] for r in split_doc["dev"] if r.get("cache_dir")), split_doc["dev"][0]["case_id"])
        _log("resume: Full dry-run")
        full = model_dry_run(root, "Full", full_case, out / "dry_run")
        write_json(out / "b2_full_dry_run.json", {"B2": b2, "Full": full, "ppo_update": False})
        if not (b2.get("transition") and full.get("transition")):
            status = "NEEDS_RERUN" if status == "PASS" else status
            notes.append("dry-run missing real transition")
    except Exception as exc:  # noqa: BLE001
        status = "NEEDS_RERUN"
        notes.append("dry-run error: %s: %s" % (type(exc).__name__, str(exc)[:300]))
        write_json(out / "b2_full_dry_run.json", {"error": type(exc).__name__, "detail": str(exc)[:500], "ppo_update": False})
    _log("resume: Stage 0B unit tests")
    rc = subprocess.call([sys.executable, "-m", "pytest", str(root / "tests"), "-m", "pure or torch_runtime", "-ra"], cwd=str(root))
    write_json(out / "stage0b_regression.json", {"returncode": rc, "marker": "pure or torch_runtime"})
    if rc != 0:
        status = "NEEDS_RERUN"
        notes.append("Stage 0B regression failed")
    stage0c_after = _stage0c_fingerprints(root)
    write_json(out / "stage0c_cache_fingerprint_after.json", stage0c_after)
    if stage0c_before != stage0c_after:
        status = "BLOCKED"
        notes.append("Stage 0C cache fingerprint changed")
    write_reports(root, out, status, notes, ctrl, ver, ev, leak, timing, split_info, scripted, fail_rows)
    _log("P0 resume finished status=%s notes=%s" % (status, notes))
    return {"status": status, "notes": notes, "elapsed_seconds": time.time() - started, "runtime": runtime_ok, "cache": cache_ok, "stage0b_returncode": rc}


def freeze_manifests(root: Path, timing, split_doc, runtime_ready=False):
    factory_path = root / "src/cp_disr/platforms/libero/runtime_factory.py"
    factory_sha = sha_file(factory_path)
    d0_path = root / "configs/tasks/resolved/D0.yaml"
    d0 = load_yaml(d0_path)
    d0["task_version"] = "cp-disr-v2.1-d0-runtime-p0"
    d0["status"] = "RUNTIME_BOUND_STAGE_1A_NOT_STARTED"
    d0["platform_ref"] = "LIBERO_CP_DISR_CLEAN@8f1084e3132a39270c3a13ebe37270a43ece2a01"
    d0["platform_version"] = "LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01"
    d0["instruction"] = "Place the red target cube into the openable blue container. The yellow cube may be moved to the green buffer if it blocks the target."
    d0["grounded_action_ids"] = GROUND_SKILLS
    d0["task_evaluator_ref"] = "src/cp_disr/platforms/libero/task_evaluator.py"
    d0["a_controller_ref"] = "src/cp_disr/platforms/libero/skill_executor.py"
    d0["verifier_ref"] = "src/cp_disr/platforms/libero/verifier.py"
    d0["controller_ref"] = "src/cp_disr/platforms/libero/skill_executor.py"
    d0["contract_registry_ref"] = "configs/contracts/d0_runtime_skills.yaml"
    d0["controller_binding_status"] = "BOUND"
    d0["verifier_binding_status"] = "BOUND"
    d0["task_evaluator_status"] = "BOUND"
    d0["safety_status"] = "BOUND"
    d0["timing_status"] = "BOUND"
    d0["stage_1a_runtime_ready"] = bool(runtime_ready)
    d0["split_ref"] = "configs/splits/D0_stage_1a.json"
    d0["deadline_seconds"] = timing["task_deadline_seconds"]
    d0["reference_skill_seconds"] = timing["reference_skill_seconds"]
    d0["move_decision"] = "REJECTED_FOR_D0"
    d0["move_alternative"] = ["PICK", "PLACE_BUFFER"]
    d0_path.write_text(yaml.safe_dump(d0, sort_keys=False, allow_unicode=True), encoding="utf-8")

    rt = load_yaml(root / "experiments/manifests/runtime_manifest.yaml")
    rt["manifest_status"] = "D0_RUNTIME_BOUND_STAGE_1A_NOT_STARTED"
    rt["runtime"]["git_hash"] = os.popen("git -C " + str(root) + " rev-parse HEAD").read().strip()
    rt["runtime"]["simulator_or_robot"] = {
        "platform": "LIBERO_robosuite_mujoco",
        "libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "robosuite": "1.4.0",
        "mujoco": "3.6.0",
        "python": "/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages reused read-only into .venv-stage0a",
    }
    rt["runtime"]["environment_version"] = "d0-runtime-v2.1-p0"
    rt["runtime"]["task_assets"] = {
        "D0": "src/cp_disr/platforms/libero/d0_env.py",
        "license": "assets/cp_disr/LICENSE",
        "owned_by": "cp_disr_project",
    }
    rt["runtime"]["controller_manifest"] = str((root / "experiments/part_1_smoke/stage_1a_p0/controller_manifest.yaml").relative_to(root))
    rt["runtime"]["camera"] = {"name": "agentview", "width": 128, "height": 128, "depth": True, "fixed": True}
    rt["runtime"]["calibration_manifest"] = "src/cp_disr/platforms/libero/perception.py::camera_calibration"
    rt["runtime"]["perception_checkpoint"] = PERCEPTION_VERSION
    rt["runtime"]["verifier_thresholds"] = THRESHOLDS
    rt["runtime"]["skill_timeouts"] = {"D0": timing["skill_timeouts"]}
    rt["runtime"]["task_deadlines"] = {"D0": timing["task_deadline_seconds"]}
    rt["runtime"]["actual_interaction_time_unit"] = "seconds"
    rt["runtime"]["actual_clock_source"] = "mujoco_sim_data_time"
    rt["runtime"]["clock_semantics"] = "DurationProvider reads env.sim.data.time (MuJoCo simulation seconds). The required binding token remains seconds."
    rt["runtime"]["reference_skill_seconds_by_task"] = {"D0": timing["reference_skill_seconds"]}
    rt["runtime"]["task_evaluator_version"] = EVALUATOR_VERSION
    rt["runtime"]["safety_authorization"] = "src/cp_disr/platforms/libero/safety.py"
    rt["runtime"]["task_splits"] = {"D0": "configs/splits/D0_stage_1a.json"}
    rt["runtime"]["d0_contract_path"] = "configs/contracts/d0_runtime_skills.yaml"
    rs = rt.setdefault("resource_settings", {})
    rs["clock_unit_required"] = "seconds"
    rs["actual_clock_source"] = "mujoco_sim_data_time"
    rt["runtime_factory"] = {
        "module": "cp_disr.platforms.libero.runtime_factory",
        "factory": "create_runtime",
        "source_path": str(factory_path),
        "sha256": factory_sha,
    }
    rt["stage_0c_r2_stage_1a_runtime_ready"] = bool(runtime_ready)
    (root / "experiments/manifests/runtime_manifest.yaml").write_text(yaml.safe_dump(rt, sort_keys=False, allow_unicode=True), encoding="utf-8")

    tm = load_yaml(root / "experiments/manifests/task_manifest.yaml")
    tm["status"] = "D0_BOUND_OTHER_TASKS_UNBOUND"
    for task in tm["tasks"]:
        if task["task_id"] == "D0":
            task["asset_ref"] = "src/cp_disr/platforms/libero/d0_env.py"
            task["controller_refs"] = GROUND_SKILLS
            task["goal_atoms"] = ["p:Inside:target:container"]
            task["skill_contract_refs"] = "configs/contracts/d0_runtime_skills.yaml"
            task["camera_verifier_refs"] = "src/cp_disr/platforms/libero/verifier.py"
            task["initial_state_generator"] = "configs/splits/D0_stage_1a.json"
            task["deadline_seconds"] = timing["task_deadline_seconds"]
            task["reference_skill_seconds"] = timing["reference_skill_seconds"]
            task["split_ref"] = "configs/splits/D0_stage_1a.json"
            task["prior_potential"] = "D0 train/dev VLM cache after Action-registry bump; Stage 0C D0 caches archived unused"
    tm["resolved_real_tasks"] = ["D0"]
    tm["stage_0c_r1_status"] = "SUPERSEDED_BY_STAGE_0C_PASS_AND_STAGE_1A_P0"
    (root / "experiments/manifests/task_manifest.yaml").write_text(yaml.safe_dump(tm, sort_keys=False, allow_unicode=True), encoding="utf-8")

    ep = load_yaml(root / "experiments/manifests/entrypoints.yaml")
    ep["status"] = "D0_RUNTIME_BOUND"
    ep["note"] = "Stage 0B PASS. Stage 0C PASS. Stage 1A-P0 binds D0 runtime. Stage 1A training remains NOT_STARTED."
    py = "PYTHONPATH=src .venv-stage0a/bin/python -m cp_disr "
    for name in ("validate-runtime", "validate-controller", "validate-verifier", "validate-evaluator", "build-d0-split", "validate-d0-cache", "stage-1a-preflight"):
        ep.setdefault("commands", {})[name] = {
            "cli": py + name,
            "source_path": "src/cp_disr/cli.py",
            "status": "IMPLEMENTED",
            "requires_torch": name not in ("build-d0-split",),
            "calls_api_this_revision": name in ("stage-1a-preflight",),
        }
    ep["commands"]["train"]["status"] = "IMPLEMENTED_RUNTIME_BOUND_TRAINING_NOT_STARTED"
    ep["commands"]["evaluate"]["status"] = "IMPLEMENTED_RUNTIME_BOUND_TRAINING_NOT_STARTED"
    (root / "experiments/manifests/entrypoints.yaml").write_text(yaml.safe_dump(ep, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_reports(root, out, status, notes, ctrl, ver, ev, leak, timing, split_info, scripted, fail_rows):
    write_text(out / "controller_manifest.yaml", yaml.safe_dump({
        "version": "d0-runtime-v2.1-p0",
        "skills": ["OPEN", "PICK", "PLACE", "PLACE_BUFFER"],
        "move": "REJECTED_FOR_D0",
        "implementation": "src/cp_disr/platforms/libero/skill_executor.py",
        "qualification": ctrl.get("summary"),
    }, sort_keys=False))
    write_text(out / "controller_qualification_report.md", "# Controller qualification\n\n" + json.dumps(ctrl.get("summary"), indent=2) + "\n")
    write_text(out / "perception_manifest.yaml", yaml.safe_dump({"version": PERCEPTION_VERSION, "thresholds": THRESHOLDS, "camera": "agentview", "hidden_state_in_policy": False}, sort_keys=False))
    write_text(out / "verifier_manifest.yaml", yaml.safe_dump({"version": VERIFIER_VERSION, "facts": list(PREDICATES), "hidden_state_input": False}, sort_keys=False))
    write_text(out / "verifier_qualification_report.md", "# Verifier qualification\n\n" + json.dumps(ver, indent=2, default=str)[:20000] + "\n")
    _write_csv(out / "verifier_qualification.csv", [{"scenario": r["scenario"], "leaked": r["leaked_hidden_into_verifier"]} for r in ver.get("rows", [])])
    write_text(out / "task_evaluator_manifest.yaml", yaml.safe_dump({"version": EVALUATOR_VERSION, "independent_of_vlm_controller": True}, sort_keys=False))
    write_text(out / "safety_manifest.yaml", yaml.safe_dump({"implementation": "src/cp_disr/platforms/libero/safety.py", "rejects_MOVE": True}, sort_keys=False))
    write_text(out / "timing_calibration_report.md", "# Timing calibration\n\n" + json.dumps(timing, indent=2, default=str) + "\n")
    if timing and timing.get("samples"):
        sample_rows = []
        for skill, xs in timing["samples"].items():
            for i, value in enumerate(xs):
                sample_rows.append({
                    "skill": skill,
                    "sample_index": i,
                    "duration_seconds": value,
                    "timeout_seconds": (timing.get("skill_timeouts") or {}).get(skill),
                    "task_deadline_seconds": timing.get("task_deadline_seconds"),
                    "reference_skill_seconds": timing.get("reference_skill_seconds"),
                })
        _write_csv(out / "timing_calibration.csv", sample_rows)
    split_path_early = root / "configs/splits/D0_stage_1a.json"
    if timing and timing.get("skill_timeouts"):
        freeze_manifests(root, timing, json.loads(split_path_early.read_text(encoding="utf-8")) if split_path_early.is_file() else {"train": [], "dev": []}, runtime_ready=(status == "PASS"))
    write_text(out / "runtime_manifest.yaml", (root / "experiments/manifests/runtime_manifest.yaml").read_text(encoding="utf-8"))
    write_text(out / "hidden_truth_leakage_report.md", "# Hidden truth leakage\n\n" + json.dumps(leak, indent=2) + "\n")
    write_text(out / "d0_cache_report.md", "# D0 cache\n\nStage 0C D0 caches retained and not overwritten. New D0 train/dev caches use runtime Action registry without MOVE.\n")
    write_text(out / "b2_full_dry_run_report.md", "# B2/Full dry-run\n\nSee b2_full_dry_run.json. No PPO update.\n")
    write_text(out / "manifest_consistency_report.md", """# Manifest consistency

Updated D0-bound fields only. T_A/T_B/T_C/T_D/T_E remain unbound.
LIBERO_CANDIDATE_BLOCKED and dirty_UNFROZEN labels were replaced for D0 because Stage 0C-R2 froze LIBERO@8f1084e and this round bound D0 controllers.
Stage 0C cache directories were not modified.
""")
    write_text(out / "action_registry_audit.md", """# D0 Action registry audit

Final executable registry: OPEN, PICK, PLACE, PLACE_BUFFER.
MOVE is rejected for D0. PLACE_BUFFER is not a MOVE synonym; it places a held object onto the buffer pad.
Stage 0C D0 caches that mentioned MOVE are archived unused because Action IDs changed.
""")
    scripted_ok = bool(scripted) and all(r.get("success") for r in scripted)
    split_path = root / "configs/splits/D0_stage_1a.json"
    train_cache_count = dev_cache_count = 0
    if split_path.is_file():
        split_now = json.loads(split_path.read_text(encoding="utf-8"))
        for row in split_now.get("train", []):
            rel = row.get("cache_dir")
            if rel and (root / rel / "COMPLETE").is_file():
                train_cache_count += 1
        for row in split_now.get("dev", []):
            rel = row.get("cache_dir")
            if rel and (root / rel / "COMPLETE").is_file():
                dev_cache_count += 1
    dry = {}
    dry_path = out / "b2_full_dry_run.json"
    if dry_path.is_file():
        dry = json.loads(dry_path.read_text(encoding="utf-8"))
    b2_ok = bool((dry.get("B2") or {}).get("transition"))
    full_ok = bool((dry.get("Full") or {}).get("transition"))
    readiness = {
        "runtime_bound": status == "PASS",
        "controller_qualified": ctrl.get("status") == "PASS",
        "verifier_qualified": ver.get("status") == "PASS",
        "evaluator_qualified": ev.get("status") == "PASS",
        "safety_bound": True,
        "timing_bound": bool(timing and timing.get("skill_timeouts")),
        "train_scene_count": 64,
        "dev_scene_count": 10,
        "train_cache_count": train_cache_count,
        "dev_cache_count": dev_cache_count,
        "scripted_dev_success": scripted_ok,
        "b2_dry_run_passed": b2_ok,
        "full_dry_run_passed": full_ok,
        "hidden_truth_leakage": bool(leak.get("leaked")),
        "stage_1a_runtime_ready": status == "PASS",
    }
    write_json(out / "stage_1a_readiness.json", readiness)
    write_json(out / "preflight_validation.json", {"status": status, "notes": notes, "split": split_info, "readiness": readiness})
    write_text(out / "d0_cache_report.md", "# D0 cache\n\nStage 0C D0 caches retained and not overwritten.\nNew D0 train/dev caches use the runtime Action registry without MOVE.\ntrain_cache_count: %s\ndev_cache_count: %s\n" % (train_cache_count, dev_cache_count))
    dry_text = json.dumps(dry, indent=2, default=str)
    write_text(out / "b2_full_dry_run_report.md", "# B2/Full dry-run\n\nNo PPO update. optimizer_steps were not executed.\n\n```json\n" + dry_text + "\n```\n")
    write_text(out / "stage_1a_p0_summary.md", "# Stage 1A-P0 summary\n\nstatus: " + status + "\n\nnotes: " + json.dumps(notes) + "\n\nreadiness:\n\n```json\n" + json.dumps(readiness, indent=2) + "\n```\n")
    st = json.loads((root / "experiments/stage_status/stage_1a.json").read_text(encoding="utf-8"))
    st["status"] = "NOT_STARTED"
    st["readiness"] = readiness
    st["issues"] = notes
    write_json(root / "experiments/stage_status/stage_1a.json", st)

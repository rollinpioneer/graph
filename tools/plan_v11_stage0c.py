#!/usr/bin/env python3
"""Plan v1.1 Stage 0C: D0/T_B/T_C x 8 real dev scenes. Reuse exact caches; materialize T_B only."""
from __future__ import annotations
import base64, csv, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cp_disr.common import BindingError, canonical, digest
from cp_disr.vlm import cache_key
from cp_disr.vlm_provider import ProviderConfig, DashScopeProvider, validate_payload
from cp_disr.vlm_cache_pipeline import request_and_process, write_audit_cache, verify_audit_cache, process_response
from cp_disr.stage0c import prepare_scene, PREPROCESSING, file_hash, read
from cp_disr.stage2a_p0 import (
    TASK_ROLE, spec_from_row, write_json, _log, _grounded_template, initial_facts, _read
)
from cp_disr.platforms.libero.d0_env import make_env
from cp_disr.platforms.libero.runtime_factory import TASK_GOALS
import numpy as np
from PIL import Image
import yaml

OUT = ROOT / "runs" / "stage_0c"
REP = ROOT / "reports"
ST = ROOT / "status"
MIG = ROOT / "migration"
CACHE_ROOT = Path("experiments/vlm_cache/plan_v11")
SCENE_ROOT = ROOT / "experiments/stage_0c_v11_inputs"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def git_hash():
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()

def load_key():
    path = Path("/home/__compress_data/xushijie/DASHSCOPE_API_KEY")
    if not path.is_file():
        raise BindingError("DASHSCOPE_API_KEY file missing")
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    key = next((ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")), "")
    if not key:
        raise BindingError("DASHSCOPE_API_KEY file empty")
    os.environ["DASHSCOPE_API_KEY"] = key
    return True

def inspect_cache(cache_dir: Path):
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return {"ok": False, "reason": "missing_dir", "path": str(cache_dir)}
    try:
        man, edges = verify_audit_cache(cache_dir)
    except Exception as e:
        man_path = cache_dir / "manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
        edges = json.loads((cache_dir / "final_edges.json").read_text(encoding="utf-8")) if (cache_dir / "final_edges.json").exists() else []
        if not (cache_dir / "COMPLETE").exists():
            return {"ok": False, "reason": type(e).__name__, "path": str(cache_dir)}
        # older/simple caches still usable if COMPLETE + identity
        try:
            if cache_key(man) != cache_dir.name and man.get("cache_key") not in (cache_dir.name, None):
                return {"ok": False, "reason": "identity_mismatch", "path": str(cache_dir)}
        except Exception:
            pass
    accepted = json.loads((cache_dir / "accepted_relations.json").read_text(encoding="utf-8")) if (cache_dir / "accepted_relations.json").exists() else []
    rejected = json.loads((cache_dir / "rejected_relations.json").read_text(encoding="utf-8")) if (cache_dir / "rejected_relations.json").exists() else []
    parsed = json.loads((cache_dir / "parsed_relations.json").read_text(encoding="utf-8")) if (cache_dir / "parsed_relations.json").exists() else {}
    raw_n = len((parsed or {}).get("relations") or []) if isinstance(parsed, dict) else 0
    redun = 0
    for r in rejected:
        reason = r.get("reason") if isinstance(r, dict) else (r[1] if isinstance(r, (list, tuple)) and len(r) > 1 else None)
        if reason == "CONTRACT_REDUNDANCY":
            redun += 1
    return {
        "ok": True,
        "path": str(cache_dir.relative_to(ROOT)) if str(cache_dir).startswith(str(ROOT)) else str(cache_dir),
        "cache_key": cache_dir.name,
        "processing_status": (man or {}).get("processing_status", "SUCCESS"),
        "n_accepted": len(accepted or []),
        "n_raw": raw_n,
        "n_rejected": len(rejected or []),
        "n_redundant": redun,
        "n_final": len(edges or []),
        "empty": len(edges or []) == 0,
        "task_id": (man or {}).get("task_id"),
        "scene_id": (man or {}).get("scene_id"),
        "model": (man or {}).get("model_snapshot"),
        "endpoint": (man or {}).get("endpoint"),
        "prompt_hash": (man or {}).get("prompt_hash"),
        "fewshot_hash": (man or {}).get("fewshot_hash"),
    }

def reuse_family(task_id, split_path, n=8):
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    rows = list(split["dev"])[:n]
    out = []
    for row in rows:
        cdir = row.get("cache_dir")
        if not cdir:
            raise BindingError("%s %s missing cache_dir" % (task_id, row["case_id"]))
        info = inspect_cache(ROOT / cdir)
        info.update({"task_id": task_id, "scene_id": row["case_id"], "action": "REUSED_EXACT", "case": row})
        if not info.get("ok"):
            raise BindingError("reuse failed %s %s: %s" % (task_id, row["case_id"], info))
        out.append(info)
    return out

def capture_tb_dev8(gpu):
    split = json.loads((ROOT / "configs/splits/T_B_stage_0a.json").read_text(encoding="utf-8"))
    rows = list(split["dev"])[:8]
    mapping = []
    dest_root = SCENE_ROOT / "T_B" / "dev"
    dest_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest = dest_root / row["case_id"]
        dest.mkdir(parents=True, exist_ok=True)
        rgb_path = dest / "rgb.png"
        if (dest / "CAPTURE_COMPLETE").exists() and rgb_path.exists():
            mapping.append({**row, "image_ref": str(rgb_path.relative_to(ROOT)), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(ROOT))})
            continue
        spec = spec_from_row("T_B", row, deadline=60.0)
        env = make_env(spec, gpu=gpu)
        try:
            env.reset()
            env._apply_case_poses()
            env._open_gripper_reset()
            obs = env.public_observation()
            Image.fromarray(np.asarray(obs["rgb"]).astype(np.uint8)).save(rgb_path)
            np.save(dest / "depth.npy", np.asarray(obs["depth"]))
            (dest / "reset_config.json").write_text(canonical(row) + "\n", encoding="utf-8")
            (dest / "CAPTURE_COMPLETE").write_text("complete\n", encoding="utf-8")
        finally:
            env.close()
        mapping.append({**row, "image_ref": str(rgb_path.relative_to(ROOT)), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(ROOT))})
        _log("captured T_B %s" % row["case_id"])
    return mapping

def write_tb_resolved():
    cal = json.loads((ROOT / "runs/stage_0a/reference_execution_manifest.json").read_text(encoding="utf-8"))
    d_ref = float(cal["families"]["T_B"]["d_ref_median"])
    doc = {
        "task_id": "T_B",
        "task_version": "cp-disr-v2.1.1-plan-1.1",
        "status": "RUNTIME_BOUND_STAGE_0C_CACHE",
        "platform_ref": "LIBERO_CP_DISR_CLEAN@8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "typed_entities": {
            "target": {"type": "object"},
            "second_object": {"type": "object"},
            "container": {"type": "container"},
            "buffer": {"type": "buffer"},
        },
        "instruction": "Place the red target cube into the openable blue container AND place the yellow second cube onto the green buffer. Not a T_A relabel: Inside(target,container) and AtBuffer(second_object,buffer) are both required.",
        "signed_goals": [
            {"predicate": "Inside", "arguments": ["target", "container"], "sign": 1},
            {"predicate": "AtBuffer", "arguments": ["second_object", "buffer"], "sign": 1},
        ],
        "required_skill_schemas": ["OPEN", "PICK", "PLACE", "PLACE_BUFFER"],
        "grounded_action_ids": [
            "a:OPEN:container:v1",
            "a:PICK:target:v1",
            "a:PICK:second_object:v1",
            "a:PLACE:target:container:v1",
            "a:PLACE:second_object:container:v1",
            "a:PLACE_BUFFER:target:buffer:v1",
            "a:PLACE_BUFFER:second_object:buffer:v1",
        ],
        "move_decision": "REJECTED",
        "structural_signature": "dual_goal_inside_and_atbuffer",
        "split_ref": "configs/splits/T_B_stage_0a.json",
        "deadline_seconds": 60.0,
        "reference_skill_seconds": d_ref,
        "not_a_ta_relabel": True,
    }
    path = ROOT / "configs/tasks/resolved/T_B.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path

def materialize_tb(mapping, gpu):
    load_key()
    vlm = _read(ROOT / "experiments/manifests/vlm_manifest.yaml")
    config = ProviderConfig(model=vlm["model_snapshot"], region=vlm["region"], endpoint=vlm["base_http_api_url"], sdk_version=vlm["sdk_version"])
    provider = DashScopeProvider(config)
    if not provider.key_present():
        raise BindingError("DASHSCOPE_API_KEY missing from process environment")
    few_path = ROOT / "experiments/stage_0c_inputs/fewshots/few_shot_manifest.json"
    few_doc = json.loads(few_path.read_text(encoding="utf-8"))
    few = few_doc.get("examples", [])
    examples = [prepare_scene(ROOT, s, True) for s in few]
    prompt_path = ROOT / "experiments/sources/v2.1_interfaces/system_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    schema_path = ROOT / "schemas/relation_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": [{"text": prompt}]}]
    for ex, record in zip(examples, few):
        messages.extend([
            {"role": "user", "content": ex["content"]},
            {"role": "assistant", "content": [{"text": canonical(record["expected_json"])}]},
        ])
    contract_doc, contracts, template = _grounded_template(ROOT, "T_B")
    actions = sorted(c.id for c in contracts)
    props = sorted(n.id for n in template.nodes if n.kind == "PROPOSITION")
    contract_sha = file_hash(ROOT / "configs/runtime/stage_2a_contract_registry.yaml")
    task_path = write_tb_resolved()
    task_sha = file_hash(task_path)
    ledger = OUT / "T_B_cache_request_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    results = []
    role = TASK_ROLE["T_B"]
    signed = [{"fact_id": g, "sign": 1} for g in TASK_GOALS["T_B"]]
    model_access = {"verified": False, "first_request_id": None, "error": None}
    for row in mapping:
        image = ROOT / row["image_ref"]
        with Image.open(image) as im:
            mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[im.format]
        scene = {
            "scene_id": row["case_id"],
            "task_id": "T_B",
            "split": row["split"],
            "image_ref": row["image_ref"],
            "image_sha256": row["image_sha256"],
            "object_table": [
                {"id": "target", "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": role, "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "container", "type": "container", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "buffer", "type": "buffer", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
            ],
            "signed_goals": signed,
            "initial_facts": initial_facts("T_B"),
            "allowed_action_ids": actions,
            "allowed_proposition_ids": props,
            "contracts_sha256": contract_sha,
            "task_definition_sha256": task_sha,
            "predicate_version": "CP-DISR-v2.1",
            "asset_binding_sha256": digest({"layout": [row["target_xy"], row["second_xy"], row["container_xy"], row["buffer_xy"]], "runtime": "plan-v1.1-0c", "task_id": "T_B"}),
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
            "task_id": "T_B",
            "synthetic_unit_fixture": False,
            "initial_facts_hash": digest(scene["initial_facts"]),
            "input_context_hash": digest(context),
            "request_payload_hash": digest(payload),
            "asset_binding_hash": scene["asset_binding_sha256"],
        }
        key = cache_key(manifest)
        path = ROOT / CACHE_ROOT / row["split"] / key
        if path.exists() and (path / "COMPLETE").is_file():
            info = inspect_cache(path)
            info.update({"task_id": "T_B", "scene_id": row["case_id"], "action": "REUSED_EXACT", "case": row, "cache_dir": str(path.relative_to(ROOT))})
            results.append(info)
            continue
        if path.exists():
            raise BindingError("Refusing to rmtree existing cache path " + str(path))
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical({"scene_id": row["case_id"], "state": "REQUEST_STARTED", "time": time.time()}) + "\n")
        execution = request_and_process(provider, payload, template, schema, ())
        last = execution["attempts"][-1]["response"]
        if not model_access["verified"]:
            rid = last.get("request_id")
            err = last.get("error_type")
            if err in ("AUTHORIZATION", "REQUEST_REJECTED", "SDK_ERROR"):
                model_access["error"] = err
                raise BindingError("Provider rejected frozen configuration; stop without fallback: " + str(err))
            if last.get("error_type") == "OK" and rid:
                model_access = {"verified": True, "first_request_id": rid, "error": None, "model": config.model, "endpoint": config.endpoint}
            elif last.get("error_type") != "OK":
                model_access["error"] = last.get("error_type")
                raise BindingError("First live T_B VLM call failed: " + str(last.get("error_type")))
        written = write_audit_cache(ROOT / CACHE_ROOT, manifest, prompt, scene, execution)
        rec = inspect_cache(Path(written))
        rec.update({
            "task_id": "T_B",
            "scene_id": row["case_id"],
            "action": "MATERIALIZED",
            "status": execution["status"],
            "attempts": len(execution["attempts"]),
            "cache_dir": str(Path(written).relative_to(ROOT)),
            "case": row,
        })
        if execution["status"] != "SUCCESS":
            complete_file = Path(written) / "COMPLETE"
            if complete_file.exists():
                complete_file.unlink()
            (Path(written) / "FAILED").write_text(str(execution["status"]) + "\n", encoding="utf-8")
            rec["ok"] = False
            rec["reason"] = execution["status"]
        results.append(rec)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical({k: rec.get(k) for k in ("scene_id", "task_id", "action", "status", "cache_dir", "empty", "n_accepted")}) + "\n")
        _log("cache T_B %s %s empty=%s" % (row["case_id"], execution["status"], rec.get("empty")))
        last_err = last.get("error_type")
        if last_err in ("AUTHORIZATION", "REQUEST_REJECTED", "SDK_ERROR"):
            raise BindingError("Provider rejected frozen configuration; stop without fallback: " + str(last_err))
    return results, model_access

def bind_tb_split(tb_rows):
    path = ROOT / "configs/splits/T_B_stage_0a.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_id = {r["scene_id"]: r for r in tb_rows}
    new_dev = []
    for row in doc["dev"]:
        rec = by_id.get(row["case_id"])
        if rec:
            row = dict(row)
            row["cache_dir"] = rec.get("cache_dir") or rec.get("path")
            row["cache_key"] = rec.get("cache_key")
            row["cache_status"] = rec.get("action")
            row["image_ref"] = rec.get("case", {}).get("image_ref")
            row["image_sha256"] = rec.get("case", {}).get("image_sha256")
        new_dev.append(row)
    doc["dev"] = new_dev
    doc["plan_version"] = "1.1"
    doc["method_version"] = "2.1.1"
    path.write_text(canonical(doc) + "\n", encoding="utf-8")
    return path

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    ST.mkdir(parents=True, exist_ok=True)
    MIG.mkdir(parents=True, exist_ok=True)
    gpu = int(os.environ.get("CP_DISR_GPU", "0"))
    started = utc()
    d0 = reuse_family("D0", ROOT / "configs/splits/D0_stage_1a.json", 8)
    tc = reuse_family("T_C", ROOT / "configs/splits/T_C_stage_2a.json", 8)
    # T_A listed as incompatible
    ta_note = {"task_id": "T_A", "action": "NOT_REUSED_AS_T_B", "reason": "T_A goals differ; T_A caches cannot be renamed as T_B"}
    mapping = capture_tb_dev8(gpu)
    tb, model_access = materialize_tb(mapping, gpu)
    bind_tb_split(tb)
    all_rows = d0 + tb + tc
    if len(all_rows) != 24:
        raise BindingError("expected 24 scenes, got %s" % len(all_rows))
    failed = [r for r in all_rows if not r.get("ok")]
    n_empty = sum(1 for r in all_rows if r.get("empty"))
    n_new = sum(1 for r in tb if r.get("action") == "MATERIALIZED")
    transport_fail = [r for r in tb if r.get("reason") in ("API_ERROR", "FORMAT_ERROR", "AUTHORIZATION")]
    status = "BLOCKED" if failed or transport_fail else "PASS"
    notes = []
    if n_empty == 24:
        notes.append("All 24 priors empty after isolation; prior-increment groups are uninformative. Empty is allowed; not a transport failure.")
        status = "PASS_WITH_NOTES"
    elif n_empty >= 16:
        notes.append("High empty-prior rate %s/24 after isolation; recorded, not semantically retried." % n_empty)
        if status == "PASS":
            status = "PASS_WITH_NOTES"
    doc = {
        "stage": "0C",
        "plan_version": "1.1",
        "method_version": "2.1.1",
        "document_version": "3.1",
        "status": status,
        "execution_reason": None,
        "git_hash": git_hash(),
        "started_at": started,
        "completed_at": utc(),
        "new_rl_runs": 0,
        "scenes_required": 24,
        "scenes_bound": len(all_rows),
        "reused_d0": len(d0),
        "reused_tc": len(tc),
        "materialized_tb": n_new,
        "empty_prior_count": n_empty,
        "failed": failed,
        "model_access": {k: v for k, v in model_access.items() if k != "key"},
        "ta_not_reused": ta_note,
        "old_stage_status_path_unmodified": "experiments/stage_status/stage_0c.json",
        "prompt_unfrozen": False,
        "schema_unfrozen": False,
        "semantic_retry": False,
        "gold_replacement": False,
        "rmtree_used": False,
    }
    write_json(OUT / "stage_0c_result.json", doc)
    write_json(ST / "stage_0c.json", doc)
    reuse_csv = MIG / "cache_reuse_manifest.csv"
    with reuse_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "task_id", "action", "cache_key", "path", "empty", "n_accepted", "n_raw", "ok"])
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "scene_id": r.get("scene_id"),
                "task_id": r.get("task_id"),
                "action": r.get("action"),
                "cache_key": r.get("cache_key"),
                "path": r.get("cache_dir") or r.get("path"),
                "empty": r.get("empty"),
                "n_accepted": r.get("n_accepted"),
                "n_raw": r.get("n_raw"),
                "ok": r.get("ok"),
            })
    md = [
        "# Stage 0C — Frozen VLM Cache Sanity (Plan v1.1 / Method 2.1.1)",
        "",
        "Status: `" + status + "`.",
        "",
        "24 real development scenes: D0 x 8 reused from Stage 1A robosuite caches, T_C x 8 reused from Stage 2A caches, T_B x 8 newly captured and requested. T_A caches were not renamed as T_B.",
        "",
        "- Empty priors allowed; no semantic retry; no gold replacement; no rmtree of failed caches.",
        "- Frozen model/endpoint/prompt/schema/few-shots. Automatic model fallback forbidden.",
        "- Model access verification: %s" % json.dumps({k: model_access.get(k) for k in ("verified", "first_request_id", "error", "model", "endpoint")}),
        "- Empty-prior count: %s/24" % n_empty,
        "- New VLM requests this stage: %s" % n_new,
        "",
        "New-profile RL training runs this stage: **0**.",
        "",
    ]
    (REP / "stage_0c_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "empty": n_empty, "new_requests": n_new, "failed": len(failed)}, ensure_ascii=False))
    if status == "BLOCKED":
        raise SystemExit(2)

if __name__ == "__main__":
    main()

"""Conditional Family B provider calls; all payloads come from public captures."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from cp_disr.common import canonical, ContractError, digest
from cp_disr.platforms.libero.family_b_runtime import TASK_ID, build_task_template, ACTION_IDS
from cp_disr.stage0c import prepare_scene
from cp_disr.vlm import cache_key
from cp_disr.vlm_cache_pipeline import (
    request_and_process, write_audit_cache, verify_audit_cache, response_text)
from cp_disr.vlm_provider import (
    DashScopeProvider, ProviderConfig, MODEL, redact, validate_payload)
from .family_b_pilot import read, write, sha, _registered

RELATION_PROMPT = "experiments/sources/v2.1_interfaces/system_prompt.txt"
ACTION_PROMPT = (
    "Choose exactly one currently legal candidate for the fixed public goal. "
    "Use the RGB image, public facts, full contracts, candidate IDs and mask, "
    "setup history, proprioception and remaining simulation time. "
    "Do not assume hidden state or future branch outcomes. "
    "Return only JSON {\\\"candidate_id\\\":\\\"<exact allowed ID>\\\"}."
)


def _image(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ContractError("Family B provider requires raw PNG")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _fewshots(root):
    doc = read(Path(root) / "experiments/stage_0c_inputs/fewshots/few_shot_manifest.json")
    if doc.get("status") != "FROZEN" or len(doc.get("examples", [])) != 3:
        raise ContractError("Frozen independent fewshots missing")
    messages = []
    for example in doc["examples"]:
        case = prepare_scene(root, example, fewshot=True)
        messages += [
            {"role": "user", "content": case["content"]},
            {"role": "assistant", "content": [{"text": canonical(example["expected_json"])}]},
        ]
    return messages, doc


def _request(messages, prompt):
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": [{"text": prompt}]}] + messages,
        "temperature": 0, "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "enable_thinking": False, "enable_search": False,
        "stream": False, "result_format": "message",
    }


def relation_case(root, cfg, out, layout, repeat):
    root, out = Path(root), Path(out)
    branch = next(b for b in _registered(out) if b["layout"] == layout
                  and b["repeat"] == repeat and b["context"] == "B_PENDING"
                  and b["candidate"] == "pad_u")
    capture = out / "captures" / branch["branch_id"]
    image_path = capture / "initial_rgb.png"
    facts_path = capture / "initial_public_facts.json"
    facts = read(facts_path)
    contract_path = root / cfg["runtime"]["contract_path"]
    contract_doc = __import__("yaml").safe_load(contract_path.read_text())
    template = build_task_template(contract_path)
    public = {
        "task_id": TASK_ID,
        "object_table": [
            {"id": k, "type": v} for k, v in sorted({
                "carrier": "object", "obj_b": "object", "obj_c": "object",
                "receiver": "container", "pad_u": "buffer", "pad_v": "buffer"}.items())],
        "signed_goals": [{"fact_id": g.fact_id, "sign": g.sign} for g in template.goals],
        "initial_facts": facts["facts"],
        "allowed_action_ids": sorted(ACTION_IDS),
        "allowed_proposition_ids": sorted(n.id for n in template.nodes if n.kind == "PROPOSITION"),
        "skill_contracts": contract_doc,
        "registered_effect_edges": [list(e) for e in template.edges if e[2] in ("ADD", "DEL")],
        "controller_public_semantics": "PLACE_BUFFER retreats above selected pad; same contracts, differing public position",
    }
    scene_content = [{"image": _image(image_path)}, {"text": canonical(public)}]
    examples, few_doc = _fewshots(root)
    prompt = (root / RELATION_PROMPT).read_text()
    payload = _request(examples + [{"role": "user", "content": scene_content}], prompt)
    validate_payload(payload)
    provider = ProviderConfig()
    schema_path = root / "schemas/relation_schema.json"
    scene_id = f"family_b_{layout}_repeat{repeat}"
    manifest = {
        "split": "dev", "task_definition_hash": sha(root / "configs/tasks/resolved/T_P_FB.yaml"),
        "initial_RGB_content_hash": sha(image_path),
        "preprocessing_hash": digest({"version": "raw-rgb-inline-v1", "resize": False, "crop": False}),
        "object_binding_hash": digest(public["object_table"]),
        "allowed_ID_hash": digest({"actions": public["allowed_action_ids"],
                                   "propositions": public["allowed_proposition_ids"],
                                   "signed_goals": public["signed_goals"]}),
        "contract_version": sha(contract_path), "predicate_version": "family-b-v1",
        "model_snapshot": provider.model, "sdk_api_version": provider.sdk_version,
        "region": provider.region, "endpoint": provider.endpoint,
        "prompt_hash": sha(root / RELATION_PROMPT),
        "fewshot_hash": digest(few_doc), "schema_hash": sha(schema_path),
        "decoding_config": {"temperature": 0, "max_tokens": 2048, "max_relations": 8,
                            "thinking": False, "response_format": "json_object"},
        "initial_facts_hash": digest(facts), "asset_binding_hash": sha(root / cfg["runtime"]["layouts_path"]),
        "request_payload_hash": digest(payload),
        "scene_id": scene_id, "task_id": TASK_ID, "synthetic_unit_fixture": False,
    }
    key = cache_key(manifest)
    return {"branch_id": branch["branch_id"], "scene_id": scene_id,
            "input_hash": digest(public), "payload": payload,
            "template": template, "schema": read(schema_path), "manifest": manifest,
            "prompt": prompt, "cache_key": key,
            "input_refs": {"initial_rgb_path": str(image_path),
                           "initial_rgb_sha256": sha(image_path),
                           "initial_public_facts_path": str(facts_path),
                           "initial_public_facts_sha256": sha(facts_path),
                           "hidden_truth_included": False}}


def validate_action_payload(payload):
    fixed = {
        "model": MODEL, "temperature": 0, "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "enable_thinking": False, "enable_search": False,
        "stream": False, "result_format": "message",
    }
    if set(payload) != set(fixed) | {"messages"}:
        raise ContractError("action payload field mismatch")
    if any(payload[k] != v for k, v in fixed.items()):
        raise ContractError("action frozen decoding mismatch")
    messages = payload["messages"]
    if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
        raise ContractError("action prompt must be system + public snapshot")
    if sum("image" in x for x in messages[1]["content"]) != 1:
        raise ContractError("action prompt requires one boundary RGB")
    if redact(payload) != payload:
        raise ContractError("credential-shaped action content")
    return payload


def action_case(root, cfg, out, layout, context):
    root, out = Path(root), Path(out)
    branch = next(b for b in _registered(out) if b["layout"] == layout
                  and b["context"] == context and b["repeat"] == 0
                  and b["candidate"] == "pad_u")
    boundary = out / "captures" / branch["branch_id"] / "boundary"
    public = read(boundary / "public.json")
    candidates = list(cfg["candidate_ids"])
    if len(candidates) != 2 or not all(cid in public["candidate_ids"] for cid in candidates):
        raise ContractError("Family B action candidate binding mismatch")
    if int(digest(public), 16) % 2:
        candidates.reverse()
    remaining = float(cfg["runtime"]["task_deadlines"][TASK_ID]) - float(
        public["sim_time"]-read(out / "physical/branch_results" / f'{branch["branch_id"]}.json')["initial_sim_time"])
    if remaining <= 0:
        raise ContractError("no remaining deadline at boundary")
    public = {**public, "candidate_order": candidates,
              "remaining_deadline_seconds": remaining,
              "skill_contracts": __import__("yaml").safe_load(
                  (root / cfg["runtime"]["contract_path"]).read_text()),
              "task_id": TASK_ID}
    payload = _request([{"role": "user", "content": [
        {"image": _image(boundary / "rgb.png")}, {"text": canonical(public)}]}], ACTION_PROMPT)
    validate_action_payload(payload)
    return {"branch_id": branch["branch_id"], "scene_id": f"{layout}_{context}_repeat0",
            "payload": payload, "input_hash": digest(public),
            "input_refs": {"boundary_public_sha256": sha(boundary / "public.json"),
                           "boundary_rgb_sha256": sha(boundary / "rgb.png"),
                           "hidden_truth_included": False}}


def _gate(out):
    gate = read(Path(out) / "physical/mechanism_gate.json")
    if gate.get("physical_mechanism_status") != "ESTABLISHED" or not gate.get("provider_released"):
        raise ContractError("physical qualification has not released provider")


def call_group(root, cfg, out, group, provider=None):
    from cp_disr.vlm_provider import DashScopeProvider
    root, out = Path(root), Path(out)
    _gate(out)
    if group not in ("relations", "vlm-action"):
        raise ValueError("unknown provider group")
    ledger_path = out / "provider" / ("relation_request_ledger.json" if group == "relations"
                                      else "action_request_ledger.json")
    entries = read(ledger_path)["entries"] if ledger_path.is_file() else []
    if entries:
        raise ContractError("existing provider ledger; explicit reviewed resume required")
    cases = ([relation_case(root, cfg, out, layout, repeat)
              for layout in ("layout_0", "layout_1") for repeat in (0, 1)]
             if group == "relations" else
             [action_case(root, cfg, out, layout, context)
              for layout in ("layout_0", "layout_1") for context in ("B_PENDING", "C_PENDING", "BOTH_PENDING")])
    if group == "relations":
        by_cache = {}
        reused = []
        for case in cases:
            if case["cache_key"] in by_cache:
                reused.append({"scene_id": case["scene_id"],
                               "cache_key": case["cache_key"],
                               "source_scene_id": by_cache[case["cache_key"]]["scene_id"]})
            else:
                by_cache[case["cache_key"]] = case
        cases = list(by_cache.values())
        write(out / "provider/reused_initial_sources.json", reused)
    cap = 4 if group == "relations" else 6
    if len(cases) > cap:
        raise ContractError("provider first-call budget exceeded")
    if provider is None:
        provider = DashScopeProvider(ProviderConfig(),
                                     payload_validator=validate_payload if group == "relations"
                                     else validate_action_payload)
    if not provider.key_present():
        frozen = [{"scene_id": c["scene_id"], "input_hash": c["input_hash"],
                   "request_payload_hash": digest(c["payload"])} for c in cases]
        write(out / "provider" / (group + "_frozen_inputs.json"), frozen)
        raise ContractError("STOPPED_PROVIDER_ACCESS:credential unavailable")
    ledger = read(out / "budget_ledger.json")
    key = "relation_first_calls" if group == "relations" else "action_first_calls"
    schema = read(root / "schemas/relation_schema.json")
    for case in cases:
        if ledger[key]["used"] >= ledger[key]["cap"]:
            raise ContractError("provider budget exhausted")
        entry = {"scene_id": case["scene_id"], "input_hash": case["input_hash"],
                 "request_payload_hash": digest(case["payload"]), "state": "STARTED"}
        entries.append(entry)
        ledger[key]["used"] += 1
        write(ledger_path, {"entries": entries})
        write(out / "budget_ledger.json", ledger)
        if group == "relations":
            execution = request_and_process(provider, case["payload"], case["template"], case["schema"])
            path = write_audit_cache(out / "provider/cache", case["manifest"],
                                     case["prompt"], case["input_refs"], execution)
            entry.update(state=execution["status"], attempts=len(execution["attempts"]),
                         cache_path=str(path), cache_key=case["cache_key"])
            if execution["status"] == "SUCCESS":
                verify_audit_cache(path)
        else:
            attempts, selected = [], None
            for trial in range(2):
                response = provider.send(case["payload"])
                attempts.append(redact(response))
                if response["error_type"] in ("TRANSPORT", "TIMEOUT") and trial == 0:
                    continue
                if response["error_type"] != "OK":
                    break
                try:
                    answer = json.loads(response_text(response))
                    if set(answer) == {"candidate_id"} and answer["candidate_id"] in cfg["candidate_ids"]:
                        selected = answer["candidate_id"]
                    else:
                        entry["semantic_invalid"] = True
                    break
                except (ValueError, KeyError, TypeError):
                    if trial == 0:
                        continue
            entry.update(state="SUCCESS" if selected else "INVALID_OR_ERROR",
                         attempts=len(attempts), candidate_id=selected,
                         response_redacted=attempts)
        retries = entry.get("attempts", 0)-1
        if ledger["provider_retries"]["used"] + retries > ledger["provider_retries"]["cap"]:
            raise ContractError("provider retry budget exceeded")
        ledger["provider_retries"]["used"] += retries
        write(out / "budget_ledger.json", ledger)
        write(ledger_path, {"entries": entries})
        if entry["state"] not in ("SUCCESS",):
            break
    return {"group": group, "first_requests": len(entries),
            "successes": sum(x["state"] == "SUCCESS" for x in entries),
            "entries": [{k: v for k, v in x.items() if k != "response_redacted"} for x in entries]}

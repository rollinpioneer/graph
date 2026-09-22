"""Stage 2A startup gate."""
from __future__ import annotations
import csv, json, os, subprocess, sys
from collections import defaultdict
from pathlib import Path
import torch
from .common import canonical
from .neural import Policy
from .torch_rl import PPO, save_checkpoint, load_checkpoint
from . import stage2a_explore as ex
from . import stage1a_smoke as s1

GATE = Path("experiments/part_2_exploration/stage_2a/startup_gate")
P0 = Path("experiments/part_2_exploration/stage_2a_p0")
STATUS = Path("experiments/stage_status/stage_2a.json")

def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def cache_audit(root):
    root = Path(root)
    out = root / GATE
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    recovery = []
    deviations = []
    for task_id, ledger_name in (("T_A", "T_A_cache_request_ledger.jsonl"), ("T_C", "T_C_cache_request_ledger.jsonl")):
        ledger = load_jsonl(root / P0 / ledger_name)
        by_scene = defaultdict(list)
        for rec in ledger:
            sid = rec.get("scene_id") or rec.get("case_id")
            by_scene[sid].append(rec)
        for scene_id, events in sorted(by_scene.items()):
            states = [e.get("state") or e.get("status") for e in events]
            started = sum(1 for s in states if s == "REQUEST_STARTED")
            success = [e for e in events if e.get("status") == "SUCCESS"]
            errors = [e for e in events if e.get("status") and e.get("status") not in ("SUCCESS", "REUSED") and e.get("state") != "REQUEST_STARTED"]
            api_errors = [e for e in events if e.get("status") == "API_ERROR"]
            cache_key = success[-1].get("cache_key") if success else (errors[-1].get("cache_key") if errors else None)
            cache_dir = success[-1].get("cache_dir") if success else (errors[-1].get("cache_dir") if errors else None)
            raw_present = False
            complete = False
            processing = None
            missing = []
            if cache_dir:
                d = root / cache_dir
                complete = (d / "COMPLETE").is_file()
                for name in ("raw_response.json", "processing_log.json", "manifest.json"):
                    if not (d / name).is_file():
                        missing.append(name)
                    elif name == "raw_response.json":
                        raw_present = True
                    elif name == "processing_log.json":
                        try:
                            processing = json.loads((d / name).read_text(encoding="utf-8")).get("status")
                        except Exception:
                            processing = "UNREADABLE"
            transport_then_success = bool(api_errors) and bool(success)
            if transport_then_success:
                deviations.append({"task_id": task_id, "scene_id": scene_id, "issue": "API_ERROR then SUCCESS after rmtree", "api_error_n": len(api_errors), "request_started_n": started})
                recovery.append({"task_id": task_id, "scene_id": scene_id, "cache_key": cache_key, "original_failed_attempt": "MISSING_ORIGINAL_EVIDENCE", "transport_failure_is_empty_prior": False})
            rows_out.append({
                "task_id": task_id, "scene_id": scene_id, "cache_key": cache_key or "",
                "request_started_n": started, "known_client_attempts": started,
                "confirmed_service_responses": len(success) + len(errors),
                "success_n": len(success), "api_error_n": len(api_errors),
                "final_status": "SUCCESS" if success else (errors[-1].get("status") if errors else "UNKNOWN"),
                "complete_marker": complete, "raw_response_present": raw_present,
                "processing_status": processing or "", "missing_files": "|".join(missing),
                "retry_budget_cross_process": "NOT_PERSISTED_HISTORICAL" if started > 1 else "SINGLE_PASS",
                "transport_failure_treated_as_empty_prior": False,
                "protocol_deviation": bool(transport_then_success or started > 1),
            })
    csv_path = out / "cache_retry_audit.csv"
    if rows_out:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
    md1 = ["# Stage 2A cache protocol deviation", "", "Historical Stage 2A-P0 deleted incomplete cache directories with shutil.rmtree and rewrote SUCCESS.", "T_C ledger has REQUEST_STARTED, API_ERROR, then later SUCCESS for the same scene.", "Transport failures are not legal EMPTY_PRIOR.", "", "deviation_rows: %s" % len(deviations), "", canonical(deviations), ""]
    write_text(out / "cache_protocol_deviation.md", chr(10).join(md1) + chr(10))
    md2 = ["# Stage 2A cache evidence recovery", "", "Recovered from JSONL ledgers and remaining SUCCESS directories.", "Original failed raw HTTP bodies for rmtree'd T_C transport failures are MISSING_ORIGINAL_EVIDENCE.", "No request IDs or error payloads were invented.", "", canonical(recovery), ""]
    write_text(out / "cache_evidence_recovery.md", chr(10).join(md2) + chr(10))
    return {"rows": len(rows_out), "deviations": len(deviations), "missing_original_evidence": len(recovery)}

def deadline_erratum(root):
    root = Path(root)
    out = root / GATE
    deadline_records = 0
    timeout_success = 0
    unknown = 0
    samples = []
    for p in list(root.glob("experiments/part_1_smoke/**/*.jsonl")) + list(root.glob("experiments/part_2_exploration/**/*.jsonl")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "DEADLINE" not in text and "deadline" not in text:
            continue
        for line in text.splitlines():
            if "DEADLINE" not in line and "deadline" not in line.lower():
                continue
            deadline_records += 1
            try:
                rec = json.loads(line)
            except Exception:
                unknown += 1
                continue
            trunc = rec.get("truncated")
            term = rec.get("terminated")
            success = rec.get("success")
            samples.append({"file": str(p.relative_to(root)), "truncated": trunc, "terminated": term, "success": success, "reason": rec.get("reason")})
            if success and (trunc or rec.get("reason") == "DEADLINE"):
                timeout_success += 1
            if trunc is None and term is None:
                unknown += 1
    lines = [
        "# Stage 1A deadline semantics erratum",
        "",
        "Old TaskEvaluator mapped task-timeout without success to truncated=true, terminated=false, reason=DEADLINE.",
        "Frozen v2.1 requires terminated=true, truncated=false, success=false, empty reward_events.",
        "Stage 1A smoke jobs and checkpoints were trained under the old mapping.",
        "This erratum does not retrain Stage 1A or treat offline target recomputation as a new training result.",
        "Historical PASS remains the then-current acceptance record and does not certify the restored termination semantics.",
        "",
        "deadline_log_lines: %s" % deadline_records,
        "timeout_success_cooccurrence: %s" % timeout_success,
        "unclassifiable_UNKNOWN: %s" % unknown,
        "",
        canonical(samples[:20]),
        "",
    ]
    write_text(out / "deadline_semantics_erratum.md", chr(10).join(lines) + chr(10))
    return {"deadline_records": deadline_records, "timeout_success": timeout_success, "unknown": unknown}

def real_checkpoint_resume_test(root, out):
    device = torch.device("cpu")
    actions = ["OPEN", "PICK", "PLACE", "PLACE_BUFFER"]
    predicates = ["GripperEmpty", "Held", "OnTable", "Open", "Inside", "AtBuffer"]
    types = ["object", "container", "buffer"]
    policy = Policy(actions, predicates, types, 48, 8, method="Full", B=0.5).to(device)
    trainer = PPO(policy)
    torch.manual_seed(0)
    loss = sum(p.square().mean() for p in policy.parameters())
    trainer.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    trainer.optimizer.step()
    p1 = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    o1 = trainer.optimizer.state_dict()
    ckpt = out / "_tmp_resume.pt"
    if ckpt.exists():
        ckpt.unlink()
    save_checkpoint(ckpt, policy, trainer.optimizer, {"kind": "startup_gate_resume", "update": 1})
    for p in policy.parameters():
        p.data.add_(1.0)
    trainer.optimizer.param_groups[0]["lr"] = 1e-8
    policy2 = Policy(actions, predicates, types, 48, 8, method="Full", B=0.5).to(device)
    trainer2 = PPO(policy2)
    load_checkpoint(ckpt, policy2, trainer2.optimizer)
    same_model = all(torch.equal(p1[k].cpu(), policy2.state_dict()[k].cpu()) for k in p1)
    st1 = o1["state"]
    st2 = trainer2.optimizer.state_dict()["state"]
    same_opt = bool(st1)
    for k, rec in st1.items():
        other = st2.get(k)
        if other is None:
            same_opt = False
            break
        for field in rec:
            a, b = rec[field], other.get(field)
            if torch.is_tensor(a) and torch.is_tensor(b):
                if not torch.equal(a.cpu(), b.cpu()):
                    same_opt = False
            elif a != b:
                same_opt = False
    payload = {
        "passed": bool(same_model and same_opt),
        "same_model": bool(same_model),
        "same_optimizer": bool(same_opt),
        "compared_json_dicts_only": False,
        "notes": "Real Policy+Adam step, save, mutate, load, compare tensors.",
    }
    (out / "real_checkpoint_resume_test.json").write_text(canonical(payload) + chr(10), encoding="utf-8")
    return payload

def ta_empty_prior_equivalence(root, out, gpu=0):
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu) if str(gpu).isdigit() else 0)
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    bundle = ex.make_bundle(root, "T_A")
    try:
        split = json.loads((root / "configs/splits/T_A_stage_2a.json").read_text(encoding="utf-8"))
        case = split["train"][0]["case_id"]
        snap = bundle.start_case(case)
        snap = s1.empty_prior(snap)
        full = ex.Policy if False else None
        from .neural import Policy
        full = Policy(**ex.model_kwargs(bundle.template, "Full", "T_A")).to(device)
        b2 = Policy(**ex.model_kwargs(bundle.template, "B2", "T_A")).to(device)
        b2.load_state_dict(full.state_dict())
        full.eval(); b2.eval()
        with torch.no_grad():
            o_full = full(snap)
            o_b2 = b2(snap)
        dp_all = []
        delta_all = []
        up_all = []
        for cid, diff in (o_full.diagnostics.get("differences") or {}).items():
            if diff is not None:
                dp_all.append(float(torch.max(torch.abs(diff.dp)).cpu()))
        for cid, up in (o_full.diagnostics.get("up") or {}).items():
            up_all.append(float(torch.max(torch.abs(up)).cpu()) if up is not None else 0.0)
        for cid, residual in (o_full.diagnostics.get("delta") or {}).items():
            if residual is not None:
                delta_all.append(abs(float(residual.detach().cpu())))
        tol = 1e-5
        m = o_full.mask
        logit_gap = float(torch.max(torch.abs(o_full.logits[m] - o_b2.logits[m])).cpu()) if m.any() else 0.0
        if m.any():
            p1 = torch.softmax(o_full.logits[m], dim=0)
            p2 = torch.softmax(o_b2.logits[m], dim=0)
            prob_gap = float(torch.max(torch.abs(p1 - p2)).cpu())
        else:
            prob_gap = 0.0
        v_gap = abs(float(o_full.value.detach().cpu()) - float(o_b2.value.detach().cpu()))
        q_gap = float(torch.max(torch.abs(o_full.q - o_b2.q)).cpu())
        payload = {
            "passed": (max(dp_all or [0]) == 0) and (max(delta_all or [0]) == 0) and logit_gap <= tol and prob_gap <= tol and v_gap <= tol and q_gap <= tol,
            "task_id": "T_A",
            "case_id": case,
            "effective_prior_edges": 0,
            "full_dp_absmax": max(dp_all or [0.0]),
            "full_up_absmax": max(up_all or [0.0]),
            "full_delta_absmax": max(delta_all or [0.0]),
            "logit_maxabs_gap": logit_gap,
            "prob_maxabs_gap": prob_gap,
            "value_abs_gap": v_gap,
            "q_maxabs_gap": q_gap,
            "tolerance": tol,
            "shared_parameters": True,
            "note": "T_A frozen caches are empty-prior. Full vs B2 compared under identical weights. Do not attribute gaps to VLM.",
        }
        (out / "ta_empty_prior_equivalence.json").write_text(canonical(payload) + chr(10), encoding="utf-8")
        return payload
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass

def run_pytest(root):
    env = os.environ.copy()
    env.pop("DASHSCOPE_API_KEY", None)
    cmd = [sys.executable, "-m", "pytest", "tests/test_deadline_semantics.py", "tests/test_smdp_boundaries.py", "tests/test_q_target_detach.py", "-q", "--tb=short"]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": (proc.stdout or "")[-4000:], "stderr": (proc.stderr or "")[-2000:]}

def runner_audit(root, tests, ckpt, ta, cache):
    src = (root / "src/cp_disr/stage2a_explore.py").read_text(encoding="utf-8")
    collector = (root / "src/cp_disr/collector.py").read_text(encoding="utf-8")
    ev = (root / "src/cp_disr/platforms/libero/task_evaluator.py").read_text(encoding="utf-8")
    torch_rl = (root / "src/cp_disr/torch_rl.py").read_text(encoding="utf-8")
    checks = {
        "explore_runner_present": ("trainer.update" in src) and ("PPO(" in src),
        "from_scratch": "from_scratch" in src,
        "real_backward_path": "optimizer.step" in torch_rl,
        "deadline_reason": "DEADLINE" in ev,
        "deadline_terminated_true": "terminated=True" in ev.replace(" ", "") or "terminated = True" in ev or "terminated=True" in ev,
        "collector_caps_remaining": "remaining" in collector,
        "checkpoint_tensors": bool(ckpt.get("passed")),
        "ta_empty_prior": bool(ta.get("passed")),
        "pytest_ok": tests.get("returncode") == 0,
        "cache_audit_written": cache.get("rows", 0) > 0,
    }
    payload = {"checks": checks, "passed": all(checks.values())}
    write_text(root / GATE / "runner_implementation_audit.md", "# Stage 2A runner implementation audit" + chr(10) + chr(10) + canonical(payload) + chr(10))
    return payload

def cmd_stage_2a_startup_gate(root, gpu=0):
    root = Path(root)
    out = root / GATE
    out.mkdir(parents=True, exist_ok=True)
    os.environ.pop("DASHSCOPE_API_KEY", None)
    cache = cache_audit(root)
    erratum = deadline_erratum(root)
    tests = run_pytest(root)
    ckpt = real_checkpoint_resume_test(root, out)
    ta = ta_empty_prior_equivalence(root, out, gpu=gpu)
    dl = {"source": "tests/test_deadline_semantics.py", "pytest_returncode": tests.get("returncode"),
          "names": ["success_in_time", "no_success_in_time", "at_deadline_unsuccessful", "goal_after_deadline", "skill_crossing", "external_truncation", "buffer_cut", "clock_across_updates", "no_bootstrap", "no_repeat_reward"]}
    (out / "deadline_semantics_tests.json").write_text(canonical(dl) + chr(10), encoding="utf-8")
    audit = runner_audit(root, tests, ckpt, ta, cache)
    gate_pass = bool(audit["passed"])
    status = json.loads((root / STATUS).read_text(encoding="utf-8"))
    if not gate_pass:
        status["status"] = "NEEDS_RERUN"
        status["issues"] = list(status.get("issues") or []) + ["startup_gate_failed"]
        ex.save_status(root, status)
        return {"status": "NEEDS_RERUN", "audit": audit, "erratum": erratum, "cache": cache, "tests": tests, "ta": ta, "ckpt": ckpt}
    return {"status": "PASS", "audit": audit, "erratum": erratum, "cache": cache, "tests": tests, "ta": ta, "ckpt": ckpt}

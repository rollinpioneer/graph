"""S1-REV1 post-recovery evidence closeout (offline, zero new samples).

Reads already-saved r1/r2/r3 evidence, provider caches, scenes and the frozen contract; writes only into its own
output directory. It never constructs a runtime/environment, never resets, never calls a provider and never
reserves/claims budget. ``install_guards`` makes any such call raise ``ZeroCallViolation``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

BASE_COMMIT = "cc1ad6474eb53b917714cf74eab715018d3cb115"
S1_REL = "runs/final_master/S1/20260928T154311Z_bea4bf0d"
ORIG_S1_REL = "runs/final_master/S1/20260928T140215Z_538ef55a"
ENUM_LIMIT = 100000
GOALS_T_A = ({"fact_id": "p:Inside:target:container", "sign": 1}, {"fact_id": "p:Inside:second_object:container", "sign": 1})
OPEN_ID = "a:OPEN:container:v1"
PICK_ID = "a:PICK:second_object:v1"
QPOS_TOL = 1e-9  # same technical tolerance as the recovery runner


class InputMissing(RuntimeError):
    pass


class AnalysisIncomplete(RuntimeError):
    pass


class ZeroCallViolation(RuntimeError):
    pass


class AuthorityMismatch(RuntimeError):
    """The authority manifest does not match the installed authoritative documents (fail closed)."""


AUTH_DIR = "docs/authoritative"
AUTH_MASTER_ID = "CP-DISR-FINAL-EXEC-3.0"
AUTH_SPECS = (("experimental_plan", "3.0", "FROZEN_PLAN"), ("research_content", "5.0", "FROZEN_AFTER_FINAL_AUDIT"))
AUTH_OK = "FOUND_AND_RECONCILED"
KEY_GATE_NOTES = {
    "E1_relation_existence": "Natural validator-admitted nonredundant relations exist in at least two independent configurations, but independent adjudication is absent; truth/utility remain UNKNOWN.",
    "E2_E3_representation": "Previously saved S1-REV1 representation values are retained as historical evidence but were not revalidated in the zero-sample closeout.",
    "E4_real_consequence": "Two independent case configurations show repeat-consistent protocol-level outcome differences under paired initialization and shared B_PLAN continuation. Cost and rework are NOT_MEASURED. PICK is evaluator CONTINUE followed by planner NO_PLAN, not evaluator failure or proof of physical infeasibility.",
    "E5_contract_insufficiency": "The nominal contract already distinguishes OPEN-first from PICK-first reachability. The observed candidate difference therefore does not establish missing soft risk/cost/relevance beyond the contract.",
    "E6_prior_classification": "Independent truth/utility/opportunity adjudication and R* evidence are absent, so no official E6 class is assigned.",
}
E4_V3_SECTION = """## 对照 Final Experimental Plan v3 §5.2

v3 E4要求：

同初始化合法候选真实执行＋共同continuation，并在至少两个独立见证上观察可靠的feasibility、cost或rework差异。

当前证据：

- T_A_dev_14、T_A_dev_18 是两个独立case配置；
- 每个case有两个paired repeat，repeat不计作独立配置；
- paired public state、qpos、qvel一致；
- OPEN分支最终由独立Evaluator确认TASK_SUCCESS；
- PICK分支首动作后Evaluator为CONTINUE，随后B_PLAN返回NO_PLAN；
- cost=NOT_MEASURED；
- rework=NOT_MEASURED；
- PICK的物理不可行性=NOT_ESTABLISHED；
- controller内部状态和RNG流=NOT_MEASURED。

因此本收口对E4只记录：

LIMITED_SUPPORT_FEASIBILITY_ONLY

它表示在冻结的共同continuation协议下存在两个独立case的可靠结果差异；
不表示PICK被Evaluator判失败，不表示物理不可行，也不表示E5的合同外soft差异已经成立。

E4与E5必须保持分离。
"""


def _front_matter(path):
    txt = Path(path).read_text(encoding="utf-8").split("\n")
    if not txt or txt[0].strip() != "---":
        return {}
    fm = {}
    for line in txt[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm


def verify_authority(root):
    """Fail-closed check of docs/authoritative/authority_manifest.json against the installed documents."""
    root = Path(root)
    mp = root / AUTH_DIR / "authority_manifest.json"
    if not mp.is_file():
        raise AuthorityMismatch("authority manifest missing: " + str(mp))
    man = json.loads(mp.read_text(encoding="utf-8"))
    problems = []
    if man.get("master_id") != AUTH_MASTER_ID:
        problems.append("manifest master_id mismatch")
    if man.get("authority_status") != AUTH_OK:
        problems.append("manifest authority_status is not " + AUTH_OK)
    docs = {}
    for key, ver, freeze in AUTH_SPECS:
        ent = man.get(key) or {}
        if ent.get("document_version") != ver:
            problems.append(f"{key} document_version != {ver}")
        rel = ent.get("path") or ""
        p = root / rel
        if not rel or not p.is_file():
            problems.append(f"{key} file missing: {rel!r}")
            continue
        h = sha256_file(p)
        if h != ent.get("sha256"):
            problems.append(f"{key} sha256 mismatch (file {h})")
        fm = _front_matter(p)
        if fm.get("master_id") != AUTH_MASTER_ID:
            problems.append(f"{key} header master_id mismatch")
        if fm.get("document_version") != ver:
            problems.append(f"{key} header document_version mismatch")
        if fm.get("freeze_status") != freeze:
            problems.append(f"{key} header freeze_status != {freeze}")
        docs[key] = {"path": rel, "sha256": h, "document_version": ver}
    if problems:
        raise AuthorityMismatch("; ".join(problems))
    return {"authority_status": AUTH_OK, "master_id": AUTH_MASTER_ID, "documents": docs, "manifest_sha256": sha256_file(mp)}


EXIT_CODES = {"OK": 0, "INPUT_MISSING": 2, "ANALYSIS_INCOMPLETE": 3, "VERIFY_FAILED": 4, "ZERO_CALL_VIOLATION": 5}


# ----------------------------------------------------------------- helpers
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def jread(path):
    p = Path(path)
    if not p.is_file():
        raise InputMissing(f"INPUT_MISSING:{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def jwrite(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("." + p.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def jsonl(path):
    p = Path(path)
    if not p.is_file():
        raise InputMissing(f"INPUT_MISSING:{p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_csv(path, fields, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def git(root, *args, check=True):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AnalysisIncomplete("git " + " ".join(args) + ": " + r.stderr.strip()[:200])
    return r.stdout.strip()


class Ctx:
    def __init__(self, root, source, output):
        self.root = Path(root).resolve()
        self.r3 = Path(source)
        self.r3 = (self.r3 if self.r3.is_absolute() else self.root / self.r3).resolve()
        self.out = Path(output)
        self.out = (self.out if self.out.is_absolute() else self.root / self.out).resolve()
        self.s1 = self.r3.parent
        self.rounds = {"r1": self.s1 / "e4_recovery_23fab0c", "r2": self.s1 / "e4_recovery_23fab0c_r2", "r3": self.r3}
        self.orig_s1 = self.root / ORIG_S1_REL

    def rel(self, p):
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)

    def need(self, *paths):
        miss = [str(p) for p in paths if not Path(p).exists()]
        if miss:
            raise InputMissing("INPUT_MISSING:" + ";".join(miss))


# ----------------------------------------------------------------- zero-call guards
def _blocked(name):
    def fn(*a, **k):
        raise ZeroCallViolation("ZERO_CALL_VIOLATION:" + name)
    fn.__name__ = "blocked_" + name
    return fn


def install_guards():
    """Make every runtime / provider / budget entry point raise. Called by every CLI command and by tests."""
    targets = [("cp_disr.runtime", ("load_runtime",)),
               ("cp_disr.baselines.b_plan", ("load_runtime",)),
               ("cp_disr.analysis.s1_integration", ("reserve_branch_attempt", "claim_branch_attempt", "finish_branch_attempt")),
               ("cp_disr.analysis.s1_revision_resume", ("reserve_branch_attempt", "claim_branch_attempt", "finish_branch_attempt", "execute_registered_branch", "run_provider_call", "load_runtime")),
               ("cp_disr.analysis.s1_e4_recovery", ("worker", "run_wave", "prepare")),
               ("cp_disr.platforms.libero.d0_env", ("make_env",)),
               ("cp_disr.platforms.libero.runtime_factory", ("make_env", "create_runtime", "create_task_runtime", "create_stage_2a_runtime"))]
    import importlib
    guarded = []
    for mod, names in targets:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for n in names:
            if hasattr(m, n):
                setattr(m, n, _blocked(f"{mod}.{n}"))
                guarded.append(f"{mod}.{n}")
    try:
        prov = importlib.import_module("cp_disr.vlm_provider")
        for cls in ("DashScopeProvider",):
            c = getattr(prov, cls, None)
            if c is not None:
                c.__init__ = _blocked(f"vlm_provider.{cls}.__init__")
                guarded.append(f"cp_disr.vlm_provider.{cls}.__init__")
    except Exception:
        pass
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"):
        os.environ.pop(k, None)
    return guarded


# ----------------------------------------------------------------- inventory
def _walk(p):
    p = Path(p)
    if p.is_file():
        yield p
    elif p.is_dir():
        for q in sorted(p.rglob("*")):
            if q.is_file() and "__pycache__" not in q.parts:
                yield q


def collect_protected_paths(c):
    paths = set()
    for f in _walk(c.orig_s1):
        paths.add(f)
    for f in _walk(c.s1):
        if c.out not in f.parents and f != c.out:
            paths.add(f)
    idx = c.s1 / "provider/cache_index.csv"
    c.need(idx)
    for row in csv.DictReader(idx.open(encoding="utf-8")):
        for f in _walk(row["cache_path"]):
            paths.add(f)
    for i in range(12, 20):
        for f in _walk(c.root / f"experiments/stage_0c_inputs/dev/T_A/scene_{i}"):
            paths.add(f)
    for rel in ("configs/runtime/stage_2a_contract_registry.yaml", "src/cp_disr/contracts.py", "src/cp_disr/baselines/b_plan.py",
                "src/cp_disr/vlm.py", "src/cp_disr/vlm_cache_pipeline.py", "src/cp_disr/facts.py", "src/cp_disr/graph.py",
                "src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/analysis/s1_e4_recovery.py"):
        paths.add(c.root / rel)
    for rel in ("prompts", "schemas", "experiments/stage_0c_inputs/fewshots"):
        for f in _walk(c.root / rel):
            paths.add(f)
    prot = c.s1 / "integration_closeout_10971a4/inventory/protected_before.json"
    if prot.is_file():
        for k in jread(prot):
            paths.add(c.root / k)
    return sorted(p for p in paths if p.is_file())


def hash_entries(c, paths):
    tracked = set(git(c.root, "ls-files").splitlines())
    entries = {}
    for p in paths:
        rel = c.rel(p)
        parts = rel.split("/")
        origin = ("run:" + "/".join(parts[:4])) if rel.startswith("runs/") else "repo"
        entries[rel] = {"realpath": str(Path(p).resolve()), "sha256": sha256_file(p), "origin": origin, "tracked_at_head": rel in tracked}
    return entries


def cmd_inventory(c):
    c.need(c.r3 / "final_summary.md", c.r3 / "paired_comparisons.json", c.r3 / "branch_results.csv", c.r3 / "budget_ledger.json")
    head = git(c.root, "rev-parse", "HEAD")
    entries = hash_entries(c, collect_protected_paths(c))
    jwrite(c.out / "inventory/source_hashes.json", {
        "generated_at_utc": now(), "delivery_commit": head, "entry_count": len(entries),
        "excludes": "this closeout output directory", "entries": entries})
    auth = jread(c.r3 / "authorization.json")
    rm = jread(c.r3 / "runtime_manifest.json")
    declared = auth.get("execution_commit")
    blobs = {}
    for f, observed in rm.get("code_sha256", {}).items():
        eb, hb = git_blob(c.root, declared, f), git_blob(c.root, "HEAD", f)
        exec_hash = hashlib.sha256(eb).hexdigest() if eb is not None else "NOT_FOUND"
        head_hash = hashlib.sha256(hb).hexdigest() if hb is not None else "NOT_FOUND"
        blobs[f] = {"observed_at_prepare_time_from_runtime_manifest": observed, "blob_at_execution_commit_declared": exec_hash,
                    "blob_at_delivery_commit": head_hash, "observed_equals_declared_commit_blob": observed == exec_hash,
                    "changed_between_execution_and_delivery": exec_hash != head_hash}
    restore_hashes = set()
    for p in sorted((c.r3 / "witnesses/restore_receipts").glob("*.json")):
        restore_hashes.add(jread(p)["restore_receipt"]["runtime_source_sha256"])
    jwrite(c.out / "inventory/execution_identity.json", {
        "delivery_commit": head, "delivery_commit_expected": BASE_COMMIT, "delivery_commit_matches_expected": head == BASE_COMMIT,
        "execution_commit_declared": declared, "execution_commit_exists": subprocess.run(["git", "-C", str(c.root), "cat-file", "-e", str(declared) + "^{commit}"]).returncode == 0,
        "execution_source_hashes_observed": blobs,
        "runtime_factory_sha256_in_restore_receipts": sorted(restore_hashes),
        "worktree_cleanliness_at_execution": "NOT_RECORDED (prepare checked clean HEAD; later worker start not recorded)",
        "note": "delivery_commit includes summary-count and formatting fixes that were not present at execution time; not back-written as runtime-verified"})
    jwrite(c.out / "inventory/command_config.json", {
        "python": sys.executable, "python_version": sys.version, "cwd": os.getcwd(),
        "PYTHONPATH": os.environ.get("PYTHONPATH"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "DASHSCOPE_key_present": bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY_FILE")),
        "authoritative_definitions": locate_authoritative(c)})
    return {"status": "OK", "entries": len(entries)}


def git_blob(root, rev, path):
    r = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def locate_authoritative(c):
    found = {}
    for label, pats in (("Final_Experimental_Plan_v3", ("*Final_Experimental_Plan_v3*", "*Experimental_Plan_v3*")),
                        ("Final_Research_Content_v5", ("*Final_Research_Content_v5*", "*Research_Content_v5*"))):
        hits = []
        for base in (c.root, c.root.parent):
            for pat in pats:
                for h in base.rglob(pat):
                    if h.is_file() and ".git" not in h.parts and "envs" not in h.parts:
                        hits.append(str(h))
            if hits:
                break
        found[label] = {"status": "FOUND" if hits else "NOT_FOUND", "paths": hits[:5],
                        "sha256": {h: sha256_file(h) for h in hits[:5]}}
    older = {}
    for rel in ("packages/cp_disr_final_v31/experiments/Experimental_Plan_v1.1.md", "packages/cp_disr_final_v31/research/CP_DISR_Paper_Oriented_Research_Specification_v3.1.md"):
        p = c.root / rel
        if p.is_file():
            older[rel] = sha256_file(p)
    found["older_documents_present_not_authoritative_for_this_card"] = older
    return found


# ----------------------------------------------------------------- budget
def _events(dirpath):
    p = Path(dirpath) / "budget_events.jsonl"
    return jsonl(p) if p.is_file() else []


def _journal_flags(journal):
    phases = [e.get("phase") for e in journal]
    err = next((e for e in journal if e.get("phase") == "error"), None)
    return {"bundle_created": "bundle_created" in phases, "action_start": "action_start" in phases,
            "action_complete": "action_complete" in phases, "error_phase": (err or {}).get("error_phase"),
            "traceback_tail": ((err or {}).get("traceback") or "")[-400:]}


def cmd_budget(c):
    c.need(c.s1 / "budget_ledger.json", *(d / "budget_ledger.json" for d in c.rounds.values()))
    rows, auth_trace = [], {}
    orig_ledger = jread(c.s1 / "budget_ledger.json")
    orig_used = orig_ledger["physical_witness_episodes"]
    # original 8: registration + physical episode csv + budget_events (no journals / receipts in that record)
    reg0 = jread(c.s1 / "witnesses/e4_branch_registration.json")["branches"]
    csv0 = {r["branch_id"]: r for r in csv.DictReader((c.s1 / "witnesses/physical_episode_ledger.csv").open(encoding="utf-8"))}
    ledger_hash = sha256_file(c.s1 / "budget_ledger.json")
    for b in reg0:
        r = csv0.get(b["branch_id"], {})
        rows.append({"round": "original", "attempt_id": b["branch_id"], "case_id": b["case_id"], "candidate_id": b["candidate_id"], "repeat": b["repeat"],
                     "reservation_record_ref": "S1-REV1/budget_ledger.json physical_witness_episodes.used=%s (per-attempt event: NOT_RECORDED)" % orig_used["used"],
                     "attempt_state": "RECORDED_IN_ORIGINAL_LEDGER_CSV" if r else "NOT_VERIFIED",
                     "environment_created_evidence": ("CSV_ROW:controller_exit=%s,verified_fact_count=%s" % (r.get("controller_exit"), r.get("verified_fact_count"))) if r else "NOT_VERIFIED",
                     "skill_executed_evidence": "NOT_VERIFIED (no journal in original record; csv only)",
                     "protocol_complete_evidence": "ORIGINAL_CSV:continuation_status=%s;later engineering review: protocol incomplete" % r.get("continuation_status", ""),
                     "authorization_ref": "NOT_LOCATED", "reservation_time": "NOT_RECORDED", "execution_time": "NOT_RECORDED", "source_hash": ledger_hash})
    total = {"original": int(orig_used["used"])}
    for name, d in c.rounds.items():
        led = jread(d / "budget_ledger.json")
        att = jread(d / "attempt_registry.json")
        events = _events(d)
        reg = jread(d / "witnesses/e4_branch_registration.json")["branches"]
        by_id = {b["branch_id"]: b for b in reg}
        total[name] = int(led["physical_witness_episodes"]["used"])
        for bid, state in sorted(att.items(), key=lambda kv: kv[0]):
            b = by_id.get(bid, {})
            ev = [e for e in events if e.get("branch_id") == bid]
            reserved = next((e["timestamp_utc"] for e in ev if e["status"] == "RESERVED"), "NOT_RECORDED")
            started = next((e["timestamp_utc"] for e in ev if e["status"] == "STARTED"), "NOT_RECORDED")
            jp = d / "witnesses" / f"branch_{bid}.jsonl"
            journal = jsonl(jp) if jp.is_file() else []
            fl = _journal_flags(journal) if journal else None
            wl = d / "worker_logs" / f"{bid}.log"
            if fl and fl["bundle_created"]:
                env_ev = f"JOURNAL:{c.rel(jp)}:bundle_created"
            elif fl and fl["error_phase"] == "bundle_create":
                env_ev = "LOG_SUPPORTED_NOT_PROCESS_VERIFIED:journal error at bundle_create before factory (%s)" % c.rel(jp)
            elif not journal and wl.is_file():
                env_ev = "LOG_SUPPORTED_NOT_PROCESS_VERIFIED:worker exited before claim (%s)" % c.rel(wl)
            else:
                env_ev = "NOT_VERIFIED"
            skill_ev = ("JOURNAL:action_complete" if fl and fl["action_complete"] else
                        ("LOG_SUPPORTED:no action_start event in journal" if fl else ("LOG_SUPPORTED:no journal (worker exited before claim)" if wl.is_file() else "NOT_VERIFIED")))
            res = d / "branch_results" / f"{bid}.json"
            proto = "NOT_APPLICABLE_NO_RESULT_FILE"
            if res.is_file():
                proto = "RESULT_JSON:protocol_complete=%s" % jread(res).get("protocol_complete")
            rec = d / "witnesses/branch_receipts" / f"{bid}.json"
            rows.append({"round": name, "attempt_id": bid, "case_id": b.get("case_id", ""), "candidate_id": b.get("candidate_id", ""), "repeat": b.get("repeat", ""),
                         "reservation_record_ref": c.rel(d / "budget_events.jsonl") + "#RESERVED", "attempt_state": state,
                         "environment_created_evidence": env_ev, "skill_executed_evidence": skill_ev, "protocol_complete_evidence": proto,
                         "authorization_ref": c.rel(d / "authorization.json"), "reservation_time": reserved, "execution_time": started,
                         "source_hash": sha256_file(rec) if rec.is_file() else "NOT_RECORDED"})
        auth_p = d / "authorization.json"
        amend_p = d / "budget_amendment.json"
        a = jread(auth_p)
        auth_trace[name] = {"authorization_path": c.rel(auth_p), "authorization_sha256": sha256_file(auth_p), "budget_amendment_sha256": sha256_file(amend_p),
                            "approval_record_present": bool(a.get("approval_text_adopted") and a.get("approval_source")),
                            "approval_external_transcript_verified": False,
                            "approval_source": a.get("approval_source"), "approval_text_adopted": a.get("approval_text_adopted"),
                            "approved_additional_physical_episodes": a.get("approved_additional_physical_episodes"),
                            "note": "recorded in a local authorization file; the originating session transcript was not independently exported or reviewed here"}
    reserved_total = sum(total.values())
    cum_max = jread(c.r3 / "budget_ledger.json")["references"].get("cumulative_max_original_plus_recovery")
    states = {}
    for r in rows:
        states.setdefault(r["round"], {})
        states[r["round"]][r["attempt_state"]] = states[r["round"]].get(r["attempt_state"], 0) + 1
    r3led = jread(c.r3 / "budget_ledger.json")
    check = {"reserved_by_round": total, "reserved_total": reserved_total, "cumulative_max_declared": cum_max,
             "remaining": (cum_max - reserved_total) if isinstance(cum_max, int) else "NOT_DECLARED",
             "attempt_states_by_round": states,
             "unknown_not_refunded": all(r["attempt_state"] == "UNKNOWN" for r in rows if r["round"] == "r2"),
             "failed_not_refunded": all(r["attempt_state"] == "FAILED" for r in rows if r["round"] == "r1"),
             "rows_in_ledger_csv": len(rows), "rows_match_reserved_total": len(rows) == reserved_total,
             "provider_new_in_r3": {"first_calls": r3led["provider_first_calls"], "retries": r3led["provider_retries"]},
             "provider_original_S1_REV1_preserved": {"first_calls": orig_ledger["provider_first_calls"], "retries": orig_ledger["provider_retries"]},
             "elastic_optimizer_rl_r3": {k: r3led[k] for k in ("elastic_attempts", "optimizer_steps", "rl_transitions")},
             "status": "CONSISTENT" if reserved_total == len(rows) and isinstance(cum_max, int) and reserved_total <= cum_max else "INCONSISTENT"}
    fields = ["round", "attempt_id", "case_id", "candidate_id", "repeat", "reservation_record_ref", "attempt_state", "environment_created_evidence",
              "skill_executed_evidence", "protocol_complete_evidence", "authorization_ref", "reservation_time", "execution_time", "source_hash"]
    write_csv(c.out / "audit/budget/cross_round_attempt_ledger.csv", fields, rows)
    jwrite(c.out / "audit/budget/authorization_trace.json", auth_trace)
    jwrite(c.out / "audit/budget/budget_check.json", check)
    return check


# ----------------------------------------------------------------- branches
def parse_journal(events):
    actions, planner, terminated, cur = [], [], None, None
    for e in events:
        ph = e.get("phase")
        if ph == "action_start":
            cur = {"candidate_id": e.get("candidate_id"), "start_relative": e.get("relative_time"), "sequence": e["sequence"]}
            actions.append(cur)
        elif ph == "action_complete" and cur is not None:
            cur.update({"controller_exit": e.get("controller_exit"), "complete_relative": e.get("relative_time"), "complete_seq": e["sequence"]})
        elif ph == "verifier_complete" and cur is not None:
            cur.update({"verified_fact_count": e.get("verified_fact_count"), "verifier_seq": e["sequence"]})
        elif ph == "evaluator_complete" and cur is not None:
            cur["evaluator"] = {"success": e.get("task_success"), "terminated": e.get("terminated"), "truncated": e.get("truncated"),
                                "reason": e.get("reason"), "relative_time": e.get("relative_time")}
            cur["evaluator_seq"] = e["sequence"]
        elif ph == "planner_return":
            planner.append({"status": e.get("status"), "plan": e.get("plan"), "relative_time": e.get("relative_time"), "expanded_nodes": e.get("expanded_nodes"), "sequence": e["sequence"]})
        elif ph == "terminated":
            terminated = e
    return actions, planner, terminated


def chain_complete(actions):
    """Every action must have action_complete < verifier_complete < evaluator_complete, in order."""
    if not actions:
        return False
    for a in actions:
        ks = ("sequence", "complete_seq", "verifier_seq", "evaluator_seq")
        if any(k not in a for k in ks):
            return False
        if not (a["sequence"] < a["complete_seq"] < a["verifier_seq"] < a["evaluator_seq"]):
            return False
    return True


def branch_outcome(r):
    """Pure function over parsed journal facts; never trusts a summary label."""
    actions, planner = r["actions"], r["planner"]
    last = actions[-1]["evaluator"] if actions and "evaluator" in actions[-1] else {}
    ev_term = bool(last.get("terminated") or last.get("truncated"))
    last_plan = planner[-1] if planner else {}
    plan_stop = (not ev_term) and last_plan.get("status") in ("NO_PLAN", "SEARCH_TIMEOUT")
    return {"last_evaluator_success": last.get("success"), "last_evaluator_reason": last.get("reason"),
            "last_evaluator_terminated": last.get("terminated"), "last_evaluator_truncated": last.get("truncated"),
            "termination_by_evaluator": ev_term, "termination_by_planner": plan_stop,
            "planner_stop_status": last_plan.get("status") if plan_stop else None,
            "action_ids": [a["candidate_id"] for a in actions], "action_count": len(actions),
            "simulation_elapsed_seconds": last.get("relative_time"),
            "per_action_evaluation_chain_complete": chain_complete(actions)}


def pair_difference(open_o, pick_o):
    """E4 difference is judged on evaluator/termination layers, never on action-sequence difference."""
    ev_success_diff = bool(open_o["last_evaluator_success"]) != bool(pick_o["last_evaluator_success"])
    term_diff = (open_o["termination_by_evaluator"], open_o["termination_by_planner"], open_o["last_evaluator_reason"]) != \
                (pick_o["termination_by_evaluator"], pick_o["termination_by_planner"], pick_o["last_evaluator_reason"])
    cost_known = open_o.get("measured_cost") is not None and pick_o.get("measured_cost") is not None
    rework_known = open_o.get("measured_rework") is not None and pick_o.get("measured_rework") is not None
    return {"evaluator_confirmed_success_difference": ev_success_diff,
            "continuation_termination_difference": term_diff,
            "measured_execution_cost_difference": ((open_o["measured_cost"] != pick_o["measured_cost"]) if cost_known else "NOT_MEASURED"),
            "measured_rework_difference": ((open_o["measured_rework"] != pick_o["measured_rework"]) if rework_known else "NOT_MEASURED"),
            "action_sequence_difference_descriptive_only": open_o["action_ids"] != pick_o["action_ids"],
            "e4_outcome_difference_established": bool(ev_success_diff or term_diff or
                                                      (cost_known and open_o["measured_cost"] != pick_o["measured_cost"]) or
                                                      (rework_known and open_o["measured_rework"] != pick_o["measured_rework"]))}


def cmd_branches(c):
    reg = jread(c.r3 / "witnesses/e4_branch_registration.json")["branches"]
    if len(reg) != 8:
        raise AnalysisIncomplete(f"expected 8 registered branches, found {len(reg)}")
    events = jsonl(c.r3 / "scheduling_events.jsonl")
    wall = {e["branch_id"]: e["wall_seconds"] for e in events if e.get("event") == "branch_finish"}
    att = jread(c.r3 / "attempt_registry.json")
    outcomes, sidecars = {}, {}
    for b in reg:
        bid = b["branch_id"]
        files = {"result": c.r3 / "branch_results" / f"{bid}.json", "journal": c.r3 / "witnesses" / f"branch_{bid}.jsonl",
                 "budget_receipt": c.r3 / "witnesses/branch_receipts" / f"{bid}.json",
                 "restore_receipt": c.r3 / "witnesses/restore_receipts" / f"{bid}.json",
                 "initial_state": c.r3 / "witnesses/initial_state_checks" / f"{bid}.json"}
        c.need(*files.values())
        res, jr = jread(files["result"]), jsonl(files["journal"])
        actions, planner, terminated = parse_journal(jr)
        oc = branch_outcome({"actions": actions, "planner": planner})
        receipt, rr, ini = jread(files["budget_receipt"]), jread(files["restore_receipt"]), jread(files["initial_state"])
        oc.update({"branch_id": bid, "case_id": b["case_id"], "candidate_id": b["candidate_id"], "repeat": b["repeat"],
                   "attempt_state": att.get(bid), "controller_exit": res.get("controller_exit"), "execution_status": res.get("execution_status"),
                   "termination_reason": res.get("termination_reason"), "task_success": res.get("task_success"), "protocol_complete": res.get("protocol_complete"),
                   "wall_seconds": wall.get(bid), "worker_wall_seconds": res.get("worker_wall_seconds"),
                   "measured_cost": None, "measured_rework": None,
                   "cost_rework_note": "NOT_MEASURED: no rework counter recorded; wall time is not used as task cost",
                   "restore_receipt_in_journal_matches_file": next((e.get("restore_receipt_sha256") for e in jr if e.get("phase") == "restore_complete"), None) == rr["restore_receipt_sha256"],
                   "budget_receipt_binds_branch": receipt.get("branch_id") == bid and receipt.get("state") == "COMPLETED",
                   "evidence_refs": {k: c.rel(v) for k, v in files.items()}, "source_hashes": {k: sha256_file(v) for k, v in files.items()},
                   "journal_event_count": len(jr), "result_trace_actions": [t.get("candidate_id") for t in res.get("trace", [])],
                   "journal_matches_result_trace": [a["candidate_id"] for a in actions] == [t.get("candidate_id") for t in res.get("trace", [])]})
        outcomes[bid] = oc
        sidecars[bid] = (rr, ini)
    # pairing
    pairs, cases = {}, sorted({b["case_id"] for b in reg})
    for b in reg:
        pairs.setdefault((b["case_id"], b["repeat"]), []).append(b)
    pair_docs = []
    for (case, rep), bs in sorted(pairs.items()):
        cands = sorted(x["candidate_id"] for x in bs)
        ok_structure = cands == sorted([OPEN_ID, PICK_ID])
        by_c = {x["candidate_id"]: x["branch_id"] for x in bs}
        doc = {"case_id": case, "repeat": rep, "candidates": cands, "structure_exactly_open_and_pick": ok_structure}
        if ok_structure:
            (rr_o, in_o), (rr_p, in_p) = sidecars[by_c[OPEN_ID]], sidecars[by_c[PICK_ID]]
            qo, qp = in_o["physical"]["qpos"], in_p["physical"]["qpos"]
            vo, vp = in_o["physical"]["qvel"], in_p["physical"]["qvel"]
            shape_ok = len(qo) == len(qp) and len(vo) == len(vp)
            qmax = max((abs(x - y) for x, y in zip(qo, qp)), default=None) if shape_ok else None
            vmax = max((abs(x - y) for x, y in zip(vo, vp)), default=None) if shape_ok else None
            rule_seed = int(hashlib.sha256(f"{case}|{rep}".encode()).hexdigest()[:8], 16)
            seeds = {rr_o["restore_receipt"]["applied_restore_seed"], rr_p["restore_receipt"]["applied_restore_seed"],
                     next(x for x in bs if x["candidate_id"] == OPEN_ID)["restore_seed"], next(x for x in bs if x["candidate_id"] == PICK_ID)["restore_seed"]}
            keys = ("normalized_reset_config_sha256", "public_facts_sha256", "candidate_ids_sha256", "candidate_mask_sha256")
            doc.update({"qpos_len": [len(qo), len(qp)], "qvel_len": [len(vo), len(vp)], "qpos_max_abs_diff": qmax, "qvel_max_abs_diff": vmax,
                        "tolerance_original": QPOS_TOL,
                        "receipt_public_hashes_equal": {k: rr_o["restore_receipt"][k] == rr_p["restore_receipt"][k] for k in keys},
                        "candidate_ids_equal": in_o["candidate_ids"] == in_p["candidate_ids"], "candidate_mask_equal": in_o["candidate_mask"] == in_p["candidate_mask"],
                        "paired_seed_values": sorted(seeds), "paired_seed_single_value": len(seeds) == 1, "paired_seed_equals_rule": seeds == {rule_seed},
                        "paired_restore_within_tolerance": bool(shape_ok and qmax is not None and qmax <= QPOS_TOL and vmax is not None and vmax <= QPOS_TOL),
                        "controller_internal_state_comparison": "NOT_MEASURED", "rng_stream_comparison": "NOT_MEASURED",
                        "restore_scope": "PUBLIC_FACTS_CANDIDATES_MASK_AND_SIM_QPOS_QVEL"})
            doc["differences"] = pair_difference(outcomes[by_c[OPEN_ID]], outcomes[by_c[PICK_ID]])
            doc["branch_ids"] = by_c
        pair_docs.append(doc)
    unit = {"independent_case_configurations": len(cases), "cases": cases, "paired_repeats_per_case": {cs: sum(1 for k in pairs if k[0] == cs) for cs in cases},
            "paired_comparisons": len(pairs), "branch_attempts": len(reg),
            "statistical_note": "repeats of one case are not independent configurations; no significance or generalisation rate is computed"}
    jwrite(c.out / "audit/branches/branch_outcomes.json", {"branches": outcomes, "unit_of_analysis": unit})
    review = review_e4(pair_docs, outcomes, unit)
    jwrite(c.out / "audit/branches/paired_outcomes_reviewed.json", {"pairs": pair_docs, "unit_of_analysis": unit, "case_review": review})
    (c.out / "evidence").mkdir(parents=True, exist_ok=True)
    (c.out / "evidence/e4_review.md").write_text(render_e4(review, pair_docs, outcomes, unit), encoding="utf-8")
    return {"status": "OK", "pairs": len(pair_docs)}


def review_e4(pairs, outcomes, unit):
    per_case = {}
    for p in pairs:
        cs = per_case.setdefault(p["case_id"], {"pairs": 0, "restore_ok": 0, "outcome_diff": 0, "ev_success_diff": 0, "chains_ok": True, "structure_ok": True})
        cs["pairs"] += 1
        cs["structure_ok"] &= bool(p.get("structure_exactly_open_and_pick"))
        if p.get("structure_exactly_open_and_pick"):
            cs["restore_ok"] += int(bool(p["paired_restore_within_tolerance"]))
            cs["outcome_diff"] += int(bool(p["differences"]["e4_outcome_difference_established"]))
            cs["ev_success_diff"] += int(bool(p["differences"]["evaluator_confirmed_success_difference"]))
            for bid in p["branch_ids"].values():
                cs["chains_ok"] &= bool(outcomes[bid]["per_action_evaluation_chain_complete"])
    out = {}
    for cs, v in per_case.items():
        reliable = v["structure_ok"] and v["chains_ok"] and v["restore_ok"] == v["pairs"] and v["outcome_diff"] == v["pairs"]
        out[cs] = {**v, "protocol_level_reliable_outcome_difference": reliable,
                   "supported_scope": "same public initial state and sim qpos/qvel; shared B_PLAN continuation; evaluator-level outcome",
                   "not_compared": ["controller internal state", "RNG stream", "actual post-action facts (not saved)"],
                   "physical_infeasibility_of_PICK_branch_established": False}
    n_reliable = sum(1 for v in out.values() if v["protocol_level_reliable_outcome_difference"])
    return {"per_case": out, "cases_with_reliable_protocol_level_difference": n_reliable,
            "frozen_E4_gate_status": ("LIMITED_SUPPORT_FEASIBILITY_ONLY" if n_reliable >= 2 else "NOT_SATISFIED_ON_AVAILABLE_EVIDENCE"),
            "note": "derived evidence status against Final Experimental Plan v3 §5.2 (not an official v3 enum); the action-sequence difference is not used as E4 evidence; cost/rework NOT_MEASURED. E4 and E5 stay separate."}


def render_e4(review, pairs, outcomes, unit):
    L = ["# E4 复核（r3，零新增样本）", "",
         f"统计单位：{unit['independent_case_configurations']} 个独立 case 配置 × 每 case {list(unit['paired_repeats_per_case'].values())} 个 paired repeats = {unit['paired_comparisons']} 组配对、{unit['branch_attempts']} 个 branch attempts；repeat 不是独立配置，不做显著性或泛化率计算。", "",
         "## 每对结果", ""]
    for p in pairs:
        d = p.get("differences", {})
        L.append(f"- {p['case_id']} repeat {p['repeat']}：evaluator 确认成功差={d.get('evaluator_confirmed_success_difference')}；终止层差={d.get('continuation_termination_difference')}；"
                 f"cost 差={d.get('measured_execution_cost_difference')}；rework 差={d.get('measured_rework_difference')}；动作序列不同（仅描述）={d.get('action_sequence_difference_descriptive_only')}；"
                 f"配对恢复(qpos/qvel≤{p.get('tolerance_original')})={p.get('paired_restore_within_tolerance')}（qpos最大差 {p.get('qpos_max_abs_diff')}，qvel最大差 {p.get('qvel_max_abs_diff')}）")
    L += ["", "## 状态层级（不合并）", "",
          "- OPEN 分支：独立 Evaluator 在最后一个动作后给出 TASK_SUCCESS（terminated=true），真实任务成功。",
          "- PICK 分支：首动作后 Evaluator 为 CONTINUE / success=false / 未终止；随后 B_PLAN 返回 NO_PLAN，runner 因 planner 停止。这不是 Evaluator 判定的失败，也不是 DEADLINE，更不是物理不可行的证明。", "",
          "## 每 case 结论", ""]
    for cs, v in review["per_case"].items():
        L.append(f"- {cs}：{v['pairs']} 组配对，配对恢复通过 {v['restore_ok']}，结果差 {v['outcome_diff']}；协议层可靠差异={v['protocol_level_reliable_outcome_difference']}。支持范围：{v['supported_scope']}；未比较：{', '.join(v['not_compared'])}。")
    L += ["", f"E4 派生证据状态：{review['frozen_E4_gate_status']}（{review['note']}）", "",
          "限制：cost/rework 无可比记录（NOT_MEASURED）；PICK 后真实 post-action facts 未保存，不能声称重演了 NO_PLAN 的运行时原因。"]
    return "\n".join(L) + "\n\n" + E4_V3_SECTION


# ----------------------------------------------------------------- contracts (pure symbolic; no environment)
def _imports():
    from cp_disr.contracts import nominal_overlay, precondition_value
    from cp_disr.common import ContractError
    from cp_disr.facts import FactRecord, FactStore, Truth
    from cp_disr.graph import Goal, build_template
    return nominal_overlay, precondition_value, ContractError, FactRecord, FactStore, Truth, Goal, build_template


def load_actual_contracts(c):
    """Ground the contracts exactly as create_task_runtime does (same production helper), from the r3 manifest."""
    rm = jread(c.r3 / "runtime_manifest.json")
    import yaml
    manifest = yaml.safe_load(Path(rm["source_manifest_path"]).read_text(encoding="utf-8"))
    rt = manifest["runtime"]
    from cp_disr.platforms.libero import runtime_factory as rf
    task = rt["active_task_id"]
    contract_path = Path(rt["repository_path"]) / rt.get("stage_2a_contract_path", "configs/runtime/stage_2a_contract_registry.yaml")
    objects = rf.TASK_OBJECTS[task]
    contracts = rf._ground_contracts_for(contract_path, rt["skill_timeouts"][task], objects)
    _, _, _, _, _, _, Goal, build_template = _imports()
    goals = tuple(Goal(g, 1) for g in rf.TASK_GOALS[task])
    template = build_template(contracts, goals, rf.PREDICATES, objects)
    return {"manifest": manifest, "runtime": rt, "task": task, "contract_path": contract_path, "objects": objects,
            "contracts": template.contracts, "template": template, "rf": rf, "manifest_path": rm["source_manifest_path"]}


def scene_dir(c, case):
    return c.root / "experiments/stage_0c_inputs/dev/T_A" / ("scene_" + case.split("_")[-1])


def bind_case_facts(c, case, branch_receipts):
    d = scene_dir(c, case)
    summ, ids, mask, reset = jread(d / "initial_fact_summary.json"), jread(d / "candidate_ids.json"), jread(d / "candidate_mask.json"), jread(d / "reset_config.json")
    values = {r["fact_id"]: r["value"] for r in summ["records"]}
    checks = {"public_facts_sha256": json_hash(values), "candidate_ids_sha256": json_hash(ids), "candidate_mask_sha256": json_hash(mask),
              "normalized_reset_config_sha256": reset["normalized_reset_config_sha256"]}
    verdicts = {}
    for k, v in checks.items():
        verdicts[k] = all(rr["restore_receipt"][k] == v for rr in branch_receipts)
    ok = all(verdicts.values())
    return {"case_id": case, "scene_dir": c.rel(d), "computed_from_scene": checks, "equals_r3_restore_receipts": verdicts,
            "binding_status": "EXACT_INITIAL_STATE_VERIFIED_AGAINST_R3_RECEIPTS" if ok else "PUBLIC_SNAPSHOT_BINDING_UNVERIFIED",
            "facts": values, "candidate_ids": ids, "candidate_mask": mask,
            "scene_hashes": {f.name: sha256_file(f) for f in (d / "initial_fact_summary.json", d / "candidate_ids.json", d / "candidate_mask.json", d / "reset_config.json")}}


def to_truth_map(values):
    _, _, _, _, _, Truth, _, _ = _imports()
    return {k: Truth(v) for k, v in values.items()}


def symbolic_closure(contracts, start, goals, derived=(), exclusive=(), limit=ENUM_LIMIT):
    """Exhaustive nominal reachability over frozen T/F/U state vectors using the production functions."""
    nominal_overlay, precondition_value, ContractError, _, _, Truth, _, _ = _imports()
    order = sorted(start)
    key = lambda vals: tuple(vals[k].value for k in order)
    sk = key(start)
    states, parent, edges = {sk: dict(start)}, {sk: None}, set()
    queue = deque([sk])
    goal_hits, status = [], "COMPLETE"
    def is_goal(vals):
        return all(vals[g["fact_id"]] == (Truth.TRUE if g["sign"] == 1 else Truth.FALSE) for g in goals)
    if is_goal(start):
        goal_hits.append(sk)
    while queue:
        k = queue.popleft()
        vals = states[k]
        for contract in sorted(contracts, key=lambda x: x.id):
            if precondition_value(contract, vals) != Truth.TRUE:
                continue
            try:
                nxt = dict(nominal_overlay(contract, vals, derived=derived, exclusive_groups=exclusive))
            except ContractError:
                continue
            nk = key(nxt)
            edges.add((k, contract.id, nk))
            if nk not in states:
                if len(states) >= limit:
                    status = "ENUMERATION_LIMIT_REACHED"
                    break
                states[nk] = nxt
                parent[nk] = (k, contract.id)
                queue.append(nk)
                if is_goal(nxt):
                    goal_hits.append(nk)
        if status != "COMPLETE":
            break
    plan = None
    if goal_hits:
        g, seq = goal_hits[0], []
        while parent[g] is not None:
            g, a = parent[g][0], parent[g][1]
            seq.append(a)
        plan = list(reversed(seq))
    reachable = bool(goal_hits)
    return {"status": status, "state_count": len(states), "edge_count": len(edges), "enumeration_limit": limit, "state_space_bound": 3 ** len(order),
            "goal_reachable": True if reachable else ("PROVEN_UNREACHABLE" if status == "COMPLETE" else "UNDETERMINED_LIMIT_REACHED"),
            "shortest_nominal_plan": plan, "goal_state_count": len(goal_hits), "predicate_order": order,
            "states_sha256": json_hash(sorted(states)), "edges_sha256": json_hash(sorted(edges))}


def atom_roles(contracts, atom_id):
    add, dele, req = [], [], []
    for ct in contracts:
        if any(a.id == atom_id for a in ct.effects.add): add.append(ct.id)
        if any(a.id == atom_id for a in ct.effects.delete): dele.append(ct.id)
        if any(a.id == atom_id for a in ct.pre_pos + ct.pre_neg): req.append(ct.id)
    return {"ADD_by": sorted(add), "DEL_by": sorted(dele), "required_by_precondition": sorted(req)}


def observed_from_journal(c, case, cand):
    reg = jread(c.r3 / "witnesses/e4_branch_registration.json")["branches"]
    b = next(x for x in reg if x["case_id"] == case and x["candidate_id"] == cand and x["repeat"] == 0)
    actions, planner, _ = parse_journal(jsonl(c.r3 / "witnesses" / f"branch_{b['branch_id']}.jsonl"))
    return {"branch_id": b["branch_id"], "first_action_complete_relative": actions[0].get("complete_relative"),
            "first_planner_return": planner[0] if planner else None}


def scan_post_action_facts(c):
    hits = []
    for p in sorted((c.r3 / "witnesses").glob("branch_*.jsonl")) + sorted((c.r3 / "branch_results").glob("*.json")):
        text = p.read_text(encoding="utf-8")
        if '"fact_id"' in text or '"facts"' in text:
            hits.append(c.rel(p))
    return {"files_with_fact_values": hits, "status": "SAVED_AND_FOUND" if hits else "NOT_RECORDED_OR_NOT_FOUND",
            "scanned": "r3 branch journals and branch_results (fact_id/facts keys); sidecars hold hashes and QA coordinates only"}


def cmd_contracts(c):
    nominal_overlay, precondition_value, ContractError, FactRecord, FactStore, Truth, Goal, build_template = _imports()
    reg = jread(c.r3 / "witnesses/e4_branch_registration.json")["branches"]
    A = load_actual_contracts(c)
    contracts, template, rt = A["contracts"], A["template"], A["runtime"]
    rf_src = Path(A["rf"].__file__).read_text(encoding="utf-8")
    contract_blob_hashes = {}
    declared = jread(c.r3 / "authorization.json").get("execution_commit")
    rel = "configs/runtime/stage_2a_contract_registry.yaml"
    for rev in (declared, "HEAD"):
        b = git_blob(c.root, rev, rel)
        contract_blob_hashes[str(rev)] = hashlib.sha256(b).hexdigest() if b is not None else "NOT_FOUND"
    id_set = sorted(ct.id for ct in contracts)
    facts_for = {}
    bindings = {"contract_file": c.rel(A["contract_path"]), "contract_file_sha256": sha256_file(A["contract_path"]), "contract_blob_sha256_by_rev": contract_blob_hashes,
                "runtime_manifest_path": A["manifest_path"], "runtime_manifest_sha256": sha256_file(A["manifest_path"]),
                "task": A["task"], "objects": A["objects"], "grounded_candidate_ids": id_set,
                "twelve_seven_candidates_align": len(id_set) == 7, "no_MOVE_DROP_PICK_FROM_BUFFER": all(n not in " ".join(id_set) for n in ("MOVE", "DROP", "PICK_FROM_BUFFER")),
                "template_derived_rules": [str(x) for x in template.derived_rules], "template_exclusive_groups": [list(g) for g in template.exclusive_groups],
                "factory_build_template_call_uses_derived_or_exclusive": ("derived_rules=" in rf_src or "exclusive_groups=" in rf_src),
                "source_files_sha256": {f: sha256_file(c.root / f) for f in ("src/cp_disr/contracts.py", "src/cp_disr/baselines/b_plan.py", "src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/graph.py", "src/cp_disr/facts.py")},
                "goals": [{"fact_id": g.fact_id, "sign": g.sign} for g in template.goals], "cases": {}}
    results, nominal, per_case_diag = {}, {}, {}
    cases = sorted({b["case_id"] for b in reg})
    pb = None
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    d_ref = float(rt["reference_skill_seconds_by_task"][A["task"]])
    deadline = float(rt["task_deadlines"][A["task"]])
    post_scan = scan_post_action_facts(c)
    for case in cases:
        rrs = [jread(p) for p in sorted((c.r3 / "witnesses/restore_receipts").glob("*.json")) if jread(p)["case_id"] == case]
        bind = bind_case_facts(c, case, rrs)
        bindings["cases"][case] = {k: v for k, v in bind.items() if k not in ("facts",)}
        facts = to_truth_map(bind["facts"])
        by_id = {ct.id: ct for ct in contracts}
        legal = {cid: precondition_value(by_id[cid], facts).value for cid in (OPEN_ID, PICK_ID)}
        mask_map = dict(zip(bind["candidate_ids"], bind["candidate_mask"]))
        succ, res = {}, {}
        goals = list(GOALS_T_A)
        for cid in (OPEN_ID, PICK_ID):
            ov = dict(nominal_overlay(by_id[cid], facts, derived=template.derived_rules, exclusive_groups=template.exclusive_groups))
            succ[cid] = {"facts": {k: v.value for k, v in ov.items()}, "changed_vs_initial": {k: v.value for k, v in ov.items() if v != facts[k]},
                         "provenance": "NOMINAL_DERIVED (nominal_overlay on saved initial facts); not a Verifier output"}
            res[cid] = symbolic_closure(contracts, ov, goals, template.derived_rules, template.exclusive_groups)
        res["initial_state_reference"] = symbolic_closure(contracts, facts, goals, template.derived_rules, template.exclusive_groups)
        nominal[case] = {"binding_status": bind["binding_status"], "legality_by_precondition_value": legal, "legality_matches_saved_mask": {cid: (legal[cid] == "TRUE") == bool(mask_map[cid]) for cid in legal},
                         "initial_open_container": bind["facts"].get("p:Open:container"), "initial_inside_goal_second": bind["facts"].get("p:Inside:second_object:container"),
                         "successors": succ}
        # B_PLAN offline on saved / nominal-derived inputs (no runtime, no real clock)
        def store(vals):
            return FactStore(tuple(FactRecord(fact_id=k, value=v, capture_time=0.0, available_time=0.0, reason="offline_analysis") for k, v in vals.items()))
        planner_runs = {}
        obs_open, obs_pick = observed_from_journal(c, case, OPEN_ID), observed_from_journal(c, case, PICK_ID)
        for label, vals, remaining, prov in (
                ("initial_saved_facts", facts, deadline, "SAVED_INITIAL_FACTS"),
                ("after_OPEN_nominal", to_truth_map(succ[OPEN_ID]["facts"]), deadline - float(obs_open["first_action_complete_relative"]), "NOMINAL_DERIVED"),
                ("after_PICK_second_nominal", to_truth_map(succ[PICK_ID]["facts"]), deadline - float(obs_pick["first_action_complete_relative"]), "NOMINAL_DERIVED")):
            planner = BPlanPlanner(SearchConfig(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=d_ref))
            pr = planner.plan(store(vals), template, remaining)
            planner_runs[label] = {"input_provenance": prov, "remaining_deadline_seconds_from_journal": remaining, "status": pr.status, "plan": list(pr.plan),
                                   "expanded_nodes": pr.expanded_nodes, "generated_nodes": pr.generated_nodes, "cpu_seconds": pr.cpu_seconds,
                                   "config": {"depth_limit": 6, "max_nodes": 4096, "cpu_time_limit_seconds": 2.0, "reference_skill_seconds": d_ref}}
        planner_runs["observed_runtime_first_planner_return"] = {"OPEN": obs_open, "PICK": obs_pick}
        results[case] = {"closure": res, "offline_b_plan": planner_runs}
        # diagnostic fields (not new official E6 categories)
        open_ok, pick = res[OPEN_ID], res[PICK_ID]
        distinguishes = open_ok["goal_reachable"] is True and pick["goal_reachable"] == "PROVEN_UNREACHABLE"
        per_case_diag[case] = {
            "CONTRACT_ALREADY_DISTINGUISHES_ACTIONS": bool(distinguishes),
            "NOMINAL_RECOVERY_GAP": bool(distinguishes and not atom_roles(contracts, "p:OnTable:second_object")["ADD_by"]),
            "SEARCH_HORIZON_LIMITATION": False if (pick["status"] == "COMPLETE" and pick["goal_reachable"] == "PROVEN_UNREACHABLE") else "NOT_ESTABLISHED",
            "SEARCH_NODE_OR_CPU_LIMITATION": ("NOT_ESTABLISHED" if planner_runs["after_PICK_second_nominal"]["status"] == "SEARCH_TIMEOUT" else False),
            "OBSERVATION_CONTRACT_MISMATCH": "NOT_ESTABLISHED" if post_scan["status"] != "SAVED_AND_FOUND" else "REQUIRES_COMPARISON_NOT_RUN",
            "EXTRA_CONTRACT_CONSEQUENCE_PLAUSIBLE": "NOT_ESTABLISHED",
            "INSUFFICIENT_SAVED_STATE": post_scan["status"] != "SAVED_AND_FOUND",
            "runtime_NO_PLAN_reason": "OBSERVED_NOT_REENACTED (actual post-action facts not saved); nominal model reachability reported separately"}
    roles = {a: atom_roles(contracts, a) for a in ("p:OnTable:second_object", "p:Held:second_object", "p:GripperEmpty", "p:Open:container", "p:Inside:second_object:container", "p:AtBuffer:second_object:buffer")}
    jwrite(c.out / "contract/input_bindings.json", bindings)
    jwrite(c.out / "contract/nominal_successors.json", nominal)
    jwrite(c.out / "contract/reachability_results.json", {"per_case": results, "diagnostic_fields": per_case_diag, "atom_roles": roles,
                                                          "actual_post_action_facts": post_scan["status"], "post_action_scan": post_scan})
    (c.out / "contract").mkdir(parents=True, exist_ok=True)
    (c.out / "contract/contract_order_explanation.md").write_text(render_contract_md(roles, results, per_case_diag, nominal), encoding="utf-8")
    (c.out / "contract/actual_vs_nominal_evidence_limits.md").write_text(render_limits_md(post_scan, nominal, results), encoding="utf-8")
    return {"status": "OK", "cases": cases}


def render_contract_md(roles, results, diag, nominal):
    L = ["# 合同层面的 OPEN / PICK 顺序解释（纯符号，NOMINAL_DERIVED，不是运行时观测）", "",
         "依据实际运行的 stage_2a 合同（无 derived 规则、无 exclusive groups）。相关效果（生产合同解析）：", ""]
    for a, r in roles.items():
        L.append(f"- `{a}`：ADD by {r['ADD_by']}；DEL by {r['DEL_by']}；作为前提被 {r['required_by_precondition']} 使用")
    L += ["", "适用条件：初态 `Open(container)=FALSE` 且 `Inside(second_object,container)=FALSE`（下表由保存的公开 facts 核验）。在此条件下：",
          "`Inside(second_object,container)` 只能由 PLACE(second_object,container) 增加，它需要 `Held(second_object)` 与 `Open(container)`；"
          "`Held(second_object)` 只能由 PICK(second_object) 增加，它需要 `OnTable(second_object)` 并永久删除之，且没有任何合同再增加 `OnTable(second_object)`；"
          "PICK 之后 `GripperEmpty=FALSE`，而 OPEN 需要 `GripperEmpty`，只有 PLACE/PLACE_BUFFER 才能恢复 GripperEmpty，二者都会删除 Held。"
          "因此名义合同下 OPEN 必须先于 PICK(second_object)。以上是名义合同的解释，不是物理不可行的证明。", "", "## 精确闭包结果", ""]
    for case, r in results.items():
        o, p = r["closure"][OPEN_ID], r["closure"][PICK_ID]
        L.append(f"- {case}：OPEN 后 goal 可达={o['goal_reachable']}（最短名义计划 {o['shortest_nominal_plan']}，闭包状态 {o['state_count']}，状态空间上界 {o['state_space_bound']}）；"
                 f"PICK(second) 后 goal 可达={p['goal_reachable']}（闭包状态 {p['state_count']}，状态 hash {p['states_sha256'][:12]}，边 hash {p['edges_sha256'][:12]}，状态: {p['status']}）。")
        bp = r["offline_b_plan"]
        L.append(f"  - 原 B_PLAN 配置离线：初态 {bp['initial_saved_facts']['status']} {bp['initial_saved_facts']['plan']}；OPEN 后(NOMINAL_DERIVED) {bp['after_OPEN_nominal']['status']}；PICK 后(NOMINAL_DERIVED) {bp['after_PICK_second_nominal']['status']}。")
    L += ["", "## 诊断字段（非新增官方 E6 类别）", ""]
    for case, d in diag.items():
        L.append(f"- {case}：" + "；".join(f"{k}={v}" for k, v in d.items()))
    L += ["", "若合同已能区分 OPEN 与 PICK 的长期可达性，这对候选不支持 E5 所需的“合同近同排序下的 soft 差异”；即便当前 mask 二者均为 true，也不代表长期合同效用相同。本轮不改合同、不补技能、不重跑物理实验。"]
    return "\n".join(L) + "\n"


def render_limits_md(post_scan, nominal, results):
    L = ["# 实际观测 vs 名义推演：证据限制", "",
         f"- 首动作后的真实 Verifier facts：{post_scan['status']}（journal 仅保存 `verified_fact_count=10`，没有 10 条 facts 的值；{post_scan['scanned']}）。",
         "- 因此本轮不声称重演了运行时 NO_PLAN 的原因；名义后继标为 NOMINAL_DERIVED，不能当成真实 Verifier 输出。",
         "- 隐藏 QA 坐标（sidecar 的 hidden_truth）只用于配对恢复 QA，未转成正式公开 facts 来补缺口。", ""]
    for case, n in nominal.items():
        L.append(f"- {case}：初态绑定 {n['binding_status']}；合法性与保存 mask 一致={n['legality_matches_saved_mask']}。")
    L += ["", "- 精确闭包只说明名义合同模型下的可达性；运行时观察到的 NO_PLAN 与之一致但不是同一证据。",
          "- 闭包使用 3^predicate_count 的严格去重状态；若超过 100000 会写 ENUMERATION_LIMIT_REACHED 而不是“不可达”。"]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------- relations
def multihop_paths(contracts, source_action, goal_fact, max_depth=4):
    """Symbolic dependency paths: action --ADD--> atom --PRE_POS--> action ... --ADD--> goal (contract-only, no runtime)."""
    by_id = {ct.id: ct for ct in contracts}
    found = []
    def walk(action_id, trail, seen):
        ct = by_id[action_id]
        for a in ct.effects.add:
            if a.id == goal_fact:
                found.append(trail + [{"action": action_id, "ADD": a.id}])
                continue
            if len(trail) >= max_depth:
                continue
            for nxt in sorted(by_id.values(), key=lambda x: x.id):
                if nxt.id in seen or nxt.id == action_id:
                    continue
                if any(p.id == a.id for p in nxt.pre_pos):
                    walk(nxt.id, trail + [{"action": action_id, "ADD": a.id, "required_by_precondition_of": nxt.id}], seen | {nxt.id})
    walk(source_action, [], {source_action})
    return found


def local_redundancy(contracts, rel, goals):
    """Mirror of the frozen validator's local redundancy rules (informational; the cache holds the production decision)."""
    by_id = {ct.id: ct for ct in contracts}
    s, t, ty, eff = rel["source_ref"], rel["target_ref"], rel["type"], rel.get("effect_fact_ref")
    if ty == "SOFT_RELEVANT_TO_GOAL":
        return bool(eff == t and t in goals)
    if ty == "SOFT_SUPPORTS" and s in by_id and t in by_id:
        cs, ct = by_id[s], by_id[t]
        added = any(a.id == eff for a in cs.effects.add)
        deleted = any(a.id == eff for a in cs.effects.delete)
        return bool((added and any(p.id == eff for p in ct.pre_pos)) or (deleted and any(p.id == eff for p in ct.pre_neg)))
    return False


def _rec(case, rid, scope, status, refs, hashes, origin, missing, limits, **extra):
    d = {"case_id": case, "record_id": rid, "evidence_scope": scope, "status": status, "evidence_refs": refs, "source_hashes": hashes,
         "measurement_origin": origin, "missing_fields": missing, "limitations": limits}
    d.update(extra)
    return d


def cmd_relations(c):
    from cp_disr.vlm_cache_pipeline import verify_audit_cache
    A = load_actual_contracts(c)
    contracts = A["contracts"]
    goals = [g.fact_id for g in A["template"].goals]
    idx = c.s1 / "provider/cache_index.csv"
    c.need(idx)
    rows = list(csv.DictReader(idx.open(encoding="utf-8")))
    if len(rows) != 8:
        raise AnalysisIncomplete(f"expected 8 provider caches, found {len(rows)}")
    records, per_case, utility = [], {}, []
    for row in rows:
        case, cdir = row["scene_id"], Path(row["cache_path"])
        manifest, final_edges = verify_audit_cache(cdir)  # raises on any hash/identity mismatch; no swallowing
        accepted = jread(cdir / "accepted_relations.json")
        rejected = jread(cdir / "rejected_relations.json")
        refs = {"cache_dir": c.rel(cdir), "index": c.rel(idx)}
        hashes = {n: sha256_file(cdir / n) for n in ("manifest.json", "accepted_relations.json", "rejected_relations.json", "final_edges.json", "raw_response.json")}
        per_case[case] = {"cache_key": row["cache_key"], "verify_audit_cache": "OK", "processing_status": manifest["processing_status"],
                          "admitted_count_from_final_edges": len(final_edges), "admitted_count_index": int(row["admitted_count"]),
                          "accepted_count": len(accepted), "rejected_count": len(rejected),
                          "counts_consistent": len(final_edges) == int(row["admitted_count"]) == len(accepted)}
        for rel in accepted:
            paths = multihop_paths(contracts, rel["source_ref"], rel["target_ref"]) if rel["source_ref"].startswith("a:") else []
            redundant = local_redundancy(contracts, rel, goals)
            explained = bool(paths)
            records.append(_rec(case, f"{case}:{rel['relation_id']}", "provider_cache_admitted_relation", "ADJUDICATION_INCOMPLETE", refs, hashes,
                                "verify_audit_cache on saved cache; contract analysis recomputed offline",
                                ["independent_truth_adjudication", "q_R", "R_star"],
                                ["truth/utility/opportunity need independent adjudication that is not in the saved data; no R* invented; provider not called"],
                                relation=rel, admitted_under_frozen_validator=True, local_contract_redundancy_result=bool(redundant),
                                multihop_contract_explanation=explained, multihop_paths=paths,
                                extra_contract_information_established=False,
                                extra_contract_information_reason=("multi-hop contract path explains the relation; not established as information beyond the contract" if explained else "not established (no independent adjudication)"),
                                truth_label="UNKNOWN", utility_label="UNKNOWN", opportunity_label="UNKNOWN",
                                admission_and_multihop_explanation_compatible=True))
            utility.append(_rec(case, f"{case}:{rel['relation_id']}:utility", "relation_utility", "UNKNOWN", refs, hashes,
                                "no independent utility measurement saved", ["q_R", "R_star", "utility_outcome"],
                                ["q_R/R* inputs missing from saved data; reported as missing, not fabricated"], utility_label="UNKNOWN"))
    nonempty = sorted(k for k, v in per_case.items() if v["admitted_count_from_final_edges"] > 0)
    summary = {"caches_verified": len(per_case), "per_case": per_case, "nonempty_cases": nonempty,
               "nonempty_counts": {k: per_case[k]["admitted_count_from_final_edges"] for k in nonempty},
               "expected_14_18_19_two_each_matches": nonempty == ["T_A_dev_14", "T_A_dev_18", "T_A_dev_19"] and all(per_case[k]["admitted_count_from_final_edges"] == 2 for k in nonempty),
               "total_admitted": sum(v["admitted_count_from_final_edges"] for v in per_case.values()),
               "case_19_has_physical_branch": False,
               "case_19_note": "T_A_dev_19 has no r3 branch; consequences of cases 14/18 are not transferred to it"}
    jwrite(c.out / "evidence/relation_adjudications.json", {"summary": summary, "records": records})
    jwrite(c.out / "evidence/utility_records.json", {"records": utility})
    L = ["# 关系裁定与多跳合同解释（离线，未调用 provider）", "",
         f"8 个 provider cache 均通过 verify_audit_cache；非空 case：{', '.join(nonempty)}；被冻结 validator 接纳的关系共 {summary['total_admitted']} 条。", "",
         "OPEN→目标/第二物体 的 SOFT_RELEVANT_TO_GOAL 被接纳，原因是本地冗余规则只在 effect fact 等于目标时判冗余（此处 effect 为 Open(container)，不等于目标）。",
         "同时存在多跳合同路径 OPEN --ADD--> Open(container) --PRE_POS--> PLACE --ADD--> Inside(...)；二者不矛盾：接纳 ≠ 关系提供了合同之外的信息。", "",
         "truth / utility / opportunity 标签均为 UNKNOWN：保存的数据里没有独立裁定、q_R 或 R*，本轮不发明。",
         "T_A_dev_19 没有物理分支，不套用 14/18 的物理后果。"]
    (c.out / "evidence").mkdir(parents=True, exist_ok=True)
    (c.out / "evidence/relation_explanations.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"status": "OK", **{k: summary[k] for k in ("caches_verified", "total_admitted", "expected_14_18_19_two_each_matches")}}


# ----------------------------------------------------------------- assemble
def _csv_rows(p):
    with Path(p).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_exit_codes(c):
    p = c.out / "raw/worker_exit_codes.txt"
    if not p.is_file():
        raise AnalysisIncomplete("worker_exit_codes.txt missing")
    codes = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            codes[k.strip()] = int(v.strip())
    if sorted(codes) != ["A", "B"] or any(v != 0 for v in codes.values()):
        (c.out / "raw/analysis_stop.txt").write_text("worker failure: " + json.dumps(codes) + "\n", encoding="utf-8")
        raise AnalysisIncomplete("worker failed: " + json.dumps(codes))
    return codes


def cmd_authority_reconcile(c):
    """Pure-file step: verify the installed v3/v5 against the manifest and record the reconciliation. No analysis is re-run."""
    a = verify_authority(c.root)
    hist = c.out / "inventory/command_config.json"
    doc = {"status": "COMPLETE", "execution_time_authority_availability": "NOT_FOUND_ON_SERVER", "current_authority_status": a["authority_status"],
           "master_id": a["master_id"],
           "experimental_plan_version": "3.0", "research_content_version": "5.0",
           "experimental_plan_path": a["documents"]["experimental_plan"]["path"], "experimental_plan_sha256": a["documents"]["experimental_plan"]["sha256"],
           "research_content_path": a["documents"]["research_content"]["path"], "research_content_sha256": a["documents"]["research_content"]["sha256"],
           "authority_manifest_sha256": a["manifest_sha256"],
           "experimental_plan_sections_applied": ["5.2", "5.3", "5.4", "5.6"], "research_content_additional_e1_e6_gate_definition": False,
           "historical_command_config_sha256_unchanged_record": sha256_file(hist) if hist.is_file() else None,
           "note": "Historical execution-time NOT_FOUND records are retained unchanged; this file records the later recovery of the authoritative documents.",
           "new_samples": 0, "scientific_result_changed": False, "tp_training_authorized": False,
           "additional_physical_attempts_authorized": 0, "next_action": "S4_RESEARCH_DECISION"}
    (c.out / "evidence").mkdir(parents=True, exist_ok=True)
    jwrite(c.out / "evidence/authority_reconciliation.json", doc)
    return {"status": "OK", "current_authority_status": a["authority_status"]}


def cmd_assemble(c):
    codes = _read_exit_codes(c)
    need = [c.out / f for f in ("inventory/source_hashes.json", "audit/budget/budget_check.json", "audit/branches/branch_outcomes.json",
                                "audit/branches/paired_outcomes_reviewed.json", "contract/reachability_results.json", "evidence/relation_adjudications.json",
                                "evidence/utility_records.json")]
    c.need(*need)
    budget = jread(c.out / "audit/budget/budget_check.json")
    outcomes = jread(c.out / "audit/branches/branch_outcomes.json")
    paired = jread(c.out / "audit/branches/paired_outcomes_reviewed.json")
    reach = jread(c.out / "contract/reachability_results.json")
    rels = jread(c.out / "evidence/relation_adjudications.json")
    missing = []
    rep = c.s1 / "representation"
    align = []
    for name in ("e2_representation_entry_rev1.csv", "e3_candidate_discriminability_rev1.csv"):
        p = rep / name
        if not p.is_file():
            missing.append({"table": "alignment_records", "missing": c.rel(p)})
            continue
        for i, r in enumerate(_csv_rows(p)):
            align.append(_rec(r["case_id"], f"{name}:{i}", "S1_REV1_saved_representation_row", "PRIOR_VALUE_RECORDED_NOT_REVALIDATED", {"file": c.rel(p)}, {name: sha256_file(p)},
                              "saved by the S1-REV1 run; copied, not recomputed here", [], ["prior PASS flag is not counted as new validation; numeric values kept for reference"],
                              values={k: r[k] for k in ("candidate_id_alignment", "mask_alignment", "goal_alignment", "node_alignment", "relative_logit_change",
                                                        "common_shift_only", "capacity_seed", "status")}))
    grad = []
    gp = rep / "gradient_reach.jsonl"
    if gp.is_file():
        for i, r in enumerate(jsonl(gp)):
            grad.append(_rec(r["case_id"], f"gradient_reach:{i}", "S1_REV1_saved_gradient_reach", "PRIOR_VALUE_RECORDED_NOT_REVALIDATED", {"file": c.rel(gp)}, {"gradient_reach.jsonl": sha256_file(gp)},
                             "saved by the S1-REV1 run", [], ["boolean reach flags only; no gradient magnitudes saved"], values=r))
    else:
        missing.append({"table": "gradient_records", "missing": c.rel(gp)})
    wit = []
    for cs, v in paired["case_review"]["per_case"].items():
        wit.append(_rec(cs, f"{cs}:r3_witness", "r3_paired_open_pick_branches", "PROTOCOL_LEVEL_DIFFERENCE_RECORDED" if v["protocol_level_reliable_outcome_difference"] else "NOT_RELIABLE",
                        {"paired_outcomes": "audit/branches/paired_outcomes_reviewed.json"}, {"paired_outcomes_reviewed.json": sha256_file(c.out / "audit/branches/paired_outcomes_reviewed.json")},
                        "recomputed offline from r3 journals and sidecars", ["controller_internal_state", "rng_stream", "actual_post_action_facts"],
                        [v["supported_scope"], "PICK branch stop is planner NO_PLAN with Evaluator CONTINUE; physical infeasibility not established"], review=v))
    cons = []
    for p in paired["pairs"]:
        cons.append(_rec(p["case_id"], f"{p['case_id']}:repeat{p['repeat']}", "r3_pair_consequence_comparison", "RECORDED", {"paired_outcomes": "audit/branches/paired_outcomes_reviewed.json"}, {},
                         "recomputed offline", ["measured_cost", "measured_rework"], ["cost/rework NOT_MEASURED; action-sequence difference is descriptive only"],
                         differences=p.get("differences"), paired_restore_within_tolerance=p.get("paired_restore_within_tolerance")))
    for name, tab in (("alignment_records", align), ("gradient_records", grad), ("witness_records", wit), ("consequence_comparisons", cons)):
        jwrite(c.out / f"evidence/{name}.json", {"records": tab})
    for name in ("relation_adjudications", "utility_records"):
        if not jread(c.out / f"evidence/{name}.json").get("records"):
            missing.append({"table": name, "missing": "no records"})
    for r in rels["records"]:
        missing.append({"table": "relation_adjudications", "record_id": r["record_id"], "missing": r["missing_fields"]})
    jwrite(c.out / "evidence/missing_evidence.json", missing)
    diag = reach["diagnostic_fields"]
    explainable = all(v["CONTRACT_ALREADY_DISTINGUISHES_ACTIONS"] is True for v in diag.values()) if diag else False
    e5 = ["# E5 复核（合同不足）", "",
          f"名义合同下 OPEN/PICK 是否已可区分（各 case，穷举 BFS + 生产 nominal 函数）：{ {k: v['CONTRACT_ALREADY_DISTINGUISHES_ACTIONS'] for k, v in diag.items()} }。", ""]
    if explainable:
        e5.append("diagnostic_reason=CONTRACT_EXPLAINABLE_BRANCH_DIFFERENCE：r3 的 OPEN 成功 / PICK 后 NO_PLAN 差异可由合同（OnTable(second) 不会被重新加入、OPEN 需 GripperEmpty）在名义层解释。这解释的是顺序结构，不是 E4 的真实后果差异；E5 的“合同之外”条件因此未被建立。")
    else:
        e5.append("合同层没有对所有 case 给出可区分结论，或搜索受限；E5 未建立也未排除（NOT_ESTABLISHED）。")
    e5 += ["", "限制：真实 post-action facts 未保存，NO_PLAN 的运行时原因只是限定假设（NOMINAL_DERIVED），未被重演证明。"]
    (c.out / "evidence/e5_review.md").write_text("\n".join(e5) + "\n", encoding="utf-8")
    (c.out / "evidence/e6_review.md").write_text("# E6 复核（先验分类）\n\n没有独立的 truth/utility/opportunity 裁定，也没有 q_R / R* 输入；分类保持 UNKNOWN，不新增官方 E6 类别。诊断字段（如 NOMINAL_RECOVERY_GAP）仅为诊断，不是 E6 类别。\n", encoding="utf-8")
    auth = verify_authority(c.root)  # fail closed; historical inventory/command_config.json is neither read for status nor modified
    c.need(c.out / "evidence/authority_reconciliation.json")
    review_now = review_e4(paired["pairs"], outcomes["branches"], paired["unit_of_analysis"])
    (c.out / "evidence/e4_review.md").write_text(render_e4(review_now, paired["pairs"], outcomes["branches"], paired["unit_of_analysis"]), encoding="utf-8")
    n_rel = review_now["cases_with_reliable_protocol_level_difference"]
    key_gates = {"E1_relation_existence": "NOT_ESTABLISHED",
                 "E2_E3_representation": "PRIOR_VALUES_RETAINED_NOT_REVALIDATED",
                 "E4_real_consequence": review_now["frozen_E4_gate_status"],
                 "E5_contract_insufficiency": "NOT_SATISFIED" if explainable else "NOT_ESTABLISHED", "E6_prior_classification": "UNKNOWN"}
    eligibility = build_eligibility(auth["authority_status"], n_rel, explainable, key_gates)
    jwrite(c.out / "evidence/reviewed_eligibility.json", eligibility)
    tp = jread(c.r3 / "throughput_report.json") if (c.r3 / "throughput_report.json").is_file() else {}
    jwrite(c.out / "evidence/scheduling_report.json", {"workers": 2, "worker_exit_codes": codes, "gpu_used": False, "speedup": "NOT_MEASURED",
                                                       "reason": "no comparable single-worker baseline of this analysis", "r3_execution_throughput_reference": tp})
    jwrite(c.out / "evidence/zero_new_calls.json", {"provider_new": 0, "env_created_new": 0, "reset_new": 0, "skill_executed_new": 0, "physical_attempt_new": 0,
                                                     "rl_transitions_new": 0, "optimizer_steps_new": 0, "elastic_new": 0,
                                                     "guards_installed": install_guards(), "budget_reserved_total": budget["reserved_total"]})
    ident = jread(c.out / "inventory/execution_identity.json")
    rows = []
    for label, rev in (("executed_declared", ident["execution_commit_declared"]), ("executed_recovery_c48c253", "c48c253"),
                       ("erroneous_delivery", "82b9e8c"), ("fix_delivery", "cc1ad64")):
        full = git(c.root, "rev-parse", "--verify", str(rev) + "^{commit}", check=False) or "NOT_FOUND"
        blob = git_blob(c.root, full, "src/cp_disr/analysis/s1_e4_recovery.py") if full != "NOT_FOUND" else None
        if blob is None:
            comp = "NOT_FOUND"
        else:
            try:
                compile(blob.decode("utf-8"), "s1_e4_recovery.py", "exec")
                comp = "COMPILES(recomputed_now_from_git_blob)"
            except SyntaxError as e:
                comp = f"SYNTAX_ERROR(recomputed_now):{e.msg}@{e.lineno}"
        rows.append({"role": label, "commit": full, "s1_e4_recovery_compile": comp, "test_result": "NOT_RECORDED", "note": "no old test count is extrapolated"})
    write_csv(c.out / "release/commit_validation_matrix.csv", ["role", "commit", "s1_e4_recovery_compile", "test_result", "note"], rows)
    sm = ["# S1-REV1 r3 证据收口摘要（零新增样本）", "",
          f"- 预算：累计预留 {budget['reserved_total']}/{budget['cumulative_max_declared']}，剩余 {budget['remaining']}，状态 {budget['status']}；UNKNOWN 未退款。",
          f"- r3：{outcomes['unit_of_analysis']['branch_attempts']} 个分支、{outcomes['unit_of_analysis']['paired_comparisons']} 组配对、{outcomes['unit_of_analysis']['independent_case_configurations']} 个独立 case 配置。",
          "- OPEN 分支：Evaluator TASK_SUCCESS；PICK 分支：Evaluator CONTINUE 后 planner NO_PLAN（不是物理不可行证明）。",
          f"- 合同层：{ {k: v['CONTRACT_ALREADY_DISTINGUISHES_ACTIONS'] for k, v in diag.items()} }；诊断原因 {eligibility['diagnostic_reason']}。",
          f"- 关系：admitted 共 {rels['summary']['total_admitted']} 条（14/18/19 各 2 条：{rels['summary']['expected_14_18_19_two_each_matches']}）；truth/utility/opportunity=UNKNOWN。",
          f"- 关键门状态：{json.dumps(key_gates, ensure_ascii=False)}", "- 资格：tp_training_authorized=false；额外物理尝试=0；方法升级=false；下一步=EXPLICIT_RESEARCH_DECISION_REQUIRED。",
          "- 权威文档：Final Experimental Plan v3 与 Research Content v5 已从原始权威附件恢复并完成对照；执行当时服务器未找到文档的历史记录保留不变。",
          "- E1：NOT_ESTABLISHED；自然合法关系存在，但缺独立可裁定truth证据。",
          "- E2/E3：沿用历史保存值，本轮未重新验证。",
          "- E4：LIMITED_SUPPORT_FEASIBILITY_ONLY；两个独立case有配对协议结果差，但cost/rework未测，PICK为Evaluator CONTINUE后planner NO_PLAN。",
          "- E5：NOT_SATISFIED；当前OPEN/PICK差异可由注册合同长期可达性解释，不满足合同外soft risk/cost/relevance条件。",
          "- E6：UNKNOWN；缺独立truth/utility/opportunity与R*证据。",
          "- S1处置不变：tp_training_authorized=false；additional_physical_attempts_authorized=0；next_action=S4_RESEARCH_DECISION。",
          "- post-action 真实事实未保存；controller 内部状态与 RNG 未比较（NOT_MEASURED）。"]
    (c.out / "evidence/reviewed_summary.md").write_text("\n".join(sm) + "\n", encoding="utf-8")
    man = {}
    for p in sorted((c.out / "evidence").glob("*")):
        if p.is_file() and p.name != "evidence_manifest.json":
            man[c.rel(p)] = sha256_file(p)
    jwrite(c.out / "evidence/evidence_manifest.json", {"files": man})
    return {"status": "OK", "key_gates": key_gates}


# ----------------------------------------------------------------- verify
def compare_protected(c, before):
    """Re-hash every protected entry recorded before the analysis; any change or disappearance is reported."""
    after, changed, missing_files = {}, [], []
    for rel, e in before["entries"].items():
        p = c.root / rel
        if not p.is_file():
            missing_files.append(rel)
            continue
        h = sha256_file(p)
        after[rel] = {"sha256": h}
        if h != e["sha256"]:
            changed.append(rel)
    return after, changed, missing_files


def build_eligibility(authority_status, n_reliable_e4, contract_explainable, key_gates):
    """No combination of evidence here can authorize training, S2 or new physical attempts. Fails closed without reconciled authority."""
    if authority_status != AUTH_OK:
        raise AuthorityMismatch("authority not reconciled: " + str(authority_status))
    return {"eligibility_decision": "DERIVED_FROM_AVAILABLE_EVIDENCE", "scientific_result": "NOT_ESTABLISHED",
            "key_gate_status": key_gates, "key_gate_notes": dict(KEY_GATE_NOTES),
            "diagnostic_reason": ("CONTRACT_EXPLAINABLE_BRANCH_DIFFERENCE" if contract_explainable else "NOT_DETERMINED"),
            "authoritative_plan_v3_status": authority_status, "authoritative_research_v5_status": authority_status,
            "plan_condition_citation": "Final Experimental Plan v3 §§5.2–5.6. Research Content v5 provides no additional E1–E6 gate definition.",
            "tp_training_authorized": False, "additional_physical_attempts_authorized": 0, "method_upgrade_authorized": False,
            "next_action": "EXPLICIT_RESEARCH_DECISION_REQUIRED", "route_if_key_gate_unmet": "S4_RESEARCH_DECISION"}


class VerifyFailed(RuntimeError):
    pass


def cmd_verify(c):
    problems = []
    before = jread(c.out / "inventory/source_hashes.json")
    after, changed, missing_files = compare_protected(c, before)
    jwrite(c.out / "inventory/source_hashes_after.json", {"generated_at_utc": now(), "entry_count": len(after), "changed": changed, "missing": missing_files, "entries": after})
    if changed or missing_files:
        problems.append(f"protected files changed={len(changed)} missing={len(missing_files)}")
    b = jread(c.out / "audit/budget/budget_check.json")
    if not (b["reserved_total"] == 20 and b["cumulative_max_declared"] == 20 and b["remaining"] == 0 and b["status"] == "CONSISTENT"):
        problems.append("budget not 20/20 consistent")
    bo = jread(c.out / "audit/branches/branch_outcomes.json")["unit_of_analysis"]
    if (bo["branch_attempts"], bo["paired_comparisons"], bo["independent_case_configurations"]) != (8, 4, 2):
        problems.append(f"unexpected denominators {bo}")
    el = jread(c.out / "evidence/reviewed_eligibility.json")
    if el["tp_training_authorized"] is not False or el["additional_physical_attempts_authorized"] != 0 or el["method_upgrade_authorized"] is not False:
        problems.append("eligibility authorizes something")
    zc = jread(c.out / "evidence/zero_new_calls.json")
    if any(zc[k] != 0 for k in ("provider_new", "env_created_new", "reset_new", "skill_executed_new", "physical_attempt_new")):
        problems.append("nonzero new calls")
    for f in ("evidence/relation_adjudications.json", "evidence/utility_records.json", "evidence/alignment_records.json", "evidence/gradient_records.json",
              "evidence/witness_records.json", "evidence/consequence_comparisons.json", "evidence/missing_evidence.json", "evidence/evidence_manifest.json",
              "contract/input_bindings.json", "contract/nominal_successors.json", "contract/reachability_results.json", "contract/contract_order_explanation.md",
              "contract/actual_vs_nominal_evidence_limits.md", "audit/budget/cross_round_attempt_ledger.csv", "audit/budget/authorization_trace.json",
              "audit/branches/paired_outcomes_reviewed.json", "evidence/e4_review.md", "evidence/e5_review.md", "evidence/e6_review.md", "release/commit_validation_matrix.csv"):
        if not (c.out / f).is_file():
            problems.append("missing output " + f)
    for f, h in jread(c.out / "evidence/evidence_manifest.json")["files"].items():
        if not (c.root / f).is_file() or sha256_file(c.root / f) != h:
            problems.append("evidence hash mismatch " + f)
    head = git(c.root, "rev-parse", "HEAD")
    anc = subprocess.run(["git", "-C", str(c.root), "merge-base", "--is-ancestor", BASE_COMMIT, head], capture_output=True)
    if anc.returncode != 0:
        problems.append("HEAD is not the expected delivery commit or a descendant of it")
    try:
        verify_authority(c.root)
    except AuthorityMismatch as e:
        problems.append("authority: " + str(e))
    ar = c.out / "evidence/authority_reconciliation.json"
    if not ar.is_file():
        problems.append("missing output evidence/authority_reconciliation.json")
    else:
        ard = jread(ar)
        if ard.get("current_authority_status") != AUTH_OK or ard.get("execution_time_authority_availability") != "NOT_FOUND_ON_SERVER" or ard.get("new_samples") != 0:
            problems.append("authority_reconciliation content unexpected")
    if el.get("authoritative_plan_v3_status") != AUTH_OK or "not located" in json.dumps(el, ensure_ascii=False):
        problems.append("eligibility still carries pre-recovery authority wording")
    res = {"status": "PASS" if not problems else "FAIL", "problems": problems, "protected_entries": len(after), "head": head}
    jwrite(c.out / "verify.json", res)
    if problems:
        raise VerifyFailed("; ".join(problems))
    return res


# ----------------------------------------------------------------- CLI
COMMANDS = {"inventory": cmd_inventory, "budget": cmd_budget, "branches": cmd_branches, "contracts": cmd_contracts,
            "relations": cmd_relations, "authority-reconcile": cmd_authority_reconcile, "assemble": cmd_assemble, "verify": cmd_verify}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    install_guards()
    c = Ctx(a.root, a.source, a.output)
    try:
        res = COMMANDS[a.command](c)
    except ZeroCallViolation as e:
        print(json.dumps({"status": "ZERO_CALL_VIOLATION", "error": str(e)})); return EXIT_CODES["ZERO_CALL_VIOLATION"]
    except InputMissing as e:
        print(json.dumps({"status": "INPUT_MISSING", "error": str(e)})); return EXIT_CODES["INPUT_MISSING"]
    except (VerifyFailed, AuthorityMismatch) as e:
        print(json.dumps({"status": "VERIFY_FAILED", "error": str(e)})); return EXIT_CODES["VERIFY_FAILED"]
    except AnalysisIncomplete as e:
        print(json.dumps({"status": "ANALYSIS_INCOMPLETE", "error": str(e)})); return EXIT_CODES["ANALYSIS_INCOMPLETE"]
    print(json.dumps({"command": a.command, **(res if isinstance(res, dict) else {"result": res})}, default=str, sort_keys=True))
    return EXIT_CODES["OK"]


if __name__ == "__main__":
    sys.exit(main())

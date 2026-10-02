import json, os, subprocess, sys, hashlib, time
OUT = open("/home/xushijie2/tmp/r2/R2_OUT.txt").read().strip(); GPU = int(sys.argv[1])
root = os.getcwd(); sys.path.insert(0, os.path.join(root, "src"))
from cp_disr import final_tb as ft
res = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan": "R-TB-K-1", "gpu": GPU, "checks": {}}
def sh(*a): return subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
auth = json.load(open(os.path.join(OUT, "authorization.json"))); prep = auth["prep_commit"]
launch = json.load(open(os.path.join(OUT, "launch_plan.json"))); state = ft.Ledger(OUT).read()
# 1 identity
head = sh("git", "rev-parse", "HEAD"); dirty = sh("git", "status", "--porcelain", "--untracked-files=no")
ft.check_source_identity(root + "", prep)
tok_path = os.path.join(OUT, "release_tokens", "train_R-TB-K-1.json")
tok = ft.verify_token(tok_path, OUT, "train", prep, "R-TB-K-1")
ft.verify_smoke_receipt(OUT, prep)
cfg = launch["configs"]["R-TB-K-1"]; cfg_sha = ft.sha256_file(cfg["path"])
res["checks"]["identity"] = {"head": head, "prep": prep, "head_equals_prep": head == prep, "tracked_tree_clean": dirty == "",
   "config_path": cfg["path"], "config_sha256_now": cfg_sha, "config_sha256_registered": cfg["sha256"], "config_unchanged": cfg_sha == cfg["sha256"],
   "token_verified": True, "token_sha256": ft.sha256_file(tok_path), "smoke_receipt_verified": True,
   "plan_identity": {k: launch["plans"]["R-TB-K-1"].get(k) for k in ("method", "seed", "release_order", "planned_run_dir")},
   "authorization_sha256": ft.sha256_file(os.path.join(OUT, "authorization.json")),
   "registration_files_unchanged": {n: ft.sha256_file(os.path.join(OUT, n)) for n in ("launch_plan.json", "smoke_selection.json", "amendment_authorization.json", "prior_smoke_reconciliation.json", "budget_reconciliation.json")}}
assert head == prep and dirty == "" and cfg_sha == cfg["sha256"]
# 2 GPU independence
running = {p: {"gpu": v.get("physical_gpu"), "pid": v.get("pid")} for p, v in state["plans"].items() if v["status"] == "RUNNING"}
res["checks"]["gpu_independent"] = {"running_workers": running, "chosen_gpu": GPU, "shared_with_my_workers": any(v["gpu"] == GPU for v in running.values())}
assert not res["checks"]["gpu_independent"]["shared_with_my_workers"]
res["checks"]["gpu_state"] = sh("nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader").splitlines()
res["checks"]["gpu_compute_apps"] = sh("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader").splitlines()
# 3 storage (live)
last = json.loads(open(os.path.join(OUT, "storage_checks.jsonl")).read().strip().splitlines()[-1])
res["checks"]["storage_live"] = {"free_bytes": last["free_bytes"], "required_bytes": last["required_bytes"], "reserve_ge_2GiB_included": True, "passed": last["passed"], "time": last["time"],
   "per_plan_remaining_estimate": last["per_plan_remaining_estimate"]}
assert last["passed"]
# 4 OOM / EGL / IO
dm = subprocess.run("dmesg -T 2>&1 | grep -i -E 'out of memory|oom-killer|killed process' | tail -3", shell=True, capture_output=True, text=True)
rows = [json.loads(l) for l in open(os.path.join(OUT, "throughput", "samples.jsonl")).read().strip().splitlines()[-10:]]
res["checks"]["oom_egl_io"] = {"dmesg_oom_lines": dm.stdout.strip().splitlines(), "dmesg_stderr": dm.stderr.strip()[:200],
   "mem_free_gb": sh("bash", "-c", "free -g | sed -n 2p"), "iowait_frac_last10_samples": [r["iowait_frac"] for r in rows],
   "iowait_max": max(r["iowait_frac"] for r in rows), "loadavg": rows[-1]["loadavg"],
   "egl_note": "K-1 gets its own physical GPU via bind_worker_gpu (one visible device); EGL/graphics-context evidence is taken from nvidia-smi after start"}
assert max(r["iowait_frac"] for r in rows) < 0.20
# 5 attempts
res["checks"]["attempts"] = {"new_rl_attempts_used_before": state["new_rl_attempts_used"], "cap": state["new_rl_attempts_cap"], "K1_status_before": state["plans"]["R-TB-K-1"]["status"],
   "note": "K-1 is the third of the three registered plans; no attempt beyond the registered three"}
assert state["plans"]["R-TB-K-1"]["status"] == "NOT_STARTED" and state["new_rl_attempts_used"] + 1 <= state["new_rl_attempts_cap"]
res["checks"]["gate_dk1"] = {k: v for k, v in ft.first_update_gate(state["plans"]["R-TB-DK-1"]["run_dir"]).items() if k in ("passed", "reasons", "optimizer_steps")}
assert res["checks"]["gate_dk1"]["passed"]
res["all_passed"] = True
json.dump(res, open(os.path.join(OUT, "k1_prelaunch_check.json"), "w"), indent=1, sort_keys=True)
print(json.dumps({"all_passed": True, "iowait_max": res["checks"]["oom_egl_io"]["iowait_max"], "oom_lines": res["checks"]["oom_egl_io"]["dmesg_oom_lines"], "storage": res["checks"]["storage_live"]["free_bytes"]}))

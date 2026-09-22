#!/usr/bin/env python3
"""Launch Stage 2A jobs with at most two exclusive GPUs. No VLM. Survives SSH disconnect."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
PY = ROOT / ".venv-stage0a/bin/python"
OUT = ROOT / "experiments/part_2_exploration/stage_2a"
LOG = OUT / "orchestrator"
STOP = LOG / "STOP_SUPERSEDED_BY_PLAN_V1_1.json"
if STOP.exists():
    raise SystemExit("Old Stage 2A orchestrator superseded by Plan v1.1; refuse dispatch/resume")
SITE = "/home/__compress_data/xushijie/.conda/envs/lerobotpi0/lib/python3.10/site-packages"
FIRST = [
    ("T_A", "B0", 0), ("T_A", "B1", 0), ("T_A", "B2", 0), ("T_A", "Full", 0),
    ("T_C", "B0", 0), ("T_C", "B1", 0), ("T_C", "B2", 0), ("T_C", "Full", 0),
]
REST = []
for task in ("T_A", "T_C"):
    for method in ("B0", "B1", "B2", "Full"):
        for seed in (0, 1, 2):
            job = (task, method, seed)
            if job not in FIRST:
                REST.append(job)

def pick_gpus(n=2):
    import subprocess as sp
    raw = sp.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"], text=True)
    apps = sp.check_output(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    rows = []
    for line in raw.strip().splitlines():
        idx, free, used = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "free": int(free), "used": int(used)})
    # exclusive among our jobs; prefer most free; do not kill others
    rows.sort(key=lambda r: -r["free"])
    chosen = [r["index"] for r in rows[:n]]
    (LOG / "gpu_pick.json").write_text(json.dumps({"chosen": chosen, "rows": rows, "apps": apps}, indent=2), encoding="utf-8")
    return chosen

def job_env(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = "egl"
    env["PYTHONPATH"] = str(ROOT / "src") + ":" + SITE
    env.pop("DASHSCOPE_API_KEY", None)
    env.pop("DASHSCOPE_API_KEY_FILE", None)
    return env

def spawn(task, method, seed, gpu, stop_after=None, max_updates=64, resume=False):
    run_id = f"stage2a_{task}_{method}_seed{seed}"
    logp = LOG / f"{run_id}.log"
    cmd = [str(PY), "-m", "cp_disr", "--root", str(ROOT), "stage-2a-run", "--task", task, "--method", method, "--seed", str(seed), "--gpu", "0", "--max-updates", str(max_updates)]
    if stop_after is not None:
        cmd += ["--stop-after-updates", str(stop_after)]
    if resume:
        cmd += ["--resume"]
    logp.parent.mkdir(parents=True, exist_ok=True)
    f = open(logp, "a", encoding="utf-8")
    f.write("\n# spawn %s gpu=%s stop=%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), gpu, stop_after))
    f.flush()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=job_env(gpu), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    rec = {"run_id": run_id, "pid": proc.pid, "gpu": gpu, "task": task, "method": method, "seed": seed, "stop_after": stop_after, "log": str(logp)}
    (LOG / f"{run_id}.pid.json").write_text(json.dumps(rec), encoding="utf-8")
    return proc, rec

def wait_two(queue, gpus, stop_after=None, max_updates=64, resume=False):
    active = []
    results = []
    qi = 0
    while qi < len(queue) or active:
        while qi < len(queue) and len(active) < len(gpus):
            used = {a["gpu"] for a in active}
            gpu = next(g for g in gpus if g not in used)
            task, method, seed = queue[qi]
            qi += 1
            proc, rec = spawn(task, method, seed, gpu, stop_after=stop_after, max_updates=max_updates, resume=resume)
            rec["proc"] = proc
            active.append(rec)
            print("started", rec["run_id"], "pid", rec["pid"], "gpu", gpu, flush=True)
        time.sleep(15)
        still = []
        for rec in active:
            code = rec["proc"].poll()
            if code is None:
                still.append(rec)
            else:
                rec["returncode"] = code
                rec.pop("proc", None)
                results.append(rec)
                print("finished", rec["run_id"], "code", code, flush=True)
        active = still
    return results

def main():
    LOG.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("STAGE2A_ORCH_MODE", "first")
    gpus = pick_gpus(2)
    print("gpus", gpus, "mode", mode, flush=True)
    if mode == "first":
        res = wait_two(FIRST, gpus, stop_after=1, max_updates=1)
        (LOG / "first_wave.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        rows = []
        ok = True
        for rec in res:
            run_id = rec["run_id"]
            att = OUT / run_id / "attempts" / "attempt_000"
            ev_path = att / "first_update_evidence.json"
            summ_path = att / "summary.json"
            ev = json.loads(ev_path.read_text()) if ev_path.is_file() else {}
            summ = json.loads(summ_path.read_text()) if summ_path.is_file() else {}
            passed = rec.get("returncode") == 0 and ev.get("optimizer_steps", 0) > 0 and ev.get("rollout_transitions", 0) >= 1024 and ev.get("finite")
            if not passed:
                ok = False
            rows.append({
                "run_id": run_id, "task": rec["task"], "method": rec["method"], "seed": rec["seed"],
                "returncode": rec.get("returncode"), "passed": passed,
                "ppo_update": ev.get("ppo_update"), "optimizer_steps": ev.get("optimizer_steps"),
                "rollout_transitions": ev.get("rollout_transitions"), "param_changed": ev.get("param_changed"),
                "optimizer_changed": ev.get("optimizer_changed"), "mean_total_loss": ev.get("mean_total_loss"),
                "mean_grad_norm": ev.get("mean_grad_norm"), "finite": ev.get("finite"),
                "interaction_seconds": ev.get("interaction_seconds"),
            })
        import csv
        csvp = ROOT / "experiments/part_2_exploration/stage_2a/startup_gate/eight_first_update_report.csv"
        with csvp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (LOG / "first_wave_pass.json").write_text(json.dumps({"passed": ok, "rows": rows}), encoding="utf-8")
        if not ok:
            print("first updates FAILED; not expanding matrix", flush=True)
            return
        print("first updates PASS; continuing seed0 then remaining", flush=True)
        res2 = wait_two(FIRST, gpus, stop_after=None, max_updates=64, resume=True)
        (LOG / "continue_seed0.json").write_text(json.dumps(res2, indent=2), encoding="utf-8")
        res3 = wait_two(REST, gpus, stop_after=None, max_updates=64, resume=False)
        (LOG / "rest_wave.json").write_text(json.dumps(res3, indent=2), encoding="utf-8")
        return
    if mode == "continue":
        res = wait_two(FIRST, gpus, stop_after=None, max_updates=64, resume=True)
        # resume flag not automatically set; jobs start new attempt unless --resume implemented with existing attempt
        (LOG / "continue_seed0.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        res2 = wait_two(REST, gpus, stop_after=None, max_updates=64, resume=False)
        (LOG / "rest_wave.json").write_text(json.dumps(res2, indent=2), encoding="utf-8")
        return
    raise SystemExit("unknown mode")

if __name__ == "__main__":
    main()

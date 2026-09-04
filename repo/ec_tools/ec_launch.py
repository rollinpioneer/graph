#!/usr/bin/env python3
"""
ec_launch.py (v2) -- single frozen entry point for the E-C rerun (PFF / SCIA)

CHANGES FROM v1  (v1 had a defect that would have wasted the whole campaign)
    v1 emitted no `task=` override. The base workspace configs default to
    pusht_lowdim, so v1 would have trained 18 seeds of the WRONG TASK -- and
    the resulting logs would have looked entirely healthy: non-zero success,
    plausible variance, plausible cost. Round 1 at least failed loudly. This
    version reads `task` from ec_spec.yaml, emits it explicitly, and refuses
    to start if it is missing or resolves to anything containing "pusht".

    v1 also carried a hardcoded FAMILIES table, i.e. a second source of truth
    beside the config files. That is the same shape as round 1's F-tf drift.
    v2 has no table: ec_spec.yaml is the only place these numbers live.

USAGE
    python ec_launch.py --archive
    python ec_launch.py --probe --family F-tf --seed 0 --run     # 20-epoch timing
    python ec_launch.py --family F-cnn --seed 0 --run            # pilot
    python ec_launch.py --family F-cnn --seeds 1-5 --run
    python ec_launch.py --manifest-only

    Omit --run for a dry run: the exact command is printed, nothing executes.
"""

__VERSION__ = "ec_launch 2026-08-25 r6 (substrate-test, heartbeat, detach)"

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

STALE_PATTERNS = ("*stale*", "*race*", "*pathfix*", "*_old", "*.bak")
REQUIRED = ("config_name", "task", "num_epochs", "checkpoint_every",
            "rollout_every", "topk_k", "monitor_n_test", "mean_rollout_sec")


def load_spec(path):
    import yaml
    with open(path) as f:
        spec = yaml.safe_load(f)
    fams = spec.get("families") or {}
    if not fams:
        sys.exit(f"  FATAL: {path} has no families block")
    for fam, f in fams.items():
        missing = [k for k in REQUIRED if f.get(k) in (None, "")]
        if missing:
            sys.exit(f"  FATAL: {fam} missing required field(s): {missing}")
        # the v1 defect, made unrepresentable
        if "pusht" in str(f["task"]).lower():
            sys.exit(f"  FATAL: {fam} task={f['task']!r} -- that is the base "
                     f"config default, not the E-C task")
        ne, ce = int(f["num_epochs"]), int(f["checkpoint_every"])
        if (ne - 1) % ce != 0:
            sys.exit(f"  FATAL: {fam} final epoch {ne-1} is not a save point "
                     f"(checkpoint_every={ce}); the trained weights would not "
                     f"persist -- this is the round-1 defect")
        if int(f["checkpoint_every"]) % int(f["rollout_every"]) != 0:
            sys.exit(f"  FATAL: {fam} checkpoint_every={f['checkpoint_every']} "
                     f"is not a multiple of rollout_every={f['rollout_every']}; "
                     f"a save epoch that is not an eval epoch has no monitor "
                     f"key in its metric dict and the save raises. This is the "
                     f"F-tf probe crash.")
        n_eval = len(range(0, ne, int(f["rollout_every"])))
        if n_eval < 2:
            sys.exit(f"  FATAL: {fam} rollout_every={f['rollout_every']} gives "
                     f"{n_eval} eval point(s); with <2 there is no progress "
                     f"signal and any later diagnostic reads the untrained model")
        # abs_action consistency: `_abs` task configs set abs_action=true
        exp = f.get("expect_abs_action")
        is_abs = str(f["task"]).endswith("_abs")
        if exp is not None and bool(exp) != is_abs:
            sys.exit(f"  FATAL: {fam} expect_abs_action={exp} but task="
                     f"{f['task']!r}; round-1 checkpoints recorded "
                     f"abs_action={exp} for this family")
    return spec



# ---------------------------------------------------------------------------
# RESTORED. A line-offset splice in the previous revision deleted archive(),
# parse_seeds() and preflight() together. The smoke test used
# `--skip-preflight --probe --seed 0`, which is precisely the flag combination
# that reaches none of the three, so all three shipped broken. Any code path
# added here must be exercised by the self-test at the bottom of this file.
# ---------------------------------------------------------------------------

def archive(runs_root, archive_root):
    os.makedirs(archive_root, exist_ok=True)
    moved = []
    for pat in STALE_PATTERNS:
        for d in glob.glob(os.path.join(runs_root, "*", pat)):
            rel = os.path.relpath(d, runs_root)
            dst = os.path.join(archive_root, rel.replace(os.sep, "__"))
            if os.path.exists(dst):
                dst += f".{int(time.time())}"
            shutil.move(d, dst)
            moved.append(rel)
            print(f"  moved {rel} -> {dst}")
    print(f"  archived {len(moved)} dir(s)" if moved else "  nothing to archive")
    return moved


def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def preflight(a, families):
    if not os.path.exists(a.preflight):
        print(f"  [WARN] preflight not found at {a.preflight}; skipping")
        return True
    cmd = [sys.executable, a.preflight, "--stage", "config",
           "--runs-root", a.runs_root, "--driver", a.driver,
           "--spec", a.spec, "--families", ",".join(families)]
    print("  " + " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


# ---------------------------------------------------------------------------
# DETACHED EXECUTION
# The probe60 run (~30 min) did not survive to completion, and an earlier
# service did not stay up either. A single F-tf seed is projected at tens of
# hours, so a synchronous subprocess tied to the session lifetime cannot work
# regardless of GPU budget. --detach writes a shell plan, launches it in its
# own session, and returns immediately; --status polls progress from the
# markers the plan writes.
# ---------------------------------------------------------------------------

PLAN = []


def write_plan(a, plan):
    """Every run gets: an unbuffered child, a banner written to train.log
    BEFORE the command starts, and a 30 s heartbeat while it lives.

    The F-tf probe left .start present, train.log at 0 bytes, and no .rc --
    which is ambiguous between (a) the child blocked in setup before flushing
    anything and (b) the whole process group was killed. The banner removes
    that ambiguity (an empty log now means the shell never reached the
    command) and the heartbeat distinguishes a live-but-silent child from a
    dead one without needing the plan pid."""
    import shlex
    ts = int(time.time())
    path = os.path.abspath(f"ec_plan_{ts}.sh")
    q = shlex.quote
    lines = ["#!/usr/bin/env bash", "set -u",
             "export PYTHONUNBUFFERED=1",
             f"cd {q(os.path.abspath(a.repo_root))}", ""]
    for tag, rd, cmd in plan:
        d = os.path.abspath(rd)
        lines += [
            f"# ---- {tag} ----",
            f"mkdir -p {q(d)}",
            f"date +%s > {q(d + '/.start')}",
            f'echo "[plan] launching {tag} at $(date -Is)" >> {q(d + "/train.log")}',
            " ".join(q(c) for c in cmd) + f" >> {q(d + '/train.log')} 2>&1 &",
            "CHILD=$!",
            f"echo $CHILD > {q(d + '/.child.pid')}",
            f"while kill -0 $CHILD 2>/dev/null; do date +%s > {q(d + '/.heartbeat')}; sleep 30; done",
            "wait $CHILD; RC=$?",
            f"echo $RC > {q(d + '/.rc')}",
            f"date +%s > {q(d + '/.end')}",
            f'echo "[plan] {tag} exited rc=$RC at $(date -Is)" >> {q(d + "/train.log")}',
            f'[ "$RC" = "0" ] || {{ echo "ABORT after {tag}"; exit 1; }}',
            "",
        ]
    lines += ['echo "ALL_DONE"', ""]
    open(path, "w").write("\n".join(lines))
    os.chmod(path, 0o755)
    return path


def substrate_test(a):
    """Answers one question and nothing else: does a detached process outlive
    the session? No GPU, no repo, no training. Launch it, end the turn, poll
    it next turn."""
    d = os.path.abspath(os.path.join(a.runs_root, "_substrate_test"))
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    mins = int(a.substrate_minutes)
    py = (f"import time,sys\n"
          f"for i in range({mins * 6}):\n"
          f"    print('tick', i, time.time(), flush=True)\n"
          f"    time.sleep(10)\n")
    script = os.path.join(d, "tick.py")
    open(script, "w").write(py)
    plan = [("substrate/tick", d, [sys.executable, "-u", script])]
    saved_repo = a.repo_root
    a.repo_root = "."
    pid = detach(a, plan)
    a.repo_root = saved_repo
    print(f"\n  substrate test running for ~{mins} min in {d}")
    print(f"  end this turn now, then next turn run:")
    print(f"    python ec_launch.py --status --runs-root {a.runs_root} "
          f"--spec {a.spec}")
    print(f"    tail -3 {os.path.join(d, 'train.log')}")
    print("  ticks spanning the turn boundary  -> the container persists; the "
          "problem is inside train.py startup")
    print("  ticks stopping at the turn boundary -> the container does not "
          "persist; no multi-hour training is possible here at all")
    return pid


def detach(a, plan):
    path = write_plan(a, plan)
    logp = path[:-3] + ".log"
    lf = open(logp, "w")
    proc = subprocess.Popen(["bash", path], stdout=lf, stderr=lf,
                            stdin=subprocess.DEVNULL, start_new_session=True,
                            cwd=os.path.abspath(a.repo_root))
    open(path + ".pid", "w").write(str(proc.pid))
    print(f"\n  detached pid={proc.pid}")
    print(f"  plan : {path}")
    print(f"  log  : {logp}")
    print(f"  poll : python {os.path.basename(__file__)} --status "
          f"--runs-root {a.runs_root} --spec {a.spec}")
    return proc.pid


def _marker(rd, name):
    p = os.path.join(rd, name)
    if not os.path.exists(p):
        return None
    try:
        return int(open(p).read().strip())
    except Exception:
        return None


def _last_epoch(rd):
    for p in (os.path.join(rd, "logs.json.txt"),
              *glob.glob(os.path.join(rd, "**", "logs.json.txt"), recursive=True)):
        if os.path.exists(p):
            last = None
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "epoch" in r:
                        last = r["epoch"]
            return last
    return None


def _plan_alive():
    """A run dir with .start and no .end looks RUNNING forever, including after
    the process was killed. Report the plan process's actual liveness so a
    session-boundary kill is visible rather than silently pending."""
    pids = sorted(glob.glob("ec_plan_*.pid"), key=os.path.getmtime)
    if not pids:
        return None, None
    pid = int(open(pids[-1]).read().strip())
    try:
        os.kill(pid, 0)
        return pid, True
    except ProcessLookupError:
        return pid, False
    except PermissionError:
        return pid, True


def status(a, spec):
    fams = spec.get("families", {})
    now = int(time.time())
    print("=" * 76)
    print("STATUS")
    print("=" * 76)
    pid, alive = _plan_alive()
    if pid is not None:
        print(f"  plan process pid={pid}: "
              f"{'alive' if alive else 'GONE'}")
        if not alive:
            print("  -> if no run below shows .end, the plan was killed rather "
                  "than finishing. That means the container does not outlive "
                  "the session and no multi-hour training is possible here; "
                  "move to a persistent host or a job scheduler before "
                  "anything else.")
    seen = {}
    for rd in sorted(glob.glob(os.path.join(a.runs_root, "*", "seed*"))):
        rel = os.path.relpath(rd, a.runs_root)
        fam = rel.split(os.sep)[0]
        base = os.path.basename(rd)
        st, en, rc = _marker(rd, ".start"), _marker(rd, ".end"), _marker(rd, ".rc")
        if st is None:
            continue
        m = re.search(r"_probe(\d+)$", base)
        target = int(m.group(1)) if m else int(fams.get(fam, {}).get("num_epochs", 0)) - 1
        ep = _last_epoch(rd)
        elapsed = (en or now) - st
        state = "done" if en is not None else "RUNNING"
        if rc not in (None, 0):
            state = f"FAILED rc={rc}"
        child = None
        cp = os.path.join(rd, ".child.pid")
        if os.path.exists(cp):
            try:
                child = int(open(cp).read().strip())
                os.kill(child, 0)
                child_alive = True
            except (ProcessLookupError, ValueError):
                child_alive = False
            except PermissionError:
                child_alive = True
        else:
            child_alive = None
        hb = _marker(rd, ".heartbeat")
        if en is None and child_alive is False:
            state = f"DEAD (child {child} gone, no .rc)"
        elif en is None and hb is not None and now - hb > 120:
            state = f"NO-HEARTBEAT {int((now-hb)//60)}m"
        elif en is None and not (alive is None or alive):
            state = "KILLED?"
        else:
            logp = os.path.join(rd, "logs.json.txt")
            if not os.path.exists(logp):
                logp = next(iter(glob.glob(os.path.join(
                    rd, "**", "logs.json.txt"), recursive=True)), None)
            if en is None and logp and \
                    now - os.path.getmtime(logp) > a.stall_sec:
                state = f"STALLED {int((now-os.path.getmtime(logp))//60)}m"
            elif en is None and not logp and elapsed > a.stall_sec:
                # JsonLogger creates logs.json.txt on entry to the epoch loop,
                # after dataset/normalizer/env setup. Its absence this long
                # means the run has not reached the training loop at all.
                state = "NO-LOG (pre-loop or dead)"
        eta = ""
        if en is None and isinstance(ep, int) and ep >= 0 and target > ep:
            per = elapsed / (ep + 1)
            eta = f"  eta~{(target - ep) * per / 3600:.1f}h"
        print(f"  {rel:28s} {state:12s} epoch {ep}/{target}  "
              f"elapsed {elapsed/3600:.2f}h{eta}")
        if m and en is not None and rc == 0:
            seen.setdefault((fam, base.split("_probe")[0]), []).append(
                (int(m.group(1)), elapsed))

    for (fam, seed), pts in seen.items():
        if len(pts) < 2:
            continue
        pts.sort()
        f = fams.get(fam, {})
        mn = int(f.get("monitor_n_test", 10))
        mrs = float(f.get("mean_rollout_sec", 1.0))
        (ns, ts_), (nl, tl) = pts[0], pts[-1]
        # each probe has exactly 2 eval points by construction
        ts_c, tl_c = ts_ - 2 * mn * mrs, tl - 2 * mn * mrs
        marginal = (tl_c - ts_c) / (nl - ns)
        fixed = ts_c - marginal * (ns + 1)
        full = int(f.get("num_epochs", 0))
        total = fixed + marginal * full
        print(f"\n  TWO-POINT FIT  {fam}/{seed}")
        print(f"    {ns+1} epochs {ts_:.0f}s   {nl+1} epochs {tl:.0f}s")
        print(f"    marginal {marginal:.2f} s/epoch   fixed startup {fixed:.0f}s")
        print(f"    @{full} epochs: {total/3600:.1f} GPU-h/seed, "
              f"{6*total/3600:.0f} GPU-h for 6 seeds")
        print(f"    implied c_s/c_e = {round(total/mrs):,}")


def overrides(f, seed, run_dir, epochs=None, ce=None, re_=None, resume=False):
    ne = int(epochs if epochs is not None else f["num_epochs"])
    ce = int(ce if ce is not None else f["checkpoint_every"])
    re_ = int(re_ if re_ is not None else f["rollout_every"])
    extra = []
    if f.get("wandb_mode"):
        extra.append(f"logging.mode={f['wandb_mode']}")
    if f.get("topk_monitor_key"):
        extra.append(f"checkpoint.topk.monitor_key={f['topk_monitor_key']}")
    return [
        f"task={f['task']}",                      # <-- the v1 omission
        f"training.seed={seed}",
        f"training.num_epochs={ne}",
        f"training.checkpoint_every={ce}",
        f"training.rollout_every={re_}",
        f"training.resume={str(bool(resume)).lower()}",   # <-- see clear_dir()
        f"training.debug={str(f.get('debug', False)).lower()}",
        f"training.max_train_steps={f.get('max_train_steps') or 'null'}",
        f"checkpoint.save_last_ckpt={str(f.get('save_last_ckpt', True)).lower()}",
        f"checkpoint.topk.k={f['topk_k']}",
        f"task.env_runner.n_test={f['monitor_n_test']}",
        f"task.env_runner.n_train={f.get('monitor_n_train', 0)}",
        f"hydra.run.dir={run_dir}",
    ] + extra


def clear_dir(run_dir, archive_root, why):
    """A stale latest.ckpt in the run dir is enough to make the workspace
    resume: training.num_epochs is a LOCAL loop count, so a resumed run does
    num_epochs MORE epochs from wherever it left off. The F-tf probe resumed
    from epoch 20 and ended at 39, which is why its 674 s covered neither 21
    epochs nor a cold start. Left unguarded in the full campaign this silently
    breaks seed homogeneity AND inflates c_s, while every gate still passes.
    Moved aside, never deleted."""
    if not os.path.isdir(run_dir) or not os.listdir(run_dir):
        return None
    os.makedirs(archive_root, exist_ok=True)
    dst = os.path.join(archive_root,
                       os.path.basename(run_dir) + f".{why}.{int(time.time())}")
    shutil.move(run_dir, dst)
    print(f"  cleared {run_dir} -> {dst}  (would otherwise auto-resume)")
    return dst


def run_once(a, fam, f, seed, run_dir, ne, ce, re_, tag, resume=False):
    os.makedirs(run_dir, exist_ok=True)
    ov = overrides(f, seed, run_dir, ne, ce, re_, resume)
    assert any(o.startswith("task=") for o in ov), "task override missing"
    cmd = [sys.executable, a.train_entry, "--config-dir", a.config_dir,
           "--config-name", f["config_name"]] + ov

    print(f"\n  --- {fam}/seed{seed} [{tag}] "
          f"epochs={ne} ckpt_every={ce} rollout_every={re_} resume={resume} ---")
    print("  " + " ".join(cmd))
    if not a.run:
        print("  [dry-run] not executed. Add --run to launch.")
        return None

    n_eval_d = len(range(0, ne, re_))
    if a.detach:
        PLAN.append((f"{fam}/seed{seed}[{tag}]", run_dir, cmd))
        print("  [queued for detached execution]")
        return {"tag": tag, "returncode": 0, "num_epochs": ne,
                "run_dir": run_dir, "wall": None, "deferred": True,
                "monitor_sec_est": n_eval_d * int(f["monitor_n_test"]) *
                float(f["mean_rollout_sec"]), "n_eval": n_eval_d}

    t0 = time.time()
    rc = subprocess.run(cmd, cwd=a.repo_root).returncode
    wall = time.time() - t0
    n_eval = len(range(0, ne, re_))
    monitor = n_eval * int(f["monitor_n_test"]) * float(f["mean_rollout_sec"])
    print(f"  rc={rc}  wall={wall:.0f}s  monitor_est={monitor:.0f}s")
    return {"tag": tag, "returncode": rc, "num_epochs": ne, "run_dir": run_dir,
            "wall": wall, "monitor_sec_est": monitor, "n_eval": n_eval}


def probe(a, fam, f, seed, sanity):
    """Two-point timing probe. A single short run cannot separate fixed startup
    (cuda init, dataset load, compile) from the marginal per-epoch cost, and at
    20 epochs the fixed part dominates. Running two lengths and taking the
    slope removes it exactly:  marginal = (t_long - t_short) / (n_long - n_short).
    Each length runs in its own fresh dir with resume disabled."""
    short, long_ = int(a.probe_short), int(a.probe_long)
    pts = []
    for n in (short, long_):
        rd = os.path.join(a.runs_root, fam, f"seed{seed}_probe{n}")
        clear_dir(rd, a.archive_root, "probe")
        r = run_once(a, fam, f, seed, rd, n + 1, n, n, f"probe{n}", resume=False)
        if r is None:
            return None
        if r["returncode"] != 0:
            print("  [BLOCK] probe returned non-zero; not projecting")
            return [r]
        pts.append(r)

    if any(p.get("deferred") for p in pts):
        print("  both probe lengths queued; run --status after they finish "
              "to get the two-point fit")
        return []

    ts, tl = pts[0]["wall"], pts[1]["wall"]
    ms, ml = pts[0]["monitor_sec_est"], pts[1]["monitor_sec_est"]
    marginal = ((tl - ml) - (ts - ms)) / (long_ - short)
    fixed = (ts - ms) - marginal * (short + 1)
    full = int(f["num_epochs"])
    proj_h = (fixed + marginal * full) / 3600.0
    rho = round((fixed + marginal * full) / float(f["mean_rollout_sec"]))

    print(f"\n  TWO-POINT FIT  {fam}/seed{seed}")
    print(f"    {short+1} epochs: {ts:.0f}s     {long_+1} epochs: {tl:.0f}s")
    print(f"    marginal = {marginal:.2f} s/epoch     fixed startup = {fixed:.0f}s")
    print(f"    projection @ {full} epochs: {proj_h:.1f} GPU-h/seed, "
          f"{6*proj_h:.0f} GPU-h for 6 seeds")
    print(f"    implied c_s/c_e = {rho:,}")
    lo, hi = sanity.get("rho_lo", 5000), sanity.get("rho_hi", 30000)
    if not (lo <= rho <= hi):
        print(f"    [WARN] outside the pre-registered band [{lo:,}, {hi:,}]")

    rec = {"family_id": fam, "seed_id": seed, "probe": True, "returncode": 0,
           "task": f["task"], "config_name": f["config_name"],
           "points": [{"epochs": p["num_epochs"], "wall_sec": round(p["wall"], 1),
                       "monitor_sec_est": round(p["monitor_sec_est"], 1)}
                      for p in pts],
           "marginal_sec_per_epoch": round(marginal, 3),
           "fixed_startup_sec": round(fixed, 1),
           "projected_full_gpu_hours_per_seed": round(proj_h, 2),
           "projected_six_seed_gpu_hours": round(6 * proj_h, 1),
           "projected_rho": rho}
    return [rec]


def launch(a, fam, f, seed, sanity):
    ne, ce, re_ = int(f["num_epochs"]), int(f["checkpoint_every"]), int(f["rollout_every"])
    run_dir = os.path.join(a.runs_root, fam, f"seed{seed}")
    ck = os.path.join(run_dir, "checkpoints", "latest.ckpt")

    resume = False
    resumed_from = None
    if os.path.exists(ck):
        if a.resume_remaining:
            # resume to a FIXED ABSOLUTE target: num_epochs is a local loop
            # count, so the remaining count must be computed explicitly or the
            # seed overshoots the frozen budget.
            from_ep = read_ckpt_epoch(ck)
            if from_ep is None:
                print("  [BLOCK] cannot read epoch from existing checkpoint")
                return None
            remaining = (ne - 1) - int(from_ep)
            if remaining <= 0:
                print(f"  {fam}/seed{seed}: already at epoch {from_ep} of "
                      f"{ne-1}; nothing to do")
                return None
            print(f"  resuming {fam}/seed{seed} from epoch {from_ep}: "
                  f"{remaining} epoch(s) remaining to reach {ne-1}")
            resume, resumed_from, ne = True, int(from_ep), remaining
        elif a.force:
            clear_dir(run_dir, a.archive_root, "force")
        else:
            print(f"  {fam}/seed{seed}: checkpoint present. Use --force for a "
                  f"clean restart or --resume-remaining to finish the budget. "
                  f"Launching as-is would auto-resume and overshoot.")
            return None

    r = run_once(a, fam, f, seed, run_dir, ne, ce, re_, "full", resume)
    if r is None:
        return None
    if r.get("deferred"):
        return None
    wall, monitor = r["wall"], r["monitor_sec_est"]
    rec = {
        "family_id": fam, "seed_id": seed, "probe": False,
        "returncode": r["returncode"], "run_dir": run_dir,
        "task": f["task"], "config_name": f["config_name"],
        "target_final_epoch": int(f["num_epochs"]) - 1,
        "epochs_this_run": ne, "resumed_from_epoch": resumed_from,
        "checkpoint_every": ce, "rollout_every": re_,
        "train_wallclock_sec_raw": round(wall, 2),
        "monitor_eval_points": r["n_eval"],
        "monitor_sec_est": round(monitor, 2),
        "monitor_frac": round(monitor / wall, 5) if wall else None,
        "train_wallclock_sec": round(wall - monitor, 2),
        "train_gpu_hours": round((wall - monitor) / 3600.0, 5),
        "sec_per_epoch": round(wall / ne, 3) if ne else None,
    }
    if resumed_from is not None:
        rec["cost_note"] = ("resumed run: train_wallclock_sec covers only this "
                            "segment; c_s must sum the segments for this seed")
    mfmax = sanity.get("monitor_frac_max", 0.02)
    if rec["monitor_frac"] and rec["monitor_frac"] > mfmax:
        print(f"  [WARN] monitoring is {rec['monitor_frac']:.1%} of wall clock")
    if r["returncode"] != 0:
        print("  [BLOCK] training returned non-zero; do not launch further seeds")
    return rec


def read_ckpt_epoch(ck):
    try:
        import torch
        try:
            import dill
        except Exception:
            dill = None
        payload = torch.load(open(ck, "rb"), map_location="cpu",
                             pickle_module=dill, weights_only=False)
        v = (payload.get("pickles", {}) or {}).get("epoch", payload.get("epoch"))
        if isinstance(v, (bytes, bytearray)):
            import pickle
            v = (dill or pickle).loads(v)
        return v
    except Exception as e:
        print(f"  [WARN] could not read epoch from {ck}: {e!r}")
        return None

def write_manifest(path, rows, spec):
    old = []
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:
            pass
    key = {(r["family_id"], r["seed_id"], r.get("probe", False)): r for r in old}
    for r in rows:
        key[(r["family_id"], r["seed_id"], r.get("probe", False))] = r
    out = sorted(key.values(),
                 key=lambda r: (r["family_id"], r["seed_id"], bool(r.get("probe"))))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  manifest: {len(out)} record(s) -> {path}")

    fams = spec.get("families", {})
    s = spec.get("sanity", {})
    lo, hi = s.get("rho_lo", 5000), s.get("rho_hi", 30000)
    for r in out:
        if r.get("probe"):
            continue
        mrs = fams.get(r["family_id"], {}).get("mean_rollout_sec")
        if mrs and r.get("train_wallclock_sec"):
            rho = r["train_wallclock_sec"] / float(mrs)
            flag = "" if lo <= rho <= hi else "   <-- OUT OF BAND"
            print(f"    {r['family_id']}/seed{r['seed_id']}: c_s/c_e = "
                  f"{rho:,.0f}{flag}")
    print(f"    pre-registered band [{lo:,}, {hi:,}]; paper's measured ratios "
          f"7,410 / 11,160 / 22,981; round 1 gave 124-154 (artifact).")


def self_test():
    """Every CLI path must resolve to a defined name. The previous revision
    shipped with archive(), parse_seeds() and preflight() deleted by a
    line-offset splice, because the smoke test only exercised
    `--skip-preflight --probe --seed 0`. This makes that class of defect
    impossible to ship silently."""
    import ast as _ast
    import builtins as _b
    src = open(__file__).read()
    tree = _ast.parse(src)
    defined = {n.name for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef)}
    # local bindings and parameters are legitimate callables too
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Store):
            defined.add(n.id)
        elif isinstance(n, _ast.arg):
            defined.add(n.arg)
        elif isinstance(n, (_ast.Import, _ast.ImportFrom)):
            for al in n.names:
                defined.add((al.asname or al.name).split(".")[0])
    called = {n.func.id for n in _ast.walk(tree)
              if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
    missing = sorted(c for c in called
                     if c not in defined and not hasattr(_b, c))
    for name in ("archive", "parse_seeds", "preflight", "probe", "launch",
                 "status", "detach", "write_plan", "run_once", "clear_dir",
                 "read_ckpt_epoch", "write_manifest", "overrides", "load_spec",
                 "substrate_test", "_plan_alive", "_marker", "_last_epoch"):
        if name not in defined:
            missing.append(f"{name} (required entry point)")
    if missing:
        print(f"  SELF-TEST FAILED: undefined names called: {missing}")
        return 1
    print(f"  self-test ok: {len(defined)} functions, no undefined calls")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="ec_spec.yaml")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--train-entry", default="train.py")
    ap.add_argument("--config-dir", default="diffusion_policy/config")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--archive-root", default="runs_archive/round1_20260824")
    ap.add_argument("--manifest", default="ec_cost_manifest.json")
    ap.add_argument("--preflight", default="ec_preflight.py")
    ap.add_argument("--driver", default="ec_eval.py")
    ap.add_argument("--family", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="e.g. 1-5 or 1,2,3")
    ap.add_argument("--probe", action="store_true",
                    help="two-point timing probe; separates fixed startup from "
                         "marginal s/epoch before committing GPU")
    ap.add_argument("--probe-short", type=int, default=20)
    ap.add_argument("--probe-long", type=int, default=60)
    ap.add_argument("--resume-remaining", action="store_true",
                    help="finish an interrupted seed to its frozen absolute "
                         "epoch target (num_epochs is a LOCAL loop count, so "
                         "the remaining count is computed explicitly)")
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--detach", action="store_true",
                    help="queue the runs into a shell plan and launch it in "
                         "its own session; returns immediately")
    ap.add_argument("--stall-sec", type=int, default=900,
                    help="no log write for this long = STALLED")
    ap.add_argument("--substrate-test", action="store_true",
                    help="detached no-GPU heartbeat job: does a process "
                         "outlive the session?")
    ap.add_argument("--substrate-minutes", type=int, default=25)
    ap.add_argument("--status", action="store_true",
                    help="poll detached runs and emit the two-point fit")
    ap.add_argument("--version", action="store_true",
                    help="print build id; run this FIRST every session so the "
                         "copy on disk is never ambiguous again")
    ap.add_argument("--self-test", action="store_true",
                    help="check every entry point is defined and reachable")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    if a.version:
        import hashlib
        h = hashlib.md5(open(__file__, "rb").read()).hexdigest()[:12]
        print(f"  {__VERSION__}")
        print(f"  path {os.path.abspath(__file__)}")
        print(f"  md5  {h}")
        flags = [x for x in ("--substrate-test", "--detach", "--status",
                             "--resume-remaining", "--probe")
                 if x in open(__file__).read()]
        print(f"  flags present: {' '.join(flags)}")
        return
    if a.self_test:
        sys.exit(self_test())

    spec = load_spec(a.spec)
    fams = spec["families"]

    if a.status:
        status(a, spec)
        return

    if a.substrate_test:
        substrate_test(a)
        return

    print("=" * 76)
    print(f"ec_launch.py v2 -- spec: {a.spec}")
    print("=" * 76)
    for fam, f in fams.items():
        ne = int(f["num_epochs"])
        ce = int(f["checkpoint_every"])
        re_ = int(f["rollout_every"])
        print(f"  {fam:8s} task={str(f['task']):20s} epochs={ne:5d} "
              f"ckpt_every={ce:5d} (final_saved={(ne-1) % ce == 0}, "
              f"{len(range(0, ne, ce))} snapshots)  rollout_every={re_} "
              f"({len(range(0, ne, re_))} eval pts)")

    if a.archive:
        print("\n" + "=" * 76 + "\nARCHIVE\n" + "=" * 76)
        archive(a.runs_root, a.archive_root)
        return
    if a.manifest_only:
        write_manifest(a.manifest, [], spec)
        return
    if not a.skip_preflight:
        print("\n" + "=" * 76 + "\nPREFLIGHT (config stage)\n" + "=" * 76)
        if not preflight(a, list(fams)):
            print("\n  BLOCKED by preflight -- nothing launched.")
            sys.exit(1)

    if not a.family or a.family not in fams:
        print(f"\n  give --family from {list(fams)}; nothing to launch.")
        return
    seeds = [a.seed] if a.seed is not None else parse_seeds(a.seeds or "")
    if not seeds:
        print("\n  no --seed / --seeds given; nothing to launch.")
        return

    rows = []
    for sd in seeds:
        if a.probe:
            recs = probe(a, a.family, fams[a.family], sd, spec.get("sanity", {}))
            if recs:
                rows += recs
                if any(r.get("returncode") for r in recs):
                    break
            continue
        rec = launch(a, a.family, fams[a.family], sd, spec.get("sanity", {}))
        if rec:
            rows.append(rec)
            if rec["returncode"] != 0:
                break
    if PLAN:
        detach(a, PLAN)
        return
    if rows:
        write_manifest(a.manifest, rows, spec)
        if not a.probe:
            print(f"\n  next: python {a.preflight} --stage pilot --runs-root "
                  f"{a.runs_root} --family {a.family} --seed {seeds[0]}")


if __name__ == "__main__":
    main()
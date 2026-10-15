#!/usr/bin/env python3
"""
ec_preflight.py -- blocking gate for the E-C rerun (PFF / SCIA project)

WHY THIS EXISTS
    Round 1 burned 18 seeds x 3 families and produced 7200 semantically empty
    rows. The proximate cause was NOT training failure and NOT an eval parse
    bug. It was this:

        training.num_epochs   = 50      -> epochs run 0..49
        training.checkpoint_every = 50  -> `epoch % 50 == 0` holds ONLY at 0

    so the save branch fired exactly once, after the first epoch, and the
    converged epoch-49 weights were never written to disk. logs.json.txt shows
    last_epoch=49 and train_loss 1.16 -> 0.0147; every latest.ckpt shows
    epoch=0, global_step~109. ec_eval.py then faithfully measured a one-epoch
    model: outcome == 0 on all 7200 rows.

    Compounding it, training.rollout_every = 1000000 meant the in-training
    rollout eval ran only at epoch 0, so the logs carried no progress signal
    that would have exposed the problem -- and no signal that a diagnostic
    could legitimately read either.

    This script makes both failures impossible to repeat, and it is designed to
    run BEFORE GPU is spent, not after.

STAGES
    --stage config   run before launching anything. Validates each family's
                     training config: save cadence reaches the final epoch,
                     eval cadence yields >=2 progress points, epoch budget is
                     not below the reference recipe, cost recording is wired,
                     workspace is clean of stale/race run dirs, eval driver
                     contract is sound. Exit 0 = cleared to launch the pilot.

    --stage pilot    run after ONE seed finishes. Verifies the saved
                     checkpoint's epoch equals num_epochs-1 (i.e. the trained
                     weights actually persisted) and that the final in-training
                     test/mean_score clears --min-score. Exit 0 = cleared to
                     launch the remaining 17 seeds.

    --stage full     run after all seeds finish, before ec_eval. Verifies every
                     seed persisted a final-epoch checkpoint, checkpoints are
                     pairwise distinct, epoch budgets are uniform within a
                     family, and training cost was recorded for every family.

USAGE
    python ec_preflight.py --stage config --runs-root runs --driver ec_eval.py \
        --families F-cnn,F-tf,F-lstm --config-glob 'conf/ec_{family}.yaml'
    python ec_preflight.py --stage pilot  --runs-root runs --family F-cnn --seed 0
    python ec_preflight.py --stage full   --runs-root runs

    Exit 0 = PASS (authorized to proceed).  Exit 1 = BLOCKED.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

BLOCKERS, WARNINGS = [], []


def block(code, msg):
    BLOCKERS.append((code, msg))
    print(f"  [BLOCK] {code}: {msg}")


def warn(code, msg):
    WARNINGS.append((code, msg))
    print(f"  [WARN ] {code}: {msg}")


def ok(code, msg):
    print(f"  [ ok  ] {code}: {msg}")


def header(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dig(cfg, dotted, default=None):
    cur = cfg
    for k in dotted.split("."):
        if cur is None:
            return default
        try:
            cur = cur[k]
        except Exception:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_ckpt_meta(ck):
    """Return (epoch, global_step, cfg). payload['pickles'][k] holds
    dill-SERIALIZED bytes -- they must be dill.loads'd. Round 1's diagnostic
    compared the raw bytes against int and silently skipped its own
    truncation check; that is why this helper exists."""
    import torch
    try:
        import dill
    except Exception:
        dill = None
    try:
        payload = torch.load(open(ck, "rb"), map_location="cpu",
                             pickle_module=dill, weights_only=False)
    except Exception:
        payload = torch.load(ck, map_location="cpu", weights_only=False)

    def unwrap(v):
        if isinstance(v, (bytes, bytearray)):
            if dill is None:
                import pickle
                return pickle.loads(v)
            return dill.loads(v)
        return v

    pk = payload.get("pickles", {}) or {}
    ep = unwrap(pk.get("epoch", payload.get("epoch")))
    gs = unwrap(pk.get("global_step", payload.get("global_step")))
    return ep, gs, payload.get("cfg")


def read_log(seed_dir):
    """Return (n_records, last_epoch, score_series, loss_first_last)."""
    cand = [os.path.join(seed_dir, "logs.json.txt")] + glob.glob(
        os.path.join(seed_dir, "**", "logs.json.txt"), recursive=True)
    log = next((p for p in cand if os.path.exists(p)), None)
    if log is None:
        return None
    rows = []
    with open(log) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    scores = [r["test/mean_score"] for r in rows if "test/mean_score" in r]
    loss = [r["train_loss"] for r in rows if "train_loss" in r]
    last_ep = max((r.get("epoch", -1) for r in rows), default=None)
    return {"path": log, "n": len(rows), "last_epoch": last_ep,
            "scores": scores, "loss": ([loss[0], loss[-1]] if loss else None)}


# ---------------------------------------------------------------------------
# STAGE: config
# ---------------------------------------------------------------------------

def load_family_specs(a):
    """Prefer the single frozen spec (ec_spec.yaml). Round 1's F-tf split into
    two behaviours across its own six seeds, which is what separate
    hand-edited per-family configs drifting apart looks like -- so one file,
    three keys, is the supported shape. The per-family glob is kept only as a
    fallback for legacy trees."""
    if a.spec and os.path.exists(a.spec):
        spec = load_yaml(a.spec)
        fams = spec.get("families") or {}
        ok("P0-SPEC", f"{a.spec}: {len(fams)} families")
        return {k: v for k, v in fams.items()}, spec
    out = {}
    for fam in [f.strip() for f in a.families.split(",") if f.strip()]:
        path = a.config_glob.replace("{family}", fam)
        hits = sorted(glob.glob(path))
        if not hits:
            block("P1-NOCFG", f"{fam}: no spec at {a.spec!r} and no config "
                              f"matched {path!r}")
            continue
        c = load_yaml(hits[0])
        out[fam] = {
            "num_epochs": dig(c, "training.num_epochs"),
            "checkpoint_every": dig(c, "training.checkpoint_every"),
            "rollout_every": dig(c, "training.rollout_every"),
            "debug": dig(c, "training.debug", False),
            "max_train_steps": dig(c, "training.max_train_steps"),
            "save_last_ckpt": dig(c, "checkpoint.save_last_ckpt", True),
            "task": dig(c, "task") or dig(c, "task.name"),
            "expect_abs_action": dig(c, "task.abs_action"),
        }
    return out, {}


def stage_config(a):
    header("P1  TRAINING CONFIG REACHABILITY  [the round-1 root cause]")
    specs, _ = load_family_specs(a)

    for fam, f in specs.items():
        ne = f.get("num_epochs")
        ce = f.get("checkpoint_every")
        re_ = f.get("rollout_every")
        dbg = f.get("debug", False)
        mts = f.get("max_train_steps")
        savelast = f.get("save_last_ckpt", True)

        # --- P1-TASK: the v1-launcher defect ------------------------------
        task = f.get("task")
        if not task:
            block("P1-TASK",
                  f"{fam}: no task named. The base workspace configs default "
                  f"to pusht_lowdim, so an unset task trains the WRONG TASK "
                  f"while producing perfectly healthy-looking logs -- non-zero "
                  f"success, plausible variance, plausible cost. Unlike round "
                  f"1, that failure is silent.")
        elif "pusht" in str(task).lower():
            block("P1-TASK", f"{fam}: task={task!r} is the base config default")
        else:
            ok("P1-TASK", f"{fam}: task={task}")
            exp = f.get("expect_abs_action")
            is_abs = str(task).endswith("_abs")
            if exp is not None and bool(exp) != is_abs:
                block("P1-ABS",
                      f"{fam}: expect_abs_action={exp} but task={task!r}; "
                      f"round-1 checkpoints recorded abs_action={exp} here")

        print(f"\n  {fam}: num_epochs={ne} checkpoint_every={ce} "
              f"rollout_every={re_} debug={dbg} max_train_steps={mts}")

        if not isinstance(ne, int) or not isinstance(ce, int):
            block("P1-TYPE", f"{fam}: num_epochs/checkpoint_every not integers")
            continue

        # --- the exact round-1 defect ------------------------------------
        saved = [e for e in range(ne) if e % ce == 0]
        if saved == [0]:
            block("P1-CADENCE",
                  f"{fam}: checkpoint_every={ce} with num_epochs={ne} saves at "
                  f"epoch 0 ONLY -- the trained weights will never reach disk. "
                  f"This is the round-1 defect verbatim.")
        elif max(saved) < ne - 1:
            block("P1-CADENCE",
                  f"{fam}: last save is at epoch {max(saved)} of {ne-1}; the "
                  f"final weights are not persisted. Set checkpoint_every so "
                  f"that (num_epochs-1) % checkpoint_every == 0.")
        else:
            ok("P1-CADENCE", f"{fam}: final epoch {ne-1} is a save point")

        if not savelast:
            block("P1-SAVELAST", f"{fam}: checkpoint.save_last_ckpt is false")

        # --- eval cadence must yield a usable progress signal --------------
        n_eval = len([e for e in range(ne) if isinstance(re_, int) and e % re_ == 0])
        if not isinstance(re_, int) or n_eval < 2:
            block("P1-ROLLOUT",
                  f"{fam}: rollout_every={re_} yields {n_eval} in-training eval "
                  f"point(s) over {ne} epochs. With <2 points there is no "
                  f"progress signal, and any later diagnostic that reads "
                  f"test/mean_score measures the untrained model (round-1 "
                  f"error B). Require rollout_every <= num_epochs//10.")
        elif re_ > ne // 10:
            warn("P1-ROLLOUT", f"{fam}: only {n_eval} eval points over {ne} epochs")
        else:
            ok("P1-ROLLOUT", f"{fam}: {n_eval} in-training eval points")

        if dbg:
            block("P1-DEBUG", f"{fam}: training.debug is true")
        if mts not in (None, 0):
            warn("P1-MAXSTEPS", f"{fam}: max_train_steps={mts} caps the epoch")

        # --- every save epoch must also be an eval epoch --------------------
        if isinstance(ce, int) and isinstance(re_, int) and re_ > 0:
            if ce % re_ != 0:
                block("P1-ALIGN",
                      f"{fam}: checkpoint_every={ce} is not a multiple of "
                      f"rollout_every={re_}. A save epoch that is not an eval "
                      f"epoch has no monitor key in its metric dict, and the "
                      f"top-k save raises. This is the F-tf probe crash; it is "
                      f"a cadence defect, not a key-name defect.")
            else:
                ok("P1-ALIGN", f"{fam}: every save epoch is an eval epoch")

        # --- epoch budget vs reference recipe (per family, from the spec) ---
        ref = f.get("reference_num_epochs") or a.ref_epochs
        if ref and ne < ref:
            block("P1-BUDGET",
                  f"{fam}: num_epochs={ne} is below the reference recipe "
                  f"({ref}). Round 1 used 50, and the implied cost ratio came "
                  f"out 124-154 against the frozen reference rho_Delta=6234 -- "
                  f"the same ~40x shortfall seen from the cost side. Fixing "
                  f"the save cadence alone will not produce a policy that ever "
                  f"succeeds.")
        elif ref and ne > ref:
            ok("P1-BUDGET", f"{fam}: num_epochs={ne} = reference {ref} + "
                            f"{ne-ref} (logged deviation: puts the final epoch "
                            f"on the checkpoint cadence)")
        elif ref:
            ok("P1-BUDGET", f"{fam}: num_epochs={ne} == reference {ref}")

    header("P2  WORKSPACE HYGIENE")
    dirty = []
    for pat in ("*stale*", "*race*", "*pathfix*", "*_old", "*.bak"):
        dirty += glob.glob(os.path.join(a.runs_root, "*", pat))
    if dirty:
        block("P2-STALE",
              f"{len(dirty)} stale/race run dir(s) under {a.runs_root}: "
              f"{[os.path.relpath(d, a.runs_root) for d in dirty[:6]]}"
              f"{' ...' if len(dirty) > 6 else ''}. Round 1 left these beside "
              f"live seeds; provenance of any cost or checkpoint read from that "
              f"tree is ambiguous. Archive them outside runs/ before launching.")
    else:
        ok("P2-STALE", "no stale/race run directories")

    for seed_dir in sorted(glob.glob(os.path.join(a.runs_root, "*", "seed*"))):
        log = read_log(seed_dir)
        if log and log["last_epoch"] is not None and log["n"] > 0:
            per_ep = log["n"] / max(log["last_epoch"] + 1, 1)
            if per_ep > a.max_records_per_epoch:
                warn("P2-APPEND",
                     f"{os.path.relpath(seed_dir, a.runs_root)}: {log['n']} log "
                     f"records over {log['last_epoch']+1} epochs "
                     f"({per_ep:.0f}/epoch) -- looks like a re-run appended to "
                     f"an existing log; train_wallclock_sec from this dir is "
                     f"not a clean c_s measurement")

    header("P3  EVAL DRIVER CONTRACT")
    check_driver(a.driver)

    header("P4  COST RECORDING")
    if a.driver and os.path.exists(a.driver):
        src = open(a.driver).read()
        for field in ("train_wallclock_sec", "train_gpu_hours",
                      "rollout_wallclock_sec"):
            if field not in src:
                warn("P4-COST",
                     f"{field} not referenced in {a.driver}; round 1 recorded "
                     f"0.0 train cost for F-tf and F-lstm, which left c_s "
                     f"unmeasurable for two of three families")
    if a.manifest and os.path.exists(a.manifest):
        rows = json.load(open(a.manifest))
        miss = [r.get("family_id") for r in rows
                if not r.get("train_wallclock_sec")]
        if miss:
            block("P4-COST", f"manifest missing train cost for {sorted(set(miss))}")


def check_driver(driver):
    if not driver or not os.path.exists(driver):
        warn("P3-MISSING", f"driver not found at {driver!r}")
        return
    src = open(driver).read()
    lines = src.splitlines()

    def code_lines():
        """Yield (lineno, text) for non-comment, non-docstring-ish lines, so a
        CLI help string mentioning 'train/' is not reported as a metric bug
        (round-1 error C: C4-TRAINPREFIX was a false positive on a click
        option's help text)."""
        in_doc = False
        for i, ln in enumerate(lines, 1):
            s = ln.strip()
            if s.startswith(('"""', "'''")):
                in_doc = not in_doc
                continue
            if in_doc or s.startswith("#"):
                continue
            yield i, ln

    for i, ln in code_lines():
        if re.search(r"\.get\(\s*[\"'][^\"']*[\"']\s*,\s*0\s*\)", ln):
            block("P3-DEFAULT0",
                  f"{driver}:{i}  .get(key, 0) default -- a renamed or missing "
                  f"metric key becomes a zero outcome instead of raising")
        if re.search(r"except\s*(Exception)?\s*:\s*pass", ln):
            # Static analysis cannot tell a swallowed OUTCOME from a swallowed
            # type conversion, so this warns rather than blocks. Round 1's
            # instance was in per_episode()'s cast, not the outcome path.
            # The question the reviewer must answer is not "is there a bare
            # except" but "what value reaches `outcome` when it fires".
            warn("P3-SWALLOW",
                 f"{driver}:{i}  `except: pass` -- confirm the value that "
                 f"reaches `outcome` when this fires. If it falls back to a "
                 f"pre-initialised 0, it is the round-1 failure by another "
                 f"route; if it skips the episode or re-raises, it is benign. "
                 f"Clear with --ack P3-SWALLOW:{i} once checked.")
        if re.search(r"\[\s*[\"']train/", ln) or re.search(r"\.get\(\s*[\"']train/", ln):
            block("P3-TRAINKEY",
                  f"{driver}:{i}  reads a train/-prefixed runner key while "
                  f"n_train=0 -- the runner emits no train/ metrics")
    if "test/" not in src:
        warn("P3-TESTKEY",
             f"{driver} never mentions a test/-prefixed key; confirm it reads "
             f"the per-episode test/sim_max_reward_* values")
    ok("P3-SCAN", f"scanned {len(lines)} lines of {driver}")


# ---------------------------------------------------------------------------
# STAGE: pilot  (one seed done -> authorize the other 17)
# ---------------------------------------------------------------------------

def stage_pilot(a):
    header(f"PILOT GATE  {a.family}/seed{a.seed}")
    seed_dir = os.path.join(a.runs_root, a.family, f"seed{a.seed}")
    ck = os.path.join(seed_dir, "checkpoints", "latest.ckpt")

    if not os.path.exists(ck):
        block("G1-CKPT", f"no checkpoint at {ck}")
        return
    ep, gs, cfg = load_ckpt_meta(ck)
    ne = dig(cfg, "training.num_epochs")
    print(f"  checkpoint: epoch={ep} global_step={gs} (num_epochs={ne})")

    if not isinstance(ep, int):
        block("G1-EPOCH", f"epoch is {type(ep).__name__}, not int -- "
                          f"deserialize payload['pickles'] with dill.loads")
    elif isinstance(ne, int) and ep < ne - 1:
        block("G1-PERSIST",
              f"saved checkpoint is at epoch {ep} of {ne-1}. The trained "
              f"weights did not persist -- this is round 1 repeating. Fix "
              f"checkpoint_every before spending the remaining seeds.")
    elif isinstance(ne, int) and ep > ne - 1:
        block("G1-OVERRUN",
              f"saved checkpoint is at epoch {ep}, PAST the frozen target "
              f"{ne-1}. training.num_epochs is a local loop count, so a stale "
              f"latest.ckpt in the run dir makes the workspace resume and run "
              f"num_epochs MORE epochs. This seed has a different training "
              f"budget from its siblings: seed homogeneity is broken and c_s "
              f"is wrong. Clear the run dir and restart this seed. (The F-tf "
              f"probe hit exactly this: resumed at 20, ended at 39.)")
    else:
        ok("G1-PERSIST", f"final-epoch checkpoint persisted (epoch={ep})")

    log = read_log(seed_dir)
    if log is None:
        block("G2-LOG", f"no logs.json.txt under {seed_dir}")
        return
    print(f"  log: {log['n']} records, last_epoch={log['last_epoch']}, "
          f"loss {log['loss']}, {len(log['scores'])} eval points")

    if len(log["scores"]) < 2:
        block("G2-SIGNAL",
              f"only {len(log['scores'])} in-training eval point(s) -- cannot "
              f"distinguish 'policy does not work' from 'never evaluated'. "
              f"This ambiguity is what produced the wrong round-1 verdict.")
        return
    best = max(log["scores"])
    final = log["scores"][-1]
    print(f"  test/mean_score: final={final} best={best}")
    if best < a.min_score:
        block("G2-SCORE",
              f"best in-training test/mean_score {best:.3f} < required "
              f"{a.min_score:.3f}. Do NOT launch the remaining seeds; audit "
              f"the recipe (dataset path, abs_action vs rotation_transformer, "
              f"n_obs_steps/horizon, normalizer) first.")
    else:
        ok("G2-SCORE", f"best={best:.3f} >= {a.min_score:.3f}")


# ---------------------------------------------------------------------------
# STAGE: full  (all seeds done -> authorize ec_eval)
# ---------------------------------------------------------------------------

def stage_full(a):
    header("FULL GATE  (run before ec_eval)")
    hashes, budgets = {}, {}
    for seed_dir in sorted(glob.glob(os.path.join(a.runs_root, "*", "seed*"))):
        rel = os.path.relpath(seed_dir, a.runs_root)
        fam = rel.split(os.sep)[0]
        ck = os.path.join(seed_dir, "checkpoints", "latest.ckpt")
        if not os.path.exists(ck):
            block("F1-CKPT", f"{rel}: missing latest.ckpt")
            continue
        ep, gs, cfg = load_ckpt_meta(ck)
        ne = dig(cfg, "training.num_epochs")
        budgets.setdefault(fam, set()).add(ne)
        if isinstance(ep, int) and isinstance(ne, int) and ep < ne - 1:
            block("F1-PERSIST", f"{rel}: checkpoint at epoch {ep} of {ne-1}")
        if isinstance(ep, int) and isinstance(ne, int) and ep > ne - 1:
            block("F1-OVERRUN",
                  f"{rel}: checkpoint at epoch {ep}, past the target {ne-1} -- "
                  f"an auto-resume ran extra epochs; this seed's budget "
                  f"differs from its siblings")

        log = read_log(seed_dir)
        if log and log["last_epoch"] is not None and isinstance(ne, int) \
                and log["last_epoch"] < ne - 1:
            block("F2-TRUNC",
                  f"{rel}: training stopped at epoch {log['last_epoch']} of "
                  f"{ne-1} (round 1: F-tf seeds 0-2 stopped near epoch 12 "
                  f"while seeds 3-5 reached 49 -- a non-homogeneous family)")

        st = os.stat(ck)
        h = hashlib.sha256()
        with open(ck, "rb") as f:
            h.update(f.read(8 << 20))
            if st.st_size > (16 << 20):
                f.seek(-(8 << 20), os.SEEK_END)
                h.update(f.read())
        hashes.setdefault(h.hexdigest()[:16], []).append(rel)
        print(f"  {rel:20s} epoch={ep} step={gs} size={st.st_size/1e6:.0f}MB")

    for hsh, keys in hashes.items():
        if len(keys) > 1:
            block("F3-DUP", f"identical checkpoint bytes across {keys} -- "
                            f"these are not independent realizations")
    for fam, b in budgets.items():
        if len(b) > 1:
            block("F4-NONUNIFORM", f"{fam}: mixed num_epochs across seeds {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["config", "pilot", "full"])
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--families", default="F-cnn,F-tf,F-lstm")
    ap.add_argument("--spec", default="ec_spec.yaml",
                    help="single frozen spec; preferred over --config-glob")
    ap.add_argument("--config-glob", default="conf/ec_{family}.yaml",
                    help="legacy fallback; {family} is substituted per family")
    ap.add_argument("--ack", action="append", default=[],
                    help="clear a reviewed finding, e.g. --ack P3-SWALLOW:39")
    ap.add_argument("--driver", default="ec_eval.py")
    ap.add_argument("--manifest", default=None,
                    help="optional JSON list of per-seed cost records")
    ap.add_argument("--ref-epochs", type=int, default=None,
                    help="reference recipe epoch count for this task; read it "
                         "off the repo's own square_lowdim train config rather "
                         "than guessing")
    ap.add_argument("--max-records-per-epoch", type=float, default=400.0)
    ap.add_argument("--family", default="F-cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-score", type=float, default=0.05)
    a = ap.parse_args()

    {"config": stage_config, "pilot": stage_pilot, "full": stage_full}[a.stage](a)

    header("GATE RESULT")
    acked = []
    for tag in a.ack:
        code, _, line = tag.partition(":")
        hit = [b for b in BLOCKERS
               if b[0] == code and (not line or f":{line} " in b[1] or
                                    b[1].endswith(f":{line}"))]
        for b in hit:
            BLOCKERS.remove(b)
            acked.append(tag)
            print(f"  [ACK  ] {tag} cleared by explicit review")
        if not hit:
            print(f"  [ACK  ] {tag} matched no open blocker (already clear?)")

    print(f"  blockers={len(BLOCKERS)}  warnings={len(WARNINGS)}  "
          f"acknowledged={len(acked)}")
    for c, m in BLOCKERS:
        print(f"    [BLOCK] {c}")
    if BLOCKERS:
        print("\n  BLOCKED -- do not spend GPU until every blocker is cleared.")
        sys.exit(1)
    print("\n  PASS -- authorized to proceed to the next stage.")
    sys.exit(0)


if __name__ == "__main__":
    main()

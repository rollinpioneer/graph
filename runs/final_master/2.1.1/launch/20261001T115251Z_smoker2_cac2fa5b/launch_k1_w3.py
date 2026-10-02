#!/usr/bin/env python
"""R2-1-CONCURRENCY-3 launcher: starts ONLY plan R-TB-K-1 with the worker cap raised 2->3 in THIS process.
No tracked file is modified.  Refuses unless the recorded amendment is present with its recorded text hash."""
import hashlib, json, os, subprocess, sys, time
OUT = open("/home/xushijie2/tmp/r2/R2_OUT.txt").read().strip()
GPU = int(sys.argv[1]); PLAN = "R-TB-K-1"
root = os.getcwd()
am = json.load(open(os.path.join(OUT, "concurrency_amendment.json")))
text = open(os.path.join(OUT, am["authorization_text_file"]), encoding="utf-8").read()
assert hashlib.sha256(text.encode("utf-8")).hexdigest() == am["authorization_text_sha256"], "amendment text hash mismatch"
assert am["change"]["max_training_workers"] == {"old": 2, "new": 3}
sys.path.insert(0, os.path.join(root, "src"))
from cp_disr import final_tb as ft
assert ft.MAX_WORKERS == 2, "unexpected MAX_WORKERS %r" % (ft.MAX_WORKERS,)
ft.MAX_WORKERS = int(am["change"]["max_training_workers"]["new"])  # in-process only
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
ev = {"event": "MAX_WORKERS_OVERRIDE", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan": PLAN, "gpu": GPU,
      "old": 2, "new": ft.MAX_WORKERS, "pid": os.getpid(), "head": head, "amendment_sha256": hashlib.sha256(open(os.path.join(OUT, "concurrency_amendment.json"), "rb").read()).hexdigest(),
      "wrapper_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
with open(os.path.join(OUT, "concurrency_override_events.jsonl"), "a") as f:
    f.write(json.dumps(ev, sort_keys=True) + "\n")
sys.exit(ft.main(["train", "--root", ".", "--out", OUT, "--plan", PLAN, "--gpu", str(GPU), "--token", os.path.join(OUT, "release_tokens", "train_%s.json" % PLAN)]))

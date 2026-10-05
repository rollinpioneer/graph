"""Start one extra CP-DISR-C1 QMARK training attempt concurrently with the running one (user-authorized pre-launch of the conditional seeds, 2026-10-05).

usage: python c1_extra_worker.py <root> <out> <plan> <gpu>
Only override: the one-worker concurrency cap (MAX_WORKERS 1 -> 3). Every other check (token, prep commit, release gate, attempt cap 3, config hash) is the unchanged CLI path.
"""
import sys

from cp_disr import final_tb_c1 as C1

root, out, plan, gpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
C1.MAX_WORKERS = 3
sys.exit(C1.main(["train", "--root", root, "--out", out, "--plan", plan, "--gpu", str(gpu), "--token", "%s/release_tokens/train_%s.json" % (out, plan)]))

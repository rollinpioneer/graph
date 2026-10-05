"""Start one CP-DISR-TB-STRUCT-GEN-V1 plan under the user-approved 12-way concurrency (in-process overrides only; tracked source untouched).

usage: python sg_extra_worker.py <root> <out> <plan> <gpu>
Overrides: MAX_WORKERS 6 -> 12 and the ledger's one-worker-per-GPU rule (the physical GPU is still recorded). Nothing else changes.
"""
import sys

from cp_disr import final_tb_structgen as S

root, out, plan, gpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
S.MAX_WORKERS = 12
_orig = S.StructgenLedger.reserve


def reserve(self, plan_id, attempt_id, run_dir, pid, physical_gpu):
    row = _orig(self, plan_id, attempt_id, run_dir, pid, None)  # None skips the shared-GPU refusal
    with self.locked():
        state = self.read()
        state["plans"][plan_id]["physical_gpu"] = physical_gpu
        state["plans"][plan_id]["shared_gpu_allowed"] = True
        S.ftb.write_json_atomic(self.state_path, state)
    return state["plans"][plan_id]


S.StructgenLedger.reserve = reserve
sys.exit(S.main(["train", "--root", root, "--out", out, "--plan", plan, "--gpu", str(gpu), "--token", "%s/release_tokens/train_%s.json" % (out, plan)]))

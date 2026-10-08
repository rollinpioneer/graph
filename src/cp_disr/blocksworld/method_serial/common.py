"""Run context, condition table, receipts, model construction and the driver manifest of C1-BW-METHOD-SERIAL-SUITE-V3."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .. import eval_a03 as E
from .. import scorer_control as SC
from .model import SerialModel, SerialPolicy

CARD = "C1-BW-METHOD-SERIAL-SUITE-V3"
BASE_COMMIT = "2d008431eb7fb069e617e4a02843bef84c6117d1"
BRANCH = "codex/cp-disr-c1-bw-method-serial-v3"
RUNBOOK_REL = "docs/c1_blocksworld/CP_DISR_C1_Method_Serial_Runbook_v3_Audit_Integrated.md"
SUITE_REL = E.BASE_REL + "/method_serial_v3"
SO_RUN_REL = E.BASE_REL + "/scale_order_prototype_v1/20261007T083206Z_ea7f69e8"
CONFIRM_NAMESPACE = "C1-BW-SERIAL-CONFIRM128-v1"
SMOKE = bool(os.environ.get("SERIAL_SMOKE"))               # end-to-end dry run of the driver chain: 2 epochs, 64 trajectories, tiny confirmation set, outputs outside the repo tree
EPOCHS, CKPT_EPOCHS = (2, (1, 2)) if SMOKE else (100, (20, 40, 60, 80, 100))
# condition -> (model kind, loss kind, family)
CONDS = {"SG_BASE": ("EVENT", "BASE", "event"), "SG_FACT": ("EVENT", "FACT", "event"), "SG_JOINT": ("EVENT", "JOINT", "event"),
         "CAL_ABS": ("CAL", "ABS", "calibration"), "CAL_REL": ("CAL", "REL", "calibration"),
         "REC_SELF": ("REC_SELF", "REC", "recurrent"), "REC_REL": ("REC_REL", "REC", "recurrent"),
         "GOAL_DENSE": ("GOAL_DENSE", "ACTION", "goal_attention"), "GOAL_REL": ("GOAL_REL", "ACTION", "goal_attention"),
         "GOAL_RAND": ("GOAL_RAND", "ACTION", "goal_attention")}          # sparsity-matched random-mask control (goal-lookahead integration card)
V3_CONDS = ("SG_BASE", "SG_FACT", "SG_JOINT", "CAL_ABS", "CAL_REL", "REC_SELF", "REC_REL", "GOAL_DENSE", "GOAL_REL")
# the 17 confirmation conditions: name -> (source trained condition or None, how it is executed)
CONFIRM = {"MG_C3": (None, "one_step"), "B_G1C3": (None, "b2_rule"), "LOOK2_MG": (None, "look_mg"), "LOOK2_COUNT": (None, "look_count"),
           "SG_BASE": ("SG_BASE", "one_step"), "SG_FACT": ("SG_FACT", "one_step"), "SG_JOINT": ("SG_JOINT", "one_step"),
           "CAL_ABS_C3": ("CAL_ABS", "one_step"), "CAL_REL_C3": ("CAL_REL", "one_step"), "CAL_ABS_LOOK2": ("CAL_ABS", "look_w"), "CAL_REL_LOOK2": ("CAL_REL", "look_w"),
           "REC_SELF_T4": ("REC_SELF", "T4"), "REC_SELF_T8": ("REC_SELF", "T8"), "REC_REL_T4": ("REC_REL", "T4"), "REC_REL_T8": ("REC_REL", "T8"),
           "GOAL_DENSE": ("GOAL_DENSE", "one_step"), "GOAL_REL": ("GOAL_REL", "one_step")}
SAME_CHECKPOINT = (("CAL_ABS_C3", "CAL_ABS_LOOK2"), ("CAL_REL_C3", "CAL_REL_LOOK2"), ("REC_SELF_T4", "REC_SELF_T8"), ("REC_REL_T4", "REC_REL_T8"))


def sha_file(p):
    return E.sha256_file(p)


def wj(p, doc):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def rj(p):
    return json.loads(Path(p).read_text())


def check_storage(path):
    p = Path(path).resolve()
    s = str(p)
    if "xushijie2" in s or "xushijie3" not in s:
        raise RuntimeError("path resolves outside the xushijie3 storage: %s" % s)
    return p


class Ctx:
    def __init__(self, root, run_root, device=None):
        self.root = check_storage(root)
        self.rr = check_storage(run_root)
        self.cfg = rj(self.rr / "suite_runtime.json")
        self.device = device or torch.device("cuda", 0)

    def p(self, *a):
        return self.rr.joinpath(*a)

    def receipt(self, task, status, outputs=(), **extra):
        outs = []
        for o in outputs:
            o = check_storage(o)
            outs.append({"path": str(o), "sha256": sha_file(o)})
        wj(self.p("receipts", "%s.json" % task), {"task": task, "status": status, "input_fingerprint": os.environ.get("TASK_FINGERPRINT"), "outputs": outs, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})

    def mg_model(self):
        pol = SC.load_scorer(self.root, "MG_C3", self.device)
        return pol.model

    def build(self, kind):
        m = SerialModel(self.mg_model(), kind)
        return m

    def load_selected(self, cond):
        from ...torch_rl import load_checkpoint
        sel = rj(self.p("runs", cond, "checkpoint_selection.json"))
        kind = CONDS[cond][0]
        m = self.build(kind)
        ck = self.root / sel["selected_checkpoint"]["path"]
        assert sha_file(ck) == sel["selected_checkpoint"]["sha256"], "checkpoint hash mismatch"
        load_checkpoint(ck, m)
        m.eval()
        return m, sel

    def load_epoch(self, cond, epoch):
        from ...torch_rl import load_checkpoint
        acct = rj(self.p("runs", cond, "training_accounting.json"))
        name = "final" if epoch == EPOCHS else "epoch%03d" % epoch
        info = acct["checkpoints"][name]
        ck = self.root / info["path"]
        assert sha_file(ck) == info["sha256"]
        m = self.build(CONDS[cond][0])
        load_checkpoint(ck, m)
        m.eval()
        return m, info

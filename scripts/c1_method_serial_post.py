#!/usr/bin/env python
"""Post-hoc descriptions for C1-BW-METHOD-SERIAL-SUITE-V3 (written after the confirmation numbers were seen; descriptive, not part of the registered stages).

    python scripts/c1_method_serial_post.py --run-root R --gpu N
Writes results/development_support_summary.csv and results/post_rec_effect_diagnostic.csv.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from cp_disr.blocksworld import imitation as I, scorer_control as SC
    from cp_disr.blocksworld.method_serial import common as C, evalkit as EK, model as MD
    from cp_disr.rl import set_suite_half_life
    rr = Path(a.run_root)
    td = json.loads((ROOT / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    set_suite_half_life(float(td["half_life"]))
    ctx = C.Ctx(ROOT, rr)
    rows = []
    _g112, m112 = EK.load_cases(ROOT, SC.FRESH_REL)
    for cond in C.CONDS:
        sel = json.loads((rr / "runs" / cond / "checkpoint_selection.json").read_text())
        b = sel["board48_by_epoch"][str(sel["selected_epoch"])]
        f = EK.summarize(EK.jl(rr / "development" / cond / "fresh112.jsonl"), m112)
        dv = EK.jl(rr / "development" / cond / "dev36.jsonl")
        bs = json.loads((rr / "development" / cond / "state_bank_summary.json").read_text())
        rows.append({"condition": cond, "selected_epoch": sel["selected_epoch"], "board48_success": b["success"], "board48_perfect": b["perfect"], "board48_h4_perfect": b["h4_perfect"], "fresh112_success": f["success"],
                     "fresh112_perfect": f["perfect"], "fresh112_h4_perfect": f["h4_perfect"], "dev36_success": sum(e["success"] for e in dv), "dev36_perfect": sum(e["decision_perfect"] for e in dv),
                     "bank_top1_optimal_of_240": bs["ALL"]["top1_optimal"], "bank_neutral_top1": json.dumps(bs["neutral_preparation"]), "bank_repaired_vs_MG": bs["ALL"].get("repaired_vs_ref"), "bank_newly_wrong_vs_MG": bs["ALL"].get("newly_wrong_vs_ref")})
    with open(rr / "results" / "development_support_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # how much does the recurrent processor change the output of the selected REC checkpoints?
    bank = EK.bank_records(ROOT / C.SO_RUN_REL)
    bcases, _bm = EK.bank_cases(ROOT)
    out = []
    for cond in ("REC_SELF", "REC_REL"):
        m, _sel = ctx.load_selected(cond)
        d48, d_no = [], []
        wo = float(m.rec.wo.weight.abs().max())
        for r in bank:
            sn = I.snapshot_at(bcases[r["case_id"]], tuple(r["state"]), 0)
            l4, l8 = m.eval_logits(sn, 4), m.eval_logits(sn, 8)
            ref = m.mg.group_scores([sn])[0]
            ok = torch.isfinite(l4)
            d48.append(float((l4[ok] - l8[ok]).abs().max()))
            d_no.append(float((l4[ok] - ref[ok]).abs().max()))
        out.append({"condition": cond, "max_abs_wo_weight": wo, "states": len(bank), "mean_max_abs_logit_diff_T4_vs_T8": statistics.mean(d48), "max_logit_diff_T4_vs_T8": max(d48),
                    "mean_max_abs_logit_diff_T4_vs_encoder_only": statistics.mean(d_no), "max_logit_diff_T4_vs_encoder_only": max(d_no),
                    "top1_changed_T4_vs_T8": None})
    with open(rr / "results" / "post_rec_effect_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

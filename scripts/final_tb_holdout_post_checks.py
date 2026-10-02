#!/usr/bin/env python
"""Post-run descriptive data-integrity check for CP-DISR-TB-INDEP-HOLDOUT-01 (reads only the recorded rows; no env, no model, no GPU).

Pairwise: how many of the 30 same-case rows are identical (success, G, steps, reason, every action id and clock) between each pair of models.
Writes pairwise_row_identity.json and appends a labelled section to holdout_summary.md (idempotent).  Not a decision input.
"""
import itertools
import json
import sys
from pathlib import Path

MARK = "## 事后数据完整性观察（描述性）"


def main(rel):
    rel = Path(rel)
    rows = {}
    for line in (rel / "per_episode_eval_row_v2.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["slot"], {})[r["case_id"]] = r["row"]
    rc = json.loads((rel / "model_hash_receipts.json").read_text())["receipts"]

    def key(r):
        return (r["success"], r["G"], r["steps"], r["reason"], tuple((a["candidate_id"], a["clock_start"], a["clock_end"], a["reward"]) for a in r["actions"]))

    slots = sorted(rows)
    out = {"document": "pairwise_row_identity", "pairs": []}
    for a, b in itertools.combinations(slots, 2):
        same = sum(1 for c in rows[a] if key(rows[a][c]) == key(rows[b][c]))
        out["pairs"].append({"a": a, "b": b, "identical_rows_of_30": same, "checkpoint_sha256_equal": rc[a]["sha256_before_first_episode"] == rc[b]["sha256_before_first_episode"],
                             "execution_class_a": rc[a]["execution_class"], "execution_class_b": rc[b]["execution_class"]})
    out["note"] = "identical rows mean identical action sequences and simulated clocks for that case; checkpoints with different SHA256 can still choose the same deterministic plan"
    (rel / "pairwise_row_identity.json").write_text(json.dumps(out, indent=1) + "\n")
    hit = [p for p in out["pairs"] if p["identical_rows_of_30"] > 0]
    lines = [MARK, "", "- 7 个模型两两比较同 case 行（success、G、steps、reason、每个动作 id 与仿真时钟全部相同才算相同），见 `pairwise_row_identity.json`。"]
    if hit:
        for p in hit:
            lines.append("- %s vs %s：%d/30 行完全相同（checkpoint SHA256 %s；执行类 %s / %s）。" % (p["a"], p["b"], p["identical_rows_of_30"], "相同" if p["checkpoint_sha256_equal"] else "不同", p["execution_class_a"][:7], p["execution_class_b"][:7]))
        lines.append("- 解读边界：这是确定性策略在同一冻结场景上选择了同一条动作序列的描述性事实；不同 SHA256 的 checkpoint 可以做出相同决策。它不改变任何模型的计分，也不能据此推出权重或训练等价。")
    else:
        lines.append("- 没有任何一对模型的同 case 行完全相同。")
    p = rel / "holdout_summary.md"
    text = p.read_text()
    if MARK not in text:
        p.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n")
    print(json.dumps({"pairs_with_identical_rows": [(p["a"], p["b"], p["identical_rows_of_30"]) for p in hit]}))


if __name__ == "__main__":
    main(sys.argv[1])

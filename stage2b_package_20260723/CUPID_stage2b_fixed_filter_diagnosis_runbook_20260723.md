# CUPID Stage 2B 操作文档：固定过滤目标诊断与保守可变数量过滤

**版本日期：2026-07-23**  
**服务器工作目录：`/home/xushijie/CUPID`**  
**当前状态：固定最低 38 条的 Bootstrap 序贯停止分支失败**  
**本阶段：只用已有 100×192 影响矩阵；不训练；不新增 Rollout；不重新计算 TRAK**

---

## 1. 当前结果说明了什么

已经确认：

- 最低 38 条名单随 Rollout 子集变化很大；
- 两个独立 50 条池的名单重合很低；
- 原 Bootstrap 停止规则必须跑满预算，因此不能节省 Rollout。

下一步不再尝试“提前恢复唯一的最低 38 条”，而是先检查：

> 是否只有一小批 Demo 在不同 Rollout 子集中始终低分，而固定删除 38 条过于激进。

若稳定低分核心不存在，应停止 Square-MH 过滤改进主线。  
若存在，则再判断 Bootstrap 稳定性是否真的比“简单少删一些最低分 Demo”更好。

---

## 2. 实验内容

本实验做三件事：

1. 检查最低 5%、10%、20%、30% 的筛选边界是否平坦；
2. 在 10、25、50、75 条 Rollout 子集中，统计每条 Demo 进入删除集合的频率；
3. 使用 50/50 不重叠 Rollout 池，比较：
   - 固定最低 38 条；
   - 固定最低 19 条；
   - Bootstrap 稳定低分核心；
   - 与稳定核心数量相同的简单最低分集合。

最后一个对照用于防止把“删得少所以更准确”误认为 Bootstrap 方法有效。

---

## 3. 禁止事项

本阶段禁止：

- 训练策略；
- 新增 Rollout；
- 重算 TRAK；
- 改变成功/失败标签；
- 修改冻结矩阵；
- 根据结果调阈值；
- 自动启动下游重训练；
- 把低分 Demo 直接称为有害 Demo。

---

## 4. 将脚本放到服务器

把随本文档提供的：

```text
stage2b_fixed_filter_diagnosis.py
```

复制到：

```text
/home/xushijie/CUPID/tools/stage2b_fixed_filter_diagnosis.py
```

然后：

```bash
chmod +x /home/xushijie/CUPID/tools/stage2b_fixed_filter_diagnosis.py
```

---

## 5. 固定路径

```bash
export CUPID_ROOT=/home/xushijie/CUPID

export MATRIX=$CUPID_ROOT/results/influence_layers/rollout_demo_influence.npy
export MANIFEST=$CUPID_ROOT/manifests/rollout100/episode_manifest.csv
export SPLIT_JSON=$CUPID_ROOT/manifests/training_split/dataset_split.json
export OFFICIAL_SCORES=$CUPID_ROOT/results/influence_layers/final_demo_scores.csv

export SCRIPT=$CUPID_ROOT/tools/stage2b_fixed_filter_diagnosis.py
export OUT=$CUPID_ROOT/results/stage2b_fixed_filter_diagnosis_v1
export LOG=$CUPID_ROOT/logs/stage2b_fixed_filter_diagnosis_v1.log
export FROZEN=$CUPID_ROOT/frozen/stage2b_fixed_filter_diagnosis_v1
```

---

## 6. 输入预检

```bash
set -euo pipefail

test -f "$MATRIX"
test -f "$MANIFEST"
test -f "$SPLIT_JSON"
test -f "$OFFICIAL_SCORES"
test -f "$SCRIPT"

if [ -e "$OUT" ]; then
    echo "输出目录已存在，禁止覆盖：$OUT"
    exit 1
fi

python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path("/home/xushijie/CUPID")
matrix = np.load(
    root / "results/influence_layers/rollout_demo_influence.npy",
    mmap_mode="r",
)
manifest = pd.read_csv(
    root / "manifests/rollout100/episode_manifest.csv"
)
split = json.loads(
    (root / "manifests/training_split/dataset_split.json")
    .read_text()
)
scores = pd.read_csv(
    root / "results/influence_layers/final_demo_scores.csv"
)

print("matrix:", matrix.shape, matrix.dtype)
print("manifest:", len(manifest), list(manifest.columns))
print("train demos:", len(split["train_demo_indices"]))
print("scores:", len(scores), list(scores.columns))

assert matrix.shape == (100, 192)
assert len(manifest) == 100
assert "episode" in manifest
assert "success" in manifest
assert len(split["train_demo_indices"]) == 192
assert len(scores) == 192
print("PRECHECK PASS")
PY
```

若分数 CSV 没有原始 Demo 编号列，脚本会停止。不得假设 CSV 行顺序自动对应矩阵列；先确认真实编号列，再在脚本的 `ID_COLUMNS` 中增加精确别名。

---

## 7. 交付脚本自检

```bash
python -m py_compile "$SCRIPT"
python "$SCRIPT" --self-test
```

必须输出：

```text
SELF-TEST PASS
```

否则不得运行真实数据。

---

## 8. 运行分析

```bash
set -euo pipefail

mkdir -p "$CUPID_ROOT/logs" "$CUPID_ROOT/results" "$CUPID_ROOT/frozen"

date | tee "$CUPID_ROOT/logs/stage2b_fixed_filter_start.txt"

/usr/bin/time -v python "$SCRIPT" \
  --matrix "$MATRIX" \
  --manifest "$MANIFEST" \
  --split-json "$SPLIT_JSON" \
  --official-scores "$OFFICIAL_SCORES" \
  --output-dir "$OUT" \
  --budgets 10 25 50 75 \
  --proportions 0.05 0.10 0.20 0.30 \
  --subsample-repeats 1000 \
  --bootstrap-reps 300 \
  --cross-pool-repeats 100 \
  --selection-size 50 \
  --seed 20260723 \
  2>&1 | tee "$LOG"

date | tee "$CUPID_ROOT/logs/stage2b_fixed_filter_end.txt"
```

预计使用 CPU 数秒至数分钟。

---

## 9. 检查输出

```bash
test -f "$OUT/audit.json"
test -f "$OUT/full100_demo_boundary_mapping.csv"
test -f "$OUT/boundary_summary.csv"
test -f "$OUT/subsample_deletion_frequency_long.csv"
test -f "$OUT/subsample_stability_summary.csv"
test -f "$OUT/cross_pool_conservative_core_repeats.csv"
test -f "$OUT/cross_pool_conservative_core_summary.csv"
test -f "$OUT/stage2b_decision_report.md"
test -f "$OUT/stage2b_decision.txt"

cat "$OUT/stage2b_decision.txt"
cat "$OUT/stage2b_decision_report.md"
```

---

## 10. 关键结果怎样看

### 10.1 `boundary_summary.csv`

底部 20% 重点看：

- 第 38 与第 39 条分数差；
- 差距相对于标准误差有多大；
- 边界附近有多少 Demo；
- 模糊 Demo 数量。

边界差很小、模糊 Demo 很多，说明唯一最低 38 条本来就难以识别。

### 10.2 `subsample_stability_summary.csv`

重点行：

```text
budget=50
proportion=0.20
```

看：

```text
stable_delete_count_p_ge_0.90
```

它表示在反复抽取 50 条 Rollout 时，有多少 Demo 在至少 90% 的抽样中都属于最低 38 条。

### 10.3 `cross_pool_conservative_core_summary.csv`

比较：

```text
fixed_bottom20
stable_core_p_ge_0.90
matched_size_bottom_for_p_ge_0.90
```

主要看：

- 平均选择数量；
- 在另一个独立池中仍属于最低 20% 的比例；
- 独立池平均排名位置。

若稳定核心不优于同数量简单最低分集合，则 Bootstrap 成员选择没有被证明有效。

---

## 11. 预注册决策

强通过必须满足：

- 80% 以上独立池评估能选出非空核心；
- 平均至少选出 5 条；
- 独立池准确率至少 70%；
- 相比固定最低 38 条，准确率至少提高 0.15；
- 独立池平均排名位于后 25%；
- 50 条 Rollout 时至少有 5 条稳定低分 Demo。

要证明 Bootstrap 成员选择本身有价值，还必须：

- 比相同数量的简单最低分集合至少提高 0.03。

脚本可能输出：

- `PASS_STABILITY_WEIGHTED_CORE_CANDIDATE`
- `PASS_VARIABLE_K_DIAGNOSIS_BOOTSTRAP_MEMBERSHIP_NOT_PROVEN`
- `PARTIAL_STABLE_CORE_EXISTS_SIMPLE_LOWEST_M_IS_BETTER`
- `PARTIAL_REFINE_CONSERVATIVE_CORE_OFFLINE_ONLY`
- `FAIL_SQUARE_FIXED_FILTER_BRANCH_CONSIDER_TRANSPORT`

无论哪一种，本脚本都不会启动训练。

---

## 12. 冻结结果

```bash
set -euo pipefail

mkdir -p "$FROZEN"

find "$OUT" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$FROZEN/output_sha256.txt"

sha256sum "$SCRIPT" \
  > "$FROZEN/script_sha256.txt"

sha256sum \
  "$MATRIX" \
  "$MANIFEST" \
  "$SPLIT_JSON" \
  "$OFFICIAL_SCORES" \
  > "$FROZEN/input_sha256.txt"

cp "$LOG" "$FROZEN/run.log"
cp "$OUT/stage2b_decision_report.md" "$FROZEN/"
cp "$OUT/stage2b_decision.txt" "$FROZEN/"

touch "$FROZEN/frozen.pass"
```

---

## 13. Codex 最终回复格式

```text
1. 执行状态：PASS / BLOCKED
2. 方法决策：stage2b_decision.txt 的精确字符串
3. 官方分数重建最大误差
4. 底部20%边界差与相对标准误差
5. 50条Rollout时稳定低分Demo数量
6. 稳定核心平均数量
7. 稳定核心独立池准确率
8. 固定最低38条独立池准确率
9. 同数量最低分集合独立池准确率
10. 是否允许下游训练
11. 决策报告绝对路径
12. 结果目录绝对路径
13. 冻结目录绝对路径
14. CPU墙钟时间
```

---

## 14. 交付前自检

交付脚本已经实际完成：

```text
Python语法检查：PASS
内置合成数据自检：PASS
成功/失败符号重建：PASS
原始Demo编号对齐：PASS
边界诊断：PASS
多预算、多比例频率分析：PASS
50/50双向独立池分析：PASS
同数量最低分对照：PASS
```

已专门防止三个逻辑错误：

1. 不把完整 100 条当作真实真值；
2. 不只与固定 38 条比较；
3. 不把“删得少导致精度上升”误判为 Bootstrap 方法有效。

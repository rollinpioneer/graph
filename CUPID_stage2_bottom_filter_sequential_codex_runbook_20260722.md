# CUPID Stage 2：最低百分之二十过滤边界与离线序贯停止操作文档

**版本日期：2026-07-22**  
**工作根目录：`/home/xushijie/CUPID`**  
**当前状态：最小复现 PASS；基础训练、100 条 Rollout、TRAK、100×192 Rollout-Demo 影响矩阵均已完成**  
**本阶段：纯离线方法可行性诊断，不训练策略，不新增 Rollout**

---

# 0. 给 Codex 的总指令

完整执行本文档，不要只返回命令。

必须遵守：

1. 不重新训练基础策略；
2. 不重新收集 Rollout；
3. 不修改冻结输入；
4. 不实现交叉拟合、主动采样、贝叶斯模型或其他附加方法；
5. 本轮只分析分数最低的 38 条训练 Demo；
6. 输入映射或官方分数重建不一致时立即停止；
7. 完整 100 条只能称为“有限池参考”，不能称为真实排序；
8. Bootstrap 停止规则只是原型，不得称为严格统计证书；
9. 即使 PASS，也不能自动启动重训练，只能提交下一阶段计划；
10. 保存脚本、输入、输出和日志哈希。

---

# 1. 当前流程位置

```text
基础策略训练                    已完成
固定 100 条 Rollout             已完成
TRAK 与 Demo 影响矩阵           已完成
Rollout 数量—整体排名稳定性      已完成并 PASS
最低 20% 删除名单稳定性          现在执行
离线序贯停止回放                 本文档执行
筛选后重训练                     尚未允许
Transport-MH 和多种子            更后面
```

下一步不是训练，而是直接复用：

```text
/home/xushijie/CUPID/results/influence_layers/rollout_demo_influence.npy
```

---

# 2. 为什么固定删除 38 条

训练 Demo 共 192 条。本轮冻结删除比例为百分之二十，并向下取整：

```text
192 × 20% = 38.4
固定删除 38 条
```

此前“最高百分之二十”不稳定，不能直接证明真正的过滤名单不稳定。当前必须专门分析最低 38 条。

---

# 3. 已修正的逻辑风险

## 3.1 防止成功失败符号重复使用

脚本按照官方净分数逻辑重建：

```text
成功 Rollout 的影响相加
失败 Rollout 的影响相减
```

重建结果必须与冻结的 `final_demo_scores.csv` 对齐，否则停止。这样可发现：

- 矩阵已带符号却再次乘符号；
- Rollout 行与成功标签错位；
- 使用了错误矩阵。

## 3.2 防止 192 列映射错 Demo

脚本读取实际训练划分，将矩阵列映射回原始 Demo 编号，并输出 `demo_column_mapping.csv`。

## 3.3 防止完整 100 条参考过度乐观

子集与完整 100 条共享 Rollout，结果会偏乐观。因此同时运行：

- 子集对完整 100 条的开发诊断；
- 两个互不重叠子池的一致性检查；
- 60 条顺序池对 40 条独立参考池的压力测试。

## 3.4 不夸大 Bootstrap 原型

当前规则只用于判断是否存在提前停止空间，不具备反复查看下的严格错误概率保证。

---

# 4. 固定输入与输出

```bash
export CUPID_ROOT=/home/xushijie/CUPID
export MATRIX=$CUPID_ROOT/results/influence_layers/rollout_demo_influence.npy
export OFFICIAL_SCORES=$CUPID_ROOT/results/influence_layers/final_demo_scores.csv
export MANIFEST=$CUPID_ROOT/manifests/rollout100/episode_manifest.csv
export SPLIT_JSON=$CUPID_ROOT/manifests/training_split/dataset_split.json
export SCRIPT=$CUPID_ROOT/tools/stage2_bottom_filter_sequential.py
export OUTPUT=$CUPID_ROOT/results/stage2_bottom_filter_sequential_v1
export LOG=$CUPID_ROOT/logs/stage2_bottom_filter_sequential_v1.log
```

检查真实文件：

```bash
set -euo pipefail
mkdir -p "$CUPID_ROOT/tools" "$CUPID_ROOT/logs" "$CUPID_ROOT/results" "$CUPID_ROOT/frozen"

for file in "$MATRIX" "$OFFICIAL_SCORES" "$MANIFEST" "$SPLIT_JSON"; do
  test -f "$file" || { echo "MISSING: $file"; exit 1; }
done

python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd

matrix = np.load('/home/xushijie/CUPID/results/influence_layers/rollout_demo_influence.npy', mmap_mode='r')
scores = pd.read_csv('/home/xushijie/CUPID/results/influence_layers/final_demo_scores.csv')
manifest = pd.read_csv('/home/xushijie/CUPID/manifests/rollout100/episode_manifest.csv')
split = json.loads(Path('/home/xushijie/CUPID/manifests/training_split/dataset_split.json').read_text())

print('matrix:', matrix.shape, matrix.dtype)
print('score columns:', list(scores.columns), 'rows:', len(scores))
print('manifest columns:', list(manifest.columns), 'rows:', len(manifest))
print('split keys:', list(split.keys()))
print('train demo count:', len(split.get('train_demo_indices', [])))

assert matrix.shape == (100, 192)
assert np.isfinite(matrix).all()
assert len(manifest) == 100
assert len(split['train_demo_indices']) == 192
print('PREFLIGHT PASS')
PY
```

若两个清单路径不同，只允许根据已有冻结产物改路径；不得重新生成数据或 Rollout。

---

# 5. 创建分析脚本

```bash
cat > /home/xushijie/CUPID/tools/stage2_bottom_filter_sequential.py <<'PY'
#!/usr/bin/env python3
"""CUPID Stage 2 offline diagnostic and sequential-stop prototype.

This script:
1. audits the frozen 100 x 192 rollout-demo influence matrix;
2. measures bottom-38 filtering-set stability;
3. runs an offline bootstrap stopping feasibility study.

Important: the full-100 ranking is a finite-pool reference, not ground truth.
The bootstrap stop rule is a prototype, not an anytime-valid guarantee.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCORE_COLS = ["net_influence", "net_influence_score", "net_score", "score", "demo_score", "influence", "net", "quality_score"]
ID_COLS = ["original_demo_index", "original_demo_idx", "demo_index", "demo_idx", "original_index", "train_demo_index", "train_demo_position", "index"]


@dataclass(frozen=True)
class StopProfile:
    name: str
    q10_jaccard: float
    max_ambiguous: int
    consecutive: int = 2
    min_success: int = 3
    min_failure: int = 3


PROFILES = [
    StopProfile("bootstrap_permissive", 0.75, 12),
    StopProfile("bootstrap_balanced", 0.80, 8),
    StopProfile("bootstrap_conservative", 0.85, 6),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--split-json", type=Path)
    p.add_argument("--official-scores", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--delete-k", type=int, default=38)
    p.add_argument("--budget-repeats", type=int, default=200)
    p.add_argument("--sequential-repeats", type=int, default=60)
    p.add_argument("--bootstrap-reps", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if not args.self_test:
        for name in ["matrix", "manifest", "split_json", "official_scores", "output_dir"]:
            if getattr(args, name) is None:
                p.error(f"--{name.replace('_', '-')} is required")
    return args


def parse_bool(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "succ", "success"}:
        return True
    if s in {"false", "0", "no", "fail", "failure"}:
        return False
    raise ValueError(f"Cannot parse boolean: {v!r}")


def load_matrix(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError(f"Expected 2D floating matrix, got shape={x.shape}, dtype={x.dtype}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Matrix contains NaN/Inf")
    return x.astype(np.float64, copy=False)


def load_manifest(path: Path, n: int) -> np.ndarray:
    df = pd.read_csv(path)
    if len(df) != n or not {"episode", "success"}.issubset(df.columns):
        raise ValueError("Manifest must have N rows and columns episode, success")
    df = df.copy()
    df["episode"] = pd.to_numeric(df["episode"], errors="raise").astype(int)
    df = df.sort_values("episode").reset_index(drop=True)
    if not np.array_equal(df["episode"].to_numpy(), np.arange(n)):
        raise ValueError("Manifest episodes must be exactly 0..N-1")
    success = np.array([parse_bool(v) for v in df["success"]], dtype=bool)
    if success.all() or not success.any():
        raise ValueError("Both success and failure rollouts are required")
    return success


def load_train_ids(path: Path, d: int) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    values = None
    for key in ["train_demo_indices", "train_demo_ids", "train_indices"]:
        if key in obj:
            values = obj[key]
            break
    if values is None:
        raise KeyError("No recognized train-demo list in split JSON")
    ids = np.asarray(values, dtype=int)
    if ids.shape != (d,) or len(np.unique(ids)) != d:
        raise ValueError("Train IDs must be unique and match matrix columns")
    return ids


def choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_official_scores(path: Path, train_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    score_col = choose_col(df, SCORE_COLS)
    if score_col is None:
        raise KeyError(f"No recognized score column; columns={list(df.columns)}")
    scores = pd.to_numeric(df[score_col], errors="raise").to_numpy(float)
    id_col = choose_col(df, ID_COLS)
    if id_col is None:
        if len(scores) != len(train_ids):
            raise ValueError("Score rows do not match matrix columns")
        return scores
    ids = pd.to_numeric(df[id_col], errors="raise").to_numpy(int)
    mapping = {int(i): float(s) for i, s in zip(ids, scores)}
    if len(mapping) != len(ids):
        raise ValueError("Duplicate IDs in score CSV")
    missing = [int(i) for i in train_ids if int(i) not in mapping]
    if missing:
        raise ValueError(f"Official score CSV missing IDs: {missing[:10]}")
    return np.array([mapping[int(i)] for i in train_ids], dtype=float)


def reconstruct_signed(raw: np.ndarray, success: np.ndarray, official: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    signed = raw * np.where(success, 1.0, -1.0)[:, None]
    rebuilt = signed.sum(axis=0)
    err = rebuilt - official
    stats = {
        "max_abs_error": float(np.max(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }
    if not np.allclose(rebuilt, official, rtol=2e-4, atol=2e-3):
        direct_rmse = float(np.sqrt(np.mean((raw.sum(axis=0) - official) ** 2)))
        raise ValueError(
            "Matrix semantics audit failed: success-sum minus failure-sum does not "
            f"match official scores. stats={stats}, direct_sum_rmse={direct_rmse:.6g}"
        )
    return signed, stats


def order(scores: np.ndarray, train_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((train_ids, scores))


def bottom(scores: np.ndarray, train_ids: np.ndarray, k: int) -> frozenset[int]:
    if not 0 < k < len(scores):
        raise ValueError("Invalid k")
    return frozenset(order(scores, train_ids)[:k].tolist())


def jac(a: frozenset[int], b: frozenset[int]) -> float:
    return len(a & b) / len(a | b)


def rho(a: np.ndarray, b: np.ndarray) -> float:
    value = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def stratified(rng: np.random.Generator, success: np.ndarray, budget: int) -> np.ndarray:
    si, fi = np.flatnonzero(success), np.flatnonzero(~success)
    ns = int(round(budget * len(si) / len(success)))
    ns = min(ns, len(si))
    nf = budget - ns
    if nf > len(fi):
        nf, ns = len(fi), budget - len(fi)
    chosen = np.concatenate([
        rng.choice(si, ns, replace=False),
        rng.choice(fi, nf, replace=False),
    ])
    return np.sort(chosen)


def budget_analysis(signed: np.ndarray, success: np.ndarray, ids: np.ndarray, k: int, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(signed)
    full_scores = signed.sum(axis=0)
    full_set = bottom(full_scores, ids, k)
    budgets = [5, 10, 25, 50, 75, 100]
    rng = np.random.default_rng(seed)
    rows = []
    for sampling in ["random", "stratified"]:
        for b in budgets:
            for r in range(repeats):
                if b == n:
                    idx = np.arange(n)
                elif sampling == "random":
                    idx = np.sort(rng.choice(n, b, replace=False))
                else:
                    idx = stratified(rng, success, b)
                scores = signed[idx].sum(axis=0)
                chosen = bottom(scores, ids, k)
                inter = len(chosen & full_set)
                rows.append({
                    "sampling": sampling,
                    "budget": b,
                    "repeat": r,
                    "spearman_to_full100": rho(scores, full_scores),
                    "bottom_jaccard_to_full100": jac(chosen, full_set),
                    "bottom_overlap": inter,
                    "bottom_mistakes": k - inter,
                })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["sampling", "budget"], as_index=False).agg(
        repeats=("repeat", "count"),
        spearman_mean=("spearman_to_full100", "mean"),
        spearman_std=("spearman_to_full100", "std"),
        bottom_jaccard_mean=("bottom_jaccard_to_full100", "mean"),
        bottom_jaccard_std=("bottom_jaccard_to_full100", "std"),
        bottom_overlap_mean=("bottom_overlap", "mean"),
        bottom_mistakes_mean=("bottom_mistakes", "mean"),
    ).fillna(0.0)
    return detail, summary


def disjoint_analysis(signed: np.ndarray, ids: np.ndarray, k: int, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(signed)
    rng = np.random.default_rng(seed)
    rows = []
    for b in [5, 10, 25, 50]:
        for r in range(repeats):
            idx = rng.choice(n, 2 * b, replace=False)
            sa, sb = signed[idx[:b]].sum(axis=0), signed[idx[b:]].sum(axis=0)
            aa, bb = bottom(sa, ids, k), bottom(sb, ids, k)
            inter = len(aa & bb)
            rows.append({
                "budget_per_pool": b,
                "repeat": r,
                "spearman": rho(sa, sb),
                "bottom_jaccard": jac(aa, bb),
                "bottom_overlap": inter,
                "bottom_mistakes_per_side": k - inter,
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("budget_per_pool", as_index=False).agg(
        repeats=("repeat", "count"),
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        bottom_jaccard_mean=("bottom_jaccard", "mean"),
        bottom_jaccard_std=("bottom_jaccard", "std"),
        bottom_overlap_mean=("bottom_overlap", "mean"),
        bottom_mistakes_mean=("bottom_mistakes_per_side", "mean"),
    ).fillna(0.0)
    return detail, summary


def bootstrap_diag(prefix: np.ndarray, current: frozenset[int], k: int, reps: int, rng: np.random.Generator) -> tuple[float, int]:
    t, d = prefix.shape
    weights = rng.multinomial(t, np.full(t, 1.0 / t), size=reps)
    scores = (weights @ prefix) / t
    selected = np.argpartition(scores, k - 1, axis=1)[:, :k]
    current_mask = np.zeros(d, dtype=bool)
    current_mask[list(current)] = True
    inter = current_mask[selected].sum(axis=1)
    jaccards = inter / (2 * k - inter)
    membership = np.bincount(selected.ravel(), minlength=d) / reps
    ambiguous = int(np.sum((membership >= 0.20) & (membership <= 0.80)))
    return float(np.quantile(jaccards, 0.10)), ambiguous


def run_sequence(seq: np.ndarray, seq_success: np.ndarray, ids: np.ndarray, k: int, reps: int, rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    checkpoints = list(range(10, len(seq) + 1, 5))
    if checkpoints[-1] != len(seq):
        checkpoints.append(len(seq))
    states = {p.name: {"count": 0, "result": None} for p in PROFILES}
    naive = {"count": 0, "result": None}
    previous = None
    last = None
    for b in checkpoints:
        prefix = seq[:b]
        current = bottom(prefix.mean(axis=0), ids, k)
        q10, ambiguous = bootstrap_diag(prefix, current, k, reps, rng)
        snap = {
            "budget": b,
            "selected": current,
            "success_count": int(seq_success[:b].sum()),
            "failure_count": int((~seq_success[:b]).sum()),
            "bootstrap_q10_jaccard": q10,
            "ambiguous_count": ambiguous,
        }
        last = snap
        if previous is not None:
            naive["count"] = naive["count"] + 1 if jac(current, previous) >= 0.90 else 0
            if naive["result"] is None and naive["count"] >= 2:
                naive["result"] = snap.copy()
        previous = current
        for p in PROFILES:
            ok = (
                snap["success_count"] >= p.min_success
                and snap["failure_count"] >= p.min_failure
                and q10 >= p.q10_jaccard
                and ambiguous <= p.max_ambiguous
            )
            states[p.name]["count"] = states[p.name]["count"] + 1 if ok else 0
            if states[p.name]["result"] is None and states[p.name]["count"] >= p.consecutive:
                states[p.name]["result"] = snap.copy()
    assert last is not None
    out = {"naive_consecutive": naive["result"] or last.copy()}
    for p in PROFILES:
        out[p.name] = states[p.name]["result"] or last.copy()
    return out


def summarize_seq(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("method", as_index=False).agg(
        repeats=("repeat", "count"),
        early_stop_rate=("stopped_early", "mean"),
        mean_budget=("stop_budget", "mean"),
        median_budget=("stop_budget", "median"),
        budget_std=("stop_budget", "std"),
        bottom_jaccard_mean=("bottom_jaccard", "mean"),
        bottom_jaccard_std=("bottom_jaccard", "std"),
        bottom_overlap_mean=("bottom_overlap", "mean"),
        bottom_mistakes_mean=("bottom_mistakes", "mean"),
    ).fillna(0.0)


def sequential_full(signed: np.ndarray, success: np.ndarray, ids: np.ndarray, k: int, repeats: int, bootstrap_reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(signed)
    target = bottom(signed.sum(axis=0), ids, k)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(repeats):
        perm = rng.permutation(n)
        results = run_sequence(signed[perm], success[perm], ids, k, bootstrap_reps, rng)
        for method, result in results.items():
            inter = len(result["selected"] & target)
            rows.append({
                "repeat": r,
                "method": method,
                "stop_budget": result["budget"],
                "stopped_early": result["budget"] < n,
                "bottom_jaccard": jac(result["selected"], target),
                "bottom_overlap": inter,
                "bottom_mistakes": k - inter,
                "bootstrap_q10_jaccard": result["bootstrap_q10_jaccard"],
                "ambiguous_count": result["ambiguous_count"],
            })
    detail = pd.DataFrame(rows)
    return detail, summarize_seq(detail)


def sequential_cross(signed: np.ndarray, success: np.ndarray, ids: np.ndarray, k: int, repeats: int, bootstrap_reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n, seq_n = len(signed), 60
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(repeats):
        perm = rng.permutation(n)
        seq_idx, ref_idx = perm[:seq_n], perm[seq_n:]
        seq, seq_success = signed[seq_idx], success[seq_idx]
        target = bottom(signed[ref_idx].sum(axis=0), ids, k)
        results = run_sequence(seq, seq_success, ids, k, bootstrap_reps, rng)
        for method, result in results.items():
            inter = len(result["selected"] & target)
            rows.append({
                "repeat": r,
                "method": method,
                "stop_budget": result["budget"],
                "stopped_early": result["budget"] < seq_n,
                "bottom_jaccard": jac(result["selected"], target),
                "bottom_overlap": inter,
                "bottom_mistakes": k - inter,
            })
        for b in [25, 50, 60]:
            chosen = bottom(seq[:b].sum(axis=0), ids, k)
            inter = len(chosen & target)
            rows.append({
                "repeat": r,
                "method": f"fixed_{b}",
                "stop_budget": b,
                "stopped_early": b < seq_n,
                "bottom_jaccard": jac(chosen, target),
                "bottom_overlap": inter,
                "bottom_mistakes": k - inter,
            })
    detail = pd.DataFrame(rows)
    return detail, summarize_seq(detail)


def make_report(audit: dict[str, Any], budget: pd.DataFrame, full: pd.DataFrame, cross: pd.DataFrame, out: Path) -> None:
    rb = budget[budget["sampling"] == "random"].set_index("budget")
    gain = float(rb.loc[50, "bottom_jaccard_mean"] - rb.loc[10, "bottom_jaccard_mean"])
    mistake_reduction = float(rb.loc[10, "bottom_mistakes_mean"] - rb.loc[50, "bottom_mistakes_mean"])
    background_pass = gain >= 0.15 or mistake_reduction >= 5.0
    fm, cm = full.set_index("method"), cross.set_index("method")
    primary = "bootstrap_balanced"
    full_pass = primary in fm.index and float(fm.loc[primary, "bottom_jaccard_mean"]) >= 0.85 and float(fm.loc[primary, "median_budget"]) <= 70
    cross_pass = primary in cm.index and "fixed_50" in cm.index and float(cm.loc[primary, "bottom_jaccard_mean"]) >= float(cm.loc["fixed_50", "bottom_jaccard_mean"]) - 0.05 and float(cm.loc[primary, "median_budget"]) <= 55
    if background_pass and full_pass and cross_pass:
        decision = "PASS_TO_ONE_DOWNSTREAM_RETRAIN"
    elif background_pass and (full_pass or cross_pass):
        decision = "PARTIAL_PASS_OFFLINE_ONLY"
    else:
        decision = "FAIL_STOP_SEQUENTIAL_BRANCH"
    text = f"""# CUPID Stage 2 决策报告

## 数据审计
- 矩阵：{audit['matrix_shape']}
- 成功 / 失败：{audit['success_count']} / {audit['failure_count']}
- 删除数量：{audit['delete_k']}
- 官方分数重建最大误差：{audit['reconstruction']['max_abs_error']:.8g}

## 逻辑限制
- 完整 100 条只是有限池参考，不是真实排序。
- 子集对完整 100 条的比较偏乐观，因为完整池包含子集。
- 60 条顺序池对 40 条独立参考池用于压力测试。
- Bootstrap 规则只是离线原型，不是严格的随时有效置信保证。

## 门槛
- 背景门槛：{'PASS' if background_pass else 'FAIL'}
- 最低集合 Jaccard 提升 10→50：{gain:.4f}
- 平均错误 Demo 减少：{mistake_reduction:.2f}
- 完整池开发诊断：{'PASS' if full_pass else 'FAIL'}
- 独立池压力测试：{'PASS' if cross_pass else 'FAIL'}

## 最终结论
**{decision}**

- PASS：才允许制定一次原 CUPID 与一次序贯方法的重训练。
- PARTIAL：只能继续离线改进，禁止重训练。
- FAIL：停止该分支，不用交叉拟合或主动采样强行挽救。
"""
    (out / "stage2_decision_report.md").write_text(text, encoding="utf-8")
    (out / "stage2_decision.txt").write_text(decision + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = load_matrix(args.matrix)
    if raw.shape != (100, 192) or args.delete_k != 38:
        raise ValueError(f"Frozen protocol expects matrix (100,192) and delete-k=38; got {raw.shape}, k={args.delete_k}")
    success = load_manifest(args.manifest, len(raw))
    ids = load_train_ids(args.split_json, raw.shape[1])
    official = load_official_scores(args.official_scores, ids)
    signed, stats = reconstruct_signed(raw, success, official)
    audit = {
        "matrix_shape": list(raw.shape),
        "success_count": int(success.sum()),
        "failure_count": int((~success).sum()),
        "delete_k": args.delete_k,
        "reconstruction": stats,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    full_scores = signed.sum(axis=0)
    ranks = np.empty(len(ids), dtype=int)
    ranks[order(full_scores, ids)] = np.arange(1, len(ids) + 1)
    pd.DataFrame({
        "matrix_column": np.arange(len(ids)),
        "original_demo_index": ids,
        "official_net_score": official,
        "reconstructed_net_score": full_scores,
        "rank_ascending": ranks,
        "delete_bottom38": ranks <= args.delete_k,
    }).to_csv(out / "demo_column_mapping.csv", index=False)

    bd, bs = budget_analysis(signed, success, ids, args.delete_k, args.budget_repeats, args.seed)
    dd, ds = disjoint_analysis(signed, ids, args.delete_k, args.budget_repeats, args.seed + 1)
    fd, fs = sequential_full(signed, success, ids, args.delete_k, args.sequential_repeats, args.bootstrap_reps, args.seed + 2)
    cd, cs = sequential_cross(signed, success, ids, args.delete_k, args.sequential_repeats, args.bootstrap_reps, args.seed + 3)
    bd.to_csv(out / "bottom_budget_repeats.csv", index=False)
    bs.to_csv(out / "bottom_budget_summary.csv", index=False)
    dd.to_csv(out / "disjoint_pair_repeats.csv", index=False)
    ds.to_csv(out / "disjoint_pair_summary.csv", index=False)
    fd.to_csv(out / "sequential_full_reference_repeats.csv", index=False)
    fs.to_csv(out / "sequential_full_reference_summary.csv", index=False)
    cd.to_csv(out / "sequential_cross_pool_repeats.csv", index=False)
    cs.to_csv(out / "sequential_cross_pool_summary.csv", index=False)
    make_report(audit, bs, fs, cs, out)
    print("AUDIT", json.dumps(audit, indent=2))
    print("\nBOTTOM BUDGET\n", bs.to_string(index=False))
    print("\nDISJOINT\n", ds.to_string(index=False))
    print("\nFULL REFERENCE\n", fs.to_string(index=False))
    print("\nCROSS POOL\n", cs.to_string(index=False))
    print("\nSaved to", out)


def self_test() -> None:
    rng = np.random.default_rng(123)
    n, d, k = 100, 20, 4
    success = np.array([True] * 70 + [False] * 30)
    rng.shuffle(success)
    truth = np.linspace(-3, 3, d)
    signed = truth[None, :] + rng.normal(0, 0.7, size=(n, d))
    raw = signed * np.where(success, 1.0, -1.0)[:, None]
    official = signed.sum(axis=0)
    ids = np.arange(100, 100 + d)
    rebuilt, stats = reconstruct_signed(raw, success, official)
    assert np.allclose(rebuilt, signed)
    assert stats["max_abs_error"] < 1e-10
    assert bottom(official, ids, k) == frozenset(range(k))
    _, bs = budget_analysis(signed, success, ids, k, 8, 1)
    hundred = bs[bs["budget"] == 100]
    assert np.allclose(hundred["spearman_mean"], 1.0)
    assert np.allclose(hundred["bottom_jaccard_mean"], 1.0)
    _, ds = disjoint_analysis(signed, ids, k, 4, 2)
    assert set(ds["budget_per_pool"]) == {5, 10, 25, 50}
    seq = run_sequence(signed, success, ids, k, 50, np.random.default_rng(3))
    assert "bootstrap_balanced" in seq
    assert all(10 <= v["budget"] <= 100 and len(v["selected"]) == k for v in seq.values())
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        np.save(p / "m.npy", raw.astype(np.float32))
        pd.DataFrame({"episode": np.arange(n), "success": success}).to_csv(p / "manifest.csv", index=False)
        (p / "split.json").write_text(json.dumps({"train_demo_indices": ids.tolist()}))
        pd.DataFrame({"original_demo_index": ids, "score": official.astype(np.float32)}).to_csv(p / "scores.csv", index=False)
        m = load_matrix(p / "m.npy")
        s = load_manifest(p / "manifest.csv", n)
        ti = load_train_ids(p / "split.json", d)
        oscore = load_official_scores(p / "scores.csv", ti)
        reconstruct_signed(m, s, oscore)
    print("SELF-TEST PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()

PY
chmod +x /home/xushijie/CUPID/tools/stage2_bottom_filter_sequential.py
```

---

# 6. 代码自检

```bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cupid

python -m py_compile "$SCRIPT"
python "$SCRIPT" --self-test | tee "$CUPID_ROOT/logs/stage2_self_test.log"
grep -q 'SELF-TEST PASS' "$CUPID_ROOT/logs/stage2_self_test.log"
```

自检覆盖：

- 成功失败符号重建；
- 最低集合方向；
- 完整预算自一致；
- 不重叠子池；
- 序贯输出预算和集合大小；
- NPY、CSV、JSON 输入解析。

若失败，不得运行真实数据。

---

# 7. 冻结脚本和输入

```bash
export FREEZE=$CUPID_ROOT/frozen/stage2_bottom_filter_sequential_v1
mkdir -p "$FREEZE"
sha256sum "$SCRIPT" | tee "$FREEZE/script_sha256.txt"
cp "$SCRIPT" "$FREEZE/"
sha256sum "$MATRIX" "$OFFICIAL_SCORES" "$MANIFEST" "$SPLIT_JSON" \
  | tee "$FREEZE/input_sha256.txt"
```

---

# 8. 执行真实离线实验

```bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cupid

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

if [ -e "$OUTPUT" ]; then
  mv "$OUTPUT" "${OUTPUT}_archived_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT"

date | tee "$CUPID_ROOT/logs/stage2_start.txt"
/usr/bin/time -v python "$SCRIPT" \
  --matrix "$MATRIX" \
  --manifest "$MANIFEST" \
  --split-json "$SPLIT_JSON" \
  --official-scores "$OFFICIAL_SCORES" \
  --output-dir "$OUTPUT" \
  --delete-k 38 \
  --budget-repeats 200 \
  --sequential-repeats 60 \
  --bootstrap-reps 150 \
  --seed 20260722 \
  2>&1 | tee "$LOG"
date | tee "$CUPID_ROOT/logs/stage2_end.txt"
```

本轮只用 CPU。不得降低重复次数后继续使用同一门槛。

---

# 9. 结果自检

```bash
python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

out = Path('/home/xushijie/CUPID/results/stage2_bottom_filter_sequential_v1')
required = [
    'audit.json', 'demo_column_mapping.csv',
    'bottom_budget_repeats.csv', 'bottom_budget_summary.csv',
    'disjoint_pair_repeats.csv', 'disjoint_pair_summary.csv',
    'sequential_full_reference_repeats.csv', 'sequential_full_reference_summary.csv',
    'sequential_cross_pool_repeats.csv', 'sequential_cross_pool_summary.csv',
    'stage2_decision_report.md', 'stage2_decision.txt',
]
for name in required:
    assert (out / name).exists(), name

audit = json.loads((out / 'audit.json').read_text())
assert audit['matrix_shape'] == [100, 192]
assert audit['success_count'] == 71
assert audit['failure_count'] == 29
assert audit['delete_k'] == 38
assert audit['reconstruction']['max_abs_error'] <= 2e-3

mapping = pd.read_csv(out / 'demo_column_mapping.csv')
assert len(mapping) == 192
assert mapping['original_demo_index'].nunique() == 192
assert mapping['delete_bottom38'].astype(bool).sum() == 38

budget = pd.read_csv(out / 'bottom_budget_summary.csv')
hundred = budget[budget['budget'] == 100]
assert len(hundred) == 2
assert np.allclose(hundred['spearman_mean'], 1.0)
assert np.allclose(hundred['bottom_jaccard_mean'], 1.0)
assert np.allclose(hundred['bottom_mistakes_mean'], 0.0)

for name in ['disjoint_pair_summary.csv', 'sequential_full_reference_summary.csv', 'sequential_cross_pool_summary.csv']:
    frame = pd.read_csv(out / name)
    assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all(), name

print('POST-RUN AUDIT PASS')
PY
```

---

# 10. 查看结果

```bash
cat "$OUTPUT/stage2_decision_report.md"
for file in \
  bottom_budget_summary.csv \
  disjoint_pair_summary.csv \
  sequential_full_reference_summary.csv \
  sequential_cross_pool_summary.csv; do
  echo "===== $file ====="
  column -s, -t < "$OUTPUT/$file" || cat "$OUTPUT/$file"
done
```

重点指标：

- `bottom_jaccard_mean`：最低 38 条名单重合；
- `bottom_mistakes_mean`：38 条中平均选错几条；
- `median_budget`：停止所需 Rollout 中位数；
- `early_stop_rate`：提前停止比例。

主原型：`bootstrap_balanced`。  
基线：`naive_consecutive`、`fixed_25`、`fixed_50`、`fixed_60`。

---

# 11. 决策门槛

## 背景门槛

满足任一：

- 最低 38 条的 Jaccard 从 10 条到 50 条提高至少 0.15；
- 平均选错 Demo 数从 10 条到 50 条减少至少 5。

## 完整池开发诊断

`bootstrap_balanced`：

- 平均 Jaccard 不低于 0.85；
- 中位预算不超过 70。

该结果偏乐观，不能单独通过。

## 独立池压力测试

60 条顺序池对 40 条独立参考池：

- 主原型平均 Jaccard 不低于固定 50 条减 0.05；
- 主原型中位预算不超过 55。

## 自动状态

- `PASS_TO_ONE_DOWNSTREAM_RETRAIN`：允许**制定**两次下游重训练计划，但不能自动训练；
- `PARTIAL_PASS_OFFLINE_ONLY`：只能继续离线调整停止规则；
- `FAIL_STOP_SEQUENTIAL_BRANCH`：停止该分支，不叠加复杂模块挽救。

---

# 12. 本轮不能证明什么

本轮不能证明：

- 100 条排名是真实排名；
- Bootstrap 是严格置信区间；
- 提前停止后的筛选数据一定提高策略；
- Square-MH 最低 38 条是真正有害数据；
- 方法可直接迁移到 Transport-MH。

---

# 13. 冻结输出

```bash
mkdir -p "$FREEZE/results"
find "$OUTPUT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$FREEZE/results/all_files_sha256.txt"
du -sh "$OUTPUT" | tee "$FREEZE/results/disk_usage.txt"
cp "$LOG" "$FREEZE/results/"
```

---

# 14. Codex 最终回复格式

```text
1. 最终状态
2. 输入矩阵语义审计是否通过
3. 官方 Demo 分数重建最大误差
4. 10、25、50、75 条下最低 38 条随机 Jaccard
5. 10、25、50 条不重叠子池 Jaccard
6. bootstrap_balanced 的完整池平均 Jaccard和中位预算
7. bootstrap_balanced 的独立池平均 Jaccard和中位预算
8. 固定 50 条的独立池平均 Jaccard
9. 是否允许制定下游重训练计划
10. 决策报告绝对路径
11. 结果目录绝对路径
12. CPU 墙钟时间
```

不得只回复“执行完成”。

---

# 15. 交付前自检结论

已实际执行：

```text
Python 语法检查：PASS
合成数据内置自检：PASS
成功失败符号重建：PASS
最低集合方向：PASS
完整预算自一致：PASS
不重叠子池采样：PASS
序贯输出范围：PASS
CSV、JSON、NPY 解析：PASS
```

尚无法预先验证的只有真实服务器文件列名和真实数据结果。脚本已兼容常见列名。若 `final_demo_scores.csv` 使用未识别列名，只允许在脚本顶部 `SCORE_COLS` 或 `ID_COLS` 增加实际列名；随后必须重新运行语法检查、`--self-test` 并重新计算脚本哈希。

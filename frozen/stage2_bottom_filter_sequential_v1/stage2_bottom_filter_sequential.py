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
ID_COLS = ["dataset_demo_index", "original_demo_index", "original_demo_idx", "demo_index", "demo_idx", "original_index", "train_demo_index", "train_demo_position", "index"]


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

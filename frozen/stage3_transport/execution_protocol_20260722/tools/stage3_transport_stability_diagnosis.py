#!/usr/bin/env python3
"""CUPID Stage 3 Transport-MH offline stability diagnosis.

This script does NOT train a policy, collect rollouts, or recompute TRAK.
It answers:
1) Is the fixed bottom-38 boundary flat?
2) Is there a smaller set of demos that stays low-scoring across rollout subsets?
3) Does a bootstrap-stable core outperform a simple same-size lowest-score set
   on an independent rollout pool?

The 100-rollout pool is finite. Outputs are diagnostics, not ground truth or
an anytime-valid statistical certificate.
"""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = (
    "net_influence", "score", "demo_score",
    "influence", "net", "quality_score",
)
ID_COLUMNS = (
    "dataset_demo_index", "original_demo_index", "demo_index", "demo_idx",
    "original_index", "train_demo_index", "index",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--split-json", type=Path)
    p.add_argument("--official-scores", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 75])
    p.add_argument(
        "--proportions", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.30],
    )
    p.add_argument("--subsample-repeats", type=int, default=1000)
    p.add_argument("--bootstrap-reps", type=int, default=300)
    p.add_argument("--cross-pool-repeats", type=int, default=100)
    p.add_argument("--selection-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if not args.self_test:
        required = (
            "matrix", "manifest", "split_json",
            "official_scores", "output_dir",
        )
        missing = [x for x in required if getattr(args, x) is None]
        if missing:
            p.error("Missing arguments: %s" % missing)
    return args


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "succ", "success"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failure"}:
        return False
    raise ValueError("Cannot parse boolean: %r" % value)


def choose_column(frame, candidates, kind):
    lower = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(
        "No recognized %s column. Available: %s"
        % (kind, list(frame.columns))
    )


def load_inputs(matrix_path, manifest_path, split_path, score_path):
    raw = np.asarray(np.load(matrix_path, allow_pickle=False))
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError("Matrix must be a 2D floating array.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Matrix contains NaN or Inf.")

    manifest = pd.read_csv(manifest_path)
    if len(manifest) != raw.shape[0]:
        raise ValueError("Manifest row count does not match matrix.")
    if "episode" not in manifest or "success" not in manifest:
        raise ValueError("Manifest needs episode and success columns.")
    manifest = manifest.copy()
    manifest["episode"] = pd.to_numeric(
        manifest["episode"], errors="raise"
    ).astype(int)
    manifest = manifest.sort_values("episode").reset_index(drop=True)
    if not np.array_equal(
        manifest["episode"].to_numpy(), np.arange(raw.shape[0])
    ):
        raise ValueError("Episode IDs must be exactly 0..N-1.")
    success = np.array(
        [parse_bool(x) for x in manifest["success"]], dtype=bool
    )
    if not success.any() or success.all():
        raise ValueError("Both success and failure rollouts are required.")

    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    ids = None
    for key in ("train_demo_indices", "train_demo_ids", "train_indices"):
        if key in split:
            ids = np.asarray(split[key], dtype=int)
            break
    if ids is None or ids.shape != (raw.shape[1],):
        raise ValueError("Training-demo IDs do not match matrix columns.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Training-demo IDs are not unique.")

    score_frame = pd.read_csv(score_path)
    score_col = choose_column(score_frame, SCORE_COLUMNS, "score")
    id_col = choose_column(score_frame, ID_COLUMNS, "original demo ID")
    score_ids = pd.to_numeric(
        score_frame[id_col], errors="raise"
    ).to_numpy(dtype=int)
    score_values = pd.to_numeric(
        score_frame[score_col], errors="raise"
    ).to_numpy(dtype=float)

    mapping = {}
    for demo_id, score in zip(score_ids, score_values):
        demo_id = int(demo_id)
        if demo_id in mapping:
            raise ValueError("Duplicate demo ID in score file: %d" % demo_id)
        mapping[demo_id] = float(score)
    missing = [int(x) for x in ids if int(x) not in mapping]
    if missing:
        raise ValueError("Score file missing IDs: %s" % missing[:10])
    official = np.array([mapping[int(x)] for x in ids], dtype=float)

    return raw.astype(np.float64), success, ids, official, score_col, id_col


def reconstruct_signed(raw, success, official):
    signed = raw * np.where(success, 1.0, -1.0)[:, None]
    rebuilt = signed.sum(axis=0)
    error = rebuilt - official
    max_abs = float(np.max(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    direct_rmse = float(
        np.sqrt(np.mean((raw.sum(axis=0) - official) ** 2))
    )
    if not np.allclose(rebuilt, official, rtol=2e-4, atol=2e-3):
        raise ValueError(
            "Matrix semantic audit failed. "
            "success-minus-failure does not reproduce official scores. "
            "max_abs=%.6g rmse=%.6g direct_sum_rmse=%.6g"
            % (max_abs, rmse, direct_rmse)
        )
    return signed, {
        "max_abs_error": max_abs,
        "rmse": rmse,
        "direct_sum_rmse": direct_rmse,
    }


def stable_order(scores, demo_ids):
    # Ascending score; original demo ID breaks ties deterministically.
    return np.lexsort((demo_ids, scores))


def proportion_to_k(proportion, demo_count):
    if not 0.0 < proportion < 1.0:
        raise ValueError("Proportion must be in (0,1).")
    return max(
        1,
        min(demo_count - 1, int(round(proportion * demo_count))),
    )


def bottom_indices(scores, demo_ids, k):
    if not 0 < k < len(scores):
        raise ValueError("Invalid k=%d." % k)
    return stable_order(scores, demo_ids)[:k]


def bootstrap_membership(pool, k, repetitions, rng):
    """Bottom-k membership frequency under ordinary rollout bootstrap."""
    n, d = pool.shape
    counts = rng.multinomial(
        n, np.full(n, 1.0 / n), size=repetitions
    )
    boot_scores = (counts @ pool) / float(n)
    selected = np.argpartition(
        boot_scores, kth=k - 1, axis=1
    )[:, :k]
    frequency = np.bincount(
        selected.ravel(), minlength=d
    ).astype(float)
    return frequency / float(repetitions)


def boundary_audit(signed, demo_ids, proportions, bootstrap_reps, seed):
    n, d = signed.shape
    scores = signed.mean(axis=0)
    standard_error = signed.std(axis=0, ddof=1) / math.sqrt(n)
    order = stable_order(scores, demo_ids)
    ranks = np.empty(d, dtype=int)
    ranks[order] = np.arange(1, d + 1)

    detail = pd.DataFrame({
        "matrix_column": np.arange(d),
        "original_demo_index": demo_ids,
        "full100_mean_signed_influence": scores,
        "standard_error": standard_error,
        "ascending_rank": ranks,
    })

    rng = np.random.default_rng(seed)
    rows = []
    for proportion in proportions:
        k = proportion_to_k(float(proportion), d)
        last_delete = int(order[k - 1])
        first_keep = int(order[k])
        gap = float(scores[first_keep] - scores[last_delete])

        pair = signed[:, first_keep] - signed[:, last_delete]
        pair_se = float(pair.std(ddof=1) / math.sqrt(n))
        gap_over_se = gap / pair_se if pair_se > 0 else float("inf")

        boundary = 0.5 * (
            scores[last_delete] + scores[first_keep]
        )
        distance = np.abs(scores - boundary) / np.maximum(
            standard_error, 1e-12
        )

        membership = bootstrap_membership(
            signed, k, bootstrap_reps, rng
        )
        detail[
            "bootstrap_delete_probability_%.2f" % float(proportion)
        ] = membership

        rows.append({
            "proportion": float(proportion),
            "delete_k": k,
            "last_delete_original_demo": int(demo_ids[last_delete]),
            "first_keep_original_demo": int(demo_ids[first_keep]),
            "full100_boundary_gap": gap,
            "pairwise_gap_standard_error": pair_se,
            "gap_divided_by_standard_error": gap_over_se,
            "demos_within_one_standard_error_of_boundary": int(
                np.sum(distance <= 1.0)
            ),
            "demos_within_two_standard_errors_of_boundary": int(
                np.sum(distance <= 2.0)
            ),
            "bootstrap_stable_delete_count_p_ge_0.90": int(
                np.sum(membership >= 0.90)
            ),
            "bootstrap_stable_keep_count_p_le_0.10": int(
                np.sum(membership <= 0.10)
            ),
            "bootstrap_ambiguous_count": int(
                np.sum(
                    (membership > 0.10)
                    & (membership < 0.90)
                )
            ),
        })

    return detail, pd.DataFrame(rows)


def subsample_frequency(
    signed, demo_ids, budgets, proportions, repeats, seed
):
    n, d = signed.shape
    rng = np.random.default_rng(seed)
    long_rows = []
    summary_rows = []

    for budget in sorted(set(int(x) for x in budgets)):
        if not 1 <= budget <= n:
            raise ValueError("Invalid budget %d." % budget)

        counters = {
            float(p): np.zeros(d, dtype=int) for p in proportions
        }
        for _ in range(repeats):
            selected = rng.choice(n, budget, replace=False)
            order = stable_order(
                signed[selected].mean(axis=0), demo_ids
            )
            for proportion in proportions:
                p = float(proportion)
                k = proportion_to_k(p, d)
                counters[p][order[:k]] += 1

        for proportion in proportions:
            p = float(proportion)
            k = proportion_to_k(p, d)
            frequency = counters[p] / float(repeats)

            summary_rows.append({
                "budget": budget,
                "proportion": p,
                "delete_k": k,
                "stable_delete_count_p_ge_0.90": int(
                    np.sum(frequency >= 0.90)
                ),
                "stable_keep_count_p_le_0.10": int(
                    np.sum(frequency <= 0.10)
                ),
                "ambiguous_count_p_between_0.10_and_0.90": int(
                    np.sum(
                        (frequency > 0.10)
                        & (frequency < 0.90)
                    )
                ),
                "strongly_ambiguous_count_p_between_0.20_and_0.80": int(
                    np.sum(
                        (frequency >= 0.20)
                        & (frequency <= 0.80)
                    )
                ),
            })

            for column, demo_id, value in zip(
                np.arange(d), demo_ids, frequency
            ):
                long_rows.append({
                    "budget": budget,
                    "proportion": p,
                    "delete_k": k,
                    "matrix_column": int(column),
                    "original_demo_index": int(demo_id),
                    "deletion_frequency": float(value),
                })

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def evaluate_selection(selected, target_scores, demo_ids, target_k):
    selected = np.asarray(selected, dtype=int)
    selected_set = set(selected.tolist())
    target_set = set(
        bottom_indices(target_scores, demo_ids, target_k).tolist()
    )

    if not selected_set:
        return {
            "selected_count": 0.0,
            "precision_against_independent_bottom20": np.nan,
            "recall_against_independent_bottom20": 0.0,
            "jaccard_against_independent_bottom20": 0.0,
            "mean_independent_rank_percentile": np.nan,
            "negative_score_rate_in_independent_pool": np.nan,
        }

    intersection = len(selected_set & target_set)
    union = len(selected_set | target_set)

    order = stable_order(target_scores, demo_ids)
    ranks = np.empty(len(target_scores), dtype=int)
    ranks[order] = np.arange(1, len(target_scores) + 1)
    selected_array = np.array(sorted(selected_set), dtype=int)

    return {
        "selected_count": float(len(selected_set)),
        "precision_against_independent_bottom20": (
            intersection / float(len(selected_set))
        ),
        "recall_against_independent_bottom20": (
            intersection / float(target_k)
        ),
        "jaccard_against_independent_bottom20": (
            intersection / float(union)
        ),
        "mean_independent_rank_percentile": float(
            np.mean(ranks[selected_array] / float(len(target_scores)))
        ),
        "negative_score_rate_in_independent_pool": float(
            np.mean(target_scores[selected_array] < 0.0)
        ),
    }


def cross_pool_core(
    signed,
    demo_ids,
    target_proportion,
    selection_size,
    repeats,
    bootstrap_reps,
    seed,
):
    n, d = signed.shape
    if not 20 <= selection_size <= n - 20:
        raise ValueError(
            "selection-size must leave at least 20 rollouts per pool."
        )
    target_k = proportion_to_k(target_proportion, d)
    rng = np.random.default_rng(seed)
    rows = []

    for repeat in range(repeats):
        permutation = rng.permutation(n)
        first = permutation[:selection_size]
        second = permutation[selection_size:]

        for direction, source, target in (
            ("A_to_B", signed[first], signed[second]),
            ("B_to_A", signed[second], signed[first]),
        ):
            source_scores = source.mean(axis=0)
            target_scores = target.mean(axis=0)
            membership = bootstrap_membership(
                source, target_k, bootstrap_reps, rng
            )

            methods = [
                (
                    "fixed_bottom20",
                    bottom_indices(
                        source_scores, demo_ids, target_k
                    ),
                ),
                (
                    "fixed_bottom10",
                    bottom_indices(
                        source_scores,
                        demo_ids,
                        proportion_to_k(0.10, d),
                    ),
                ),
            ]

            for threshold in (0.80, 0.90, 0.95):
                core = np.flatnonzero(membership >= threshold)
                methods.append((
                    "stable_core_p_ge_%.2f" % threshold,
                    core,
                ))
                matched = (
                    bottom_indices(
                        source_scores, demo_ids, len(core)
                    )
                    if len(core) > 0
                    else np.array([], dtype=int)
                )
                methods.append((
                    "matched_size_bottom_for_p_ge_%.2f"
                    % threshold,
                    matched,
                ))

            for method, selected in methods:
                metrics = evaluate_selection(
                    selected, target_scores, demo_ids, target_k
                )
                rows.append({
                    "repeat": repeat,
                    "direction": direction,
                    "source_pool_size": len(source),
                    "target_pool_size": len(target),
                    "method": method,
                    **metrics,
                })

    detail = pd.DataFrame(rows)

    def nonempty_rate(series):
        return float(np.mean(series.to_numpy(dtype=float) > 0.0))

    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            evaluations=("repeat", "count"),
            nonempty_rate=("selected_count", nonempty_rate),
            selected_count_mean=("selected_count", "mean"),
            selected_count_median=("selected_count", "median"),
            precision_mean=(
                "precision_against_independent_bottom20",
                "mean",
            ),
            precision_std=(
                "precision_against_independent_bottom20",
                "std",
            ),
            recall_mean=(
                "recall_against_independent_bottom20",
                "mean",
            ),
            jaccard_mean=(
                "jaccard_against_independent_bottom20",
                "mean",
            ),
            independent_rank_percentile_mean=(
                "mean_independent_rank_percentile",
                "mean",
            ),
            independent_negative_score_rate_mean=(
                "negative_score_rate_in_independent_pool",
                "mean",
            ),
        )
        .fillna(0.0)
    )
    return detail, summary


def decide(subsample_summary, cross_summary):
    table = cross_summary.set_index("method")
    required = (
        "stable_core_p_ge_0.90",
        "matched_size_bottom_for_p_ge_0.90",
        "fixed_bottom20",
    )
    if any(name not in table.index for name in required):
        return "FAIL_INPUT_OR_OUTPUT_MISSING", {}

    core = table.loc["stable_core_p_ge_0.90"]
    matched = table.loc[
        "matched_size_bottom_for_p_ge_0.90"
    ]
    fixed = table.loc["fixed_bottom20"]

    stable50 = subsample_summary[
        (subsample_summary["budget"] == 50)
        & np.isclose(
            subsample_summary["proportion"], 0.20
        )
    ]
    stable_count = (
        int(
            stable50.iloc[0][
                "stable_delete_count_p_ge_0.90"
            ]
        )
        if len(stable50) == 1 else 0
    )

    gain_fixed = float(
        core["precision_mean"] - fixed["precision_mean"]
    )
    gain_matched = float(
        core["precision_mean"] - matched["precision_mean"]
    )

    diagnostics = {
        "core90_nonempty_rate": float(core["nonempty_rate"]),
        "core90_selected_count_mean": float(
            core["selected_count_mean"]
        ),
        "core90_precision_mean": float(core["precision_mean"]),
        "fixed20_precision_mean": float(
            fixed["precision_mean"]
        ),
        "matched90_precision_mean": float(
            matched["precision_mean"]
        ),
        "precision_gain_over_fixed20": gain_fixed,
        "precision_gain_over_matched_size": gain_matched,
        "core90_independent_rank_percentile_mean": float(
            core["independent_rank_percentile_mean"]
        ),
        "stable_delete_count_at_budget50_bottom20": stable_count,
    }

    robust_core = (
        float(core["nonempty_rate"]) >= 0.80
        and float(core["selected_count_mean"]) >= 5.0
        and float(core["precision_mean"]) >= 0.70
        and gain_fixed >= 0.15
        and float(
            core["independent_rank_percentile_mean"]
        ) <= 0.25
        and stable_count >= 5
    )

    if robust_core and gain_matched >= 0.03:
        return "PASS_STABILITY_WEIGHTED_CORE_CANDIDATE", diagnostics
    if robust_core and gain_matched >= -0.02:
        return (
            "PASS_VARIABLE_K_DIAGNOSIS_"
            "BOOTSTRAP_MEMBERSHIP_NOT_PROVEN",
            diagnostics,
        )
    if robust_core:
        return (
            "PARTIAL_STABLE_CORE_EXISTS_"
            "SIMPLE_LOWEST_M_IS_BETTER",
            diagnostics,
        )

    if (
        "stable_core_p_ge_0.80" in table.index
        and "matched_size_bottom_for_p_ge_0.80"
        in table.index
    ):
        core80 = table.loc["stable_core_p_ge_0.80"]
        gain80 = float(
            core80["precision_mean"]
            - fixed["precision_mean"]
        )
        diagnostics.update({
            "core80_nonempty_rate": float(
                core80["nonempty_rate"]
            ),
            "core80_selected_count_mean": float(
                core80["selected_count_mean"]
            ),
            "core80_precision_mean": float(
                core80["precision_mean"]
            ),
            "core80_precision_gain_over_fixed20": gain80,
        })
        if (
            float(core80["nonempty_rate"]) >= 0.80
            and float(core80["selected_count_mean"]) >= 5.0
            and float(core80["precision_mean"]) >= 0.65
            and gain80 >= 0.10
        ):
            return (
                "PARTIAL_REFINE_CONSERVATIVE_CORE_"
                "OFFLINE_ONLY",
                diagnostics,
            )

    return (
        "FAIL_TRANSPORT_FILTER_BRANCH_STOP",
        diagnostics,
    )


def write_report(
    output_dir,
    audit,
    boundary_summary,
    subsample_summary,
    cross_summary,
    decision,
    diagnostics,
):
    boundary20 = boundary_summary[
        np.isclose(boundary_summary["proportion"], 0.20)
    ]
    stable50 = subsample_summary[
        (subsample_summary["budget"] == 50)
        & np.isclose(
            subsample_summary["proportion"], 0.20
        )
    ]

    report = """# CUPID Stage 3 Transport-MH 稳定性决策报告

## 输入审计

- Rollout：{n}
- Demo：{d}
- 成功 / 失败：{succ} / {fail}
- 官方分数重建最大误差：{err:.8g}

## 重要限制

- 完整 100 条只是有限池参考，不是真实价值。
- 独立 50/50 池也有噪声，因此只用于压力测试。
- Bootstrap 删除概率是离线稳定性指标，不是严格置信证书。
- 本阶段不把低分 Demo 直接解释成有害 Demo。
- 本阶段不允许自动启动重训练。

## 底部 20% 边界

```text
{boundary}
```

## 50 条 Rollout 下的底部 20% 稳定性

```text
{stable}
```

## 50/50 独立池结果

```text
{cross}
```

## 决策诊断

```json
{diag}
```

## 最终决策

**{decision}**
""".format(
        n=audit["num_rollouts"],
        d=audit["num_demos"],
        succ=audit["success_count"],
        fail=audit["failure_count"],
        err=audit["reconstruction"]["max_abs_error"],
        boundary=boundary20.to_string(index=False),
        stable=stable50.to_string(index=False),
        cross=cross_summary.to_string(index=False),
        diag=json.dumps(
            diagnostics, indent=2, ensure_ascii=False
        ),
        decision=decision,
    )

    (output_dir / "stage2b_decision_report.md").write_text(
        report, encoding="utf-8"
    )
    (output_dir / "stage2b_decision.txt").write_text(
        decision + "\n", encoding="utf-8"
    )


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)

    raw, success, demo_ids, official, score_col, id_col = load_inputs(
        args.matrix,
        args.manifest,
        args.split_json,
        args.official_scores,
    )
    signed, reconstruction = reconstruct_signed(
        raw, success, official
    )

    if raw.shape != (100, 192):
        raise ValueError(
            "Expected frozen matrix shape (100,192), got %s."
            % (raw.shape,)
        )
    if int(success.sum()) < 5 or int((~success).sum()) < 5:
        raise ValueError(
            "Transport class-balance gate requires at least 5/5, got %d/%d."
            % (int(success.sum()), int((~success).sum()))
        )

    audit = {
        "matrix_path": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "split_json_path": str(args.split_json.resolve()),
        "split_json_sha256": sha256(args.split_json),
        "official_scores_path": str(
            args.official_scores.resolve()
        ),
        "official_scores_sha256": sha256(
            args.official_scores
        ),
        "num_rollouts": int(raw.shape[0]),
        "num_demos": int(raw.shape[1]),
        "success_count": int(success.sum()),
        "failure_count": int((~success).sum()),
        "official_score_column": score_col,
        "official_id_column": id_col,
        "reconstruction": reconstruction,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    detail, boundary = boundary_audit(
        signed,
        demo_ids,
        args.proportions,
        args.bootstrap_reps,
        args.seed,
    )
    detail.to_csv(
        args.output_dir / "full100_demo_boundary_mapping.csv",
        index=False,
    )
    boundary.to_csv(
        args.output_dir / "boundary_summary.csv",
        index=False,
    )

    frequency, frequency_summary = subsample_frequency(
        signed,
        demo_ids,
        args.budgets,
        args.proportions,
        args.subsample_repeats,
        args.seed + 1,
    )
    frequency.to_csv(
        args.output_dir
        / "subsample_deletion_frequency_long.csv",
        index=False,
    )
    frequency_summary.to_csv(
        args.output_dir / "subsample_stability_summary.csv",
        index=False,
    )

    cross_detail, cross_summary = cross_pool_core(
        signed,
        demo_ids,
        target_proportion=0.20,
        selection_size=args.selection_size,
        repeats=args.cross_pool_repeats,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed + 2,
    )
    cross_detail.to_csv(
        args.output_dir
        / "cross_pool_conservative_core_repeats.csv",
        index=False,
    )
    cross_summary.to_csv(
        args.output_dir
        / "cross_pool_conservative_core_summary.csv",
        index=False,
    )

    decision, diagnostics = decide(
        frequency_summary, cross_summary
    )
    write_report(
        args.output_dir,
        audit,
        boundary,
        frequency_summary,
        cross_summary,
        decision,
        diagnostics,
    )

    print("\nAUDIT")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\nBOUNDARY SUMMARY")
    print(boundary.to_string(index=False))
    print("\nSUBSAMPLE SUMMARY")
    print(frequency_summary.to_string(index=False))
    print("\nCROSS-POOL SUMMARY")
    print(cross_summary.to_string(index=False))
    print("\nDECISION:", decision)
    print("OUTPUT:", args.output_dir)


def self_test():
    rng = np.random.default_rng(1234)
    n, d = 100, 20
    success = np.zeros(n, dtype=bool)
    success[:70] = True
    rng.shuffle(success)

    means = np.concatenate([
        np.full(4, -2.5),
        np.linspace(-0.4, 0.4, 8),
        np.full(8, 2.0),
    ])
    signed = (
        means[None, :]
        + rng.normal(0.0, 0.7, size=(n, d))
    )
    raw = signed * np.where(
        success, 1.0, -1.0
    )[:, None]
    official = signed.sum(axis=0)
    demo_ids = np.arange(100, 100 + d)

    rebuilt, stats = reconstruct_signed(
        raw, success, official
    )
    assert np.allclose(rebuilt, signed)
    assert stats["max_abs_error"] < 1e-10

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        np.save(temp / "matrix.npy", raw.astype(np.float32))
        pd.DataFrame({
            "episode": np.arange(n),
            "success": success,
        }).to_csv(temp / "manifest.csv", index=False)
        (temp / "split.json").write_text(
            json.dumps({
                "train_demo_indices": demo_ids.tolist()
            }),
            encoding="utf-8",
        )
        pd.DataFrame({
            "original_demo_index": demo_ids,
            "score": official.astype(np.float32),
        }).to_csv(temp / "scores.csv", index=False)

        loaded = load_inputs(
            temp / "matrix.npy",
            temp / "manifest.csv",
            temp / "split.json",
            temp / "scores.csv",
        )
        assert loaded[0].shape == (n, d)

    mapping, boundary = boundary_audit(
        signed, demo_ids, [0.20], 60, 1
    )
    assert len(mapping) == d
    assert len(boundary) == 1

    long_frame, summary = subsample_frequency(
        signed, demo_ids, [25, 50], [0.20], 40, 2
    )
    assert len(long_frame) == 2 * d
    assert long_frame["deletion_frequency"].between(
        0.0, 1.0
    ).all()

    cross_detail, cross_summary = cross_pool_core(
        signed, demo_ids, 0.20, 50, 8, 50, 3
    )
    assert len(cross_detail) > 0
    methods = set(cross_summary["method"])
    assert "stable_core_p_ge_0.90" in methods
    assert (
        "matched_size_bottom_for_p_ge_0.90"
        in methods
    )

    print("SELF-TEST PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()

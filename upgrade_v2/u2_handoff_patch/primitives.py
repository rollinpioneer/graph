"""Small, dependency-light primitives used by the U2 handoff patch.

The functions in this module deliberately operate on already materialised
arrays and rows.  They do not know about checkpoints or invoke a model, which
makes the audit path deterministic and safe to run on CPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_inventory(paths: Iterable[Path], root: Path | None = None) -> dict[str, Any]:
    """Return a deterministic digest for a finite collection of input files.

    The returned aggregate hash covers both each path and each file digest.  It
    is deliberately a compact provenance record: result-status JSON files can
    prove the exact cached inputs used without embedding thousands of hashes.
    Missing paths are returned separately rather than silently dropped.
    """

    digest = hashlib.sha256()
    present: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in sorted({Path(value) for value in paths}, key=lambda value: str(value)):
        try:
            label = str(path.resolve().relative_to(root.resolve())) if root else str(path.resolve())
        except ValueError:
            label = str(path.resolve())
        if not path.is_file():
            missing.append(label)
            continue
        file_digest = sha256_file(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        present.append({"path": label, "size_bytes": path.stat().st_size, "sha256": file_digest})
    return {
        "file_count": len(present),
        "sha256": digest.hexdigest(),
        "files": present,
        "missing_paths": missing,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    materialised = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in materialised:
            keys.update(row)
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialised)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _better(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Whether score *left* is preferred to score *right*.

    A score is ``(number_of_matches, total_absolute_error)``.  Matching count
    is the primary objective, then error.  The DP uses this ordering rather
    than a greedy nearest-neighbour rule, so an early prediction cannot steal
    a gold event needed by a later prediction.
    """

    return left[0] > right[0] or (left[0] == right[0] and left[1] < right[1])


def match_events(predicted_times: Sequence[int], gold_times: Sequence[int], tolerance: int) -> dict[str, Any]:
    """Match sorted event positions one-to-one within ``tolerance``.

    The dynamic program is monotone over time.  For ordered events and an
    absolute-distance objective, an optimal matching always has no crossings,
    so the compact ``O(n*m)`` table is sufficient.  Inputs are sorted here as
    a defensive measure; duplicate positions remain separate observations but
    each can match at most one counterpart.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    predicted = sorted(int(x) for x in predicted_times)
    gold = sorted(int(x) for x in gold_times)
    n, m = len(predicted), len(gold)
    score: list[list[tuple[int, int]]] = [[(0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    decision: list[list[str]] = [["" for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = score[i - 1][j]
            best_decision = "skip_pred"
            if _better(score[i][j - 1], best):
                best, best_decision = score[i][j - 1], "skip_gold"
            distance = abs(predicted[i - 1] - gold[j - 1])
            if distance <= tolerance:
                candidate = (score[i - 1][j - 1][0] + 1, score[i - 1][j - 1][1] + distance)
                if _better(candidate, best) or candidate == best:
                    best, best_decision = candidate, "match"
            score[i][j] = best
            decision[i][j] = best_decision

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        action = decision[i][j]
        if action == "match":
            pairs.append((predicted[i - 1], gold[j - 1]))
            i -= 1
            j -= 1
        elif action == "skip_gold":
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    errors = [abs(pred - target) for pred, target in pairs]
    tp = len(pairs)
    return {
        "tp": tp,
        "fp": len(predicted) - tp,
        "fn": len(gold) - tp,
        "errors": errors,
        "pairs": pairs,
        "predicted_support": len(predicted),
        "gold_support": len(gold),
    }


def metric_from_counts(tp: int, fp: int, fn: int, errors: Sequence[float] = ()) -> dict[str, Any]:
    denom_p = tp + fp
    denom_r = tp + fn
    precision = tp / denom_p if denom_p else None
    recall = tp / denom_r if denom_r else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    error_values = [float(x) for x in errors]
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": (sum(error_values) / len(error_values)) if error_values else None,
        "error_sum": float(sum(error_values)),
        "matched_count": len(error_values),
        "estimability": "estimable" if denom_p or denom_r else "not_estimable",
    }


def incoming_segment_return(phi: Sequence[float], start: int, end: int) -> float:
    """Return all stored transitions entering observations ``[start, end]``.

    The first stored observation has no recorded predecessor, hence ``t=0``
    contributes no separately fabricated transition.  For a singleton at
    ``t>0`` the return is ``phi[t]-phi[t-1]`` rather than zero.
    """

    if not 0 <= start <= end < len(phi):
        raise ValueError("invalid segment bounds")
    predecessor = max(start - 1, 0)
    return float(phi[end] - phi[predecessor])

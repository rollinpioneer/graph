from __future__ import annotations

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None

def conservative_hit_rate(rows, key):
    return mean(1.0 if r[key] else 0.0 for r in rows)
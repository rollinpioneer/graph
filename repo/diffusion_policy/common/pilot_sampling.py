"""Frozen per-demo allocation construction for the D25 dev pilot."""
import hashlib
import json
from pathlib import Path
import numpy as np

# D25 freeze-1b (CONTRACT.json, normalized path).  Keep this beside the
# deterministic score construction so every generated allocation artifact
# identifies the exact preregistration it implements.
CONTRACT_SHA256 = "21e2ebffc3caf2f223b63407fbec841b240e4344fa3f7f01c3a9d2ca4af95e18"
SCORE_SEEDS = {"R1": 20260726001, "R2": 20260726002}

def calibrated_softmax(scores, kappa=5.0):
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = 1e-8, 1e8
    target = float(kappa)
    for _ in range(120):
        tau = (lo + hi) / 2
        z = (scores - scores.max()) / tau
        p = np.exp(z); p /= p.sum()
        if p.max() / p.min() > target:
            lo = tau
        else:
            hi = tau
    tau = (lo + hi) / 2
    z = (scores - scores.max()) / tau
    p = np.exp(z); p /= p.sum()
    return tau, p

def random_score_target(arm, n_demo):
    if arm not in SCORE_SEEDS:
        raise ValueError(f"unsupported pilot arm: {arm}")
    rng = np.random.default_rng(SCORE_SEEDS[arm])
    scores = rng.normal(size=n_demo)
    tau, p = calibrated_softmax(scores)
    return scores, tau, p

def sha256_array(x):
    return hashlib.sha256(np.asarray(x, dtype=np.float64).tobytes()).hexdigest()

def write_target_artifact(directory, arm, scores, tau, p):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arm}_target.json"
    payload = {
        "contract_sha256": CONTRACT_SHA256, "arm": arm,
        "score_seed": SCORE_SEEDS[arm], "kappa": 5.0,
        "tau": float(tau), "n_demo": int(len(p)),
        "score_sha256": sha256_array(scores), "target_sha256": sha256_array(p),
        "scores": np.asarray(scores).tolist(), "p": np.asarray(p).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path

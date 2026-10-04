"""Generate the frozen ALFWorld prior cache (needs DASHSCOPE_API_KEY in the environment; never printed).

  smoke : python scripts/alfworld_prior_cache.py --smoke 10 --out runs/alfworld_prior_reliance/prior_smoke.json
  full  : python scripts/alfworld_prior_cache.py --out runs/alfworld_prior_reliance/prior_cache.json [--resume]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cp_disr.platforms.alfworld import data  # noqa: E402
from cp_disr.platforms.alfworld.prior_provider import DashScopeTextProvider, cache_report, generate_cache  # noqa: E402


def all_keys(rows):
    return sorted({(r["goal_otype"], tuple(sorted({x["rtype"] for x in r["receptacles"]}))) for r in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/alfworld_prior_reliance")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.7-flash-2026-07-15")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-parse", action="store_true")
    ap.add_argument("--rpm", type=int, default=55)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.rpm > 55:
        raise SystemExit("rpm must be <= 55")
    rows = data.load_catalog(os.path.join(a.run_dir, "catalog.json"))
    keys = all_keys(rows)
    if a.smoke:
        step = max(1, len(keys) // a.smoke)
        keys = keys[::step][: a.smoke]
    prov = DashScopeTextProvider(model=a.model)
    if not prov.key_present():
        raise SystemExit("DASHSCOPE_API_KEY not set in the environment")
    cache = generate_cache(keys, prov, a.out, rpm=a.rpm, workers=a.workers, resume=a.resume, retry_parse=a.retry_parse)
    rep = cache_report(cache)
    rep["n_distinct_keys_requested"] = len(keys)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

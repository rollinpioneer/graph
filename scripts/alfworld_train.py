"""One ALFWorld training run: python scripts/alfworld_train.py --method Full --seed 0 --prior CACHE.json ..."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cp_disr.platforms.alfworld import data  # noqa: E402
from cp_disr.platforms.alfworld.prior_provider import PriorCache, SyntheticPrior  # noqa: E402
from cp_disr.platforms.alfworld.trainer import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["B2", "A_STAT", "Full", "A_CAT", "PRIOR_BIAS"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default="runs/alfworld_prior_reliance")
    ap.add_argument("--data-root", default="/home/xushijie2/xsj2_alf/data")
    ap.add_argument("--prior", default=None, help="frozen prior cache JSON; omit only together with --synthetic-prior")
    ap.add_argument("--synthetic-prior", action="store_true", help="ENGINEERING throughput run only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-total", type=int, default=2048)
    ap.add_argument("--rollout", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=10**9)
    ap.add_argument("--eval-repeats", type=int, default=2)
    ap.add_argument("--save-at", type=int, nargs="*", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    splits = json.load(open(a.run_dir + "/splits.json"))
    tables = data.Tables(json.load(open(a.run_dir + "/tables.json")))
    prior = SyntheticPrior() if a.synthetic_prior else PriorCache.load(a.prior)
    games = splits["train"][: a.max_games or None]
    cfg = dict(method=a.method, seed=a.seed, games=games, eval_games=splits["dev"], tables=tables, prior=prior, data_root=a.data_root,
               out_dir=a.out, n_total=a.n_total, rollout=a.rollout, eval_every=a.eval_every, eval_repeats=a.eval_repeats, save_at=a.save_at, device=a.device)
    train(cfg)


if __name__ == "__main__":
    main()

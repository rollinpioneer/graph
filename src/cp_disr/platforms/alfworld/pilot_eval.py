"""Evaluation-episode recorder for the E2-rescue pilot (same rollout logic as trainer.run_episode in eval mode,
plus CHECK count, prior-induced TV and the per-decision residual distribution). Learning code is untouched."""
import torch

from ...torch_rl import prefix_hidden
from .adapter import AlfEpisode
from .pddl_contracts import candidate_key
from .snapshot import build_snapshot
from .trainer import ENV_ID, H, full_prior_edges


def eval_episode_record(policy, game_path, tables, prior, game, repeat, prior_mode="original"):
    ep = AlfEpisode(game_path, tables)
    prefix, decision, n_checks = [], 0, 0
    tv, spread, absmean = [], [], []
    try:
        pub = ep.public()
        edges = full_prior_edges(pub, prior, tables) if prior_mode == "original" else ()
        snap = build_snapshot(pub, edges, ENV_ID, "eval|%s|%d" % (game, repeat), decision)
        while pub.legal and not ep.done:
            with torch.no_grad():
                hidden = prefix_hidden(policy, prefix)
                out = policy(snap, hidden)
                policy.shadow_zero_prior = True
                shadow = policy(snap, hidden)
                policy.shadow_zero_prior = False
                tv.append(0.5 * float((out.distribution.probs - shadow.distribution.probs).abs().sum()))
                res = [float(v) for v in out.diagnostics["delta"].values()]
                if res:
                    spread.append(max(res) - min(res))
                    absmean.append(sum(abs(v) for v in res) / len(res))
            cand, _ = out.select(False)
            kind, target = candidate_key(next(c for c in snap.template.contracts if c.id == cand))
            ep.execute(kind, target)
            n_checks += int(kind == "CHECK")
            prefix.append(snap)
            pub = ep.public()
            decision += 1
            snap = build_snapshot(pub, edges, ENV_ID, "eval|%s|%d" % (game, repeat), decision)
        return {"game": game, "repeat": repeat, "prior_mode": prior_mode, "won": bool(ep.won), "raw_steps": ep.raw_steps,
                "J": (2.0 ** (-ep.success_step / H) if ep.won else 0.0), "n_checks": n_checks, "decisions": decision,
                "tv_mean": (sum(tv) / len(tv) if tv else None), "resid_spread": spread, "resid_abs_mean": absmean}
    finally:
        ep.close()

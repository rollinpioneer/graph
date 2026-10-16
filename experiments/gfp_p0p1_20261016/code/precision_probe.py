"""How much does GPU matmul precision move Q and the guidance weight g? (no training)

On A100, JAX's default float32 matmul precision may use TF32. The guidance is
g = sigmoid((lam/eta) * (Q_data - Q_ref)) with lam/eta ~ 1/(0.001*|Q|) ~ 6 for |Q|~160,
so a Q error of 0.1 moves the sigmoid argument by ~0.6. This compares Q and g on
the probe's fixed dataset states under the default precision and under 'highest'.

usage: python precision_probe.py STATES_NPZ OUT_JSON NAME:STEP [...]
"""
import os
import sys
import json

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from common import load_agent

ETA = 1e-3
CHUNK = 5000


def qs(agent, obs, act, z):
    a_pi = jnp.clip(agent.network.select("actor")(obs, z), -1, 1)
    a_flow = agent.compute_flow_actions(obs, z)
    return dict(Qt_pi=agent.critic_eval(obs, a_pi), Qt_flow=agent.critic_eval(obs, a_flow),
                Qt_data=agent.critic_eval(obs, act),
                Qo_pi=agent.network.select("critic")(obs, actions=a_pi).mean(0), a_pi=a_pi)


def run(agent, m, precision):
    f = jax.jit(qs)
    out = []
    with jax.default_matmul_precision(precision):
        for k in range(0, len(m["obs"]), CHUNK):
            out.append(jax.device_get(f(agent, m["obs"][k:k + CHUNK], m["act"][k:k + CHUNK],
                                        m["z"][k:k + CHUNK])))
    r = {key: np.concatenate([o[key] for o in out]) for key in out[0]}
    lam = 1.0 / np.abs(r["Qo_pi"]).mean()
    x = (lam / ETA) * (r["Qt_data"] - np.minimum(r["Qt_pi"], r["Qt_flow"]))
    r["g"] = 1 / (1 + np.exp(-np.clip(x, -60, 60)))
    return r


def main():
    m = dict(np.load(sys.argv[1]))
    out = sys.argv[2]
    res = {}
    for spec in sys.argv[3:]:
        name, step = spec.split(":")
        agent, _, _ = load_agent(name, int(step))
        a = run(agent, m, "default")
        b = run(agent, m, "highest")
        d = {}
        for k in ("Qt_pi", "Qt_flow", "Qt_data"):
            d[f"{k}_mean_abs_diff"] = float(np.abs(a[k] - b[k]).mean())
            d[f"{k}_p99_abs_diff"] = float(np.quantile(np.abs(a[k] - b[k]), 0.99))
        dg = np.abs(a["g"] - b["g"])
        d.update(g_default_mean=float(a["g"].mean()), g_highest_mean=float(b["g"].mean()),
                 g_mean_abs_diff=float(dg.mean()), g_frac_diff_gt_0p1=float((dg > 0.1).mean()),
                 g_frac_diff_gt_0p25=float((dg > 0.25).mean()),
                 g_corr=float(np.corrcoef(a["g"], b["g"])[0, 1]),
                 a_pi_mean_l2_diff=float(np.linalg.norm(a["a_pi"] - b["a_pi"], axis=1).mean()))
        res[spec] = d
        print(spec, json.dumps(d), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

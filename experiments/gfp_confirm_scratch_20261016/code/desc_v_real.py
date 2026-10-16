"""Stage-2 confirmation, card section 4.3 (descriptive only): start-region V at the 1M checkpoint vs the realized
discounted return of the 1M training-time evaluation, both arms.

The runs have no frozen evaluation with per-episode records, so the realized return is rebuilt from eval.csv
(success rate p and mean episode length at 1M), with the critic_check.py convention: reward -1 per step, 0 on the
success step, discount 0.995, failures run 2000 steps (G = -(1 - 0.995**2000) / 0.005 = -199.99).
Successful episodes get one common length Ls = (mean_len - (1 - p) * 2000) / p. G is concave in the length, so this
overstates the success return slightly (Jensen); it is a description, not a judgement.
V_start is the scan's start-region mean of Q(s, clip(actor(s, z))) over the fixed probe, not the exact init state.

usage: python desc_v_real.py SCAN_JSON OUT_JSON RUN_DIR:NAME [RUN_DIR:NAME ...]
"""
import sys
import csv
import json

GAMMA, H = 0.995, 2000


def disc(T, r_last):
    return -(1 - GAMMA ** (T - 1)) / (1 - GAMMA) + r_last * GAMMA ** (T - 1)


def main():
    scan = json.load(open(sys.argv[1]))
    out = {}
    for spec in sys.argv[3:]:
        rd, name = spec.split(":")
        row = [r for r in csv.DictReader(open(f"{rd}/eval.csv")) if int(float(r["step"])) == 1000000][0]
        p, L = float(row["Actor/success"]), float(row["Actor/episode.length"])
        g_fail = disc(H, -1.0)
        g = (1 - p) * g_fail
        Ls = None
        if p > 0:
            Ls = (L - (1 - p) * H) / p
            g += p * disc(Ls, 0.0)
        v = scan[f"{name}:1000000"]["V_start"]
        out[name] = dict(success_1M=p, mean_len_1M=L, Ls=Ls, G_real=g, V_start=v, gap=v - g)
        print(name, json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in out[name].items()}))
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()

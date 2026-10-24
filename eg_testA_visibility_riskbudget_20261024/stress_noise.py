"""Exploratory stress test (C3): how much of the risk-budget advantage survives when the friction signal is noisy?
Noise is added to I1 itself, so both L3 and the risk-budget rule see the same corrupted friction estimate.
usage: python3 -I stress_noise.py <per_pose_dir> <i1_dir> <frozen_models.json> <out_dir>"""
import sys, json, glob, os
import numpy as np

PP, I1D, FROZEN, OUT = sys.argv[1:5]
FM = json.load(open(FROZEN)); os.makedirs(OUT, exist_ok=True)


def cyc(nom):
    k = np.arange(12)[None, :]
    return np.minimum((k - nom[:, None]) % 12, (nom[:, None] - k) % 12)


rows = []
for f in sorted(glob.glob(f'{PP}/C3_*.npz')):
    a = np.load(f); s, fr = map(int, os.path.basename(f)[:-4].split('_')[1:])
    v = np.load(f'{I1D}/{s:04d}_{fr:04d}.npz')
    d = {k: a[k] for k in ['original_angle', 'collision', 'empty', 'mu', 'ref_occ']}; d['I1'] = v['I1']
    rows.append(d)
A = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
N = len(A['original_angle']); IX = np.arange(N); NOM = A['original_angle']; DIST = cyc(NOM)
OLDMU = A['mu'][IX, NOM]; GOOD = (OLDMU > 0) & (OLDMU <= .4); WEAK = OLDMU > .4
FIN = np.isfinite(A['I1'])


def outcomes(ang):
    mu = A['mu'][IX, ang]; co = A['collision'][IX, ang]; em = A['empty'][IX, ang]
    feas = ~co & ~em & (mu > 0) & (mu <= .4)
    return np.array([feas.sum(), (WEAK & feas).sum(), (GOOD & ~feas).sum(), (co & ~em).sum()])


def risk(name):
    m = FM[name]; X = [(DIST == j).astype(float) for j in range(1, 7)]
    if name == 'R_do':
        v = A['ref_occ']; C = np.stack([v, v - v[IX, NOM][:, None]], -1)
        C = (C - np.array(m['mean'])) / np.array(m['std']); X += [C[..., 0], C[..., 1]]
    return 1 / (1 + np.exp(-(np.stack(X, -1) @ np.array(m['w']))))


P = {k: risk(k) for k in ['R_d', 'R_do']}
TIEKEY = DIST * 12 + np.arange(12)[None, :]; TIEKEY[IX, NOM] = -1
ORDKEY = DIST * 12 + np.arange(12)[None, :]
LAMS = np.concatenate([[0.0], np.logspace(-5, 0, 251)])


def l3(I):
    valid = (DIST <= 1) & FIN
    mn = np.min(np.where(valid, I, np.inf), 1)
    ties = valid & np.isclose(I, mn[:, None], rtol=0, atol=1e-7)
    return np.argmin(np.where(ties, TIEKEY, 999), 1)


def budget(I, p, lam):
    sc = (I[IX, NOM][:, None] - I) - lam * p
    sc[~FIN] = -np.inf; sc[IX, NOM] = 0.0
    best = sc.max(1); cand = (sc == best[:, None]) & (sc > 0)
    return np.where(cand.any(1), np.argmin(np.where(cand, ORDKEY, 999), 1), NOM)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# noise scale: std of the true per-variant gain over all valid non-original variants
m = FIN.copy(); m[IX, NOM] = False
TRUE_GAIN = (A['I1'][IX, NOM][:, None] - A['I1'])
SD = float(np.nanstd(TRUE_GAIN[m]))
rng = np.random.default_rng(20261024)
DRAWS = 20
out = []
for rel in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    acc = {k: [] for k in ['L3', 'R_d', 'R_do', 'R_do_lamC3']}; cs = []
    for t in range(DRAWS if rel > 0 else 1):
        I = A['I1'] + (rng.normal(0, rel * SD, A['I1'].shape) if rel > 0 else 0)
        cs.append(corr((I[IX, NOM][:, None] - I)[m], TRUE_GAIN[m]))
        oL = outcomes(l3(I)); acc['L3'].append(oL); target = oL[1]
        for name in ['R_d', 'R_do']:
            best = None
            for lam in LAMS:
                o = outcomes(budget(I, P[name], lam)); g = abs(o[1] - target)
                if best is None or g <= best[0]: best = (g, o)
            acc[name].append(best[1])
        acc['R_do_lamC3'].append(outcomes(budget(I, P['R_do'], FM['R_do']['C3_matched_lambda'])))
    row = dict(noise_rel_sd=rel, gain_corr_with_truth=float(np.mean(cs)))
    for k, v in acc.items():
        mv = np.mean(v, 0); row[k] = dict(feasible=float(mv[0]), recovery=float(mv[1]), harm=float(mv[2]), collision=float(mv[3]))
    out.append(row)
    print(rel, round(row['gain_corr_with_truth'], 3), {k: [round(x, 1) for x in row[k].values()] for k in acc})
json.dump(dict(gain_sd=SD, draws=DRAWS, rows=out, note='exploratory; columns feasible/recovery/harm/collision; matched to L3 recovery under the same noise'),
          open(f'{OUT}/stress_noise.json', 'w'), indent=1)

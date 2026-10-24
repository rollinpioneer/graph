"""Frozen confirmation of the risk-budget rule on an unseen frame split (design doc section 8).
usage: python3 -I confirm_split.py <per_pose_dir> <i1_dir> <split> <frozen_models.json> <out_dir>
No fitting happens here: coefficients come from frozen_models.json (training frames only)."""
import sys, json, glob, os
import numpy as np

PP, I1D, SPLIT, FROZEN, OUT = sys.argv[1:6]
SEED = 20261024
os.makedirs(OUT, exist_ok=True)
FM = json.load(open(FROZEN))


def cyc(nom):
    k = np.arange(12)[None, :]
    return np.minimum((k - nom[:, None]) % 12, (nom[:, None] - k) % 12)


rows = []
for f in sorted(glob.glob(f'{PP}/{SPLIT}_*.npz')):
    a = np.load(f)
    s, fr = map(int, os.path.basename(f)[:-4].split('_')[1:])
    v = np.load(f'{I1D}/{s:04d}_{fr:04d}.npz')
    for k in ['candidate_id', 'original_angle', 'mu', 'collision', 'empty']:
        assert np.array_equal(a[k], v[k]), (s, fr, k)
    d = {k: a[k] for k in ['candidate_id', 'original_angle', 'collision', 'empty', 'mu', 'ref_occ', 'R1', 'R3']}
    d['I1'] = v['I1']; d['scene'] = np.full(len(d['candidate_id']), s); d['frame'] = np.full(len(d['candidate_id']), fr)
    rows.append(d)
A = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
ix = np.arange(len(A['original_angle'])); nom = A['original_angle']
clean = ~A['collision'][ix, nom] & ~A['empty'][ix, nom]
A = {k: v[clean] for k, v in A.items()}
N = len(A['original_angle']); IX = np.arange(N); NOM = A['original_angle']; DIST = cyc(NOM)
OLDMU = A['mu'][IX, NOM]; GOOD = (OLDMU > 0) & (OLDMU <= .4); WEAK = OLDMU > .4
orig_i1_missing = int((~np.isfinite(A['I1'][IX, NOM])).sum())
SCENES = np.array(sorted(set(A['scene']))); SI = np.searchsorted(SCENES, A['scene'])
rng = np.random.default_rng(SEED)
BOOT = np.concatenate([rng.choice(np.flatnonzero((SCENES >= lo) & (SCENES < hi)), (10000, 4), replace=True)
                       for lo, hi in [(100, 130), (130, 160), (160, 190)]], axis=1)


def outcomes(ang):
    mu = A['mu'][IX, ang]; co = A['collision'][IX, ang]; em = A['empty'][IX, ang]
    feas = ~co & ~em & (mu > 0) & (mu <= .4)
    return dict(feasible=feas, recovery=WEAK & feas, harm=GOOD & ~feas, collision=co & ~em, empty=em,
                coll_noobs=co & ~em & (A['R3'][IX, ang] == 0))


def counts(o): return {k: int(v.sum()) for k, v in o.items()}
def rate(x): return np.bincount(SI, weights=x.astype(float), minlength=len(SCENES)) / np.bincount(SI, minlength=len(SCENES))
def interval(x):
    b = x[BOOT].mean(1); return dict(mean=float(x.mean()), low=float(np.quantile(b, .025)), high=float(np.quantile(b, .975)))


TIEKEY = DIST * 12 + np.arange(12)[None, :]; TIEKEY[IX, NOM] = -1
def z0_choose(allowed):
    valid = allowed & np.isfinite(A['I1'])
    mn = np.min(np.where(valid, A['I1'], np.inf), 1)
    ties = valid & np.isclose(A['I1'], mn[:, None], rtol=0, atol=1e-7)
    out = np.argmin(np.where(ties, TIEKEY, 999), 1); fb = ~valid.any(1); out[fb] = NOM[fb]
    return out


def risk(name):
    m = FM[name]; X = [(DIST == j).astype(float) for j in range(1, 7)]
    extra = {'R_do': 'ref_occ', 'R_de': 'R1'}.get(name)
    if extra:
        v = A[extra].astype(float); C = np.stack([v, v - v[IX, NOM][:, None]], -1)
        C = (C - np.array(m['mean'])) / np.array(m['std']); X = X + [C[..., 0], C[..., 1]]
    z = np.stack(X, -1) @ np.array(m['w'])
    return 1 / (1 + np.exp(-z))


GAIN = A['I1'][IX, NOM][:, None] - A['I1']
ORDKEY = DIST * 12 + np.arange(12)[None, :]
def choose(p, lam):
    sc = GAIN - lam * p
    sc[~np.isfinite(A['I1'])] = -np.inf
    sc[IX, NOM] = 0.0
    best = sc.max(1); cand = (sc == best[:, None]) & (sc > 0)
    out = np.where(cand.any(1), np.argmin(np.where(cand, ORDKEY, 999), 1), NOM)
    out[~np.isfinite(A['I1'][IX, NOM])] = NOM[~np.isfinite(A['I1'][IX, NOM])]
    return out


L3 = z0_choose(DIST <= 1); oL3 = outcomes(L3); TARGET = int(oL3['recovery'].sum())
LAMS = np.concatenate([[0.0], np.logspace(-5, 0, 501)])
res = dict(split=SPLIT, candidates_clean=N, weak=int(WEAK.sum()), good=int(GOOD.sum()), orig_I1_missing=orig_i1_missing,
           L3=counts(oL3), target_recovery=TARGET)
sel = {}
for name in ['R_d', 'R_do', 'R_de']:
    p = risk(name)
    rec = [int(outcomes(choose(p, l))['recovery'].sum()) for l in LAMS]
    gap = np.abs(np.array(rec) - TARGET); i = np.flatnonzero(gap == gap.min())[-1]
    o = outcomes(choose(p, LAMS[i])); sel[name] = o
    ot = outcomes(choose(p, FM[name]['C3_matched_lambda']))
    res[name] = dict(matched_lambda=float(LAMS[i]), matched=counts(o), transferred_lambda=FM[name]['C3_matched_lambda'], transferred=counts(ot))
    sel[name + '_transferred'] = ot
cmp = {}
for a, b in [('R_do', 'L3'), ('R_do', 'R_d'), ('R_d', 'L3'), ('R_de', 'R_d'), ('R_do', 'R_de'), ('R_do_transferred', 'L3')]:
    oa = sel[a]; ob = oL3 if b == 'L3' else sel[b]
    cmp[f'{a}_minus_{b}'] = {m: interval(rate(oa[m]) - rate(ob[m])) for m in ['collision', 'harm', 'feasible', 'recovery']}
res['comparisons'] = cmp
cd, cl, co = res['R_d']['matched']['collision'], res['L3']['collision'], res['R_do']['matched']['collision']
hd, hl = res['R_do']['matched']['harm'], res['L3']['harm']
P1 = (co <= 0.6 * cl) and (hd < hl) and (cmp['R_do_minus_L3']['collision']['high'] < 0)
P2 = (co <= 0.8 * cd) and (cmp['R_do_minus_R_d']['collision']['high'] < 0)
res['gate_P1'] = bool(P1); res['gate_P2'] = bool(P2)
res['verdict'] = 'CONFIRMED' if (P1 and P2) else 'PARTIAL' if P1 else 'NOT-CONFIRMED'
json.dump(res, open(f'{OUT}/confirm_{SPLIT}.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ['split', 'candidates_clean', 'L3', 'target_recovery', 'R_d', 'R_do', 'R_de', 'gate_P1', 'gate_P2', 'verdict']}, indent=1))

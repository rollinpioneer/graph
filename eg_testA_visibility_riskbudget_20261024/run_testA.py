"""Test A: does visibility make angle changes safer? CPU-only, frozen protocol (design doc section 7)."""
import sys, json, glob, os, hashlib
import numpy as np

ROOT = sys.argv[1]
OUT = sys.argv[2]
PP = f'{ROOT}/eg_n0_nref_main_20261024/per_pose'
I1D = f'{ROOT}/eg_physical_diagnostics_20261024/eg_i1_angle_physics_20261024/evaluation'
SEED = 20261024
os.makedirs(OUT, exist_ok=True)


def cyc(nom):
    k = np.arange(12)[None, :]
    return np.minimum((k - nom[:, None]) % 12, (nom[:, None] - k) % 12)


def load(split):
    rows = []
    for f in sorted(glob.glob(f'{PP}/{split}_*.npz')):
        a = np.load(f)
        s, fr = map(int, os.path.basename(f)[:-4].split('_')[1:])
        d = {k: a[k] for k in ['candidate_id', 'original_angle', 'collision', 'empty', 'mu', 'ref_unk', 'ref_occ', 'R3']}
        if split == 'C3':
            v = np.load(f'{I1D}/{s:04d}_{fr:04d}.npz')
            for k in ['candidate_id', 'original_angle', 'mu', 'collision', 'empty']:
                assert np.array_equal(d[k], v[k]), (s, fr, k)
            d['I1'] = v['I1']
        d['scene'] = np.full(len(d['candidate_id']), s)
        rows.append(d)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def features(A, kind, stats=None):
    n = len(A['original_angle']); ix = np.arange(n); nom = A['original_angle']
    D = cyc(nom)
    u = A['ref_unk']; o = A['ref_occ']
    du = u - u[ix, nom][:, None]; do = o - o[ix, nom][:, None]
    cols = [(D == j).astype(float) for j in range(1, 7)]
    cont = []
    if kind in ('dv', 'dvo'):
        cont += [u, du]
    if kind == 'dvo':
        cont += [o, do]
    X = np.stack(cols + cont, -1)  # (n,12,F)
    nd = len(cols)
    if stats is None and cont:
        flat = X[..., nd:].reshape(-1, len(cont))
        stats = (flat.mean(0), flat.std(0) + 1e-12)
    if cont:
        X[..., nd:] = (X[..., nd:] - stats[0]) / stats[1]
    return X, stats


def fit_logit(X, y, l2=1e-3, iters=100):
    # one-hot distance columns act as per-distance intercepts; no separate intercept needed
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + pen * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def auroc(score, y):
    order = np.argsort(score, kind='mergesort'); s = score[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    r = np.empty(len(s)); r[order] = ranks
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def variant_mask(A, clean_only=True):
    n = len(A['original_angle']); ix = np.arange(n); nom = A['original_angle']
    m = np.ones((n, 12), bool); m[ix, nom] = False
    m &= ~A['empty']
    if clean_only:
        clean = ~A['collision'][ix, nom] & ~A['empty'][ix, nom]
        m &= clean[:, None]
    return m


TR = load('train'); C3 = load('C3')
N = len(C3['original_angle']); IX = np.arange(N); NOM = C3['original_angle']; DIST = cyc(NOM)
OLDMU = C3['mu'][IX, NOM]; GOOD = (OLDMU > 0) & (OLDMU <= .4); WEAK = OLDMU > .4
assert N == 1229 and GOOD.sum() == 641 and WEAK.sum() == 588
SCENES = np.array(sorted(set(C3['scene']))); SI = np.searchsorted(SCENES, C3['scene'])
rng = np.random.default_rng(SEED)
BOOT = np.concatenate([rng.choice(np.flatnonzero((SCENES >= lo) & (SCENES < hi)), (10000, 4), replace=True)
                       for lo, hi in [(100, 130), (130, 160), (160, 190)]], axis=1)


def outcomes(ang):
    mu = C3['mu'][IX, ang]; co = C3['collision'][IX, ang]; em = C3['empty'][IX, ang]
    feas = ~co & ~em & (mu > 0) & (mu <= .4)
    return dict(feasible=feas, recovery=WEAK & feas, harm=GOOD & ~feas, collision=co & ~em, empty=em,
                coll_noobs=co & ~em & (C3['R3'][IX, ang] == 0), coll_obs=co & ~em & (C3['R3'][IX, ang] > 0))


def counts(o):
    return {k: int(v.sum()) for k, v in o.items()}


def scene_rate(x):
    return np.bincount(SI, weights=x.astype(float), minlength=len(SCENES)) / np.bincount(SI, minlength=len(SCENES))


def interval(x):
    b = x[BOOT].mean(1)
    return dict(mean=float(x.mean()), low=float(np.quantile(b, .025)), high=float(np.quantile(b, .975)))


# --- pre-check: replicate L3 with Z0's rule
TIEKEY = DIST * 12 + np.arange(12)[None, :]; TIEKEY[IX, NOM] = -1
def z0_choose(allowed):
    valid = allowed & np.isfinite(C3['I1'])
    mn = np.min(np.where(valid, C3['I1'], np.inf), 1)
    ties = valid & np.isclose(C3['I1'], mn[:, None], rtol=0, atol=1e-7)
    out = np.argmin(np.where(ties, TIEKEY, 999), 1); fb = ~valid.any(1); out[fb] = NOM[fb]
    return out
L3 = z0_choose(DIST <= 1); L3c = counts(outcomes(L3))
assert L3c['recovery'] == 141 and L3c['collision'] == 87, L3c
ALL12 = z0_choose(np.ones((N, 12), bool)); ALLc = counts(outcomes(ALL12))

# --- fit risk models on train
models = {}; train_auc = {}
mtr = variant_mask(TR); ytr = TR['collision'][mtr].astype(float)
for kind in ['d', 'dv', 'dvo']:
    X, st = features(TR, kind)
    w = fit_logit(X[mtr], ytr)
    models[kind] = (w, st)
    train_auc[kind] = auroc(X[mtr] @ w, ytr)
mc3 = variant_mask(C3); yc3 = C3['collision'][mc3].astype(float)
c3_auc = {}
P = {}
for kind, (w, st) in models.items():
    X, _ = features(C3, kind, st)
    z = X @ w; P[kind] = 1 / (1 + np.exp(-z))
    c3_auc[kind] = auroc(z[mc3], yc3)

GAIN = C3['I1'][IX, NOM][:, None] - C3['I1']
ORDKEY = DIST * 12 + np.arange(12)[None, :]
def choose(p, lam):
    sc = GAIN - lam * p
    sc[~np.isfinite(C3['I1'])] = -np.inf
    sc[IX, NOM] = 0.0
    best = sc.max(1)
    cand = (sc == best[:, None]) & (sc > 0)
    out = np.where(cand.any(1), np.argmin(np.where(cand, ORDKEY, 999), 1), NOM)
    return out

LAMS = np.concatenate([[0.0], np.logspace(-5, 0, 501)])
curves = {}; matched = {}
for kind in models:
    rec = []
    for lam in LAMS:
        o = outcomes(choose(P[kind], lam)); rec.append((lam, int(o['recovery'].sum()), int(o['collision'].sum()), int(o['harm'].sum())))
    curves[kind] = rec
    gap = np.array([abs(r[1] - 141) for r in rec])
    best = np.flatnonzero(gap == gap.min()); i = best[-1]  # tie -> larger lambda
    lam = rec[i][0]; ang = choose(P[kind], lam); o = outcomes(ang)
    matched[kind] = dict(lam=lam, ang=ang, o=o, c=counts(o))

def coll_at(kind, target):
    rec = curves[kind]; gap = np.array([abs(r[1] - target) for r in rec]); i = np.flatnonzero(gap == gap.min())[-1]
    return dict(recovery=rec[i][1], collision=rec[i][2], harm=rec[i][3], lam=rec[i][0])

diff = scene_rate(matched['dv']['o']['collision']) - scene_rate(matched['d']['o']['collision'])
diff_feas = scene_rate(matched['dv']['o']['feasible']) - scene_rate(matched['d']['o']['feasible'])
diff_dvo = scene_rate(matched['dvo']['o']['collision']) - scene_rate(matched['d']['o']['collision'])
diff_L3 = scene_rate(matched['dv']['o']['collision']) - scene_rate(outcomes(L3)['collision'])
ci = interval(diff)
cd, cv = matched['d']['c']['collision'], matched['dv']['c']['collision']
gate = (cv <= 0.8 * cd) and (ci['high'] < 0)

stats = dict(
    L3=L3c, unconstrained_12=ALLc,
    train_rows=int(mtr.sum()), train_pos_rate=float(ytr.mean()),
    coefficients={k: dict(w=models[k][0].tolist(), mean=(models[k][1][0].tolist() if models[k][1] else None),
                          std=(models[k][1][1].tolist() if models[k][1] else None)) for k in models},
    train_auroc=train_auc, c3_auroc=c3_auc,
    matched={k: dict(lam=v['lam'], **v['c'], mean_shift_deg=float(15 * DIST[IX, v['ang']].mean()),
                     moved=int((v['ang'] != NOM).sum())) for k, v in matched.items()},
    dv_minus_d_collision_rate=ci, dv_minus_d_feasible_rate=interval(diff_feas),
    dvo_minus_d_collision_rate=interval(diff_dvo), dv_minus_L3_collision_rate=interval(diff_L3),
    other_levels={k: {t: coll_at(k, t) for t in [100, 120, 160]} for k in models},
    ratio_dv_over_d=cv / cd if cd else None, gate_A_main=bool(gate), verdict='VIS-HELPS' if gate else 'VIS-NO',
)
json.dump(stats, open(f'{OUT}/statistics.json', 'w'), indent=2)
with open(f'{OUT}/curves.csv', 'w') as f:
    f.write('model,lambda,recovery,collision,harm\n')
    for k, rec in curves.items():
        for r in rec:
            f.write(f'{k},{r[0]:.6g},{r[1]},{r[2]},{r[3]}\n')
np.savez(f'{OUT}/selected_angles.npz', **{k: v['ang'] for k, v in matched.items()}, L3=L3)
print(json.dumps({k: stats[k] for k in ['L3', 'unconstrained_12', 'train_auroc', 'c3_auroc', 'matched',
                                         'dv_minus_d_collision_rate', 'dvo_minus_d_collision_rate',
                                         'dv_minus_L3_collision_rate', 'ratio_dv_over_d', 'verdict']}, indent=1))

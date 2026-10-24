"""Firmness learnability (prereg eg_firmness_learnability_prereg_20261024.md). CPU-only numpy.
usage: python3 -I firmness_probe.py <config.json> <out_dir>
config keys: features_dir, probe_eval_dir, cal_eval_dir, c3_eval_dir, p2_eval_dir, per_pose_dir, frozen_models
File layout (one npz per frame):
  features_dir/{split}_{s:04d}_{f:04d}.npz : candidate_id, original_angle, vp (n,256), ang (n,64), logit (n,12)
  *_eval_dir/{s:04d}_{f:04d}.npz           : candidate_id, original_angle, mu (n,12), collision, empty, I1
  per_pose_dir/{train,C3,P2}_{s:04d}_{f:04d}.npz (N0, d36ec75): candidate_id, original_angle, collision, empty, R1
splits for features: probe, cal, C3, P2."""
import sys, os, glob, json
import numpy as np

CFG = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SEED = 20261024
N0_TRAIN = [67, 4, 50, 56, 14, 65, 62, 30, 45, 1, 96, 71]
N0_EXCLUDED = [2, 9, 31, 33, 64, 68, 76, 82]
POOL = sorted(set(range(100)) - set(N0_TRAIN) - set(N0_EXCLUDED))
PERM = np.random.default_rng(SEED).permutation(POOL)
TRAIN_SCENES, VAL_SCENES = sorted(PERM[:70].tolist()), sorted(PERM[70:].tolist())
LAMS = np.concatenate([[0.0], np.logspace(-4, 2, 601), [np.inf]])  # inf = never switch


def cyc(nom):
    k = np.arange(12)[None, :]
    return np.minimum((k - nom[:, None]) % 12, (nom[:, None] - k) % 12)


def frame_key(path):
    parts = os.path.basename(path)[:-4].split('_')
    return int(parts[-2]), int(parts[-1])


def firmness(mu, collision, empty):
    q = np.full(mu.shape, np.nan)
    ok = ~collision & ~empty
    q[ok & (mu > 0)] = (1.4 - mu[ok & (mu > 0)]) / 1.2
    q[ok & ~(mu > 0) & np.isfinite(mu)] = 0.0
    return q  # NaN = masked (collision, empty or unknown mu)


def load_split(split, eval_dir, pose_split=None):
    rows = []
    for fp in sorted(glob.glob(f"{CFG['features_dir']}/{split}_*.npz")):
        s, f = frame_key(fp); F = np.load(fp); E = np.load(f'{eval_dir}/{s:04d}_{f:04d}.npz')
        assert np.array_equal(F['candidate_id'], E['candidate_id']) and np.array_equal(F['original_angle'], E['original_angle']), (split, s, f)
        d = dict(X=np.concatenate([F['vp'], F['ang'], F['logit']], 1).astype(np.float64), logit=F['logit'].astype(np.float64),
                 nom=E['original_angle'], mu=E['mu'], col=E['collision'], emp=E['empty'], I1=E['I1'],
                 scene=np.full(len(F['candidate_id']), s))
        if 'R1' in E.files:  # v3: fresh A100 export carries the EG ratio per variant
            d['R1'] = E['R1']
        elif pose_split:
            P = np.load(f"{CFG['per_pose_dir']}/{pose_split}_{s:04d}_{f:04d}.npz")
            for k in ['candidate_id', 'original_angle', 'collision', 'empty']:
                assert np.array_equal(P[k], E[k]), (split, s, f, k)
            d['R1'] = P['R1']
        rows.append(d)
    assert rows, split
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


# ---------------- small MLP in numpy ----------------
class MLP:
    def __init__(self, din, dh, dout, drop, rng):
        self.W1 = rng.normal(0, np.sqrt(2 / din), (din, dh)); self.b1 = np.zeros(dh)
        self.W2 = rng.normal(0, np.sqrt(1 / dh), (dh, dout)); self.b2 = np.zeros(dout)
        self.drop = drop; self.rng = rng
        self.params = ['W1', 'b1', 'W2', 'b2']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}; self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}; self.t = 0

    def forward(self, X, train=False):
        z = X @ self.W1 + self.b1; h = np.maximum(z, 0)
        mask = None
        if train and self.drop > 0:
            mask = (self.rng.random(h.shape) >= self.drop) / (1 - self.drop); h = h * mask
        return h @ self.W2 + self.b2, (X, z, h, mask)

    def backward(self, cache, gy):
        X, z, h, mask = cache
        g = {'W2': h.T @ gy, 'b2': gy.sum(0)}
        gh = gy @ self.W2.T
        if mask is not None: gh = gh * mask
        gz = gh * (z > 0)
        g['W1'] = X.T @ gz; g['b1'] = gz.sum(0)
        return g

    def step(self, g, lr=1e-3, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            w = getattr(self, p); gp = g[p] + (wd * w if p.startswith('W') else 0)
            self.m[p] = b1 * self.m[p] + (1 - b1) * gp; self.v[p] = b2 * self.v[p] + (1 - b2) * gp * gp
            mh = self.m[p] / (1 - b1 ** self.t); vh = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, w - lr * mh / (np.sqrt(vh) + eps))

    def state(self): return {p: getattr(self, p).copy() for p in self.params}
    def load(self, st):
        for p, v in st.items(): setattr(self, p, v.copy())


def loss_grad(y, q):
    valid = np.isfinite(q); nv = max(valid.sum(), 1)
    diff = np.where(valid, y - np.nan_to_num(q), 0.0)
    mse = (diff ** 2).sum() / nv; g = 2 * diff / nv
    rows = valid.sum(1) >= 2; nr = max(rows.sum(), 1)
    yl = np.where(valid, y, -np.inf); ql = np.where(valid, np.nan_to_num(q) / 0.1, -np.inf)
    p = np.exp(yl - yl.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    t = np.exp(ql - ql.max(1, keepdims=True)); t /= t.sum(1, keepdims=True)
    p = np.where(valid, p, 0); t = np.where(valid, t, 0)
    ce = -(t * np.log(np.where(valid, p, 1) + 1e-12)).sum(1)
    ce = (ce * rows).sum() / nr
    g = g + np.where(rows[:, None], (p - t) / nr, 0)
    return mse + ce, g


def train(Xtr, Qtr, Xva, Qva, dh, drop, seed):
    rng = np.random.default_rng(seed); net = MLP(Xtr.shape[1], dh, 12, drop, rng)
    best, best_state, bad, hist = np.inf, net.state(), 0, []
    for ep in range(30):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), 512):
            b = order[i:i + 512]; y, c = net.forward(Xtr[b], train=True); _, g = loss_grad(y, Qtr[b]); net.step(net.backward(c, g))
        vl = loss_grad(net.forward(Xva)[0], Qva)[0]; hist.append(float(vl))
        if vl < best - 1e-6: best, best_state, bad = vl, net.state(), 0
        else:
            bad += 1
            if bad >= 3: break
    net.load(best_state); return net, hist


# ---------------- data ----------------
PR = load_split('probe', CFG['probe_eval_dir'])
CAL = load_split('cal', CFG['cal_eval_dir'], 'train')
C3 = load_split('C3', CFG['c3_eval_dir'], 'C3')
P2 = load_split('P2', CFG['p2_eval_dir'], 'P2')
Qpr = firmness(PR['mu'], PR['col'], PR['emp'])
tr = np.isin(PR['scene'], TRAIN_SCENES); va = np.isin(PR['scene'], VAL_SCENES)
assert tr.any() and va.any() and not np.isin(PR['scene'], N0_TRAIN + N0_EXCLUDED + list(range(100, 190))).any()
mean, std = PR['X'][tr].mean(0), PR['X'][tr].std(0) + 1e-6
lmean, lstd = PR['logit'][tr].mean(0), PR['logit'][tr].std(0) + 1e-6
models = {'M_feat': [], 'M_logit': []}; curves = {}
for i in range(3):
    net, h = train((PR['X'][tr] - mean) / std, Qpr[tr], (PR['X'][va] - mean) / std, Qpr[va], 256, 0.1, SEED + i)
    models['M_feat'].append(net); curves[f'M_feat_{i}'] = h
    net, h = train((PR['logit'][tr] - lmean) / lstd, Qpr[tr], (PR['logit'][va] - lmean) / lstd, Qpr[va], 64, 0.0, SEED + i)
    models['M_logit'].append(net); curves[f'M_logit_{i}'] = h


def predict(A, name):
    X = (A['X'] - mean) / std if name == 'M_feat' else (A['logit'] - lmean) / lstd
    return np.mean([m.forward(X)[0] for m in models[name]], 0)


FM = json.load(open(CFG['frozen_models']))['R_de']


def prepare(A):
    ix = np.arange(len(A['nom'])); clean = ~A['col'][ix, A['nom']] & ~A['emp'][ix, A['nom']]
    B = {k: v[clean] for k, v in A.items()}
    n = len(B['nom']); IX = np.arange(n); NOM = B['nom']; D = cyc(NOM)
    v = B['R1'].astype(float); C = np.stack([v, v - v[IX, NOM][:, None]], -1)
    C = (C - np.array(FM['mean'])) / np.array(FM['std'])
    Z = np.stack([(D == j).astype(float) for j in range(1, 7)] + [C[..., 0], C[..., 1]], -1) @ np.array(FM['w'])
    B.update(IX=IX, D=D, p=1 / (1 + np.exp(-Z)))
    mu0 = B['mu'][IX, NOM]; B['GOOD'] = (mu0 > 0) & (mu0 <= .4); B['WEAK'] = mu0 > .4
    gains = {}
    for name in models:
        y = predict(B, name); gains[name] = y - y[IX, NOM][:, None]
    gains['B1_logit'] = B['logit'] - B['logit'][IX, NOM][:, None]
    gains['ref_I1'] = B['I1'][IX, NOM][:, None] - B['I1']
    B['gains'] = gains
    return B


def choose(B, gain, lam):
    sc = gain - (lam * B['p'] if np.isfinite(lam) else np.inf); sc = np.where(np.isfinite(sc), sc, -np.inf); sc[B['IX'], B['nom']] = 0.0
    key = B['D'] * 12 + np.arange(12)[None, :]
    best = sc.max(1); cand = (sc == best[:, None]) & (sc > 0)
    return np.where(cand.any(1), np.argmin(np.where(cand, key, 999), 1), B['nom'])


def outcomes(B, ang):
    IX = B['IX']; mu = B['mu'][IX, ang]; co = B['col'][IX, ang]; em = B['emp'][IX, ang]
    feas = ~co & ~em & (mu > 0) & (mu <= .4)
    return dict(feasible=feas, recovery=B['WEAK'] & feas, harm=B['GOOD'] & ~feas, collision=co & ~em, moved=ang != B['nom'])


CB, C3B, P2B = prepare(CAL), prepare(C3), prepare(P2)
lam = {}
for name in CB['gains']:
    best = None
    for l in LAMS:
        fv = outcomes(CB, choose(CB, CB['gains'][name], l))['feasible'].sum()
        if best is None or fv >= best[0]: best = (fv, l)
    lam[name] = float(best[1])


def boot_setup(B):
    sc = np.array(sorted(set(B['scene']))); si = np.searchsorted(sc, B['scene']); r = np.random.default_rng(SEED)
    bt = np.concatenate([r.choice(np.flatnonzero((sc >= lo) & (sc < hi)), (10000, 4), replace=True) for lo, hi in [(100, 130), (130, 160), (160, 190)]], 1)
    cnt = np.bincount(si, minlength=len(sc))
    def rate(x): return np.bincount(si, weights=x.astype(float), minlength=len(sc)) / cnt
    def interval(x):
        b = x[bt].mean(1); return dict(mean=float(x.mean()), low=float(np.quantile(b, .025)), high=float(np.quantile(b, .975)))
    return rate, interval


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


res = dict(train_scenes=TRAIN_SCENES, val_scenes=VAL_SCENES, probe_candidates=int(len(PR['nom'])), probe_masked_frac=float(np.isnan(Qpr).mean()),
           lambda_from_cal=lam, curves=curves, splits={})
for tag, B in [('C3', C3B), ('P2', P2B)]:
    rate, interval = boot_setup(B); n = len(B['nom']); O = {}
    base = int(B['GOOD'].sum())  # do-nothing feasible count on this panel
    for name, g in B['gains'].items():
        O[name] = outcomes(B, choose(B, g, lam[name]))
    nonorig = np.ones((n, 12), bool); nonorig[B['IX'], B['nom']] = False
    qtrue = firmness(B['mu'], B['col'], B['emp']); qgain = qtrue - qtrue[B['IX'], B['nom']][:, None]
    sp = dict(candidates=n, do_nothing_feasible=base, min_gain=int(np.ceil(0.025 * n)),
              counts={k: {m: int(v.sum()) for m, v in o.items()} for k, o in O.items()},
              corr_with_true_q_gain={k: corr(g[nonorig], qgain[nonorig]) for k, g in B['gains'].items()},
              corr_with_true_I1_gain={k: corr(g[nonorig], B['gains']['ref_I1'][nonorig]) for k, g in B['gains'].items()},
              feat_minus_nothing=interval(rate(O['M_feat']['feasible']) - rate(B['GOOD'])),
              feat_minus_logitprobe=interval(rate(O['M_feat']['feasible']) - rate(O['M_logit']['feasible'])),
              feat_minus_B1=interval(rate(O['M_feat']['feasible']) - rate(O['B1_logit']['feasible'])),
              collision_rate_feat=float(O['M_feat']['collision'].mean()))
    res['splits'][tag] = sp
c3, p2 = res['splits']['C3'], res['splits']['P2']
F1 = c3['counts']['M_feat']['feasible'] - c3['do_nothing_feasible'] >= c3['min_gain'] and c3['feat_minus_nothing']['low'] > 0
F2 = c3['feat_minus_logitprobe']['low'] > 0
F3 = p2['counts']['M_feat']['feasible'] - p2['do_nothing_feasible'] >= p2['min_gain'] and p2['feat_minus_nothing']['low'] > 0
G = c3['collision_rate_feat'] <= 0.04 and p2['collision_rate_feat'] <= 0.04
res['gates'] = dict(F1=bool(F1), F2=bool(F2), F3=bool(F3), G=bool(G))
res['verdict'] = ('LEARNABLE' if F1 and F2 and F3 and G else 'LOGIT-ONLY' if F1 and F3 and G else 'HOLD' if F1 else 'NOT-YET')
json.dump(res, open(f'{OUT}/firmness_results.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ['lambda_from_cal', 'gates', 'verdict']}, indent=1))
for tag in ['C3', 'P2']:
    print(tag, res['splits'][tag]['counts'], res['splits'][tag]['corr_with_true_q_gain'])

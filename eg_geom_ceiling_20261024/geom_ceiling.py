"""Single-view firmness ceiling test (prereg eg_geom_ceiling_prereg_20261024.md). CPU-only numpy, no learning.
usage: python3 -I geom_ceiling.py <config.json> <out_dir>
config keys: eval_dir (v3/eval, files {split}_{s:04d}_{f:04d}.npz with mu, collision, empty, I1, R1),
             obs_dir (files {split}_{s:04d}_{f:04d}.npz with candidate_id, original_angle, I1obs_r3, found_r3, ...),
             frozen_models (frozen_models_P2.json),
             expect (firmness_results.json: the I1 reference and do-nothing counts must reproduce exactly, else stop)."""
import sys, os, glob, json
import numpy as np

CFG = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SEED = 20261024
LAMS = np.concatenate([[0.0], np.logspace(-4, 2, 601), [np.inf]])
FM = json.load(open(CFG['frozen_models']))['R_de']
VARIANTS = ['I1obs_r3', 'I1obs_r2', 'I1obs_r5', 'I1obs_seg_r3']  # primary first


def cyc(nom):
    k = np.arange(12)[None, :]
    return np.minimum((k - nom[:, None]) % 12, (nom[:, None] - k) % 12)


def firmness(mu, col, emp):
    q = np.full(mu.shape, np.nan); ok = ~col & ~emp
    q[ok & (mu > 0)] = (1.4 - mu[ok & (mu > 0)]) / 1.2
    q[ok & ~(mu > 0) & np.isfinite(mu)] = 0.0
    return q


def load(split):
    rows = []
    for fp in sorted(glob.glob(f"{CFG['eval_dir']}/{split}_*.npz")):
        name = os.path.basename(fp); s = int(name.split('_')[1])
        E = np.load(fp); O = np.load(f"{CFG['obs_dir']}/{name}")
        assert np.array_equal(E['candidate_id'], O['candidate_id']) and np.array_equal(E['original_angle'], O['original_angle']), name
        d = dict(nom=E['original_angle'], mu=E['mu'], col=E['collision'], emp=E['empty'], I1=E['I1'], R1=E['R1'],
                 found=O['found_r3'], scene=np.full(len(E['candidate_id']), s))
        for v in VARIANTS: d[v] = O[v]
        n = len(E['candidate_id'])  # report-only contact diagnostics, NaN when not exported
        d['gt_pts'] = O['gt_pts_cam'] if 'gt_pts_cam' in O.files else np.full((n, 12, 2, 3), np.nan)
        d['obs_pts'] = O['obs_pts_r3'] if 'obs_pts_r3' in O.files else np.full((n, 12, 2, 3), np.nan)
        rows.append(d)
    assert rows, split
    A = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    ix = np.arange(len(A['nom'])); clean = ~A['col'][ix, A['nom']] & ~A['emp'][ix, A['nom']]
    B = {k: v[clean] for k, v in A.items()}
    n = len(B['nom']); IX = np.arange(n); NOM = B['nom']; D = cyc(NOM)
    v = B['R1'].astype(float); C = np.stack([v, v - v[IX, NOM][:, None]], -1)
    C = (C - np.array(FM['mean'])) / np.array(FM['std'])
    Z = np.stack([(D == j).astype(float) for j in range(1, 7)] + [C[..., 0], C[..., 1]], -1) @ np.array(FM['w'])
    B.update(IX=IX, D=D, p=1 / (1 + np.exp(-Z)))
    mu0 = B['mu'][IX, NOM]; B['GOOD'] = (mu0 > 0) & (mu0 <= .4); B['WEAK'] = mu0 > .4
    gains = {}
    for vname in VARIANTS:
        I = B[vname]; g = I[IX, NOM][:, None] - I
        gains[vname] = np.where(np.isfinite(g), g, 0.0)  # cannot tell -> no gain, rule keeps original
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


def boot(B):
    sc = np.array(sorted(set(B['scene']))); si = np.searchsorted(sc, B['scene']); r = np.random.default_rng(SEED)
    bt = np.concatenate([r.choice(np.flatnonzero((sc >= lo) & (sc < hi)), (10000, 4), replace=True) for lo, hi in [(100, 130), (130, 160), (160, 190)]], 1)
    cnt = np.bincount(si, minlength=len(sc))
    rate = lambda x: np.bincount(si, weights=x.astype(float), minlength=len(sc)) / cnt
    def interval(x):
        b = x[bt].mean(1); return dict(mean=float(x.mean()), low=float(np.quantile(b, .025)), high=float(np.quantile(b, .975)))
    def corr_ci(a, ref, b, frame_scene):
        # scene bootstrap of the pooled Pearson correlation and of its share of the GT reference correlation
        out = []
        for row in bt[:2000]:
            w = np.concatenate([np.flatnonzero(frame_scene == sc[j]) for j in row])
            c, cr = corr(a[w], b[w]), corr(ref[w], b[w]); out.append((c, c / cr))
        out = np.array(out)
        return dict(corr=[float(np.quantile(out[:, 0], .025)), float(np.quantile(out[:, 0], .975))],
                    retained=[float(np.quantile(out[:, 1], .025)), float(np.quantile(out[:, 1], .975))])
    return rate, interval, corr_ci


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


CAL, C3, P2 = load('cal'), load('C3'), load('P2')
COL_CAP = 0.04
lam, lam_free = {}, {}  # lam: max cal feasible s.t. cal new-collision rate <= COL_CAP (lam=inf always qualifies); lam_free: unconstrained
for name in CAL['gains']:
    best = bestc = None
    for l in LAMS:
        o = outcomes(CAL, choose(CAL, CAL['gains'][name], l)); fv = o['feasible'].sum()
        if best is None or fv >= best[0]: best = (fv, l)
        if o['collision'].mean() <= COL_CAP and (bestc is None or fv >= bestc[0]): bestc = (fv, l)
    lam_free[name] = float(best[1]); lam[name] = float(bestc[1])

res = dict(lambda_from_cal=lam, lambda_from_cal_unconstrained=lam_free, splits={})
for tag, B in [('C3', C3), ('P2', P2)]:
    rate, interval, corr_ci = boot(B); n = len(B['nom'])
    base = int(B['GOOD'].sum())
    q = firmness(B['mu'], B['col'], B['emp']); qg = q - q[B['IX'], B['nom']][:, None]
    m = np.isfinite(qg); m[B['IX'], B['nom']] = False
    sc_rep = np.repeat(B['scene'][:, None], 12, 1)
    found = B['found'][m]; both = ((B['found'] == 2) & (B['found'][B['IX'], B['nom']] == 2)[:, None])[m]
    gp, op = B['gt_pts'][m], B['obs_pts'][m]; dist = np.linalg.norm(gp - op, axis=-1)  # (k, 2) per side
    has_gt = np.isfinite(gp).all(-1)
    sp = dict(candidates=n, do_nothing_feasible=base, min_gain=int(np.ceil(0.025 * n)),
              found_fraction={str(k): float((found == k).mean()) for k in [0, 1, 2]},
              found_fraction_original={str(k): float((B['found'][B['IX'], B['nom']] == k).mean()) for k in [0, 1, 2]},
              both_found_pairs=float(both.mean()),
              gt_contact_seen_within_5mm={str(j): (float((dist[:, j][has_gt[:, j]] <= 0.005).mean()) if has_gt[:, j].any() else None) for j in [0, 1]},
              corr_q={k: corr(g[m], qg[m]) for k, g in B['gains'].items()},
              corr_q_both_found={k: corr(g[m][both], qg[m][both]) for k, g in B['gains'].items()},
              corr_true_I1={k: corr(g[m], B['gains']['ref_I1'][m]) for k, g in B['gains'].items()})
    ref0 = np.nan_to_num(B['gains']['ref_I1'], nan=0.0)  # GT gain with the same "cannot tell -> 0" fill as the observed gains
    sp['corr_q_ref_zero'] = corr(ref0[m], qg[m])
    sp['retained'] = {k: sp['corr_q'][k] / sp['corr_q_ref_zero'] for k in VARIANTS}
    sp['primary_ci'] = corr_ci(B['gains']['I1obs_r3'][m], ref0[m], qg[m], sc_rep[m])
    O = {k: outcomes(B, choose(B, g, lam[k])) for k, g in B['gains'].items()}
    sp['counts'] = {k: {mm: int(v.sum()) for mm, v in o.items()} for k, o in O.items()}
    sp['counts_unconstrained'] = {k: {mm: int(v.sum()) for mm, v in outcomes(B, choose(B, g, lam_free[k])).items()} for k, g in B['gains'].items()}
    sp['obs_minus_nothing'] = interval(rate(O['I1obs_r3']['feasible']) - rate(B['GOOD']))
    sp['collision_rate_obs'] = float(O['I1obs_r3']['collision'].mean())
    sp['collision_rate_ref_I1'] = float(O['ref_I1']['collision'].mean())  # oracle gain under the same rule and cap: the safety yardstick
    res['splits'][tag] = sp
EXP = json.load(open(CFG['expect']))['splits']
for tag in ['C3', 'P2']:  # pipeline check against the published firmness-probe reference
    got = res['splits'][tag]; want = EXP[tag]
    assert got['do_nothing_feasible'] == want['do_nothing_feasible'] and got['counts_unconstrained']['ref_I1'] == want['counts']['ref_I1'], (tag, got['counts_unconstrained']['ref_I1'], want['counts']['ref_I1'])
c3, p2 = res['splits']['C3'], res['splits']['P2']
r3 = min(c3['retained']['I1obs_r3'], p2['retained']['I1obs_r3'])
res['ceiling'] = 'HIGH' if r3 >= 0.82 else 'LOW' if r3 < 0.70 else 'MID'
res['geom_rule'] = bool(c3['counts']['I1obs_r3']['feasible'] - c3['do_nothing_feasible'] >= c3['min_gain'] and c3['obs_minus_nothing']['low'] > 0
                        and p2['counts']['I1obs_r3']['feasible'] - p2['do_nothing_feasible'] >= p2['min_gain'] and p2['obs_minus_nothing']['low'] > 0
                        and c3['collision_rate_obs'] <= c3['collision_rate_ref_I1'] and p2['collision_rate_obs'] <= p2['collision_rate_ref_I1'])
json.dump(res, open(f'{OUT}/geom_ceiling_results.json', 'w'), indent=1)
print(json.dumps(dict(lambda_from_cal=lam, lambda_from_cal_unconstrained=lam_free, ceiling=res['ceiling'], geom_rule=res['geom_rule']), indent=1))
for tag in ['C3', 'P2']:
    s = res['splits'][tag]
    print(tag, 'found', s['found_fraction'], 'corr_q', {k: round(v, 3) for k, v in s['corr_q'].items()}, 'ref0', round(s['corr_q_ref_zero'], 3),
          'retained', {k: round(v, 3) for k, v in s['retained'].items()}, 'ci', s['primary_ci'])
    print(tag, 'counts', s['counts'])

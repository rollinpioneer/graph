from core import *
d=json.loads((R/'results/gate_calibration.json').read_text());rng=np.random.default_rng(20261018);draw=rng.integers(0,4,(10000,4));out=[]
for row in d['rows']:
 n=np.array([x['Net'] for x in row['scenes']]);v=np.array([np.nan if x['Damage'] is None else x['Damage'] for x in row['scenes']]);nb=n[draw].mean(1);db=np.nanmean(v[draw],axis=1) if np.isfinite(v).any() else np.full(10000,np.nan)
 out.append(dict(seed=row['seed'],tau=row['tau'],Net_ci95=np.quantile(nb,[.025,.975]).tolist(),Damage_ci95=np.nanquantile(db,[.025,.975]).tolist() if np.isfinite(db).any() else None,valid_damage_scenes=int(np.isfinite(v).sum())))
write(R/'results/gate_uncertainty.json',dict(note='Descriptive four-scene cluster bootstrap; calibration decision remains frozen point-estimate rule, not an independent safety test.',replicates=10000,rows=out))

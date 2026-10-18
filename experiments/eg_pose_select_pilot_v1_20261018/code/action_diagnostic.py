from core import *
scenes=CFG['gate_cal'];ys=[];ids=[]
for s in scenes:
 for f in CFG['fit_frames']:
  with np.load(R/f'labels/{s:04d}_{f:04d}.npz') as z:ys.append(utility(z['q']));ids.append(z['actions'])
y=np.concatenate(ys);actions=np.concatenate(ids);within_y=y-y.mean(1,keepdims=True);base_variance=float((within_y**2).mean());rows=[]
for seed in SEEDS:
 with np.load(R/f'heads/{seed}/gate_predictions.npz') as z:q=z['pred']
 within_q=q-q.mean(1,keepdims=True);within_mse=float(((within_q-within_y)**2).mean());between_mse=float(((q.mean(1)-y.mean(1))**2).mean());total=float(((q-y)**2).mean());assert np.isclose(total,within_mse+between_mse)
 chosen=select(q,actions,'S4',0);best=y[:,SETS['S4']].max(1);pick=y[np.arange(len(y)),chosen];gain=pick-y[:,0]
 rows.append(dict(seed=seed,groups=len(y),predicted_six_action_range_quantiles=np.quantile(np.ptp(q,axis=1),[0,.25,.5,.75,.95,1]).tolist(),true_six_action_range_quantiles=np.quantile(np.ptp(y,axis=1),[0,.25,.5,.75,.95,1]).tolist(),target_within_group_variance=base_variance,within_group_mse=within_mse,between_group_mse=between_mse,total_mse=total,within_group_variance_explained=1-within_mse/base_variance,within_group_pearson=float(np.corrcoef(within_q.ravel(),within_y.ravel())[0,1]),S4_zero_mean_utility_gain=float(gain.mean()),S4_zero_regret=float((best-pick).mean()),S4_original_regret=float((best-y[:,0]).mean()),true_actions_all_equal_groups=int((np.ptp(y,axis=1)==0).sum())))
write(R/'results/action_diagnostic.json',dict(scope='Exploratory analysis on already-used gate-cal only; no retraining or changed selection. Variance decomposition is descriptive, not a causal attribution to architecture.',rows=rows))
print([(x['seed'],x['predicted_six_action_range_quantiles'][2],x['within_group_variance_explained'],x['within_group_pearson']) for x in rows])

"""D4 input coverage: descriptive joint state-goal support, no policy queries."""
from scipy.spatial import cKDTree
from common import *


def main():
    summary={}
    for kind in ('navigate','stitch'):
        batch=np.load(ROOT/'runs'/f'{kind}_sampler_probe.npz')
        features=np.c_[batch['observations'],batch['high_actor_goals']]
        tree=cKDTree(features)
        train_nn=tree.query(features,k=2)[0][:,1]
        final=json.load(open(ROOT/'results'/f'{kind}_FINAL_EVAL.json'))
        eval_features=np.asarray([r['initial']+r['goal'] for r in final['records']])
        distance,index=tree.query(eval_features,k=1)
        cutoff=float(np.quantile(train_nn,.99))
        eval_horizon=np.linalg.norm(eval_features[:,:2]-eval_features[:,2:],axis=1)
        train_horizon=np.linalg.norm(features[:,:2]-features[:,2:],axis=1)
        summary[kind]=dict(training_style_probes=4096,final_initial_state_goal_pairs=len(eval_features),
                           train_leave_one_out_joint_nn_distance_quantiles=np.quantile(train_nn,[.5,.9,.99,1]),
                           final_joint_nn_distance_quantiles=np.quantile(distance,[0,.5,.9,.99,1]),
                           fraction_above_train_99_percentile=float(np.mean(distance>cutoff)),
                           training_goal_euclidean_distance_quantiles=np.quantile(train_horizon,[0,.5,.9,.99,1]),
                           final_goal_euclidean_distance_quantiles=np.quantile(eval_horizon,[0,.5,.9,.99,1]),
                           note='finite 4096-probe joint nearest-neighbor coverage proxy; no proof that the full million-transition dataset has zero support; distance is not optimal value')
        from diagnose import rows_to_csv
        rows_to_csv(ROOT/'results'/f'{kind}_D4_INPUT_COVERAGE.csv',[
            dict(task=r['task_id'],episode=r['episode'],reset_seed=r['reset_seed'],success=r['success'],
                 joint_nn_distance=float(d),nearest_probe_id=int(idx),above_train_99_percentile=bool(d>cutoff))
            for r,d,idx in zip(final['records'],distance,index)])
    write_json(ROOT/'results/D4_INPUT_COVERAGE.json',summary)
    print('INPUT_COVERAGE_DONE',flush=True)


if __name__=='__main__':main()

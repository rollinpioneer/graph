"""Observe the unchanged official sampler and pilot exactly 1000 updates."""
import argparse, copy, importlib.metadata
from common import *

class TraceSampler(HGCDataset):
    def get_observations(self, idx):
        self.indices_seen.append(np.asarray(idx).copy())
        return super().get_observations(idx)
    def sample(self,n):
        self.indices_seen=[]; self.rand_seen=[]
        idx=self.dataset.get_random_idxs(n); self.state_idx=idx.copy()
        original=np.random.rand
        def observed(*shape):
            x=original(*shape); self.rand_seen.append(x.copy()); return x
        np.random.rand=observed
        try: result=super().sample(n,idx,evaluation=True)
        finally: np.random.rand=original
        self.pick=self.rand_seen[-1]<self.config['actor_p_randomgoal']
        return result

def audit():
    report={}; identities=[]
    old=json.loads((OLD/'results/DATA_AUDIT.json').read_text())
    for kind in ('navigate','stitch'):
        report[kind]={}
        for file in old[kind]['files']:
            actual=sha(file['path']); assert actual==file['sha256']
            identities.append(dict(path=file['path'],sha256=actual))
        env,data,val,b=load(f'pointmaze-large-{kind}-v0')
        a=get_config().to_dict()
        diff={k:[a[k],b[k]] for k in a if a[k]!=b[k]}
        assert diff=={'actor_p_trajgoal':[1.0,0.5],'actor_p_randomgoal':[0.0,0.5]}
        report[kind]['config_diff']=diff
        for variant,conf in [('A',a),('B',b)]:
            tracer=TraceSampler(data.dataset,conf)
            with separate_numpy(1030131+(kind=='stitch')): batch=tracer.sample(4096)
            i=tracer.state_idx; vg,low,g,w=tracer.indices_seen; pick=tracer.pick
            end=tracer.terminal_locs[np.searchsorted(tracer.terminal_locs,i)]
            checks=dict(valid_transition=bool(np.all(data.dataset['valids'][i]>0)),
                low_waypoint=bool(np.array_equal(low,np.minimum(i+25,end))),
                future_bounds=bool(np.all((g[~pick]>=i[~pick])&(g[~pick]<=end[~pick]))),
                future_waypoint=bool(np.array_equal(w[~pick],np.minimum(i[~pick]+25,g[~pick]))),
                random_waypoint=bool(np.array_equal(w[pick],np.minimum(i[pick]+25,end[pick]))),
                random_goal_valid=bool(np.all(data.dataset['valids'][g[pick]]>0)),
                rewards=bool(np.array_equal(batch['rewards'],(i==vg).astype(float)-1)),
                masks=bool(np.array_equal(batch['masks'],(i!=vg).astype(float))),
                random_fraction=bool(.47<=pick.mean()<=.53) if variant=='B' else bool(pick.mean()==0))
            assert all(checks.values()),checks
            raw=OLD/'vendor/data'/f'pointmaze-large-{kind}-v0.npz'
            with np.load(raw) as r:
                terminals=np.flatnonzero(r['terminals'])
                traj=np.searchsorted(terminals,i); gt=np.searchsorted(terminals,g)
            report[kind][variant]=dict(checks=checks,random_fraction=float(pick.mean()),
                cross_original_trajectory_fraction=float(np.mean(traj!=gt)),
                high_goal_distance_quantiles=np.quantile(np.linalg.norm(batch['high_actor_goals']-batch['observations'],axis=1),[.5,.9,.99]),
                boundary_rule='official compact loader retained; original terminal/predecessor mapping not changed')
            np.savez_compressed(ROOT/'runs'/f'{kind}_{variant}_sampler.npz',**batch,state_idx=i,goal_idx=g,waypoint_idx=w,pick_random=pick)
        env.close()
    write_json(ROOT/'results/SAMPLER_AUDIT.json',report)
    write_json(ROOT/'results/DATA_IDENTITIES.json',identities)
    write_json(ROOT/'results/DEPENDENCIES.json',{p:importlib.metadata.version(p) for p in ('jax','jaxlib','flax','numpy','ogbench','mujoco','optax')})
    print('SAMPLER_AUDIT_PASS',flush=True)


if __name__=='__main__':audit()

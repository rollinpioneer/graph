"""CPU D1 coverage and goal distribution analysis; no policy calls."""
from collections import Counter, deque
import csv
from common import *


def distances(u, goal):
    target=u.xy_to_ij(goal)
    d=np.full(u.maze_map.shape,np.inf)
    if not (0<=target[0]<d.shape[0] and 0<=target[1]<d.shape[1]) or u.maze_map[target]: return d
    d[target]=0; q=deque([target])
    while q:
        i,j=q.popleft()
        for x,y in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
            if 0<=x<d.shape[0] and 0<=y<d.shape[1] and u.maze_map[x,y]==0 and not np.isfinite(d[x,y]):
                d[x,y]=d[i,j]+1; q.append((x,y))
    return d


def geodesic(u, states, goal):
    d=distances(u,goal)
    cells=np.asarray([u.xy_to_ij(s) for s in states])
    valid=(cells[:,0]>=0)&(cells[:,0]<d.shape[0])&(cells[:,1]>=0)&(cells[:,1]<d.shape[1])
    out=np.full(len(states),np.inf)
    out[valid]=d[cells[valid,0],cells[valid,1]]*u._maze_unit
    return out


def main():
    result={}
    for kind in ('navigate','stitch'):
        env,data,val,config=load(f'pointmaze-large-{kind}-v0'); u=env.unwrapped
        raw=np.load(ROOT/'vendor/data'/f'pointmaze-large-{kind}-v0.npz')
        terminal=np.flatnonzero(raw['terminals']); starts=np.r_[0,terminal[:-1]+1]
        obs=raw['observations']; act=raw['actions']; valid=np.flatnonzero(1-raw['terminals'])
        cells=np.asarray([u.xy_to_ij(s) for s in obs],dtype=np.int16)
        occupancy=Counter(map(tuple,cells[valid]))
        transitions=Counter((tuple(cells[i]),tuple(cells[i+1])) for i in valid if not np.array_equal(cells[i],cells[i+1]))
        ends=data.terminal_locs; indices=np.loadtxt(ROOT/'results'/f'{kind}_sampler_indices.csv',delimiter=',',skiprows=1,dtype=int)
        _,i,vg,low,g,w=indices.T
        high_horizon=g-i; waypoint_horizon=w-i
        goalgeo=[]; waypoint_progress=[]; turns=[]
        for si,gi,wi in zip(i,g,w):
            pair=geodesic(u,obs[[si,wi]],obs[gi]); goalgeo.append(pair[0]); waypoint_progress.append(pair[0]-pair[1])
            route=cells[si:wi+1]; changes=np.diff(route,axis=0); changes=changes[np.any(changes,axis=1)]
            turns.append(int(sum(not np.array_equal(a,b) for a,b in zip(changes[:-1],changes[1:]))))
        goalgeo=np.asarray(goalgeo); progress=np.asarray(waypoint_progress)
        support=[]
        for task in u.task_infos:
            d=distances(u,task['goal_xy']); start=tuple(task['init_ij']); goal=tuple(task['goal_ij'])
            graph={cell:[] for cell in occupancy}
            for (a,b),n in transitions.items(): graph.setdefault(a,[]).append(b)
            seen={start}; queue=deque([start])
            while queue:
                a=queue.popleft()
                for b in graph.get(a,[]):
                    if b not in seen: seen.add(b); queue.append(b)
            support.append(dict(task=task['task_name'],map_distance_cells=float(d[start]),
                                directed_empirical_cell_reachable=goal in seen,start_occupancy=occupancy.get(start,0),goal_occupancy=occupancy.get(goal,0),
                                interpretation='cell aggregate support only; not a proof of full-state policy feasibility'))
        summary=dict(valid_transitions=len(valid),trajectories=len(terminal),free_cells=int(np.sum(u.maze_map==0)),occupied_cells=len(occupancy),
                     directed_intercell_edges=len(transitions),action_min=act[valid].min(axis=0),action_max=act[valid].max(axis=0),
                     qvel_abs_max=float(np.abs(raw['qvel']).max()),qpos_observation_max_abs_error=float(np.abs(raw['qpos']-obs).max()),
                     high_goal_horizon_quantiles=np.quantile(high_horizon,[0,.25,.5,.75,.9,1]),
                     waypoint_horizon_quantiles=np.quantile(waypoint_horizon,[0,.25,.5,.75,.9,1]),
                     waypoint_positive_cell_progress_fraction=float(np.mean(progress>0)),waypoint_negative_cell_progress_fraction=float(np.mean(progress<0)),
                     turn_within_waypoint_fraction=float(np.mean(np.asarray(turns)>0)),high_goal_same_cell_fraction=float(np.mean(goalgeo==0)),
                     task_support=support,geom_metric_note='privileged cell BFS; discontinuous, ignores within-cell position and velocity, not discounted value ground truth')
        with open(ROOT/'results'/f'{kind}_cell_support.csv','w',newline='') as f:
            writer=csv.writer(f,lineterminator='\n'); writer.writerow(['i','j','visits'])
            for cell,n in sorted(occupancy.items()): writer.writerow([*cell,n])
        with open(ROOT/'results'/f'{kind}_sampler_geometry.csv','w',newline='') as f:
            writer=csv.writer(f,lineterminator='\n'); writer.writerow(['sample_id','state_idx','goal_idx','waypoint_idx','high_horizon','waypoint_horizon','cell_distance','cell_progress','turns'])
            for row in zip(indices[:,0],i,g,w,high_horizon,waypoint_horizon,goalgeo,progress,turns): writer.writerow(row)
        result[kind]=summary; env.close()
    result['comparison']=dict(same_map=True,same_tasks=True,warning='dataset horizon, coverage and collection policy differ simultaneously; this is not a single-factor stitching causal experiment')
    write_json(ROOT/'results/D1_COVERAGE.json',result)
    print('D1_CPU_DONE',flush=True)


if __name__=='__main__': main()

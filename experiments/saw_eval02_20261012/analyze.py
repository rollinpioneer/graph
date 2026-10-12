"""Registered fixed-cell uncertainty and complete scene/failure comparisons."""
from collections import Counter
from scipy.stats import ks_2samp
from eval_core import *

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]
def readrows(path):
    rows=list(csv.DictReader(open(path)))
    for row in rows:
        for k in ('task','episode','stream','steps'):row[k]=int(row[k])
        for k in ('success','initial_x','initial_y','initial_vx','initial_vy','goal_x','goal_y','initial_distance','final_distance','path_length'):row[k]=float(row[k])
    return rows
def failure(env,tr,cal):
    xy=tr['states'];a=tr['actions'];request=.2*np.linalg.norm(a,axis=1);motion=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    no=(request>cal['min_requested_displacement'])&(motion/(request+1e-12)<cal['no_effect_ratio_threshold'])
    win=np.convolve(no.astype(int),np.ones(10,dtype=int),mode='valid') if len(no)>=10 else np.array([])
    centers=np.asarray([env.unwrapped.ij_to_xy(tuple(c)) for c in np.argwhere(env.unwrapped.maze_map==1)])
    delta=np.maximum(np.abs(xy[:-1,None,:]-centers[None,:,:])-2,0);wall=np.linalg.norm(delta,axis=2).min(axis=1)<cal['wall_proximity']
    cells=[env.unwrapped.xy_to_ij(s) for s in xy];compressed=[cells[0]]+[b for a,b in zip(cells,cells[1:]) if a!=b]
    return dict(persistent_no_effect=bool((win>=8).any()),no_effect_steps=int(no.sum()),wall_no_effect_steps=int((wall&no).sum()),
                revisit_fraction=(len(compressed)-len(set(compressed)))/len(compressed),small_command_fraction=float(np.mean(request<=cal['min_requested_displacement'])))

def main():
    start=time.monotonic();cpu=time.process_time();size=json.load(open(ROOT/'SAMPLE_SIZE.json'));streams=size['streams']
    scores=[];comparisons=[];scenario_stats=[];matching=[];failure_rows=[];overall={};notes={}
    cal=json.load(open(OLD/'results/EVENT_CALIBRATION.json'))
    for model in ('navigate','stitch'):
        allrows={};raws={};env=env_new(model)
        for s in streams:
            for c in ('O','U'):
                name=f'B_{model}_{c}_{s}';rows=readrows(ROOT/'results'/f'{name}.csv');allrows[c,s]=rows
                assert len(rows)==250
                bundle=pickle.load(open(ROOT/'runs'/f'{name}.pkl','rb'));raws[c,s]=bundle['trajectories']
                for row,tr in zip(rows,bundle['trajectories']):failure_rows.append(dict(model=model,condition=c,stream=s,task=row['task'],episode=row['episode'],success=row['success'],**failure(env,tr,cal)))
                for task in range(1,6):
                    cell=[r for r in rows if r['task']==task];k=sum(r['success'] for r in cell)
                    lo,hi=wilson(k,len(cell));scores.append(dict(model=model,condition=c,stream=s,task=task,n=len(cell),successes=int(k),success_rate=k/len(cell),wilson95_low=lo,wilson95_high=hi))
        env.close()
        rng=np.random.default_rng(57001+(model=='stitch'));boots={};means={}
        for c in ('O','U'):
            boots[c]=[];means[c]=[]
            for s in streams:
                for t in range(1,6):
                    r=[x for x in allrows[c,s] if x['task']==t];p=np.mean([x['success'] for x in r]);means[c].append(p)
                    boots[c].append(rng.binomial(len(r),p,20000)/len(r))
            boots[c]=np.asarray(boots[c]).reshape(len(streams),5,20000)
        diffboot=boots['O']-boots['U'];diff=float(np.mean(means['O'])-np.mean(means['U']));ci=np.quantile(diffboot.mean(axis=(0,1)),[.025,.975]).tolist()
        sd={str(s):float(np.mean([r['success'] for r in allrows['O',s]])-np.mean([r['success'] for r in allrows['U',s]])) for s in streams}
        large=abs(diff)>=.1 and (ci[0]>0 or ci[1]<0) and all(x*diff>0 for x in sd.values())
        near=ci[0]>-.1 and ci[1]<.1
        overall[model]=dict(O=float(np.mean(means['O'])),U=float(np.mean(means['U'])),O_minus_U=diff,bootstrap95=ci,stream_differences=sd,
                            stream_difference_range=[min(sd.values()),max(sd.values())],large_effect=large,interval_within_10pp=near,
                            interpretation='LARGE_RESET_EFFECT' if large else 'NO_LARGE_EFFECT_DETECTED_CONDITIONAL_ON_STREAMS' if near else 'UNCERTAIN')
        for task in range(1,6):
            d=float(np.mean([means['O'][j*5+task-1]-means['U'][j*5+task-1] for j in range(len(streams))]));low,high=np.quantile(diffboot[:,task-1,:].mean(axis=0),[.025,.975])
            comparisons.append(dict(model=model,task=task,O_minus_U=d,bootstrap95_low=float(low),bootstrap95_high=float(high)))
            for c in ('O','U'):
                cell=[r for s in streams for r in allrows[c,s] if r['task']==task];k=sum(r['success'] for r in cell);lo,hi=wilson(k,len(cell))
                scores.append(dict(model=model,condition=c,stream='pooled',task=task,n=len(cell),successes=int(k),success_rate=k/len(cell),wilson95_low=lo,wilson95_high=hi))
            for s in streams:
                o=[r for r in allrows['O',s] if r['task']==task];u=[r for r in allrows['U',s] if r['task']==task]
                for key in ('initial_x','initial_y','initial_vx','initial_vy','goal_x','goal_y','initial_distance'):
                    x=np.array([r[key] for r in o]);y=np.array([r[key] for r in u]);pooled=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
                    ks=ks_2samp(x,y)
                    scenario_stats.append(dict(model=model,stream=s,task=task,feature=key,O_mean=float(x.mean()),U_mean=float(y.mean()),O_sd=float(x.std(ddof=1)),U_sd=float(y.std(ddof=1)),
                        mean_difference=float(x.mean()-y.mean()),standardized_mean_difference=float((x.mean()-y.mean())/pooled) if pooled else None,
                        O_q05=float(np.quantile(x,.05)),O_q50=float(np.quantile(x,.5)),O_q95=float(np.quantile(x,.95)),U_q05=float(np.quantile(y,.05)),U_q50=float(np.quantile(y,.5)),U_q95=float(np.quantile(y,.95)),KS_statistic=float(ks.statistic),KS_pvalue_descriptive=float(ks.pvalue)))
        for s in streams:
            for j,(o,u) in enumerate(zip(allrows['O',s],allrows['U',s])):
                a,b=raws['O',s][j],raws['U',s][j]
                same_steps=len(a['actions'])==len(b['actions']);same_traj=same_steps and np.array_equal(a['states'],b['states']) and np.array_equal(a['actions'],b['actions']) and a['flags']==b['flags']
                matching.append(dict(model=model,stream=s,task=o['task'],episode=o['episode'],full_initial_state_equal=o['full_scene_sha256']==u['full_scene_sha256'],
                     position_goal_equal=o['position_goal_sha256']==u['position_goal_sha256'],initial_qvel_max_abs=float(np.max(np.abs(a['initial']['qvel']-b['initial']['qvel']))),
                     states_actions_flags_exact_equal=same_traj,success_equal=o['success']==u['success'],steps_equal=same_steps))
    csv_rows(ROOT/'results/B_TASK_SCORES.csv',scores);csv_rows(ROOT/'results/B_TASK_DIFFERENCES.csv',comparisons)
    csv_rows(ROOT/'results/B_SCENE_DISTRIBUTIONS.csv',scenario_stats);csv_rows(ROOT/'results/B_SCENE_MATCHING.csv',matching);csv_rows(ROOT/'results/B_FAILURE_FEATURES.csv',failure_rows)
    write(ROOT/'results/B_SUMMARY.json',dict(overall=overall,total_formal_episodes=size['total_formal_episodes'],reset_flow_pairs=len(matching),
          exact_position_goal_pairs=sum(x['position_goal_equal'] for x in matching),exact_full_state_pairs=sum(x['full_initial_state_equal'] for x in matching),
          exact_states_actions_flags_pairs=sum(x['states_actions_flags_exact_equal'] for x in matching),
          uncertainty='Fixed-checkpoint fixed-task fixed-stream independent-cell bootstrap; no training-seed or all-stream interval. Descriptive task intervals unadjusted.',
          wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0))
    print('ANALYSIS_DONE',flush=True)

if __name__=='__main__':main()

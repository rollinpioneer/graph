"""Actual two evaluator functions in separate matched simulator instances."""
from unittest.mock import patch
from eval_core import *

def diff(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
class OneTaskView:
    def __init__(self,u,task):self.u=u;self.task_infos=[copy.deepcopy(u.task_infos[task-1])]
    def __getattr__(self,name):return getattr(self.u,name)
class OneTaskTrace(Trace):
    def __init__(self,env,scene):
        super().__init__(env,scene);self.view=OneTaskView(env.unwrapped,scene['task'])
    @property
    def unwrapped(self):return self.view
def main():
    start=time.monotonic();cpu=time.process_time();cases=[];scenes=[]
    for kind in ('navigate','stitch'):
        agent,identity=load(kind);base=54001+(kind=='stitch');generator=env_new(kind);setup_stream(generator,base)
        selected=[]
        for task in range(1,6):
            for ep in range(5):
                seed=base+task*10000+ep;generator.action_space.seed(seed+2000000)
                _,info=generator.reset(seed=seed,options={'task_id':task,'render_goal':False})
                selected.append((task,ep,snapshot(generator,info,seed)))
        generator.close()
        for task,ep,scene in selected:
            name=f'{kind}_task{task}_scene{ep}';scenes.append(dict(case=name,scene=scene))
            official=Trace(env_new(kind),scene)
            _,official_traj,_=official_evaluate(agent,official,task_id=task,config={'discrete':False},num_eval_episodes=1,num_video_episodes=0,eval_temperature=0)
            official.close()
            custom=OneTaskTrace(env_new(kind),scene)
            with patch.object(rd.ogbench,'make_env_and_datasets',return_value=custom):
                # RD01 function iterates all five tasks; replay only this one official task.
                result=rd.evaluate(agent,f'pointmaze-large-{kind}-v0',base,episodes=1)
            a,b=official.traces[0],custom.traces[0]
            initial={key:diff(a['initial'][key],b['initial'][key]) for key in ('qpos','qvel','goal','goal_xy','integration')}
            rng_equal=all(digest(a['initial'][key])==digest(b['initial'][key]) for key in ('env_rng','action_rng','numpy_rng','python_rng','wrappers'))
            step_rows=[];maxa=maxp=maxv=0.;flags_ok=True;independent_ok=True
            for j,(x,y) in enumerate(zip(a['steps'],b['steps'])):
                da,dp,dv=diff(x['action'],y['action']),diff(x['qpos'],y['qpos']),diff(x['qvel'],y['qvel'])
                maxa=max(maxa,da);maxp=max(maxp,dp);maxv=max(maxv,dv)
                flags=all(x[k]==y[k] for k in ('reward','terminated','truncated','success'));flags_ok &= flags
                for z in (x,y):
                    succ=float(np.linalg.norm(z['qpos']-scene['goal_xy'])<=1.0)
                    expected_term=bool(succ and official.unwrapped._terminate_at_goal)
                    expected_trunc=(j+1)>=1000
                    independent_ok &= (z['success']==succ and z['terminated']==expected_term and z['truncated']==expected_trunc and z['reward']==succ)
                step_rows.append(dict(step=j+1,action_max_abs=da,qpos_max_abs=dp,qvel_max_abs=dv,
                     official_reward=x['reward'],custom_reward=y['reward'],official_terminated=x['terminated'],custom_terminated=y['terminated'],
                     official_truncated=x['truncated'],custom_truncated=y['truncated'],official_success=x['success'],custom_success=y['success'],
                     flags_equal=flags,official_qpos_x=float(x['qpos'][0]),official_qpos_y=float(x['qpos'][1]),
                     custom_qpos_x=float(y['qpos'][0]),custom_qpos_y=float(y['qpos'][1])))
            passed=max(initial.values())<=1e-12 and rng_equal and maxa<=1e-6 and maxp<=1e-5 and maxv<=1e-5 and flags_ok and independent_ok and len(a['steps'])==len(b['steps'])
            row=dict(case=name,model=kind,task=task,scene_episode=ep,initial_max_abs=max(initial.values()),initial_qpos_max_abs=initial['qpos'],
                     initial_qvel_max_abs=initial['qvel'],initial_goal_max_abs=initial['goal'],initial_integration_max_abs=initial['integration'],
                     initial_rng_equal=rng_equal,action_max_abs=maxa,qpos_max_abs=maxp,qvel_max_abs=maxv,
                     official_steps=len(a['steps']),custom_steps=len(b['steps']),flags_equal=flags_ok,independent_judgment_pass=independent_ok,passed=passed)
            cases.append(row);csv_rows(ROOT/'results'/f'A_steps_{name}.csv',step_rows)
            dump(ROOT/'runs'/f'A_{name}.pkl',dict(scene=scene,official=a,custom=b,checkpoint=identity))
            print('A_CASE',name,'PASS' if passed else 'FAIL',flush=True)
    dump(ROOT/'runs/A_MATCHED_SCENES.pkl',scenes);csv_rows(ROOT/'results/A_CASES.csv',cases)
    report=dict(status='PASS' if all(x['passed'] for x in cases) and len(cases)==50 else 'FAIL',cases=len(cases),
                max_action_difference=max(x['action_max_abs'] for x in cases),max_qpos_difference=max(x['qpos_max_abs'] for x in cases),
                max_qvel_difference=max(x['qvel_max_abs'] for x in cases),max_initial_difference=max(x['initial_max_abs'] for x in cases),
                initial_rng_all_equal=all(x['initial_rng_equal'] for x in cases),flags_all_equal=all(x['flags_equal'] for x in cases),
                independent_judgments_all_pass=all(x['independent_judgment_pass'] for x in cases),wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0)
    write(ROOT/'results/A_AUDIT.json',report);print('A_END',report['status'],flush=True)
    if report['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()

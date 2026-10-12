"""Separate action-key/video-designated random-stream consumption check."""
from eval_core import *
start=time.monotonic();cpu=time.process_time();agent,identity=load('navigate');paths={}
for videos in (0,1):
    env=Trace(env_new('navigate'));setup_stream(env,56001)
    # Rendering is explicitly stubbed in this isolated technical instance.
    env.unwrapped.render=lambda *args,**kwargs:np.zeros((4,4,3),dtype=np.uint8)
    first=[];counts=[]
    for task in (1,2):
        before=len(env.traces)
        _,_,rendered=official_evaluate(agent,env,task_id=task,config={'discrete':False},num_eval_episodes=2,
              num_video_episodes=videos if task==1 else 0,eval_temperature=0)
        first.append(env.traces[before]['initial']);counts.append(len(env.traces))
    paths[f'official_video_{videos}']=dict(first_scenes=first,reset_counts=counts,traces=env.traces,
                                         final_global_numpy=digest(np.random.get_state()))
    env.close()
env=env_new('navigate');setup_stream(env,56001);rows=[];raw=[]
for task in (1,2):
    for ep in (0,1):
        row,tr=episode(agent,env,56001,task,ep,'O');rows.append(row);raw.append(tr)
env.close();paths['unified_no_action_key_draw']=dict(rows=rows,raw=raw)
a,b=paths['official_video_0'],paths['official_video_1']
summary=dict(model='navigate',seed=56001,rendering='zero-image placeholder; actual extra reset/actions executed, no graphics context',
             official_no_video_reset_count=a['reset_counts'][-1],official_extra_video_reset_count=b['reset_counts'][-1],unified_reset_count=4,
             statistical_episodes_per_task=2,
             task1_same_without_before_video=digest(a['first_scenes'][0]['qpos'])==digest(b['first_scenes'][0]['qpos']),
             task2_position_changed_after_video=digest(a['first_scenes'][1]['qpos'])!=digest(b['first_scenes'][1]['qpos']),
             task2_goal_changed_after_video=digest(a['first_scenes'][1]['goal'])!=digest(b['first_scenes'][1]['goal']),
             task1_position_changed_by_official_action_key_draw=digest(a['first_scenes'][0]['qpos'])!=digest(raw[0]['initial']['qpos']),
             official_no_video_final_numpy=a['final_global_numpy'],official_video_final_numpy=b['final_global_numpy'],
             raw_rollouts=13,wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0)
dump(ROOT/'runs/TECHNICAL_RNG.pkl',dict(checkpoint=identity,paths=paths));write(ROOT/'results/TECHNICAL_RNG.json',summary)
print('TECHNICAL_RNG_DONE',flush=True)

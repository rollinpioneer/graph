"""Timing only, without reporting or inspecting success rates."""
from eval_core import *
start=time.monotonic();cpu=time.process_time();records=[]
for kind in ('navigate','stitch'):
    agent,identity=load(kind);env=env_new(kind);base=53001+(kind=='stitch');setup_stream(env,base)
    s,info=env.reset(seed=base+50000,options={'task_id':5,'render_goal':False})
    t=time.monotonic();jax.block_until_ready(mode(agent,s,info['goal']));compile_seconds=time.monotonic()-t
    t=time.monotonic();steps=episode(agent,env,base,5,0,'U',keep=False);seconds=time.monotonic()-t;env.close()
    records.append(dict(model=kind,steps=steps,seconds=seconds,seconds_per_step=seconds/steps,compile_seconds=compile_seconds,identity=identity))
perstep=max(r['seconds_per_step'] for r in records)
predicted=2000*1000*perstep+sum(r['compile_seconds']+r['identity']['load_seconds'] for r in records)+200
remaining=DEADLINE-time.time()-120
streams=2 if 2*predicted<=remaining else 1 if 2*(predicted/2+200)<=remaining else 0
write(ROOT/'receipts/TIMING_PROBE.json',dict(records=records,planned_2000_episode_upper_seconds=predicted,remaining_wall_seconds=remaining,
         safety_factor=2,wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0,success_metrics='not collected, printed or inspected'))
write(ROOT/'SAMPLE_SIZE.json',dict(streams=[55001,65001][:streams],episodes_per_task=50,tasks=5,models=2,conditions=['O','U'],total_formal_episodes=1000*streams,
         decision='pre-outcome timing-only freeze',reason='2x conservative maximum-step forecast versus remaining 4h wall cap',gpu_seconds=0))
print('TIMING_ONLY',perstep,'PREDICTED',predicted,'STREAMS',streams,flush=True)
if streams==0:raise SystemExit(3)

from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'evaluate.py';s=p.read_text()
s=s.replace("choices=['pilot','calibrate','A']","choices=['pilot','diagnostic','P1','validation']")
s=s.replace("a=p.parse_args()","p.add_argument('--scene-file',default='scenes_all.json');p.add_argument('--groups',default='12,13');p.add_argument('--arms',default='B0');a=p.parse_args()")
x=s.index("if a.mode=='A':");y=s.index('\ndef dump',x)
s=s[:x]+'''if a.mode in ['P1','validation']:
    frozen=json.loads((ROOT/('freeze_'+a.mode+'.json')).read_text())
    assert a.steps==200
    for relative,digest in frozen['source_and_scenes_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
'''+s[y:]
x=s.index('scenes=[]');y=s.index('if scenes:',x)
s=s[:x]+'''groups=[int(x) for x in a.groups.split(',')]
scenes=[s for s in json.loads((ROOT/'results'/a.scene_file).read_text()) if s['group'] in groups and s['condition']==a.condition]
arms=a.arms.split(',')
if a.mode=='validation':
    scenes=[dict(s,arm=arm) for arm in arms for s in scenes]
assert len(scenes)==a.parallel,(len(scenes),a.parallel)
'''+s[y:]
s=s.replace("if a.mode=='A' and t in [100,200]:","if a.mode in ['P1','validation','diagnostic'] and t in [100,200]:")
s=s.replace("'group':s['group'],'condition':a.condition","'arm':s.get('arm','B0'),'group':s['group'],'condition':a.condition")
s=s.replace("if t%10==0:print(json.dumps({'t':t,'elapsed_s':time.monotonic()-start,'fraction':(dd<.04).mean(-1).tolist()}),flush=True)","if t%10==0:print(json.dumps({'t':t,'elapsed_s':time.monotonic()-start}),flush=True)")
s=s.replace("obs,_,dd=step(infer(obs))","proposed=infer(obs)\n    executed=intervene(proposed,t)\n    obs,_,dd=step(executed)")
# Remove unused snapshot branch and cache interception so baseline observations stay official.
x=s.index('tensor_names=');y=s.index("print('LOADED'",x);s=s[:x]+s[y:]
x=s.index('    if a.mode==\'calibrate\' and t==20:');y=s.index('    if t%10==0:',x);s=s[:x]+s[y:]
x=s.index("if a.mode=='calibrate':");y=s.index("np.savez_compressed(out/'trajectory.npz'",x);s=s[:x]+s[y:]
s=s.replace("50 if a.mode=='calibrate' else a.steps","a.steps").replace("40 if a.mode=='calibrate' else a.steps","a.steps")
# Physical-time sampled net forces + body states. Net force cannot name a contact counterpart.
x=s.index('def infer(ob):')
s=s[:x]+'''net=gymtorch.wrap_tensor(raw.gym.acquire_net_contact_force_tensor(raw.sim)).view(a.parallel,-1,3)
body_names=raw.gym.get_actor_rigid_body_names(raw.envs[0],raw.frankas[0]) if hasattr(raw,'frankas') else []
body_map=[]
for actor in range(raw.gym.get_actor_count(raw.envs[0])):
    for local,name in enumerate(raw.gym.get_actor_rigid_body_names(raw.envs[0],actor)):
        body_map.append({'actor':actor,'name':name,'index_env':raw.gym.get_actor_rigid_body_index(raw.envs[0],actor,local,gymapi.DOMAIN_ENV)})
contacts_available=False
try:
    contacts=raw.gym.get_env_rigid_contacts(raw.envs[0]);contacts_available=True
except Exception as exc:
    contacts_reason=str(exc)
# Unsupported GPU pipeline pair API must not be invoked during formal run.
dump('contact_source.json',{'source':'net contact force tensor','counterpart_identifiable':False,'pair_api_initial_probe':contacts_available,'pair_probe_only_not_used':True,'sampling':'every official _refresh, including each controller/physics substep','sim_dt':cfg.get('sim',{}).get('dt'),'body_map':body_map})
physical={'time':[],'policy_step':[],'body_state':[],'net_force':[],'q':[],'qd':[],'effort':[]}
recording=[False];control_index=[0];refresh_official=raw._refresh
def measured_refresh(*args,**kwargs):
    answer=refresh_official(*args,**kwargs)
    if recording[0]:
        raw.gym.refresh_net_contact_force_tensor(raw.sim)
        physical['time'].append(raw.gym.get_sim_time(raw.sim));physical['policy_step'].append(control_index[0])
        physical['body_state'].append(raw._rigid_body_state.detach().cpu().numpy().copy())
        physical['net_force'].append(net.detach().cpu().numpy().copy())
        for key,attr in [('q','_q'),('qd','_qd'),('effort','_effort_control')]:physical[key].append(getattr(raw,attr).detach().cpu().numpy().copy())
    return answer
raw._refresh=measured_refresh
original_step=raw.pre_physics_step
# Baseline pre_physics_step remains untouched; measurement wraps only reads.
triggers=[None]*a.parallel
intervention_log=[]
proposed_actions=[]
def intervene(ac,t):
    proposed_actions.append(ac.copy());ex=ac.copy()
    if a.mode=='validation':
        rule=json.loads((ROOT/'intervention.json').read_text())
        for i,scene in enumerate(scenes):
            arm=scene.get('arm','B0')
            if triggers[i] is None and len(history['states'])>=21:
                st=np.array(history['states'][-21:])[:,i];dd=np.array(history['distances'][-21:])[:,i]
                motion=np.linalg.norm(st[...,:3]-st[0,:,:3],axis=-1).max()
                improvement=dd[0].mean()-dd.mean(-1).min()
                if not (st[...,2]<.975).any() and not (dd[-1]<.04).all() and motion<.001 and improvement<.001:
                    triggers[i]=t;intervention_log.append({'slot':i,'group':scene['group'],'arm':arm,'first_action_step':t,'detected_state_step':t-1})
            if triggers[i] is not None and 0<=t-triggers[i]<rule['duration_steps']:
                if arm=='B1':ex[i]=np.array(rule['B1_action'])
                elif arm=='B2':ex[i]=np.array(rule['B2_action'])
    return ex
'''+s[x:]
s=s.replace("ob,_,_,info=env.step(ac)","control_index[0]+=1;recording[0]=True\n    ob,_,_,info=env.step(ac)\n    recording[0]=False")
s=s.replace("np.savez_compressed(out/'trajectory.npz'", "np.savez_compressed(out/'physical.npz',**{k:np.array(v) for k,v in physical.items()})\ndump('interventions.json',intervention_log)\nnp.savez_compressed(out/'proposed_actions.npz',actions=np.array(proposed_actions))\nnp.savez_compressed(out/'trajectory.npz'")
p.write_text(s)
p=R/'run_job.sh';s=p.read_text().replace('/home/xushijie3/work/pc_scale02_20261011/experiments/pc_scale02_20261011',str(R));p.write_text(s)
print('ADAPTED')

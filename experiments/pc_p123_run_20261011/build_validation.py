from pathlib import Path
R=Path(__file__).resolve().parent;s=(R/'evaluate.py').read_text();x=s.index('def intervene(ac,t):');y=s.index('def infer(ob):',x)
s=s[:x]+'''def intervene(ac,t):
    proposed_actions.append(ac.copy());ex=ac.copy()
    if a.mode=='validation':
        rule=json.loads((ROOT/'intervention.json').read_text())
        for i,scene in enumerate(scenes):
            if scene.get('arm','B0')=='B1':ex[i]=ac[i]*rule['action_scale']
        if t==1:
            for i,scene in enumerate(scenes):
                intervention_log.append({'slot':i,'group':scene['group'],'arm':scene.get('arm','B0'),'start_action_step':1,'end_action_step':200,'action_scale':rule['action_scale'] if scene.get('arm','B0')=='B1' else 1.0,'trigger':'all control steps, no ground truth trigger'})
    return ex
'''+s[y:]
s=s.replace("main=yaml.safe_load","cfg['sim']['physx']['contact_collection']=1 # readout collection only; solver, dt, substeps unchanged\nmain=yaml.safe_load")
(R/'evaluate_validation.py').write_text(s)
s=(R/'run_job.sh').read_text().replace('evaluate.py','evaluate_validation.py');(R/'run_job_validation.sh').write_text(s)
s=(R/'launch_job.py').read_text().replace("ROOT/'run_job.sh'","ROOT/'run_job_validation.sh'");(R/'launch_validation.py').write_text(s)
print('P3_VALIDATION_RUNNER_BUILT')

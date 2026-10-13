"""Exactly one frozen D rollout per admitted condition and existing model."""
from core import *
import subprocess
def morphology(xy,actions):
    v=np.diff(xy,axis=0)
    def reverse(v):
        if len(v)<2:return 0
        n=np.linalg.norm(v,axis=1); den=n[:-1]*n[1:]
        return int(np.sum((n[:-1]>.01)&(n[1:]>.01)&(np.sum(v[:-1]*v[1:],axis=1)/np.maximum(den,1e-30)<-.5)))
    repeat=sum(bool(np.any(np.linalg.norm(xy[:t-2]-xy[t],axis=1)<=.1)) for t in range(3,len(xy)))
    return reverse(v),reverse(actions),repeat
def main():
    guard(); assert (ROOT/'SIGNAL_SAMPLE_FREEZE.json').exists(), 'ALL_SAMPLES_MUST_FREEZE_BEFORE_MODEL_QUERY'
    bank=pickle.load(open(ROOT/'runs/SCENE_GOAL_BANK.pkl','rb')); manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    assert sha(ROOT/'runs/SCENE_GOAL_BANK.pkl')==manifest['bank_sha256']
    e=env_new(); geo=Geometry(e); output=[]; code=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    write(ROOT/'receipts/D_STARTED.json',dict(epoch=time.time(),code_commit=code,bank_sha256=manifest['bank_sha256'],sample_freeze_sha256=sha(ROOT/'SIGNAL_SAMPLE_FREEZE.json')))
    for entry in manifest['models']:
        agent=load_agent(entry); variant=entry['variant']; seed=entry['seed']; out=ROOT/f'runs/D_{variant}_s{seed}'; out.mkdir()
        policy=lambda s,g:raw_mode(agent,np.asarray(s,dtype=np.float32),np.asarray(g,dtype=np.float32))
        for x in bank:
            if not x['admitted']:continue
            item=dict(state_id=x['condition_id'],group=x['bank'],cluster=str(x['start_cell']),source=x['source'],cell=x['start_cell'],scene=x['scene'],goals=[x['goal_xy']],goal_cells=[x['goal_cell']],firsts=[x['firsts']],routes=[geo.route(x['start_cell'],x['goal_cell'])])
            metric,trace=rollout(e,geo,item,0,policy,'formal_D'); xy=trace['states']; actions=trace['actions']; norms=np.linalg.norm(actions,axis=1); dr,ar,rv=morphology(xy,actions)
            endcell=geo.cell(xy[-1]); progress=x['distance']-geo.distances[x['goal_cell']][endcell] if endcell in geo.free else None
            metric.update(condition_id=x['condition_id'],bank=x['bank'],role=x['role'],start_cell=json.dumps(x['start_cell']),goal_cell=json.dumps(x['goal_cell']),region=x['region'],distance=x['distance'],firsts=json.dumps(x['firsts']),variant=variant,seed=seed,median_action_norm=float(np.median(norms)),graph_progress=progress,displacement_reversals=dr,action_reversals=ar,repeated_visit_steps=rv,small_command_candidate=bool(norms.mean()<=.25),near_stationary=bool(metric['maximum_displacement']<=.1),backtracking_evidence=bool(metric['path_length']>=1 and dr>=3 and rv>=2),first_exit_right_censored=metric['first_exit_step'] is None,first_exit_time_or_censor=metric['first_exit_step'] or 25,checkpoint_sha256=entry['sha256'],code_commit=code)
            trace.update(metric=metric,goal_xy=x['goal_xy']); dump(out/(x['condition_id']+'.pkl'),trace); output.append(metric)
        assert sha(entry['path'])==entry['sha256']; rows(ROOT/'BEHAVIOR_ROLLOUTS.csv',output)
        print('D_MODEL_COMPLETE',variant,seed,len(output),flush=True); del agent; jax.clear_caches()
    assert len(output)<=768; assert json.loads((ROOT/'receipts/CONTROL_BUDGET.json').read_text())['by_phase'].get('formal_D',0)<=19200
    e.close(); write(ROOT/'receipts/D_EXIT.json',dict(exit_code=0,epoch=time.time(),rollouts=len(output),csv_sha256=sha(ROOT/'BEHAVIOR_ROLLOUTS.csv')))
if __name__=='__main__':main()

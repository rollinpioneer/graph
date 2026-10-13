"""Deterministic CPU outputs, historical consistency, <=25-step frozen diagnostics."""
from core import *
import gc

def main():
    assert not (ROOT/'receipts/EXECUTION_STARTED.json').exists(),'NO_FORMAL_RERUN'
    frozen=json.loads((ROOT/'receipts/BANK_FROZEN.json').read_text());prepared=json.loads((ROOT/'receipts/PREPARE_EXIT.json').read_text())
    assert sha(ROOT/'runs/FROZEN_BANK.pkl')==frozen['bank_sha256']==prepared['bank_sha256']
    assert sha(ROOT/'PREREGISTRATION.json')==frozen['preregistration_sha256']==prepared['preregistration_sha256']
    assert sha(ROOT/'ADMISSION_AUDIT.json')==prepared['admission_sha256']
    admission=json.loads((ROOT/'ADMISSION_AUDIT.json').read_text());bank=pickle.load(open(ROOT/'runs/FROZEN_BANK.pkl','rb'));bank=[x for x in bank if x['state_id'] in admission['accepted']]
    eligible=admission['group_counts']['P1']>=16 and admission['P1_junctions']>=4 and admission['group_counts']['P2']>=8
    write(ROOT/'receipts/EXECUTION_STARTED.json',dict(epoch=time.time(),bank_sha256=frozen['bank_sha256'],eligible_main_mechanism=eligible,states=len(bank),formal_outputs_already_observed=False))
    e=env_new();geo=Geometry(e);entries=json.loads((ROOT/'MODEL_IDENTITY.json').read_text())['models'];sens=[];rolls=[];history=[]
    for item in entries:
        agent=load_agent(item);before=int(agent.network.step);label=f'{item["variant"]}_s{item["seed"]}'
        histpath=MECH/'runs'/f'Ceval_stitch_{item["variant"]}_s{item["seed"]}_1000000_full_trajectories.pkl'
        archived=pickle.load(open(histpath,'rb'));checks=[]
        for task in (1,3):
            record=next(x for x in archived if x['task']==task)
            assert record['checkpoint_sha256']==item['sha256']
            s,err=restore(e,record['initial']);assert np.max(np.abs(s-record['states'][0]))<=1e-10
            predicted=np.clip(np.asarray(raw_mode(agent,s,record['initial']['goal'])),-1,1);delta=float(np.max(np.abs(predicted-record['actions'][0])))
            assert delta<=1e-6,'CPU_HISTORICAL_ACTION_MISMATCH'
            checks.append(dict(task=task,scene_id=record['scene_id'],action_max_abs_error=delta,restore_error=err))
        history.append(dict(seed=item['seed'],variant=item['variant'],source_path=str(histpath),source_sha256=sha(histpath),checks=checks))
        del archived
        for state in bank:
            guard();actions=[];stats=[]
            for gi,g in enumerate(state['goals']):
                s,err=restore(e,state['scene'],g);raw=np.asarray(raw_mode(agent,s,g));a=np.clip(raw,-1,1);actions.append((raw,a))
                data_action=state['data_action'];ref=geo.xy(state['firsts'][gi][0])-geo.xy(state['cell'])
                stats.append(dict(raw=json.dumps(raw.tolist()),clipped=json.dumps(a.tolist()),raw_norm=float(np.linalg.norm(raw)),clipped_norm=float(np.linalg.norm(a)),direction_deg=angle(a),reference_direction_deg=angle(ref),reference_angle_error=between(a,ref),saturation_count=int(np.sum(np.abs(raw)>=1)),offline_action_cosine=None if data_action is None or np.linalg.norm(a)*np.linalg.norm(data_action)<=1e-12 else float(np.dot(a,data_action)/(np.linalg.norm(a)*np.linalg.norm(data_action))),offline_index=state['offline_index'],offline_alignment_status='EXACT_RECORDED_OBS_QPOS_QVEL_RECONSTRUCTED_PHYSICS' if data_action is not None else 'UNMEASURABLE'))
            d_raw=float(np.linalg.norm(actions[0][0]-actions[1][0]));d_clip=float(np.linalg.norm(actions[0][1]-actions[1][1]));pair_angle=between(actions[0][1],actions[1][1])
            for gi in (0,1):
                sens.append(dict(seed=item['seed'],variant=item['variant'],state_id=state['state_id'],group=state['group'],cluster=state['cluster'],source=state['source'],goal_index=gi,D_raw=d_raw,D_clipped=d_clip,goal_action_angle_deg=pair_angle,**stats[gi]))
                policy=lambda s,g:np.asarray(raw_mode(agent,s,g))
                metric,trace=rollout(e,geo,state,gi,policy,'formal');metric=dict(seed=item['seed'],variant=item['variant'],checkpoint_sha256=item['sha256'],**metric)
                trace['metric']=metric;path=ROOT/'runs'/f'{label}_{state["state_id"]}_g{gi}.pkl';dump(path,trace);rolls.append(dict(**metric,trace_path=str(path),trace_sha256=sha(path)))
            assert int(agent.network.step)==before==1000001
        assert sha(item['path'])==item['sha256'];rows(ROOT/'GOAL_SENSITIVITY.csv',sens);rows(ROOT/'BRANCH_ROLLOUTS.csv',rolls)
        write(ROOT/'receipts'/f'{label}_EXIT.json',dict(exit_code=0,epoch=time.time(),network_counter_before=before,network_counter_after=int(agent.network.step),model_sha256=item['sha256'],states=len(bank),rollouts=2*len(bank),CPU_devices=[str(x) for x in jax.devices()],affinity=sorted(os.sched_getaffinity(0)),new_updates=0,GPU_MiB=0));del agent;gc.collect();print('MODEL_COMPLETE',label,len(bank)*2,flush=True)
    write(ROOT/'receipts/CPU_HISTORY_AUDIT.json',dict(status='PASS',tolerance=1e-6,models=history))
    write(ROOT/'receipts/EXECUTION_EXIT.json',dict(exit_code=0,epoch=time.time(),models=len(entries),sensitivity_rows=len(sens),rollout_rows=len(rolls),control_steps=json.loads((ROOT/'receipts/CONTROL_BUDGET.json').read_text())['used'],new_training_updates=0,GPU_hours=0,GPU_MiB=0))
    e.close();print('EXECUTION_COMPLETE',len(rolls),flush=True)

if __name__=='__main__':main()

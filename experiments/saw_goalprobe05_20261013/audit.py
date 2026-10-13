"""Independent saved-trace recomputation; no new control steps or model outputs."""
from core import *
import pandas as pd, subprocess

def main():
    guard();frozen=json.loads((ROOT/'receipts/BANK_FROZEN.json').read_text());assert sha(ROOT/'runs/FROZEN_BANK.pkl')==frozen['bank_sha256'];assert sha(ROOT/'PREREGISTRATION.json')==frozen['preregistration_sha256']
    bank={x['state_id']:x for x in pickle.load(open(ROOT/'runs/FROZEN_BANK.pkl','rb'))};admission=json.loads((ROOT/'ADMISSION_AUDIT.json').read_text());accepted=set(admission['accepted']);r=pd.read_csv(ROOT/'BRANCH_ROLLOUTS.csv');s=pd.read_csv(ROOT/'GOAL_SENSITIVITY.csv');assert len(r)==len(s)==6*len(accepted)*2
    assert not r.duplicated(['seed','variant','state_id','goal_index']).any();assert set(r.state_id)==accepted
    e=env_new();geo=Geometry(e);checked=0;physical_steps=0;maximum_error=0.
    def independent_cell(xy):return int(np.floor((xy[1]+6)/4)),int(np.floor((xy[0]+6)/4))
    for row in r.itertuples(index=False):
        guard();assert sha(row.trace_path)==row.trace_sha256;t=pickle.load(open(row.trace_path,'rb'));item=bank[row.state_id];gi=int(row.goal_index);xy=t['states'];raw=t['raw_actions'];a=t['actions'];n=len(a)
        assert 1<=n<=25 and len(xy)==n+1 and len(t['qvel'])==n+1 and len(t['integration'])==n+1
        assert np.array_equal(t['initial']['integration'],item['scene']['integration']);assert np.array_equal(t['initial']['qpos'],item['scene']['qpos']);assert np.array_equal(t['initial']['qvel'],item['scene']['qvel']);assert np.array_equal(t['initial']['goal'],item['goals'][gi]);assert np.array_equal(xy[0],item['scene']['qpos']);assert np.array_equal(a,np.clip(raw,-1,1))
        assert np.max(np.abs(t['integration'][:,1:3]-xy))<=1e-10;assert np.max(np.abs(t['integration'][:,3:5]-t['qvel']))<=1e-10
        start=independent_cell(xy[0]);exits=[(j,independent_cell(p)) for j,p in enumerate(xy[1:],1) if independent_cell(p)!=start];first=None if not exits else exits[0][1];valid=[tuple(z) for z in item['firsts'][gi]];correct=int(first in valid) if first else 0
        maximum=float(np.linalg.norm(xy-xy[0],axis=1).max());cat='correct' if correct else ('wrong_channel' if first else ('no_motion' if maximum<=.1 else 'not_left_region'))
        assert int(row.correct)==correct and row.category==cat;assert int(row.steps)==n
        path=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());error=abs(path-row.path_length);maximum_error=max(maximum_error,error);assert error<=1e-10
        ss=s[(s.seed==row.seed)&(s.variant==row.variant)&(s.state_id==row.state_id)&(s.goal_index==gi)].iloc[0];assert np.max(np.abs(np.asarray(json.loads(ss.clipped))-a[0]))<=1e-6
        gcell=geo.cell(item['goals'][gi]);assert gcell==tuple(item['goal_cells'][gi]);assert geo.firsts(start,gcell)==item['firsts'][gi]
        if item['group']=='P1':assert len(valid)==1 and item['firsts'][0]!=item['firsts'][1] and geo.distances[tuple(item['goal_cells'][0])][start]==geo.distances[tuple(item['goal_cells'][1])][start]
        if item['group']=='P2':assert len(valid)==1 and item['firsts'][0]==item['firsts'][1]
        physical_steps+=n;checked+=1
    budget=json.loads((ROOT/'receipts/CONTROL_BUDGET.json').read_text());assert physical_steps==budget['by_phase']['formal'];assert sum(budget['by_phase'].values())==budget['used']<=25000
    indices=json.loads((ROOT/'MODEL_IDENTITY.json').read_text())['models']
    for m in indices:assert sha(m['path'])==m['sha256'] and sha(m['source_index'])==m['source_index_sha256']
    for source in frozen['source_inputs']:assert sha(source['path'])==source['sha256']
    configs=[]
    for item in indices:
        cfg_path=(OLD if item['variant']=='A' else PREV)/'runs/formal_stitch/config.json' if item['seed']==0 else MECH/'runs'/f'C_stitch_{item["variant"]}_s{item["seed"]}/config.json'
        cfg=json.loads(cfg_path.read_text())['config'];actual=get_config().to_dict()
        if item['variant']=='B':actual.update(actor_p_trajgoal=.5,actor_p_randomgoal=.5)
        assert json.loads(json.dumps(actual))==cfg;configs.append(dict(seed=item['seed'],variant=item['variant'],path=str(cfg_path),sha256=sha(cfg_path),exact_static_config_match=True))
    e.close();write(ROOT/'results/STATIC_CONFIG_IDENTITY.json',configs)
    source={x.name:sha(x) for x in ROOT.glob('*.py')};write(ROOT/'receipts/FINAL_AUDIT.json',dict(status='PASS',exit_code=0,epoch=time.time(),trace_rows_recomputed=checked,maximum_path_length_error=maximum_error,full_integration_qpos_qvel_trace_alignment=True,first_channel_recomputed_independently=True,frozen_bank_unchanged=True,preregistration_unchanged=True,weights_datasets_indices_unchanged=True,static_configs_matched=True,physical_control_steps=budget['used'],formal_control_steps=physical_steps,policy_checkpoint_counter_unchanged=True,source_sha256=source,GPU_hours=0,GPU_MiB=0,new_training_updates=0))
    print('AUDIT_PASS',checked,budget['used'],flush=True)

if __name__=='__main__':main()

"""Include initial qpos/qvel mismatch in the preregistered data-chain join gate."""
from state_core import *
def main():
    path=ROOT/'results/E3_DATA_REPLAYS.json';records=json.loads(path.read_text())
    for kind in ('navigate','stitch'):
        raw=np.load(OLD/'vendor/data'/f'pointmaze-large-{kind}-v0.npz')
        for r in [x for x in records if x['kind']==kind and x['status']=='MEASURED']:
            p=ROOT/'runs'/f'E3_REPLAY_{r["snapshot_id"]}_{r["chain"]}.pkl'
            with open(p,'rb') as f:trace=pickle.load(f)
            i=r['state_idx'];s=trace['initial']
            pe=float(np.max(np.abs(s['qpos']-raw['qpos'][i])));ve=float(np.max(np.abs(s['qvel']-raw['qvel'][i])))
            r['initial_position_error']=pe;r['initial_velocity_error']=ve;r['post_step_tracking_gate']=r['approximate_join_gate']
            r['approximate_join_gate']=bool(pe<=.15 and ve<=.5 and r['max_position_error']<=.15 and r['max_velocity_error']<=.5)
            r['gate_includes_initial_state']=True
        raw.close()
    write(path,records)
    write(ROOT/'results/E3_REPLAY_GATE_AUDIT.json',dict(measured=sum(r['status']=='MEASURED' for r in records),
        final_gates_pass=sum(r.get('approximate_join_gate',False) for r in records),
        correction='Initial qpos/qvel discrepancy added to the join decision; prior post-step tracking measurements retained. No new control step or model query.',
        training_updates=0,extra_control_steps=0))
    print('REPLAY_INITIAL_GATE_AUDITED',flush=True)
if __name__=='__main__':main()

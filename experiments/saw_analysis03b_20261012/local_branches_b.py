"""B endpoint replay of all four A gate-qualified failures, frozen before B results."""
from local_branches import *
def main():
    start=time.time()
    with open(ROOT/'runs/E2_SNAPSHOTS.pkl','rb') as f:snaps={x['id']:x for x in pickle.load(f)}
    with open(ROOT/'runs/E2_CANDIDATES.pkl','rb') as f:targets={x['snapshot_id']:x for x in pickle.load(f)}
    ids=['navigate_t2_e2_s300','navigate_t2_e3_s300','navigate_t5_e2_s300','navigate_t5_e3_s300']
    gates={x['snapshot_id']:x for x in json.loads((ROOT/'results/E2_GATES.json').read_text())}
    assert all(gates[k]['status']=='PASS' and not snaps[k]['source_success'] for k in ids)
    agent,identity=load_checkpoint('navigate','B',1000000);early,eid=load_checkpoint('navigate','B',300000);env=env_new('navigate');rows=[];raw=[]
    for sid in ids:
        check_time();snap=snaps[sid];target=targets[sid];restore_gate(env,snap['scene'])
        for c in ('C0','C1','C2','C3','C4'):
            row,trace=branch(env,snap,agent,early,c,target['near']['goal'],target['control']['goal']);row['variant']='B';trace['variant']='B';rows.append(row);raw.append(trace)
        print('E2_B',sid,'DONE',flush=True)
    env.close();csv_rows(ROOT/'results/E2_B_BRANCHES.csv',rows);dump(ROOT/'runs/E2_B_BRANCH_TRAJECTORIES.pkl',raw)
    old=list(csv.DictReader((ROOT/'results/E2_BRANCHES.csv').open()));paired=[]
    for b in rows:
        a=next(x for x in old if x['snapshot_id']==b['snapshot_id'] and x['condition']==b['condition'])
        paired.append(dict(snapshot_id=b['snapshot_id'],condition=b['condition'],A_local_reached=a['local_reached'],B_local_reached=b['local_reached'],
            A_far_success=a['original_far_success'],B_far_success=b['original_far_success'],A_displacement=float(a['net_displacement']),B_displacement=b['net_displacement'],
            A_far_progress=float(a['far_progress']),B_far_progress=b['far_progress'],A_ineffective_steps=int(a['ineffective_steps']),B_ineffective_steps=b['ineffective_steps']))
    csv_rows(ROOT/'results/E2_AB_MATCHED_FAILED.csv',paired)
    write(ROOT/'receipts/E2_B_exit.json',dict(status='COMPLETED',snapshots=len(ids),branches=len(rows),control_steps=sum(x['steps'] for x in rows),
        seconds=time.time()-start,checkpoint=identity,early_checkpoint=eid,scope='pre-B-endpoint fixed explanatory replay of all A gate-qualified failures; no new unique snapshots'))
if __name__=='__main__':main()

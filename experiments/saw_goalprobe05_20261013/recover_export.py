"""Recover serialization only, reuse immutable bank and completed physical admission."""
from prepare import *

def main():
    assert not (ROOT/'receipts/PREPARE_EXIT.json').exists()
    assert (ROOT/'receipts/PREPARE_PROCESS_EXIT.txt').read_text().strip()=='1'
    frozen=json.loads((ROOT/'receipts/BANK_FROZEN.json').read_text());assert sha(ROOT/'runs/FROZEN_BANK.pkl')==frozen['bank_sha256'];assert sha(ROOT/'PREREGISTRATION.json')==frozen['preregistration_sha256']
    bank=pickle.load(open(ROOT/'runs/FROZEN_BANK.pkl','rb'));e=env_new();geo=Geometry(e);audit=[];accepted=[]
    for item in bank:
        repetitions=[]
        for _ in range(2):
            restore(e,item['scene'],item['goals'][0]);step(e,np.array([.123,-.234]),'export_recovery_restore_selfcheck');repetitions.append(snapshot(e,item['goals'][0],item['scene']['reset_seed']))
        assert equal(repetitions[0],repetitions[1])
        results=[]
        for gi in (0,1):
            p=ROOT/'runs'/f'admission_{item["state_id"]}_g{gi}.pkl';trace=pickle.load(open(p,'rb'));initial=trace['initial'];assert np.array_equal(initial['integration'],item['scene']['integration']);assert np.array_equal(initial['goal'],item['goals'][gi]);results.append(trace['metric'])
        valid=all(x['correct'] for x in results);item['admitted']=valid
        audit.append(dict(state_id=item['state_id'],group=item['group'],cluster=item['cluster'],restore_error=max(x['restore_error'] for x in results),full_integration_repeat_exact=True,admitted=valid,goals=results,reason='PASS' if valid else 'REFERENCE_CHANNEL_NOT_ENTERED_WITHIN_25'))
        if valid:accepted.append(item['state_id'])
    export(bank,audit,accepted,e,geo);e.close();write(ROOT/'receipts/EXPORT_RECOVERY_EXIT.json',dict(exit_code=0,epoch=time.time(),policy_outputs_before_recovery=0,reused_admission_rollouts=len(bank)*2,new_admission_rollouts=0,repeated_restore_steps=len(bank)*2,bank_unchanged=True,preregistration_unchanged=True,change='native JSON encoding for NumPy integer geometry labels only'))
    print('EXPORT_RECOVERY_COMPLETE',len(bank),len(accepted),flush=True)
if __name__=='__main__':main()

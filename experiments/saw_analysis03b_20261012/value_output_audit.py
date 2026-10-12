"""Compare already saved matched queries; no new rollout, model call or update."""
from analyze import *

def main():
    rows=[]
    fields=('v1','v2','w1','w2','waypoint_adv','log_rho','rho','one_step_adv','log_awr','awr')
    for suffix,label in (('CANDIDATES','fixed16'),('VERIFIED_TARGETS','verified_near_control')):
        a=read(ROOT/'results'/f'E3_A_{suffix}.csv');b=read(ROOT/'results'/f'E3_B_{suffix}.csv')
        discriminator='candidate' if suffix=='CANDIDATES' else 'target'
        key=lambda r:(r['kind'],r['snapshot_id'],r['checkpoint_step'],r[discriminator])
        aa={key(r):r for r in a};bb={key(r):r for r in b};assert len(aa)==len(a) and len(bb)==len(b)
        for kind in ('navigate','stitch'):
            for step in (100000,300000,500000,1000000):
                keys=sorted(k for k in aa if k in bb and k[0]==kind and int(k[2])==step)
                if not keys:continue
                for k in keys:
                    for name in ('state_idx','goal_idx','candidate_idx','candidate_x','candidate_y','waypoint_idx'):
                        if name in aa[k]:assert aa[k][name]==bb[k][name],(k,name)
                maximum=max(abs(float(aa[k][f])-float(bb[k][f])) for k in keys for f in fields)
                head_max=max(abs(float(aa[k][f])-float(bb[k][f])) for k in keys for f in ('v1','v2','w1','w2'))
                weight_max=max(abs(float(aa[k][f])-float(bb[k][f])) for k in keys for f in ('rho','awr'))
                actor=max(abs(float(aa[k][f'actor_{d}'])-float(bb[k][f'actor_{d}'])) for k in keys for d in range(2))
                teacher=max(abs(float(aa[k][f'teacher_{d}'])-float(bb[k][f'teacher_{d}'])) for k in keys for d in range(2))
                rows.append(dict(kind=kind,checkpoint_step=step,query_set=label,matched_queries=len(keys),value_head_max_abs=head_max,clipped_weight_max_abs=weight_max,value_and_weight_max_abs=maximum,
                    value_and_weight_exact_equal=maximum==0,actor_action_max_abs=actor,teacher_action_max_abs=teacher,
                    scope='existing fixed queries only; no proof of value correctness or unique failure cause'))
    save(ROOT/'results/VALUE_OUTPUT_IDENTITY.csv',rows)
    write(ROOT/'receipts/VALUE_OUTPUT_AUDIT.json',dict(status='COMPLETED',stage_query_sets=len(rows),additional_model_queries=0,additional_control_steps=0,training_updates=0))
    for r in rows:print(r['kind'],r['checkpoint_step'],r['query_set'],r['matched_queries'],r['value_and_weight_max_abs'])
if __name__=='__main__':main()

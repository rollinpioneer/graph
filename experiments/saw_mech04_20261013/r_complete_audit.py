"""Explicit negative denominators and identical-weight value-query audit."""
from summarize import *
def main():
    negative=[];identity=[]
    for kind in ['navigate','stitch']:
        for x in read(P/'results'/f'R_SUPPORT_{kind}.csv'):
            if x['status']!='PASS':negative.append(dict(kind=kind,case_id=x['case_id'],anchor=x['anchor'],negative_type=x['status'],detail=f"neighbors_eligible={x['neighbors_eligible']}; no tolerance relaxation"))
        for x in read(P/'results'/f'R_EARLIEST_DIVERGENCE_{kind}.csv'):
            if x['triggered']!='True':negative.append(dict(kind=kind,case_id=x['case_id'],anchor='',negative_type='NO_PRESET_ACTION_DIVERGENCE',detail='Prespecified descriptive anchor retained'))
    categories=read(P/'results/R_NEGATIVE_SELECTION_stitch.csv')
    for x in categories:negative.append(dict(kind='stitch',case_id='',anchor='',negative_type='SELECTION_DENOMINATOR',detail=json.dumps(x,ensure_ascii=False)))
    table(P/'results/R_NEGATIVE_CASES.csv',negative)
    tuples=json.loads((P/'results/R_TUPLE_IDENTITIES_stitch.json').read_text())
    for x in tuples:
        prefix=f"{x['case_id']}_t{x['anchor']}"
        for gt in ['F','R']:
            a=np.load(P/'runs'/f'{prefix}_A_1000000_{gt}_signals.npz');b=np.load(P/'runs'/f'{prefix}_B_1000000_{gt}_signals.npz')
            errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['v1','v2','target_v1','target_v2','log_awr','log_wp','AWR_weight','WP_weight']}
            assert max(errors.values())==0
            identity.append(dict(case_id=x['case_id'],anchor=x['anchor'],goal_type=gt,n=1024,all_value_target_and_weight_queries_exact=True,max_abs_error=0))
    table(P/'results/R_VALUE_QUERY_IDENTITY.csv',identity)
    mechanism=[dict(hypothesis='R1',support='same-state clipped action divergence; matched early-late behavioral loss; 9/10->0/10 new seed0 scenes',counterexample='Task1/2 are controls; no cross-training-seed law yet',unidentified='which training signal causes mapping drift'),dict(hypothesis='R2',support='native F/R weights, data/teacher actions and original loss gradients measured at eligible anchors',counterexample='most gradients not negatively aligned; local targets often feasible',unidentified='causal assignment to AWR/Waypoint/weights'),dict(hypothesis='H1',support='same future component can be isolated in prepared T1/T2',counterexample='full A/B changes future dose and random supervision simultaneously',unidentified='short intervention required'),dict(hypothesis='H2',support='Stitch physical data chains and actions justify a controlled T3-T1 test',counterexample='local random-waypoint teacher success13/13 or12/13 is not proof of harmful random targets',unidentified='net effect on far-goal learned skills until T'),dict(hypothesis='H3',support='same values yield same tuple weights; all selected tuples retained',counterexample='parameter equality does not prove values or weights correct on random targets',unidentified='no uniquely identified submechanism')]
    table(P/'results/R_MECHANISM_MATRIX.csv',mechanism)
    print('R_NEGATIVE_VALUE_AUDIT_PASS',len(negative),len(identity))
if __name__=='__main__':main()

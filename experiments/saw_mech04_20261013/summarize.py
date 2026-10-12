"""All prespecified tasks/seeds/stages; finite-weight scene uncertainty is not training-seed uncertainty."""
import csv,json,pathlib,numpy as np
P=pathlib.Path(__file__).resolve().parent;G=np.random.default_rng(2030130)
def read(p):return list(csv.DictReader(open(p)))
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(p,rows):
    if not rows:return
    with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def wilson(y):
    n=len(y);m=float(np.mean(y));z=1.959963984540054;d=1+z*z/n;c=(m+z*z/(2*n))/d;h=z*np.sqrt(m*(1-m)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)
def paired(x,y):
    a={r['scene_id']:r for r in x};b={r['scene_id']:r for r in y};assert set(a)==set(b)
    ids=sorted(a);assert all(a[k]['scene_sha256']==b[k]['scene_sha256'] for k in ids)
    delta=np.array([int(a[k]['success'])-int(b[k]['success']) for k in ids]);cells={}
    for j,k in enumerate(ids):cells.setdefault((a[k]['task'],a[k]['group']),[]).append(j)
    draws=np.zeros(10000)
    for idx in cells.values():draws+=delta[G.choice(idx,(10000,len(idx)),replace=True)].sum(axis=1)/len(ids)
    lo,hi=np.quantile(draws,[.025,.975]);groups={str(g):float(np.mean([int(a[k]['success'])-int(b[k]['success']) for k in ids if int(a[k]['group'])==g])) for g in (0,1)}
    return dict(n=len(ids),difference_pp=float(delta.mean()*100),ci95_low_pp=float(lo*100),ci95_high_pp=float(hi*100),first_success_second_failure=sum(int(a[k]['success'])==1 and int(b[k]['success'])==0 for k in ids),first_failure_second_success=sum(int(a[k]['success'])==0 and int(b[k]['success'])==1 for k in ids),group0_difference_pp=groups['0']*100,group1_difference_pp=groups['1']*100)
def main():
    cdata={};scores=[];stream_scores=[]
    for p in sorted((P/'results').glob('Ceval_*_episodes.csv')):
        rows=read(p);r=rows[0];key=(r['kind'],r['variant'],int(r['training_seed']),int(r['checkpoint_step']));cdata[key]=rows
        for task in [0,1,2,3,4,5]:
            if task==0 and r['scope']!='full':continue
            selected=rows if task==0 else [x for x in rows if int(x['task'])==task]
            if not selected:continue
            full=all(x['complete']=='True' for x in selected);y=[int(x['success']) for x in selected if x['complete']=='True'];lo,hi=wilson(y) if y else (None,None)
            scores.append(dict(kind=key[0],variant=key[1],training_seed=key[2],checkpoint_step=key[3],scope=r['scope'],task=task,n=len(selected),completed_n=len(y),success=sum(y),success_rate=float(np.mean(y)) if y and full else None,ci95_low=lo,ci95_high=hi,valid_full_endpoint=full and r['scope']=='full' and len(rows)==500,seed_role='discovery' if key[2]==0 else 'independent_training_confirmation'))
            for group in (0,1):
                z=[x for x in selected if int(x['group'])==group];yy=[int(x['success']) for x in z if x['complete']=='True'];a,b=wilson(yy) if yy else (None,None)
                stream_scores.append(dict(kind=key[0],variant=key[1],training_seed=key[2],checkpoint_step=key[3],scope=r['scope'],task=task,group=group,stream=z[0]['scene_stream'] if z else '',n=len(z),completed_n=len(yy),success=sum(yy),success_rate=float(np.mean(yy)) if yy and len(yy)==len(z) else None,ci95_low=a,ci95_high=b))
    table(P/'results/C_TASK_SCORES.csv',scores)
    table(P/'results/C_STREAM_TASK_SCORES.csv',stream_scores)
    seed_summary=[]
    for kind,variant in [('stitch','A'),('stitch','B'),('navigate','A')]:
        for task in range(6):
            for scope,seeds in [('confirmation_seeds1_2',[1,2]),('all_observed_including_discovery_seed0',[0,1,2])]:
                z=[x for x in scores if x['kind']==kind and x['variant']==variant and x['task']==task and x['training_seed'] in seeds and x['checkpoint_step']==1000000 and x['valid_full_endpoint']]
                if not z:continue
                values=[x['success_rate']*100 for x in z]
                seed_summary.append(dict(kind=kind,variant=variant,task=task,analysis_scope=scope,n_training_seeds=len(z),all_requested_seeds_available=len(z)==len(seeds),mean_score_pp=float(np.mean(values)),sample_sd_pp=float(np.std(values,ddof=1)) if len(z)>1 else None,min_score_pp=min(values),max_score_pp=max(values),raw_seed_scores_pp=json.dumps({str(x['training_seed']):x['success_rate']*100 for x in z})))
    table(P/'results/C_TRAINING_SEED_SUMMARY.csv',seed_summary)
    contrasts=[];cs=[];cn=[]
    for seed in (0,1,2):
        aa=cdata.get(('stitch','A',seed,1000000));bb=cdata.get(('stitch','B',seed,1000000))
        if aa and bb and len(aa)==len(bb)==500 and all(r['complete']=='True' for r in aa+bb):
            for task in range(6):
                x=aa if task==0 else [r for r in aa if int(r['task'])==task];y=bb if task==0 else [r for r in bb if int(r['task'])==task]
                out=dict(kind='stitch',training_seed=seed,contrast='A_minus_B_1M',task=task,**paired(x,y));contrasts.append(out)
                if task==0:cs.append(dict(training_seed=seed,A=float(np.mean([int(r['success']) for r in aa]))*100,B=float(np.mean([int(r['success']) for r in bb]))*100,delta_pp=out['difference_pp']))
    for seed in (1,2):
        early=cdata.get(('navigate','A',seed,300000));late=cdata.get(('navigate','A',seed,1000000))
        if early and late:
            x=[r for r in early if int(r['task'])==5];y=[r for r in late if int(r['task'])==5]
            if len(x)==len(y)==100 and all(r['complete']=='True' for r in x+y):
                out=dict(kind='navigate',training_seed=seed,contrast='Task5_300k_minus_1M',task=5,**paired(x,y));contrasts.append(out);e=float(np.mean([int(r['success']) for r in x]))*100;l=float(np.mean([int(r['success']) for r in y]))*100
                def stage_rate(step):
                    rr=[r for r in cdata.get(('navigate','A',seed,step),[]) if r['task']=='5' and r['complete']=='True']
                    return float(np.mean([int(r['success']) for r in rr]))*100 if len(rr)==100 else None
                at100=stage_rate(100000);at500=stage_rate(500000)
                description='CAPABILITY_ALREADY_LOST_BEFORE_300K_PRIMARY_WINDOW' if e<30 and at100 is not None and at100>=30 else 'CAPABILITY_NOT_ESTABLISHED_AT_PRIMARY_300K' if e<30 else 'PREREGISTERED_DECLINE_CRITERION_MET' if e-l>=20 else 'CAPABILITY_RETAINED_OR_PRIMARY_DROP_BELOW_20PP'
                cn.append(dict(training_seed=seed,Task5_100k_descriptive=at100,Task5_300k=e,Task5_500k_descriptive=at500,Task5_1M=l,drop_pp=e-l,criterion=e>=30 and e-l>=20,stage_description=description))
    table(P/'results/C_PAIRED_CONTRASTS.csv',contrasts)
    nz=[x['drop_pp'] for x in cn]
    table(P/'results/C_N_DROP_SEED_SUMMARY.csv',[dict(n_training_seeds=len(nz),mean_drop_pp=float(np.mean(nz)) if nz else None,sample_sd_drop_pp=float(np.std(nz,ddof=1)) if len(nz)>1 else None,min_drop_pp=min(nz) if nz else None,max_drop_pp=max(nz) if nz else None,raw_drop_pp=json.dumps({str(x['training_seed']):x['drop_pp'] for x in cn}))])
    confirm=[x for x in cs if x['training_seed']>0]
    sc='PENDING' if len(confirm)!=2 else 'LARGE_DIFFERENCE_REPLICATED_TWO_SEEDS' if all(x['delta_pp']>=20 for x in confirm) else 'DIRECTION_CONSISTENT_MAGNITUDE_UNSTABLE' if all(x['delta_pp']>0 for x in confirm) else 'SEED_SENSITIVE_ORIGINAL_LARGE_GAP_UNSTABLE'
    nc='PENDING' if len(cn)!=2 else 'DECLINE_REPLICATED_TWO_SEEDS' if all(x['criterion'] for x in cn) else 'PREREGISTERED_300K_TO_1M_PATTERN_SEED_DEPENDENT' if any(x['Task5_300k']<30 for x in cn) else 'SEED0_STAGE_DECLINE_UNSTABLE'
    summary=dict(C_S=dict(status=sc,all_seed_scores=cs,confirmation_seed_count=len(confirm),delta_mean_pp=float(np.mean([x['delta_pp'] for x in confirm])) if confirm else None,delta_sample_sd_pp=float(np.std([x['delta_pp'] for x in confirm],ddof=1)) if len(confirm)>1 else None,delta_range_pp=[min(x['delta_pp'] for x in confirm),max(x['delta_pp'] for x in confirm)] if confirm else None),C_N=dict(status=nc,confirmation=cn),note='Wilson and paired stratified bootstrap describe these fixed weights/scenes. Independent training units are new seeds1/2, not 500 episodes. Seed0 discovery scores on new scenes do not replace original03B scores.')
    write(P/'results/C_SUMMARY.json',summary)
    td={};ts=[];tc=[]
    for p in sorted((P/'results').glob('Teval_*_episodes.csv')):
        rows=read(p);r=rows[0];key=(int(r['arm']),int(r['student_step']));td[key]=rows
        for task in range(6):
            z=rows if task==0 else [x for x in rows if int(x['task'])==task]
            if z and all(x['complete']=='True' for x in z):
                y=[int(x['success']) for x in z];lo,hi=wilson(y);ts.append(dict(arm=key[0],student_step=key[1],task=task,n=len(z),success=sum(y),success_rate=float(np.mean(y)),ci95_low=lo,ci95_high=hi))
    table(P/'results/T_TASK_SCORES.csv',ts)
    for first,second in [(1,0),(2,0),(2,1),(3,1),(3,0)]+[(a,-1) for a in range(4)]:
        aa=td.get((first,100000));bb=td.get((second,0 if second==-1 else 100000))
        if not aa or not bb or len(aa)!=500 or len(bb)!=500 or any(x['complete']!='True' for x in aa+bb):continue
        for task in range(6):
            x=aa if task==0 else [r for r in aa if int(r['task'])==task];y=bb if task==0 else [r for r in bb if int(r['task'])==task];out=paired(x,y)
            meaningful=abs(out['difference_pp'])>=15 and out['group0_difference_pp']*out['group1_difference_pp']>0 and out['ci95_low_pp']*out['ci95_high_pp']>0
            tc.append(dict(first_arm=first,second_arm=second,task=task,prespecified_main_task=task in (1,3),meaningful_short_effect=meaningful,**out))
    table(P/'results/T_PAIRED_CONTRASTS.csv',tc)
    approved=(P/'receipts/T_AUTHORIZATION.json').exists() and json.loads((P/'receipts/T_AUTHORIZATION.json').read_text())['approved']
    write(P/'results/T_SUMMARY.json',dict(status='NOT_EXECUTED_AWAITING_SEPARATE_APPROVAL' if not approved else 'FOUR_FINALS_COMPLETE' if all((a,100000) in td and len(td[a,100000])==500 and all(x['complete']=='True' for x in td[a,100000]) for a in range(4)) else 'PARTIAL_OR_PENDING',contrasts=tc,single_common_training_seed=0))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()

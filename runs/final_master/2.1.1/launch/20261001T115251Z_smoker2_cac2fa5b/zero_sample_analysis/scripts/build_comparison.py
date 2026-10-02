import json,glob,os,csv,calendar,time,statistics as st
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip()
ANA=OUT+'/zero_sample_analysis'; os.makedirs(ANA,exist_ok=True)
R='/home/xushijie2/graph_cp_disr_final_tb_launch/runs/final_master/2.1.1/T_B/'
RUNS=[('R-TB-E-0','B1-K+E','0','B1-K+E/seed_0/R-TB-E-0-*'),('R-TB-DK-1','B2','1','B2/seed_1/R-TB-DK-1-*'),('R-TB-K-1','B1-K','1','B1-K/seed_1/R-TB-K-1-*')]
ep=lambda s: calendar.timegm(time.strptime(s,'%Y-%m-%dT%H:%M:%SZ'))
manifest={}
for l in open(OUT+'/artifact_manifest.jsonl'):
    j=json.loads(l); manifest[j['path']]=j
state=json.load(open(OUT+'/launch_state.json'))['plans']
epoch=json.load(open(OUT+'/throughput/epoch_analysis.json'))
res=[]
for plan,method,seed,pat in RUNS:
    d=glob.glob(R+pat)[0]; rel=os.path.relpath(d,'/home/xushijie2/graph_cp_disr_final_tb_launch')
    js=json.load(open(d+'/job_summary.json'))
    r=dict(plan=plan,method=method,seed=int(seed),run_dir=rel)
    # eval points
    pts=[]
    files=sorted(glob.glob(d+'/eval_n_*.json'))+[d+'/eval_final.json']
    meta={int(row['update']):row for row in js['eval']}
    for f in files:
        e=json.load(open(f))
        rows=e['rows']
        succ=[x for x in rows if x['success']]
        ss=[x['success_seconds'] for x in succ]
        cp=e['checkpoint']; cprel=os.path.relpath(cp,'/home/xushijie2/graph_cp_disr_final_tb_launch')
        # find job_summary eval row by checkpoint path
        jr=[row for row in js['eval'] if row['checkpoint']==cp]
        pts.append(dict(file=os.path.basename(f),file_sha256=manifest[os.path.relpath(f,'/home/xushijie2/graph_cp_disr_final_tb_launch')]['sha256'],
            update=jr[0]['update'] if jr else None,skill_transitions=jr[0]['skill_transitions'] if jr else None,
            checkpoint=os.path.basename(cp),checkpoint_sha256=manifest.get(cprel,{}).get('sha256'),
            success_n=e['success_n'],n=e['n'],success_rate=e['success_rate'],mean_discounted_return=e['mean_discounted_return'],
            mean_success_seconds=(round(st.mean(ss),4) if ss else None),success_seconds_per_success=ss,
            per_case=[dict(case_id=x['case_id'],success=x['success'],G=x['G'],steps=x['steps'],reason=x['reason'],success_seconds=x.get('success_seconds')) for x in rows],
            reasons={k:sum(1 for x in rows if x['reason']==k) for k in sorted(set(x['reason'] for x in rows))},
            eval_action=e.get('eval_action'),label=e.get('label')))
    r['eval_points']=pts
    r['train']=dict(valid_transitions_N=js['valid_transitions'],interaction_seconds_T=js['interaction_seconds'],Ncap=16384,Tcap=js['Tcap'],
        stop_reason=js['stop_reason'],complete_updates=js['complete_updates'],fragment_updates=js['fragment_updates'],optimizer_steps=js['optimizer_steps'],
        train_success_episodes=js['train_success_episodes'],NaN_n=js['NaN_n'],hard_fail=js['hard_fail'],H=js['H'],d_ref=js['d_ref'],
        selected_checkpoint=os.path.basename(js['selected_checkpoint']))
    s=state[plan]; a=ep(s['started']); b=ep(s['finished'])
    ppo=[x['ppo_min'] for x in epoch[plan] if x.get('ppo_min') is not None]; col=[x['collect_min'] for x in epoch[plan]]
    r['compute']=dict(started_utc=s['started'],finished_utc=s['finished'],wall_hours=round((b-a)/3600,3),
        sum_complete_update_ppo_minutes=round(sum(ppo),1),sum_collection_minutes_between_updates=round(sum(col),1),
        note='wall clock on a shared 112-core server with shared GPUs and 1-3 concurrent workers; not a controlled compute-cost measurement; PPO/collection minutes exclude start-up, eval, final fragment update and fresh-load (see wall_hours)')
    prof=d+'/profiling_summary.json'
    if os.path.exists(prof): r['compute']['profiling_summary_first_update']=json.load(open(prof))
    res.append(r)
json.dump(res,open(ANA+'/comparison.json','w'),indent=1)
# CSV long table
with open(ANA+'/comparison_eval_points.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['plan','method','seed','eval_file','update','skill_transitions_N','success_n','n','mean_discounted_return','mean_success_seconds','reasons','checkpoint','checkpoint_sha256'])
    for r in res:
        for p in r['eval_points']:
            w.writerow([r['plan'],r['method'],r['seed'],p['file'],p['update'],p['skill_transitions'],p['success_n'],p['n'],p['mean_discounted_return'],p['mean_success_seconds'],json.dumps(p['reasons']),p['checkpoint'],p['checkpoint_sha256']])
print(json.dumps([{k:v for k,v in r.items() if k not in('eval_points',)} for r in res],indent=1)[:3500])
for r in res:
    print(r['plan'])
    for p in r['eval_points']:
        print('  ',p['file'],p['update'],p['skill_transitions'],p['success_n'],'/',p['n'],round(p['mean_discounted_return'],6),p['mean_success_seconds'],p['reasons'],p['checkpoint'])

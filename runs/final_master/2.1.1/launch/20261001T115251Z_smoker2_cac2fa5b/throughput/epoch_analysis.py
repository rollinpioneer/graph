import json,glob,os,calendar,time,statistics as st
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip()
R='/home/xushijie2/graph_cp_disr_final_tb_launch/runs/final_master/2.1.1/'
ep=lambda s: calendar.timegm(time.strptime(s,'%Y-%m-%dT%H:%M:%SZ'))
life={'R-TB-E-0':('2026-10-01T12:01:59Z','2026-10-01T20:55:11Z'),'R-TB-DK-1':('2026-10-01T12:40:36Z','2026-10-02T00:33:22Z'),'R-TB-K-1':('2026-10-01T13:59:06Z','2026-10-01T21:54:48Z')}
lifes={p:(ep(a),ep(b)) for p,(a,b) in life.items()}
def nlive(t0,t1):
    m=(t0+t1)/2; return sum(1 for a,b in lifes.values() if a<=m<=b)
pats={'R-TB-E-0':'T_B/B1-K+E/seed_0/R-TB-E-0-*','R-TB-DK-1':'T_B/B2/seed_1/R-TB-DK-1-*','R-TB-K-1':'T_B/B1-K/seed_1/R-TB-K-1-*'}
res={}
for p,pat in pats.items():
    d=glob.glob(R+pat)[0]
    pub={};T={}
    for l in open(d+'/persistence/raw_events.jsonl'):
        j=json.loads(l)
        if j.get('event')=='checkpoint_generation_published': pub[j['N']]=ep(j['time']); 
    start={}
    for l in open(OUT+'/logs/train_%s.log'%p):
        if 'PPO complete update' in l:
            n=int(l.split('N=')[1].split()[0]); start[n]=ep(l.split()[1])
    ns=sorted(pub); rows=[]
    # collection interval k: from pub[N_k] to start[N_{k+1}]
    for a,b in zip(ns,ns[1:]):
        if b in start:
            span=(start[b]-pub[a])/60.0; n=b-a
            rows.append(dict(N_from=a,N_to=b,collect_min=round(span,1),collect_per_min=round(n/span,1),workers_live_mid=nlive(pub[a],start[b]),ppo_min=round((pub[b]-start[b])/60,1) if b in pub else None))
    res[p]=rows
json.dump(res,open(OUT+'/throughput/epoch_analysis.json','w'),indent=1)
for p,rows in res.items():
    print(p)
    for r in rows: print('  ',r)
# per-worker mean collection rate by live-worker count (intervals fully within a single count)
print('--- mean collect_per_min by live workers (intervals entirely within one epoch only)')
for p,rows in res.items():
    by={}
    for r in rows: by.setdefault(r['workers_live_mid'],[]).append(r['collect_per_min'])
    print(p,{k:(round(st.mean(v),1),len(v)) for k,v in sorted(by.items())})

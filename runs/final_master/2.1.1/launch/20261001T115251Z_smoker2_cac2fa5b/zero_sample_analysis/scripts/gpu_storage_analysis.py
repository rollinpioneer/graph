import json,glob,calendar,time,statistics as st,os
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip()
R='/home/xushijie2/graph_cp_disr_final_tb_launch/runs/final_master/2.1.1/'
ep=lambda s: calendar.timegm(time.strptime(s,'%Y-%m-%dT%H:%M:%SZ'))
rows=[json.loads(l) for l in open(OUT+'/throughput/samples.jsonl') if l.strip()]
# E-0 PPO windows
d=glob.glob(R+'T_B/B1-K+E/seed_0/R-TB-E-0-*')[0]
pub={}
for l in open(d+'/persistence/raw_events.jsonl'):
    j=json.loads(l)
    if j.get('event')=='checkpoint_generation_published': pub[j['N']]=ep(j['time'])
start={}
for l in open(OUT+'/logs/train_R-TB-E-0.log'):
    if 'PPO complete update' in l: start[int(l.split('N=')[1].split()[0])]=ep(l.split()[1])
out={'E-0_PPO_windows_gpu0':[]}
for n in sorted(start):
    if n in pub:
        rs=[r for r in rows if start[n]<=r['t']<=pub[n]]
        if rs:
            u=[r['gpus']['0'][0] for r in rs]; m=[r['gpus']['0'][1] for r in rs]
            out['E-0_PPO_windows_gpu0'].append(dict(N=n,ppo_min=round((pub[n]-start[n])/60,1),gpu0_util_mean=round(st.mean(u),1),gpu0_mem_mib_mean=round(st.mean(m)),samples=len(rs),iowait_mean=round(st.mean(r['iowait_frac'] for r in rs),4),load1_mean=round(st.mean(r['loadavg'][0] for r in rs),1)))
# storage excursions: free < 7e9 (guard resume threshold)
exc=[];cur=None
for r in rows:
    f=r['free_bytes']
    if f<7.0e9:
        if cur is None: cur=dict(start=r['utc'],min_free=f,min_at=r['utc'])
        if f<cur['min_free']: cur['min_free']=f;cur['min_at']=r['utc']
        cur['end']=r['utc']
    else:
        if cur: exc.append(cur);cur=None
if cur: exc.append(cur)
out['storage_samples_free_below_7GB_intervals']=exc
out['storage_overall_min_free_bytes']=min(r['free_bytes'] for r in rows)
out['n_samples']=len(rows); out['first_sample_utc']=rows[0]['utc']; out['last_sample_utc']=rows[-1]['utc']
out['iowait_overall']={'mean':round(st.mean(r['iowait_frac'] for r in rows),4),'max':max(r['iowait_frac'] for r in rows)}
out['load1_overall_max']=max(r['loadavg'][0] for r in rows)
# guard events
ev={}
for l in open(OUT+'/storage_guard.jsonl'):
    j=json.loads(l); ev[j['event']]=ev.get(j['event'],0)+1
out['guard_event_counts']=ev
out['guard_min_heartbeat_free_bytes']=min(json.loads(l)['free'] for l in open(OUT+'/storage_guard.jsonl') if '"HEARTBEAT"' in l)
# per-run dir MB
for p,pat in {'R-TB-E-0':'T_B/B1-K+E/seed_0/R-TB-E-0-*','R-TB-DK-1':'T_B/B2/seed_1/R-TB-DK-1-*','R-TB-K-1':'T_B/B1-K/seed_1/R-TB-K-1-*'}.items():
    dd=glob.glob(R+pat)[0]
    out.setdefault('run_dir_MB',{})[p]=json.load(open(OUT+'/%s_completion_check.json'%{'R-TB-E-0':'e0','R-TB-DK-1':'dk1','R-TB-K-1':'k1'}[p]))['run_dir_MB']
json.dump(out,open(OUT+'/throughput/gpu_storage_analysis.json','w'),indent=1)
print(json.dumps(out,indent=1))

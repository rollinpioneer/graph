import json,glob,os,time,calendar,csv,torch,re
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip(); ANA=OUT+'/zero_sample_analysis'
REPO='/home/xushijie2/graph_cp_disr_final_tb_launch'; R=REPO+'/runs/final_master/2.1.1/T_B/'
man={}
for l in open(OUT+'/artifact_manifest.jsonl'):
    j=json.loads(l); man[j['path']]=j
rel=lambda p: os.path.relpath(p,REPO)
ep=lambda s: calendar.timegm(time.strptime(s,'%Y-%m-%dT%H:%M:%SZ'))
samples=[json.loads(l) for l in open(OUT+'/throughput/samples.jsonl') if l.strip()]
def free_at(t):
    s=min(samples,key=lambda r:abs(r['t']-t)); return s['free_bytes'] if abs(s['t']-t)<=45 else None
LOW_A,LOW_B=ep('2026-10-02T00:14:46Z'),ep('2026-10-02T00:17:17Z')
res={}
for plan,pat in [('R-TB-E-0','B1-K+E/seed_0/R-TB-E-0-*'),('R-TB-DK-1','B2/seed_1/R-TB-DK-1-*'),('R-TB-K-1','B1-K/seed_1/R-TB-K-1-*')]:
    d=glob.glob(R+pat)[0]; js=json.load(open(d+'/job_summary.json')); r={}
    # jsonl parse
    jl={}
    for fn in ['transition_log.jsonl','decision_log.jsonl','episode_log.jsonl','episode_priors.jsonl','fixed_mechanism_scope.jsonl','persistence/raw_events.jsonl']:
        p=d+'/'+fn
        if not os.path.exists(p): continue
        n=bad=0
        with open(p,'rb') as f:
            for line in f:
                if not line.strip(): continue
                n+=1
                try: json.loads(line)
                except Exception: bad+=1
        endsnl=open(p,'rb').read()[-1:]==b'\n'
        jl[fn]={'lines':n,'unparsable':bad,'ends_with_newline':endsnl}
    r['jsonl']=jl
    r['transition_lines_equals_N']=jl['transition_log.jsonl']['lines']==js['valid_transitions']
    # json/csv
    badj=[]
    for p in glob.glob(d+'/**/*.json',recursive=True):
        try: json.load(open(p))
        except Exception as e: badj.append(rel(p))
    r['unparsable_json_files']=badj; r['n_json_files']=len(glob.glob(d+'/**/*.json',recursive=True))
    rows=0
    for p in glob.glob(d+'/*.csv')+glob.glob(d+'/persistence/*.csv'):
        rows+=len(list(csv.reader(open(p))))
    r['csv_rows_total']=rows
    # checkpoints dir vs persistence copies
    mism=[];cmp=0
    for gdir in sorted(glob.glob(d+'/persistence/generations/*')):
        gen=os.path.basename(gdir)
        for src,dst in [('model.pt','.pt'),('model.json','.json'),('rng.json','.rng.json'),('episode.pkl','.episode.pkl')]:
            a=gdir+'/'+src; b=d+'/checkpoints/'+gen+dst
            if os.path.exists(a) and os.path.exists(b):
                cmp+=1
                if man[rel(a)]['sha256']!=man[rel(b)]['sha256']: mism.append(gen+dst)
            else: mism.append('MISSING '+gen+dst)
    r['generation_copy_compares']=cmp; r['generation_copy_mismatches']=mism
    # load all .pt
    okpt=0; badpt=[]
    for p in sorted(glob.glob(d+'/checkpoints/*.pt')):
        try:
            o=torch.load(p,map_location='cpu',weights_only=False); assert 'model' in o and 'optimizer' in o; okpt+=1
        except Exception as e: badpt.append((os.path.basename(p),str(e)[:80]))
    r['pt_loaded_ok']=okpt; r['pt_load_failures']=badpt
    # free space at each published generation
    ev=[json.loads(l) for l in open(d+'/persistence/raw_events.jsonl')]
    pubs=[]
    for e in ev:
        if e.get('event')=='checkpoint_generation_published':
            t=ep(e['time']); f=free_at(t); pubs.append({'generation':e['generation'],'time':e['time'],'free_GB_nearest_sample':None if f is None else round(f/1e9,2)})
    r['publish_events']=pubs
    r['min_free_GB_at_publish']=min([x['free_GB_nearest_sample'] for x in pubs if x['free_GB_nearest_sample'] is not None] or [None])
    # files modified inside the <2GiB window
    inwin=[]
    for root,_,fs in os.walk(d):
        for fn in fs:
            p=os.path.join(root,fn); m=os.path.getmtime(p)
            if LOW_A-120<=m<=LOW_B+120: inwin.append({'file':rel(p).split('/')[-1] if 'persistence' not in p and 'checkpoints' not in p else rel(p).split(plan.split('-')[-1])[-1][-60:],'mtime_utc':time.strftime('%FT%TZ',time.gmtime(m))})
    r['files_with_mtime_in_low_window_plus_minus_2min']=inwin
    # ENOSPC scan in logs
    pats=re.compile(r'No space left|ENOSPC|Errno 28|Disk quota|Input/output error|OSError|Traceback|nan|OOM|out of memory',re.I)
    hits={}
    for p in [OUT+'/logs/train_%s.log'%plan]+glob.glob(d+'/*.log'):
        if os.path.exists(p):
            h=[l.strip()[:160] for l in open(p,errors='replace') if pats.search(l)]
            hits[os.path.basename(p)]=h[:5]
    r['error_pattern_hits_in_logs']=hits
    res[plan]=r
json.dump(res,open(ANA+'/integrity_lowdisk_check.json','w'),indent=1)
for p,r in res.items():
    print(p,{k:r[k] for k in ('jsonl','transition_lines_equals_N','unparsable_json_files','n_json_files','generation_copy_compares','generation_copy_mismatches','pt_loaded_ok','pt_load_failures','min_free_GB_at_publish','error_pattern_hits_in_logs')})
    print('   in-window files',r['files_with_mtime_in_low_window_plus_minus_2min'])

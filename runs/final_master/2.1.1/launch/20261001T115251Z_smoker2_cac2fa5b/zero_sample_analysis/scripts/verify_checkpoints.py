import json,glob,os,hashlib,time
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip()
ANA=OUT+'/zero_sample_analysis'
REPO='/home/xushijie2/graph_cp_disr_final_tb_launch'
R=REPO+'/runs/final_master/2.1.1/T_B/'
RUNS=[('R-TB-E-0','B1-K+E/seed_0/R-TB-E-0-*'),('R-TB-DK-1','B2/seed_1/R-TB-DK-1-*'),('R-TB-K-1','B1-K/seed_1/R-TB-K-1-*')]
man={}
for l in open(OUT+'/artifact_manifest.jsonl'):
    j=json.loads(l); man[j['path']]=j
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''): h.update(b)
    return h.hexdigest()
rel=lambda p: os.path.relpath(p,REPO)
res={}
for plan,pat in RUNS:
    d=glob.glob(R+pat)[0]; js=json.load(open(d+'/job_summary.json'))
    fin=[f for f in os.listdir(d+'/checkpoints') if f.startswith('final_n_') and f.endswith('.pt')]
    assert len(fin)==1; fin=fin[0]; gen=fin[:-3]
    r={'plan':plan,'final_generation':gen,'checks':{}}
    ck=d+'/checkpoints/'+fin
    now=sha(ck); r['final_pt_sha256_recomputed_now']=now; r['final_pt_bytes']=os.path.getsize(ck)
    r['checks']['sha_now_equals_manifest']=(now==man[rel(ck)]['sha256'])
    # persistence generation copy
    gdir=d+'/persistence/generations/'+gen
    r['persistence_generation_files']=sorted(os.listdir(gdir)) if os.path.isdir(gdir) else None
    cmp={}
    if os.path.isdir(gdir):
        for fn in sorted(os.listdir(gdir)):
            a=man.get(rel(gdir+'/'+fn),{}).get('sha256'); b=man.get(rel(d+'/checkpoints/'+gen+'.'+fn.split('.',1)[1] if '.' in fn else ''),{}) .get('sha256') if False else None
        # compare same-named suffix files
        for suf in ['pt','json','rng.json','episode.pkl']:
            pa=gdir+'/'+[x for x in os.listdir(gdir) if x.endswith(suf)][0] if [x for x in os.listdir(gdir) if x.endswith(suf)] else None
            pb=d+'/checkpoints/'+gen+'.'+suf
            if pa and os.path.exists(pb):
                cmp[suf]={'generation_copy':os.path.basename(pa),'sha_equal':man[rel(pa)]['sha256']==man[rel(pb)]['sha256']}
    r['persistence_vs_checkpoints_dir']=cmp
    meta=json.load(open(d+'/checkpoints/'+gen+'.json'))['manifest']
    r['checkpoint_manifest']={k:meta.get(k) for k in ('N','T','Ncap','Tcap','H','attempt_id')}
    for k in meta:
        if 'hash' in k.lower() or 'sha' in k.lower() or 'commit' in k.lower(): r['checkpoint_manifest'][k]=meta[k]
    r['checks']['ckpt_manifest_N_equals_job_summary']=(meta.get('N')==js['valid_transitions'])
    r['checks']['ckpt_manifest_T_equals_job_summary']=(abs(meta.get('T')-js['interaction_seconds'])<1e-9)
    r['checks']['ckpt_manifest_attempt_id_equals_run']=(meta.get('attempt_id')==js['run_id'])
    fl=glob.glob(d+'/fresh_load_final_*.json'); flj=json.load(open(fl[0]))
    r['fresh_load']={'file':os.path.basename(fl[0]),'content':flj,'mtime_utc':time.strftime('%FT%TZ',time.gmtime(os.path.getmtime(fl[0]))),'sha256':man[rel(fl[0])]['sha256']}
    r['checks']['fresh_load_generation_equals_final']=(flj.get('generation')==gen and flj.get('model') is True and flj.get('adam') is True and flj.get('fresh_process') is True)
    r['final_pt_mtime_utc']=time.strftime('%FT%TZ',time.gmtime(os.path.getmtime(ck)))
    r['checks']['fresh_load_written_after_final_pt']=os.path.getmtime(fl[0])>=os.path.getmtime(ck)
    ef=json.load(open(d+'/eval_final.json'))
    r['eval_final_checkpoint']=os.path.basename(ef['checkpoint']); r['checks']['eval_final_ckpt_is_final_pt']=(os.path.realpath(ef['checkpoint'])==os.path.realpath(ck))
    r['eval_final_ident']={k:ef['hashes'].get(k) for k in('git_commit','plan_id','attempt_id')}
    r['checks']['eval_final_identity_matches_run']=(ef['hashes'].get('attempt_id')==js['run_id'] and ef['hashes'].get('git_commit')=='cac2fa5b3cafb46f1f74370314189f4bca4878fa')
    r['job_summary_selected_checkpoint']=os.path.basename(js['selected_checkpoint'])
    # published event
    ev=[json.loads(l) for l in open(d+'/persistence/raw_events.jsonl')]
    pubfin=[e for e in ev if e.get('generation')==gen]
    r['published_event']=pubfin
    r['checks']['published_N_T_match']=bool(pubfin) and pubfin[0]['N']==js['valid_transitions'] and abs(pubfin[0]['T']-js['interaction_seconds'])<1e-9
    r['all_checks_true']=all(r['checks'].values())
    res[plan]=r
json.dump(res,open(ANA+'/checkpoint_identity.json','w'),indent=1)
for p,r in res.items():
    print(p,r['final_generation'],'all_checks_true=',r['all_checks_true'])
    print('   sha256',r['final_pt_sha256_recomputed_now'],r['final_pt_bytes'])
    print('   checks',r['checks'])
    print('   persist-vs-ckpt',r['persistence_vs_checkpoints_dir'])
    print('   fresh_load',r['fresh_load']['content'],r['fresh_load']['mtime_utc'],'final_pt mtime',r['final_pt_mtime_utc'])
    print('   ckpt manifest extra keys',{k:v for k,v in r['checkpoint_manifest'].items() if k not in('N','T','Ncap','Tcap','H','attempt_id')})

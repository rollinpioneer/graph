import glob,json,hashlib,sys,time
for d in sorted(glob.glob('runs/final_master/c1_route_b/mech_confirm_v1/runs/B1-K+QMARK/seed_*/R-TB-C1-QMARK-*')):
    p=d+'/checkpoint_manifest.json'
    import os
    if os.path.exists(p): print('exists',d); continue
    files={}
    for ck in sorted(glob.glob(d+'/checkpoints/*.pt')):
        h=hashlib.sha256()
        with open(ck,'rb') as f:
            for b in iter(lambda:f.read(1<<20),b''): h.update(b)
        files[os.path.basename(ck)]=h.hexdigest()
    json.dump({'attempt_id':os.path.basename(d),'checkpoint_files':files,'created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'note':'sha256 recorded after training finished and before any Fresh Confirm evaluation'},open(p,'w'),indent=2,sort_keys=True)
    print('wrote',os.path.basename(d),len(files))

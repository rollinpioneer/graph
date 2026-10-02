import json,hashlib,os
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip(); REPO='/home/xushijie2/graph_cp_disr_final_tb_launch'
bad=[];n=0;missing=[]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''): h.update(b)
    return h.hexdigest()
for l in open(OUT+'/artifact_manifest.jsonl'):
    j=json.loads(l); p=REPO+'/'+j['path']
    if not os.path.exists(p): missing.append(j['path']); continue
    n+=1
    if os.path.getsize(p)!=j['bytes'] or sha(p)!=j['sha256']: bad.append(j['path'])
json.dump({'checked':n,'missing':missing,'changed':bad},open(OUT+'/zero_sample_analysis/manifest_reverify.json','w'),indent=1)
open('/home/xushijie2/tmp/r2/reverify.done','w').write('done')

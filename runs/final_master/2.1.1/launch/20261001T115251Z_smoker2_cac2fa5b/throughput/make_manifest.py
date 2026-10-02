import os,glob,hashlib,json,sys
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip()
R='/home/xushijie2/graph_cp_disr_final_tb_launch/runs/final_master/2.1.1/T_B/'
dirs=sorted(glob.glob(R+'*/seed_*/R-TB-*-20261001T115252Z-cac2fa5b'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''): h.update(b)
    return h.hexdigest()
tmp=OUT+'/artifact_manifest.jsonl.part'
with open(tmp,'w') as o:
    for d in dirs:
        for root,_,fs in os.walk(d):
            for fn in sorted(fs):
                p=os.path.join(root,fn); rel=os.path.relpath(p,'/home/xushijie2/graph_cp_disr_final_tb_launch')
                s=os.path.getsize(p)
                committed = s<=1<<20 and '/checkpoints/' not in rel.replace(d,'') + '/' and '/persistence/generations' not in p
                # checkpoint json metadata are small: commit them
                if '/checkpoints/' in p and p.endswith('.json'): committed = s<=1<<20
                o.write(json.dumps({'path':rel,'bytes':s,'sha256':sha(p),'in_git_commit':committed})+'\n')
os.rename(tmp,OUT+'/artifact_manifest.jsonl')
open(OUT+'/artifact_manifest.DONE','w').write('done\n')

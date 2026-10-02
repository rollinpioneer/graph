import json,glob,os,sys,hashlib
import torch
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip()
ANA=OUT+'/zero_sample_analysis'
REPO='/home/xushijie2/graph_cp_disr_final_tb_launch'
R=REPO+'/runs/final_master/2.1.1/T_B/'
man={}
for l in open(OUT+'/artifact_manifest.jsonl'):
    j=json.loads(l); man[j['path']]=j
rel=lambda p: os.path.relpath(p,REPO)
def load(p):
    try: return torch.load(p,map_location='cpu',weights_only=True)
    except Exception as e: return torch.load(p,map_location='cpu',weights_only=False)
def flat(sd,prefix=''):
    out={}
    if isinstance(sd,dict):
        for k,v in sd.items(): out.update(flat(v,prefix+'/'+str(k)))
    elif torch.is_tensor(sd): out[prefix]=sd
    return out
res={}
for plan,pat in [('R-TB-E-0','B1-K+E/seed_0/R-TB-E-0-*'),('R-TB-DK-1','B2/seed_1/R-TB-DK-1-*'),('R-TB-K-1','B1-K/seed_1/R-TB-K-1-*')]:
    d=glob.glob(R+pat)[0]; ck=d+'/checkpoints/'
    fin=[f for f in os.listdir(ck) if f.startswith('final_n_') and f.endswith('.pt')][0]
    r={}
    frag=ck+'update_fragment_1.pt'
    r['final_sha_equals_update_fragment_1_sha']=man[rel(ck+fin)]['sha256']==man[rel(frag)]['sha256']
    names=['n_000000.pt','n_004096.pt','n_008192.pt',fin]
    obj={n:load(ck+n) for n in names}
    top=obj[names[0]]
    r['checkpoint_top_level_type']=type(top).__name__
    r['checkpoint_top_level_keys']=list(top.keys())[:10] if isinstance(top,dict) else None
    mk=[k for k in (top.keys() if isinstance(top,dict) else []) if 'model' in k.lower() or 'policy' in k.lower() or 'state' in k.lower()]
    r['model_like_keys']=mk
    key=mk[0] if mk else None
    F={n:flat(obj[n][key]) if key else flat(obj[n]) for n in names}
    r['n_model_tensors']=len(F[names[0]])
    rows=[]
    import itertools
    for a,b in itertools.combinations(names,2):
        keys=F[a].keys()&F[b].keys()
        mx=0.0; nd=0; tot=0
        for k in keys:
            x,y=F[a][k],F[b][k]
            if x.shape!=y.shape or not x.is_floating_point(): continue
            dd=(x-y).abs().max().item(); mx=max(mx,dd); tot+=1; nd+= (dd>0)
        rows.append({'a':a,'b':b,'tensors_compared':tot,'tensors_differing':nd,'max_abs_diff':mx})
    r['pairwise_model_weight_diff']=rows
    res[plan]=r
json.dump(res,open(ANA+'/weights_identity.json','w'),indent=1)
for p,r in res.items():
    print(p,'final==fragment1 sha:',r['final_sha_equals_update_fragment_1_sha'],'type',r['checkpoint_top_level_type'],r['checkpoint_top_level_keys'],'model keys',r['model_like_keys'],'n tensors',r['n_model_tensors'])
    for x in r['pairwise_model_weight_diff']: print('   ',x['a'],x['b'],x['tensors_differing'],'/',x['tensors_compared'],'max|d|=%.3e'%x['max_abs_diff'])

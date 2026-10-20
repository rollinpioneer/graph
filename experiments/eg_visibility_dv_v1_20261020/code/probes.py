from common import *
import joblib,warnings
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor,as_completed

SETS=['F0','FC','FG','FGV','FGK','FGVK','shuffle','align']
LITE=np.array([j*22+i for j in range(4) for i in [0,1,2,3,4,9,10,11,12,13,14,19,20,21]])
def designs(f,a,raw,v,k,align,scene,frame,kmode):
 base=np.concatenate([f,a],1);k=k if kmode=='full' else k[:,LITE]
 perm=np.random.default_rng(seed(scene,frame)).permutation(len(f))
 def missing(x):return np.concatenate([x,(~np.isfinite(x)).astype(np.float32)],1)
 fg=np.concatenate([base,missing(raw)],1)
 return dict(F0=base,FC=np.concatenate([base,missing(raw[:,CIND])],1),FG=fg,FGV=np.concatenate([fg,v],1),FGK=np.concatenate([fg,missing(k)],1),FGVK=np.concatenate([fg,v,missing(k)],1),shuffle=np.concatenate([fg,v[perm],missing(k[perm])],1),align=np.concatenate([base,missing(align)],1))
def load_head():
 z=np.load(RP/'prepared.npz');kmode=json.load(open(R/'results/DV0.json'))['K_mode'];features={k:[] for k in SETS}
 ys=[];sc=[];fr=[];role=[];cd=[];co=[]
 for i,(scene,frame) in enumerate(zip(z['scene'],z['frame'])):
  zz=np.load(R/f'cache/head/{scene:04d}/{frame:04d}.npz');raw=np.load(LG/f'geometry/head/{scene:04d}/{frame:04d}.npz')['raw'][:,0]
  assert np.array_equal(zz['ids'],z['ids'][i]);assert np.allclose(zz['mu'],z['mu'][i,:,0])
  ds=designs(z['feature'][i],z['action'][i,:,0],raw,zz['V'],zz['K'],zz['align'],int(scene),int(frame),kmode)
  for k in SETS:features[k].append(ds[k])
  mu=zz['mu'];ys.append(np.stack([~zz['collision']&~zz['empty'],(mu>0)&(mu<=.400001),utility(mu)],1))
  cd.extend(zz['cd']);co.extend(zz['collision']);sc.extend([scene]*128);fr.extend([frame]*128);role.extend([z['role'][i]]*128)
 return {k:np.concatenate(v).astype(np.float32) for k,v in features.items()},np.concatenate(ys),np.array(sc),np.array(fr),np.array(role),np.array(cd),np.array(co),kmode
def train_one(task):
 key,kind,target=task;t=time.time();X=TRAIN_X[key];fit=TRAIN_ROLE=='fit'
 active=fit&(~TRAIN_CD if target==0 else ~TRAIN_CO if target==1 else np.ones(len(fit),bool))
 median=np.nanmedian(X[fit],0);median=np.nan_to_num(median)
 full=np.where(np.isfinite(X),X,median);mean=full[fit].mean(0);std=full[fit].std(0);std[std<1e-6]=1;full=((full-mean)/std).astype(np.float32)
 args=dict(hidden_layer_sizes=(64,),activation='relu',solver='adam',alpha=.0001,batch_size=512,learning_rate_init=.001,max_iter=50,early_stopping=False,n_iter_no_change=100,tol=0,random_state=20261020)
 if kind=='linear':model=LogisticRegression(C=1,max_iter=300,random_state=20261020) if target<2 else Ridge(alpha=1)
 else:model=MLPClassifier(**args) if target<2 else MLPRegressor(**args)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');model.fit(full[active],TRAIN_Y[active,target])
 pred=model.predict_proba(full)[:,1] if target<2 else model.predict(full)
 name=f'{key}_{kind}_t{target+1}';joblib.dump(dict(model=model,median=median,mean=mean,std=std,key=key,kind=kind,target=target),R/f'probes/{name}.joblib')
 np.save(R/f'probes/{name}_head.npy',pred.astype(np.float32))
 record=dict(name=name,seconds=time.time()-t,n_fit=int(active.sum()),iterations=np.asarray(getattr(model,'n_iter_',[])).tolist(),warnings=[str(w.message) for w in caught])
 write(R/f'results/probe_jobs/{name}.json',record);return record
def train():
 global TRAIN_X,TRAIN_Y,TRAIN_ROLE,TRAIN_CD,TRAIN_CO
 TRAIN_X,TRAIN_Y,sc,fr,TRAIN_ROLE,TRAIN_CD,TRAIN_CO,kmode=load_head()
 (R/'probes').mkdir(exist_ok=True);np.savez_compressed(R/'probes/head_index.npz',scene=sc,frame=fr,role=TRAIN_ROLE,y=TRAIN_Y,cd=TRAIN_CD,collision=TRAIN_CO)
 jobs=[(k,m,t) for k in SETS for m in ['linear','mlp'] for t in range(3)]
 # Fork shares read-only diagnostic matrices; all BLAS threads remain one.
 import multiprocessing
 context=multiprocessing.get_context('fork')
 base=json.load(open(R/'results/dispatch_status.json'))['worker_seconds'];spent=0
 with ProcessPoolExecutor(max_workers=8,mp_context=context) as pool:
  futures={pool.submit(train_one,j):j for j in jobs}
  for f in as_completed(futures):
   v=f.result();spent+=v['seconds'];print('PROBE_DONE',v['name'],v['seconds'],flush=True)
   write(R/'results/probe_status.json',dict(status='RUNNING',completed=len(list((R/'results/probe_jobs').glob('*.json'))),worker_seconds=spent,extraction_seconds=base))
   if base+spent>7*3600:raise RuntimeError('Budget reserve exhausted')
 write(R/'results/probe_status.json',dict(status='COMPLETE',completed=len(jobs),worker_seconds=spent,extraction_seconds=base,K_mode=kmode))
def evaluate_scene(scene):
 start=time.time();models=[joblib.load(p) for p in sorted((R/'probes').glob('*.joblib'))];kmode=json.load(open(R/'results/DV0.json'))['K_mode'];result=[]
 for frame in range(256):
  z=np.load(R/f'cache/c3/{scene:04d}/{frame:04d}.npz');ids=z['ids']
  with np.load(export(scene,frame)) as e:f=e['feature320'][ids];a=e['original_action'][ids]
  raw=np.load(LG/f'geometry/C3/{scene:04d}/{frame:04d}.npz')['raw'][ids,0]
  ds=designs(f,a,raw,z['V'],z['K'],z['align'],scene,frame,kmode)
  ys=[~z['collision']&~z['empty'],(z['mu']>0)&(z['mu']<=.400001),utility(z['mu'])]
  for m in models:
   x=ds[m['key']];x=(np.where(np.isfinite(x),x,m['median'])-m['mean'])/m['std'];t=m['target']
   score=m['model'].predict_proba(x)[:,1] if t<2 else m['model'].predict(x)
   mask=~z['cd'] if t==0 else ~z['collision'] if t==1 else np.ones(len(ids),bool)
   y=ys[t][mask];pr=score[mask];val=float(roc_auc_score(y,pr)) if t<2 and len(np.unique(y))==2 else float(spearmanr(y,pr).statistic) if t==2 and np.std(y)>0 and np.std(pr)>0 else np.nan
   result.append([scene,frame,m['key'],m['kind'],t+1,len(y),val])
  risk=np.max(z['V'][:,[2,3]]+z['V'][:,[12,13]]+z['V'][:,[22,23]],axis=1)+z['V'][:,39]
  y=ys[0][~z['cd']];score=-risk[~z['cd']]
  val=float(roc_auc_score(y,score)) if len(np.unique(y))==2 else np.nan
  result.append([scene,frame,'conservative','fixed',1,len(y),val])
 import pandas as pd
 pd.DataFrame(result,columns=['scene','frame','features','probe','target','n','metric']).to_csv(R/f'results/c3_probe_{scene}.csv',index=False)
 write(R/f'results/predict_jobs/{scene}.json',dict(seconds=time.time()-start,scene=scene));return scene,time.time()-start
def evaluate():
 with ProcessPoolExecutor(max_workers=12) as pool:
  for s,t in pool.map(evaluate_scene,C3):print('C3_PROBES',s,t,flush=True)
if __name__=='__main__':
 os.nice(10)
 if sys.argv[1]=='train':train()
 else:evaluate()

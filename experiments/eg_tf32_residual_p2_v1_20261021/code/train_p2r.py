from p2_residual import *
import argparse,resource
p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();start=time.time();os.nice(10)
cfg=json.loads((R/'frozen_config.json').read_text());gate=json.loads((R/'results/invariance_checks.json').read_text());assert gate['status']=='PASS' and gate['config_sha256']==sha(R/'frozen_config.json') and gate['residual_code_sha256']==sha(R/'code/p2_residual.py')
assert gate['head_inputs_sha256']==sha(R/'head_inputs.npz');d=dict(np.load(R/'head_inputs.npz'));fit=d['role']=='fit';init=json.loads((R/'results/readout_init.json').read_text());cal=init['calibration'];x=d['T'][fit].reshape(-1,64);s=d['S'][fit].reshape(-1);y=(d['Y'][:,:,0] if a.family!='Epot' else d['Y'].max(-1))[fit].reshape(-1);logit=d[f'logit_{a.seed}'][fit].reshape(-1);assert len(x)==24576
device='cuda:0';torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();xx=torch.tensor(x,device=device);ss=torch.tensor(s,device=device);yy=torch.tensor(y,device=device);ll=torch.tensor(logit.astype(float),device=device);theta=torch.nn.Parameter(torch.zeros(65,dtype=torch.float64,device=device));assert theta.numel()==65
opt=torch.optim.AdamW([theta],lr=.001,weight_decay=.0001);rng=np.random.default_rng(a.seed);order=rng.permutation(len(x));cursor=0;stream=hashlib.sha256();losses=[]
for step in range(1,1001):
 if cursor+128>len(order):order=rng.permutation(len(x));cursor=0
 ix=order[cursor:cursor+128];cursor+=128;stream.update(ix.tobytes());it=torch.tensor(ix,device=device);delta=xx[it]@theta[:64]+theta[64]
 pred=torch.sigmoid(ll[it]+delta) if a.family=='Gcap65' else cal[a.family]['alpha']*(ss[it]+delta)+cal[a.family]['beta']
 loss=((pred-yy[it])**2).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.item()))
 if step%100==0:print('TRAIN',a.family,a.seed,step,float(loss.item()),flush=True)
old=torch.load(L/f'heads/G/{a.seed}/step1000.pt',map_location='cpu',weights_only=False);assert stream.hexdigest()==old['sample_stream_sha256']
value=theta.detach().cpu().numpy();path=R/f'heads/{a.family}/{a.seed}/step1000.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,theta=value,seed=a.seed,family=a.family,step=1000)
with torch.no_grad():
 delta=xx@theta[:64]+theta[64];q=torch.sigmoid(ll+delta) if a.family=='Gcap65' else cal[a.family]['alpha']*(ss+delta)+cal[a.family]['beta'];ref=q.cpu().numpy()
reload=predict(a.family,np.load(path)['theta'],x,s,logit,cal);err=float(abs(ref-reload).max());assert err<=1e-9
out=dict(status='PASS',family=a.family,seed=a.seed,steps=1000,parameters=65,zero_initialized=True,frozen_backbone_and_G=True,sample_stream_sha256=stream.hexdigest(),sample_stream_matches_G=True,checkpoint=str(path),checkpoint_sha256=sha(path),config_sha256=sha(R/'frozen_config.json'),invariance_checks_sha256=sha(R/'results/invariance_checks.json'),training_code_sha256=sha(Path(__file__)),fit_MSE=float(np.mean((reload-y)**2)),reload_max_error=err,seconds=time.time()-start,start=start,end=time.time(),GPU=torch.cuda.get_device_name(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),losses=losses,C4_read=False)
write(R/f'results/train_{a.family}_{a.seed}.json',out);print('DONE',a.family,a.seed,out['fit_MSE'],flush=True)

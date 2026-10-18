from common_v2 import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args();seed=args.seed;os.nice(10);start=time.time();cpu=time.process_time();val=load('model_val');gate=load('gate_cal');history=[];preds={};m=old_model()
for step in range(100,1001,100):
 ck=torch.load(P/f'heads/{seed}/step{step}.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'])
 if step==100:xv=inputs(val,ck['mean'],ck['std']);xg=inputs(gate,ck['mean'],ck['std'])
 q=predict(m,xv);res=metrics(q,val);history.append(dict(step=step,checkpoint_sha256=sha(P/f'heads/{seed}/step{step}.pt'),**res));preds[step]=q
old=next(x for x in json.loads((P/'results/selected_heads.json').read_text()) if x['seed']==seed);selected=min(history,key=lambda x:(x['regret'],x['step']));chosen={}
folder=R/f'predictions/H0/{seed}';folder.mkdir(parents=True,exist_ok=True)
for rule,step in [('mse',old['selected_step']),('regret',selected['step'])]:
 ck=torch.load(P/f'heads/{seed}/step{step}.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);q=preds[step];gq=predict(m,xg);np.savez_compressed(folder/f'{rule}.npz',val=q,gate=gq);chosen[rule]=dict(step=step,checkpoint_sha256=sha(P/f'heads/{seed}/step{step}.pt'),thresholds=thresholds(q),model_val=next(x for x in history if x['step']==step))
write(R/f'results/H0_{seed}.json',dict(seed=seed,history=history,selected=chosen,seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,device='CPU inference of original GPU-trained checkpoints'))
print('H0_DONE',seed,old['selected_step'],selected['step'],flush=True)

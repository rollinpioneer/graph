"""Authorized T student updates only; frozen full reference and inherited Adam moments."""
from t_core import *
import argparse,signal,csv

def main(arm):
    label=f'T{arm}';assert json.loads((ROOT/'results/T_CPU_ADMISSION.json').read_text())['status']=='PASS'
    authorization=json.loads((ROOT/'receipts/T_AUTHORIZATION.json').read_text());assert authorization['approved'] is True
    manifest=json.loads((ROOT/'results/T_SHARED_STREAM.json').read_text());identity=manifest['source'];assert sha(identity['path'])==identity['sha256']
    env,data,val,cfg=load('pointmaze-large-stitch-v0','A');template=init_agent(data,cfg,0)
    with open(identity['path'],'rb') as f:source=pickle.load(f)
    ref=flax.serialization.from_state_dict(template,source['agent']);ref=jax.device_put(ref,jax.devices('gpu')[0]);params=subset(ref.network.params);opt=actor_optimizer(ref);rng=ref.rng
    np.random.set_state(source['numpy_rng']);random.setstate(source['python_rng'])
    directory=ROOT/'runs'/label;directory.mkdir(exist_ok=False);stop=[False];signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True));start=time.monotonic();step=0;cps=[];status='RUNNING'
    initial=flax.serialization.to_state_dict(jax.device_get(ref));initial_actor=flax.serialization.to_state_dict(jax.device_get(params));initial_opt=flax.serialization.to_state_dict(jax.device_get(opt))
    def save(step):
        sync((params,opt));agent=policy(ref,params).replace(rng=rng);s=flax.serialization.to_state_dict(jax.device_get(agent));assert all(np.array_equal(x,y) for key in s['network']['params'] if key not in ACTOR_KEYS for x,y in zip(jax.tree_util.tree_leaves(s['network']['params'][key]),jax.tree_util.tree_leaves(initial['network']['params'][key])))
        assert __import__('pickle').dumps(s['network']['opt_state'],4)==__import__('pickle').dumps(initial['network']['opt_state'],4)
        p=directory/f'params_{step}.pkl';state=dict(agent=s,reference=identity,student_step=step,reference_native_step=int(ref.network.step),actor_opt_state=flax.serialization.to_state_dict(jax.device_get(opt)),actor_rng=np.asarray(rng),numpy_rng=np.random.get_state(),python_rng=random.getstate(),stream_cursor=step,stream_manifest_sha256=sha(ROOT/'results/T_SHARED_STREAM.json'),latent_rng_path=manifest['chunks'][max(0,(step-1)//1000)]['rng_path'] if step else identity['path'],latent_rng_meaning='source NumPy state at step0; sealed chunk end thereafter; cursor locates exact candidates')
        with open(p,'xb') as f:pickle.dump(state,f,4)
        cps.append(dict(step=step,path=str(p),sha256=sha(p),frozen_parameters_and_reference_optimizer_equal=True,actor_optimizer_count=int(opt[0].count),reference_native_step=int(ref.network.step)))
        write_json(ROOT/'results'/f'{label}_checkpoints.json',cps)
    logfile=open(ROOT/'results'/f'{label}_training.csv','x',newline='');logger=csv.DictWriter(logfile,fieldnames=['step','elapsed_seconds','loss','awr_loss','waypoint_loss'],lineterminator='\n');logger.writeheader()
    save(0)
    try:
        for block,item in enumerate(manifest['chunks']):
            assert sha(item['path'])==item['sha256'];indices=np.load(item['path'],mmap_mode='r')
            for j in range(1000):
                if stop[0]:status='BUDGET_STOP';break
                batch=make_batch(data,indices[j],arm);params,opt,info=update(ref,params,opt,batch,arm);rng=jax.random.split(rng)[0];step+=1
                if step%1000==0:
                    sync(info);assert bool(info['gradient_finite']) and bool(info['parameter_finite']);assert all(np.isfinite(float(info[k])) for k in ('loss','awr_loss','waypoint_loss'))
                    row=dict(step=step,elapsed_seconds=time.monotonic()-start,**{k:float(info[k]) for k in ('loss','awr_loss','waypoint_loss')});logger.writerow(row);logfile.flush();write_json(ROOT/'receipts'/f'{label}_progress.json',row);print(label,'TRAIN',step,flush=True)
                if step in (25000,50000,100000):save(step)
            if stop[0]:break
        if step==100000 and not stop[0]:status='COMPLETE_100K'
    except Exception:status='FAILED';raise
    finally:
        if not any(x['step']==step for x in cps):save(step)
        write_json(ROOT/'results'/f'{label}_training_status.json',dict(status=status,student_updates=step,arm=arm,seconds=time.monotonic()-start,source=identity,shared_stream_manifest_sha256=sha(ROOT/'results/T_SHARED_STREAM.json'),initial_actor_sha256=hashlib.sha256(pickle.dumps(initial_actor,4)).hexdigest(),initial_actor_optimizer_sha256=hashlib.sha256(pickle.dumps(initial_opt,4)).hexdigest(),initial_agent_rng=initial['rng'],checkpoints=cps))
        logfile.close();env.close()
    print('T_DONE',label,status,step,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',type=int,choices=range(4),required=True);main(p.parse_args().arm)

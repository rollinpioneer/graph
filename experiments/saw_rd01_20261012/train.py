"""Joint original SAW training with complete checkpoints and isolated evaluation RNG."""
import argparse
import csv
import signal
from common import *

stop_requested=False


def stop_handler(signum, frame):
    global stop_requested
    stop_requested=True


def main(kind):
    signal.signal(signal.SIGUSR1,stop_handler)
    signal.signal(signal.SIGTERM,stop_handler)
    name=f'pointmaze-large-{kind}-v0'
    out=ROOT/'runs'/f'formal_{kind}'
    if out.exists():
        raise FileExistsError(f'Protect existing result directory: {out}')
    out.mkdir()
    env,data,val,config=load(name)
    env.action_space.seed(0); env.reset(seed=0,options={'task_id':1})
    agent=init_agent(data,config)
    sync(agent)
    write_json(out/'config.json',dict(config=config,training_seed=0,name=name,train_steps=1000000,
                                   eval_interval=100000,eval_episodes=50,eval_tasks=5,temperature=0,video_episodes=0,
                                   logging_interval=5000,eval_seed_rule='development: 71000 + 1000*dataset_index + update_counter',
                                   final_seed='confirmation: navigate 91001, stitch 91002; separate process after freeze',
                                   sampler='unmodified official HGCDataset; NumPy seed 0, validation/evaluation isolated',
                                   fresh_initialization=True,official_commit='3535a16c901e5de1233ac5a586401b3ddfa5e30d'))
    initial=checkpoint(agent,env,0,out)
    start=time.monotonic(); last=start; step=0; checkpoints=[initial]; evals=[]
    csv_file=open(ROOT/'results'/f'{kind}_training.csv','w',newline='')
    logger=None
    status='RUNNING'
    try:
        for step in range(1,1000001):
            if stop_requested: step-=1; status='BUDGET_STOP'; break
            batch=data.sample(config['batch_size'])
            agent,info=agent.update(batch)
            if step%5000==0:
                sync((agent,info))
                train={k:float(v) for k,v in info.items()}
                with separate_numpy(81000+step):
                    loss,vinfo=agent.total_loss(val.sample(config['batch_size'])); sync((loss,vinfo))
                train.update({'validation/'+k:float(v) for k,v in vinfo.items()})
                now=time.monotonic()
                row=dict(step=step,elapsed_seconds=now-start,block_seconds=now-last,**train)
                if logger is None: logger=csv.DictWriter(csv_file,fieldnames=list(row)); logger.writeheader()
                logger.writerow(row); csv_file.flush(); last=now
                write_json(ROOT/'receipts'/f'{kind}_progress.json',dict(step=step,elapsed_seconds=now-start,status='RUNNING',metrics=train))
                print(kind,'TRAIN',step,round(now-start,2),train['value/value_loss'],train['waypoint/waypoint_loss'],flush=True)
            if step==1 or step%100000==0:
                result=evaluate(agent,name,71000+(1000 if kind=='stitch' else 0)+step,50,out,f'development_{step}',lambda:stop_requested)
                evals.append(dict(step=step,success=result['overall_success'],tasks=result['tasks'],seconds=result['seconds']))
                write_json(ROOT/'results'/f'{kind}_development_eval.json',evals)
                print(kind,'DEVELOPMENT_EVAL',step,result['overall_success'],flush=True)
            if step in (100000,300000,500000,1000000):
                cp=checkpoint(agent,env,step,out); checkpoints.append(cp)
                write_json(ROOT/'results'/f'{kind}_checkpoints.json',checkpoints)
        if step==1000000 and not stop_requested: status='SINGLE_SEED_BASELINE'
    except RuntimeError as e:
        if str(e)=='RESOURCE_STOP_REQUESTED': status='BUDGET_STOP'
        else: status='FAILED'; raise
    finally:
        if status not in ('SINGLE_SEED_BASELINE',):
            cp=checkpoint(agent,env,step,out,label='stopped'); checkpoints.append(cp)
        write_json(ROOT/'results'/f'{kind}_checkpoints.json',checkpoints)
        write_json(ROOT/'results'/f'{kind}_training_status.json',dict(status=status,step=step,elapsed_seconds=time.monotonic()-start,
                                                                  stop_requested=stop_requested,checkpoints=checkpoints))
        csv_file.close(); env.close()
    print('FORMAL_DONE',kind,status,step,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=['navigate','stitch'],required=True)
    main(p.parse_args().kind)

"""Fresh confirmation rollout after complete 1M checkpoint; CPU-only process."""
import argparse
from common import *
from diagnose import restore


def main(kind):
    started=time.monotonic(); cpu=time.process_time()
    env,data,val,config,agent,identity=restore(kind)
    out=ROOT/'runs'/f'final_{kind}'
    if out.exists(): raise FileExistsError(str(out))
    out.mkdir()
    result=evaluate(agent,f'pointmaze-large-{kind}-v0',91001+(kind=='stitch'),50,out,'confirmation')
    result['checkpoint']=identity; result['cpu_process_seconds']=time.process_time()-cpu
    result['wall_seconds']=time.monotonic()-started; result['gpu_seconds']=0
    write_json(ROOT/'results'/f'{kind}_FINAL_EVAL.json',result)
    env.close();print('FINAL_EVAL_DONE',kind,result['overall_success'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['navigate','stitch'],required=True)
    main(p.parse_args().kind)

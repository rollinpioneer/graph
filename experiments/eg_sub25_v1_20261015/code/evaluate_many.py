import os,sys,json,argparse,concurrent.futures
from pathlib import Path
from evaluate_partition import task
R=Path(__file__).resolve().parents[1];D=R/'data_recovery'
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['dev','diagnostic']);args=parser.parse_args();os.nice(10);cfg=json.loads((D/'evaluation_run_config.json').read_text())
 parts=['dev'] if args.scope=='dev' else ['diagnostic_seen','diagnostic_novel'];specs=[(role,part,scene) for part in parts for role in cfg['inference_models'][part] for scene in cfg['partitions'][part]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['api_eval_processes']) as pool:
  futures=[pool.submit(task,s) for s in specs]
  for f in concurrent.futures.as_completed(futures):print('SCENE_COMPLETE',json.dumps(f.result()),flush=True)
 print('EVALUATION_SCOPE_COMPLETE',args.scope,flush=True)

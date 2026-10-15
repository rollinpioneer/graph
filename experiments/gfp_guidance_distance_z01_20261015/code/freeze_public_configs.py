from pathlib import Path
import csv,json
from omegaconf import OmegaConf
R=Path(__file__).resolve().parents[1]; U=R/'sources/gfp'
OmegaConf.register_new_resolver('eval',lambda x:eval(x,{'__builtins__':{}},{}))
output={}
for alg in ['gfp','rebrac']:
 for dataset in ['navigate','stitch']:
  for task in [1,4]:
   name=f'humanoidmaze-medium-{dataset}-singletask-task{task}-v0'
   cfg=OmegaConf.load(U/'config/main.yaml');del cfg.defaults
   cfg.agent=OmegaConf.merge(cfg.agent,OmegaConf.load(U/f'config/agent/{alg}.yaml'))
   cfg.env_name=name;cfg.is_final=True
   processed='OGBench_Env_'+name.replace('-','_')
   rows=[row for row in csv.DictReader(open(U/f'hyperparams/{alg}_params.csv')) if row['benchmark'] in processed and row['env_name'] in processed]
   assert rows,(alg,name)
   row=sorted(rows,key=lambda x:len(x['env_name']))[-1]
   for k,v in row.items():
    if k=='offline_steps':cfg.offline_steps=int(v)
    if k in cfg.agent:cfg.agent[k]=type(cfg.agent[k])(v)
   output[alg+'_'+name]={'config':OmegaConf.to_container(cfg,resolve=True),'matched_csv_row':row,'execution':'NOT_RUN; public protocol snapshot only','seed_ids':'Not encoded by result CSV; eight values available','reported_result_aggregation':'mean of 800k,900k,1M native successes; across eight seeds; population SD (ddof=0), verified from eight published values'}
(R/'configs/public_task_configs.json').write_text(json.dumps(output,indent=2))
print([(k,v['config']['agent']['discount']) for k,v in output.items()])

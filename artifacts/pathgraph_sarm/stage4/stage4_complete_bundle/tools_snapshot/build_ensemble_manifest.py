import argparse,json
from pathlib import Path
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--output',default='ensemble_manifest.json'); p.add_argument('--selection-lock'); a,_=p.parse_known_args(); lock={}
 if a.selection_lock and Path(a.selection_lock).exists(): lock=json.loads(Path(a.selection_lock).read_text())
 seeds=lock.get('selected_checkpoints',{}); obj={'bundle_version':'stage4-1.0','history_steps':32,'statistics_unit':'content_group_id','checkpoints':[{'seed':int(s),'path':v} for s,v in seeds.items()],'temperatures':{str(s):{'node':1.0,'edge_type':1.0,'edge_id':1.0} for s in seeds}}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(obj,indent=2))

"""Read-only bounded inventory. Never load checkpoints, credentials or environments."""
import os,json,time,hashlib,subprocess
from pathlib import Path
R=Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1');BASE=R.parent;O=R/'experiments/part_0_validation/stage_0c'
names=['LIBERO/images','LIBERO/libero','graph_github_upload','graph_pathgraph_p1_v6gm1_data','graph_pathgraph_p1_v6gm1_worktree','graph_pathgraph_p2c_rl_data','graph_pathgraph_p2c_rl_evaluation_repair_v1_worktree','OG_ea_v3_s2_pool','OG_ea_v3_s3_mechanism','OG_ea_v4_outcome_pilot','ea_v3_root_store','artifacts','downloads']
results=[]
for name in names:
 root=BASE/name;start=time.monotonic();row={'root':str(root),'exists':root.is_dir(),'images':[],'data_containers':[],'binding_metadata_candidates':[],'licenses':[],'files_seen':0,'complete_within_declared_depth':True,'depth_limit':8}
 if not root.is_dir():results.append(row);continue
 try:row['git_commit']=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True,timeout=3).strip()
 except Exception:row['git_commit']=None
 for current,dirs,files in os.walk(root):
  if time.monotonic()-start>12 or row['files_seen']>60000:row['complete_within_declared_depth']=False;break
  depth=len(Path(current).relative_to(root).parts)
  dirs[:]=[d for d in dirs if d not in ['.git','.ssh','.cache','.venv','node_modules','__pycache__','wandb','checkpoints'] and not d.startswith('.') and depth<8]
  for filename in files:
   row['files_seen']+=1;p=Path(current)/filename;suffix=p.suffix.lower();low=filename.lower()
   if suffix in ['.png','.jpg','.jpeg','.webp'] and len(row['images'])<40:
    stat=p.stat();entry={'absolute_path':str(p),'size_bytes':stat.st_size,'provenance':'Existing file in '+str(root),'split':'UNVERIFIED','license':'UNVERIFIED','task_mapping':'NOT_BOUND','eligible':False,'reason':'No certified CP-DISR dev split, D0/T_A/T_C asset/controller/object mapping and upload permission'}
    if stat.st_size<20000000:entry['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    row['images'].append(entry)
   if suffix in ['.hdf5','.h5','.zarr','.mp4'] and len(row['data_containers'])<25:row['data_containers'].append(str(p))
   if suffix in ['.json','.yaml','.csv'] and any(x in low for x in ['split','scene_manifest','dataset_info','task_manifest','few_shot','fewshot','object_binding']) and not any(x in low for x in ['key','credential','secret']) and len(row['binding_metadata_candidates'])<30:row['binding_metadata_candidates'].append(str(p))
   if low.startswith(('license','copying')) and len(row['licenses'])<10:row['licenses'].append(str(p))
 results.append(row);print(name,len(row['images']),len(row['data_containers']),row['complete_within_declared_depth'],flush=True)
(O/'input_discovery.json').write_text(json.dumps({'scope':'Readonly bounded inventory; not an exhaustive server-wide absence claim','roots':results,'bound_formal_scenes':0,'bound_independent_fewshots':0,'api_requests':0,'runtime_actions':0},indent=2)+'\n')

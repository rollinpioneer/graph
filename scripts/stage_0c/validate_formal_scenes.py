"""Complete offline validator for 24 scenes and 3 independent few-shots."""
import argparse,json,hashlib,csv
from pathlib import Path
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args(); root=a.root.resolve(); base=root/'experiments/stage_0c_inputs'; rows=[]; errors=[]; image_hashes=[]; config_hashes=[]
 required=['rgb.png','depth.npy','scene_manifest.json','reset_config.json','task_definition.json','object_bindings.json','initial_fact_summary.json','allowed_action_ids.json','allowed_proposition_ids.json','contract_refs.json','camera_config.json','capture_log.json','provenance.json','hidden_truth_qa.json']
 for task in ('D0','T_A','T_C'):
  for i in range(8):
   d=base/'dev'/task/f'scene_{i:02d}'; row={'split':'dev','task_id':task,'scene_id':f'{task}_dev_{i:02d}','status':'PASS'}
   if not d.is_dir() or any(not (d/x).is_file() for x in required): row['status']='MISSING_REQUIRED_FILES'; errors.append(row.copy()); rows.append(row); continue
   try:
    with Image.open(d/'rgb.png') as im:
     if im.mode!='RGB' or min(im.size)<32: raise ValueError('RGB')
     im.verify()
    m=json.loads((d/'scene_manifest.json').read_text()); cfg=json.loads((d/'reset_config.json').read_text()); qa=json.loads((d/'hidden_truth_qa.json').read_text());
    if m['synthetic_unit_fixture'] is not False or qa.get('hidden_truth_used_for_prompt') is not False: raise ValueError('hidden_truth')
    row['image_sha256']=m['image_sha256']; row['scene_config_sha256']=m['scene_config_hash']; image_hashes.append(m['image_sha256']); config_hashes.append(m['scene_config_hash'])
   except Exception as e: row['status']='INVALID_'+type(e).__name__; errors.append(row.copy())
   rows.append(row)
 few=[]
 for i in range(1,4):
  d=base/'fewshots'/f'fs_0{i}'; mfile=d/'scene_manifest.json'; row={'split':'independent_dev','scene_id':f'fs_0{i}','status':'PASS'}
  if not mfile.is_file() or any(not (d/x).is_file() for x in required): row['status']='MISSING_REQUIRED_FILES'; errors.append(row.copy()); few.append(row); continue
  try:
   with Image.open(d/'rgb.png') as im: im.verify()
   m=json.loads(mfile.read_text()); qa=json.loads((d/'hidden_truth_qa.json').read_text());
   if m['synthetic_unit_fixture'] is not False or qa.get('hidden_truth_used_for_prompt') is not False: raise ValueError('hidden_truth')
   row['image_sha256']=m['image_sha256']; row['scene_config_sha256']=m['scene_config_hash']; image_hashes.append(m['image_sha256']); config_hashes.append(m['scene_config_hash'])
  except Exception as e: row['status']='INVALID_'+type(e).__name__; errors.append(row.copy())
  few.append(row)
 if len(set(image_hashes))!=27: errors.append({'error':'IMAGE_HASH_DUPLICATE','unique':len(set(image_hashes))})
 if len(set(config_hashes))!=27: errors.append({'error':'CONFIG_HASH_DUPLICATE','unique':len(set(config_hashes))})
 if {r['scene_id'] for r in rows}&{r['scene_id'] for r in few}: errors.append({'error':'SPLIT_OVERLAP'})
 try:
  import sys; sys.path.insert(0,str(root/'src')); from cp_disr.stage0c import prepare_matrix; from cp_disr.vlm_provider import DashScopeProvider
  pr=DashScopeProvider(); pr.key_present=lambda:True; prepare_matrix(root,pr); preflight='PASS'
 except Exception as e: preflight='FAIL_'+type(e).__name__; errors.append({'error':'PROVIDER_PREFLIGHT','detail':str(e)})
 result={'status':'PASS' if not errors and len(rows)==24 and len(few)==3 and preflight=='PASS' else 'BLOCKED','formal_scene_count':len(rows),'fewshot_count':len(few),'unique_image_hashes':len(set(image_hashes)),'unique_scene_config_hashes':len(set(config_hashes)),'provider_preflight':preflight,'errors':errors,'rows':rows+few,'api_requests':0}
 out=root/'experiments/part_0_validation/stage_0c_r2'; out.mkdir(parents=True,exist_ok=True); (out/'input_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 with (out/'input_validation.csv').open('w',newline='') as f:
  cols=['split','task_id','scene_id','status','image_sha256','scene_config_sha256']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows({k:r.get(k,'') for k in cols} for r in result['rows'])
 print(json.dumps({'status':result['status'],'scenes':len(rows),'fewshots':len(few),'unique_images':len(set(image_hashes)),'unique_configs':len(set(config_hashes)),'preflight':preflight,'errors':len(errors)})); return 0 if result['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())

"""Offline validator for formal scenes; reports missing inputs and never calls a service."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args();base=a.root/'experiments/stage_0c_inputs';rows=[]
 for task in ('D0','T_A','T_C'):
  for i in range(8):
   d=base/'dev'/task/f'scene_{i:02d}';row={'task_id':task,'scene_id':f'{task}_dev_{i:02d}','status':'MISSING'}
   if d.is_dir() and (d/'scene_manifest.json').is_file() and (d/'rgb.png').is_file():
    try:
     with Image.open(d/'rgb.png') as im: im.verify()
     row.update(status='CANDIDATE_UNVALIDATED',image_sha256=hashlib.sha256((d/'rgb.png').read_bytes()).hexdigest())
    except Exception as e: row.update(status='INVALID_IMAGE',error=type(e).__name__)
   rows.append(row)
 out=a.root/'experiments/part_0_validation/stage_0c/input_validation.json';out.write_text(json.dumps({'status':'PASS' if all(x['status']=='CANDIDATE_UNVALIDATED' for x in rows) else 'BLOCKED','rows':rows,'api_requests':0},indent=2)+'\n');print(json.dumps({'status':'PASS' if all(x['status']=='CANDIDATE_UNVALIDATED' for x in rows) else 'BLOCKED','count':len(rows)}))
if __name__=='__main__':main()

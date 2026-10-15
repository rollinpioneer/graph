import json,hashlib,argparse,importlib.util
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--component',choices=['all','models'],default='all');A=P.parse_args();R=Path(__file__).parent
spec=importlib.util.spec_from_file_location('official_index',R/'official_zip_index.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
expected=json.loads((R/'official_zip_crc_summary.json').read_text())['groups'];names=['models.tar'] if A.component=='models' else ['models.tar','grasp_label.tar','collision_label.tar','scenes.tar']
if A.component=='models':expected={k:v for k,v in expected.items() if k.startswith('models/')}
actual={}
for n in names:
 p=R/(n+'.extraction.json');assert p.exists() and json.loads(p.read_text())['status']=='EXTRACTED',n+' extraction not complete'
 with (R/(n+'.files.jsonl')).open() as f:
  for line in f:
   record=json.loads(line);matched=mod.group_for(record['path'])
   if matched:
    group,path=matched;actual.setdefault(group,[]).append({'path':path,'bytes':record['bytes'],'crc32':record['crc32']})
errors=[];summaries={}
for group,e in expected.items():
 records=actual.get(group,[]);canonical=''.join(f"{x['path']}\t{x['bytes']}\t{x['crc32']}\n" for x in sorted(records,key=lambda x:x['path']));a={'files':len(records),'bytes':sum(x['bytes'] for x in records),'sha256_of_path_size_crc32_list':hashlib.sha256(canonical.encode()).hexdigest()};summaries[group]=a
 if a!=e:errors.append({'group':group,'expected':e,'actual':a})
report={'component':A.component,'status':'PASS' if not errors else 'FAIL','official_groups':len(expected),'verified_files':sum(x['files'] for x in summaries.values()),'verified_bytes':sum(x['bytes'] for x in summaries.values()),'method':'Every selected path, uncompressed size and CRC32 matched against independent official Google Drive ZIP central directories; mirror TAR full SHA256 also required.','errors':errors,'actual_groups':summaries}
(R/('OFFICIAL_CRC_VERIFICATION_'+A.component+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='actual_groups'}),flush=True);raise SystemExit(0 if not errors else 2)

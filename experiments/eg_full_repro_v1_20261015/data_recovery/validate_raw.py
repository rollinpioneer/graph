from pathlib import Path
import sys,json,concurrent.futures,time,subprocess
R=Path(__file__).parent;D=R.parent/'archive_worktree/experiments/eg_full_repro_v1_20261015';sys.path.insert(0,str(D));from audit_data import audit
ROOT=R.parent/'data/graspnet'
def scene_shard(pair):
 start,end=pair;out=audit(ROOT,True,False,start,end,False);p=R/f'raw_scene_audit_{start}_{end}.json';p.write_text(json.dumps(out));return out
if __name__=='__main__':
 started=time.time();proof=json.loads((R/'OFFICIAL_CRC_VERIFICATION_all.json').read_text());assert proof['status']=='PASS'
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(scene_shard,[(0,48),(48,96),(96,144),(144,190)]))
 obj=sorted(set(i for x in results for i in x['required_object_ids']));assets=audit(ROOT,True,False,0,0,True,obj);(R/'raw_asset_audit.json').write_text(json.dumps(assets))
 summary={'status':'PASS' if all(x['complete'] for x in results) and assets['complete'] else 'FAIL','scene_count':sum(x['scene_count'] for x in results),'training_scene_count':sum(x['training_scene_count'] for x in results),'test_scene_count':sum(x['test_scene_count'] for x in results),'image_files_present':sum(x['image_files_present'] for x in results),'supervision_generated':False,'required_training_object_ids':obj,'scene_error_counts':[len(x['errors']) for x in results],'asset_error_count':len(assets['errors']),'wall_seconds':time.time()-started,'official_source_crc_verified':True}
 assert summary['scene_count']==190 and summary['training_scene_count']==100 and summary['test_scene_count']==90 and summary['image_files_present']==145920
 (R/'RAW_DATA_VALIDATION.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);raise SystemExit(0 if summary['status']=='PASS' else 2)

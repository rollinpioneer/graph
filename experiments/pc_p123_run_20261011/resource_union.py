import pathlib,json,datetime,subprocess
R=pathlib.Path(__file__).resolve().parent;intervals={};jobs=[]
for f in (R/'runs').glob('*/guard_result.json'):
 rec=json.loads(f.read_text());start=datetime.datetime.fromisoformat(rec['start_utc']).timestamp();end=datetime.datetime.fromisoformat(rec['end_utc']).timestamp();last=0;lastactive=[]
 rows=[json.loads(l) for l in (f.parent/'gpu_memory.jsonl').read_text().splitlines()]
 for row in rows:
  now=row['seconds'];active=row['active_gpu_uuids']
  for uuid in set(active)|set(lastactive):intervals.setdefault(uuid,[]).append((start+last,start+now))
  last=now;lastactive=active
 for uuid in lastactive:intervals.setdefault(uuid,[]).append((start+last,end))
 jobs.append({'name':f.parent.name,'start_utc':rec['start_utc'],'end_utc':rec['end_utc'],'primary_gpu':rec['gpu'],'charged_seconds':rec['charged_gpu_seconds']})
merged={}
for uuid,parts in intervals.items():
 output=[]
 for lo,hi in sorted(parts):
  if output and lo<=output[-1][1]:output[-1][1]=max(output[-1][1],hi)
  else:output.append([lo,hi])
 merged[uuid]={'resident_seconds':sum(b-a for a,b in output),'intervals_utc':[[datetime.datetime.fromtimestamp(a,datetime.timezone.utc).isoformat(),datetime.datetime.fromtimestamp(b,datetime.timezone.utc).isoformat()] for a,b in output]}
x={'method':'Union of task-resident GPU UUID intervals across all own workers, expanded by adjacent telemetry rows; shared renderer GPU not counted twice. Sampling estimate; conservative per-process full-time upper bound controls stopping.','jobs':jobs,'gpu_intervals':merged,'sampled_union_gpu_seconds':sum(v['resident_seconds'] for v in merged.values()),'gpu_map':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)}
x['sampled_union_gpu_hours']=x['sampled_union_gpu_seconds']/3600;x['wall_first_job_to_last_job_s']=max(datetime.datetime.fromisoformat(j['end_utc']).timestamp() for j in jobs)-min(datetime.datetime.fromisoformat(j['start_utc']).timestamp() for j in jobs)
(R/'results/resource_union.json').write_text(json.dumps(x,indent=2));print('GPU_UNION',x['sampled_union_gpu_hours'],x['wall_first_job_to_last_job_s'])

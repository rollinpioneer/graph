"""Seal management logs after their writers exit; no science CSV changes."""
import os,json,time,hashlib,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.sched_getaffinity(0)=={75}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
started=time.time();d=read(ROOT/'receipts/ADVANCE_PROCESS.json');assert d['status']=='COMPLETE' and not Path('/proc',str(d['pid'])).exists()
for phase in ['I']+(['II'] if d['H_R_pass'] else []):
    m=read(ROOT/'receipts'/f'{phase}_WORKFLOW_MONITOR.json');assert m['status']=='COMPLETE' and m['all_owned_children_exited']
    assert all(not Path('/proc',str(x['pid'])).exists() for x in m['jobs']+[m['cpu_evaluation']])
before={str(p):sha(p) for p in ROOT.rglob('*.csv')}
idx=read(ROOT/'RAW_ARTIFACT_INDEX.json');old=dict(idx)
raw=[]
for base in [ROOT/'runs',ROOT/'source_ref',ROOT/'results/P_continuation/runs']:
    for p in sorted(base.rglob('*')):
        if p.is_file():raw.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
idx.update(raw_files=raw,total_new_raw_bytes=sum(x['bytes'] for x in raw),pending_active_management_logs=[],requires_postprocess_seal_after_all_workers_exit=False,sealed_epoch=time.time(),all_raw_writers_exited=True)
assert idx['total_new_raw_bytes']<50*1024**3
write(ROOT/'receipts/RAW_INDEX_BEFORE_MANAGEMENT_LOG_SEAL.json',old);write(ROOT/'RAW_ARTIFACT_INDEX.json',idx)
after={str(p):sha(p) for p in ROOT.rglob('*.csv')};assert before==after
r=read(ROOT/'RESOURCE_AND_EXIT.json');r.update(final_advance_driver_exit='CONFIRMED',advance_process_wall_seconds=d['elapsed_seconds'],advance_subprocesses=d['jobs'],sealing_seconds=time.time()-started,sealing_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_round_bytes_snapshot=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()))
assert r['new_round_bytes_snapshot']<50*1024**3;write(ROOT/'RESOURCE_AND_EXIT.json',r)
write(ROOT/'receipts/FINAL_RAW_INDEX_SEAL.json',dict(status='PASS',raw_files=len(raw),all_raw_writers_exited=True,all_science_CSV_unchanged=True,science_CSV_sha256=before,new_model_queries=0,new_training_updates=0,new_physics_steps=0,elapsed_seconds=time.time()-started))
print('SEALED_ALL_RAW',len(raw),idx['total_new_raw_bytes'],flush=True)

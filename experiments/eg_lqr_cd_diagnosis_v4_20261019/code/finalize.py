"""Single final artifact/metric sanity check and measured resource settlement."""
from common import *
import math,time,datetime,subprocess

pipeline=json.loads((R/'results/pipeline_status.json').read_text())
assert pipeline['status']=='COMPLETE','Do not finalize a partial confirmation'
decision=json.loads((R/'results/confirmation_decision.json').read_text())
lock=json.loads((R/'C2_method_lock.json').read_text());scenes=sum([lock['selected'][s] for s in SPLITS],[])
assert lock['config_sha256']==sha256(R/'frozen_config.json')
assert lock['reservation_sha256']==sha256(R/'C2_reservation.json')
assert lock['resource_amendment_sha256']==sha256(R/'resource_amendment.json')
expected={'R0','R0_CD'}|{f'F05_{s}{tail}' for s in SEEDS for tail in ['', '_CD']}
total=0;receipts=[];unknown=0
for scene in scenes:
 folder=R/f'evaluation/confirmation/{scene:04d}';p=folder/'COMPLETE.json';done=json.loads(p.read_text())
 assert done['status']=='PASS' and done['frames']==list(range(256)) and done['configurations_per_frame']==8
 receipt={'scene':scene,'sha256':sha256(p),'seconds':done['seconds'],'cpu_seconds':done['cpu_seconds'],'rss_kb':done['rss_kb'],'end_mtime':p.stat().st_mtime};receipts.append(receipt)
 export_done=json.loads((R/f'exports/subset/scene_{scene:04d}/COMPLETE.json').read_text());assert export_done['method_lock_sha256']==sha256(R/'C2_method_lock.json') and export_done['frames']==256
 for frame in range(256):
  p=folder/f'{frame:04d}.json';rows=json.loads(p.read_text());assert len(rows)==8 and {x['configuration'] for x in rows}==expected
  first=rows[0]
  for key in ['geometry_sha256','input_sha256','export_sha256','raw_observation_sha256','backbone_sha256','native_eval_source_sha256']:
   assert len({v[key] for v in rows})==1
  assert first['backbone_sha256']==CFG['backbone_sha256']
  with np.load(p.with_suffix('.npz')) as cache:
   identity=json.loads(str(cache['identity']));assert all(first[k]==v for k,v in identity.items());mu=cache['quality'];coll=cache['collision'];empty=cache['empty'];assert mu.shape==coll.shape==empty.shape==(1024,);assert not (empty&~coll).any()
   unknown+=int((~np.isfinite(mu)).sum())
   for seed in SEEDS:assert cache[f'Q_{seed}'].shape==(1024,) and np.isfinite(cache[f'Q_{seed}']).all()
   for row in rows:
    assert row['scene']==scene and row['frame']==frame
    for key in ['ap','ap04','ap08']:assert math.isfinite(row[key]) and 0<=row[key]<=100
    ids=cache['selected_'+row['configuration']];assert np.all((ids>=0)&(ids<1024)) and np.isfinite(mu[ids]).all()
    assert row['top_candidate']==(int(ids[0]) if len(ids) else None)
    assert row['top1_valid']==bool(len(ids) and 0<mu[ids[0]]<=.4000001)
    assert row['top1_collision']==bool(len(ids) and coll[ids[0]])
    assert row['top1_empty']==bool(len(ids) and empty[ids[0]])
  total+=8
assert total==24576
with (R/'results/confirmation_per_frame.csv').open(newline='') as f:assert sum(1 for _ in csv.DictReader(f))==total
with (R/'results/confirmation_per_scene.csv').open(newline='') as f:scene_rows=list(csv.DictReader(f))
assert len(scene_rows)==96
table={(int(v['scene']),v['configuration']):float(v['ap']) for v in scene_rows}
head_delta=np.mean([[[table[scene,f'F05_{seed}_CD']-table[scene,'R0_CD'] for scene in lock['selected'][split]] for seed in SEEDS] for split in SPLITS],axis=(0,2))
assert np.isclose(head_delta.mean(),decision['primary']['delta'],atol=1e-10) and np.allclose(head_delta,decision['primary']['seed_delta'],atol=1e-10)
x=decision['primary'];target=x['delta']>=1 and x['ci95'][0]>0 and all(v>0 for v in x['seed_delta']) and x['positive_scenes']>=8
assert (decision['status']=='AP_CONFIRMED_TARGET')==target
raw=json.loads((R/'results/C2_raw_observation_audit.json').read_text());assert raw['status']=='PASS_EXACT_RAW_OBSERVATION_CHECK' and raw['unique_current']==3072 and raw['overlap_with_prior_C1']==0
validation=dict(status='PASS',frames=3072,scenes=12,configuration_records=total,checks=['Exactly eight frozen configurations per frame','Full0..255 frames and completed export/model lock identity','Shared input/export/geometry/raw-observation identity','All selected candidates have finite actual-pose quality','Stored Top1 flags match quality/collision/empty arrays','Frame CSV count and independent scene/head mean recomputation','Exact raw-observation uniqueness and prior-C1 disjointness'],uncomputed_candidate_slots=unknown,unknown_quality_is_not_failure=True,scene_receipts=receipts,confirmation_status=decision['status'])
write('results/confirmation_validation.json',validation)
resource={};cpu=0;max_rss=0;envelopes={}
for name,folder in [('diagnosis_final','diagnosis'),('diagnosis_retained_attempt','diagnosis_attempt1_mixed_dtype'),('exploration','evaluation/exploration'),('confirmation','evaluation/confirmation')]:
 points=[];seconds=0
 for p in (R/folder).glob('*/COMPLETE.json'):
  v=json.loads(p.read_text());seconds+=v['seconds'];cpu+=v['cpu_seconds'];max_rss=max(max_rss,v['rss_kb']);points.append((p.stat().st_mtime-v['seconds'],p.stat().st_mtime))
 resource[name+'_completed_worker_seconds']=seconds
 if points:
  envelopes[name]={'evaluation_envelope_seconds':max(t[1] for t in points)-min(t[0] for t in points),'all_scene_body_overlap_seconds':max(0,min(t[1] for t in points)-max(t[0] for t in points)),'overlap_scope':'Receipt mtime minus measured worker duration; approximate body timing, not continuous process sampling'}
export=json.loads((R/'results/C2_export_status.json').read_text());assert export['status']=='PASS'
resource.update(GPU_export_process_seconds=export['wall_seconds'],GPU_torch_peak_allocated_bytes=export['peak_torch_gpu_bytes'],measured_main_worker_CPU_seconds=cpu+export['cpu_seconds'],unmeasured_failed_and_auxiliary_conservative_reserve_seconds=360,actual_directory_bytes=int(subprocess.check_output(['du','-sb',str(R)]).split()[0]),max_single_worker_RSS_GiB=max_rss/1024**2,new_training_runs=0,backbone_updates=0,evaluation_timing=envelopes,C2_wall_seconds=(R/'results/pipeline_status.json').stat().st_mtime-(R/'results/C2_resource_launch.json').stat().st_mtime,concurrency_limit=13,observed_peak_own_compute_processes=13,concurrency_evidence='Live checks with12 CPU evaluators and controller; overlapping measured bodies in E/C2 receipts support count. No other users tasks counted.',threads_per_worker=1)
resource['conservative_accounted_worker_hours']=(sum(resource[k] for k in resource if k.endswith('_completed_worker_seconds'))+export['wall_seconds']+360)/3600
assert resource['conservative_accounted_worker_hours']<=CFG['budget']['worker_wall_hours']
assert export['wall_seconds']<=CFG['budget']['gpu_process_hours']*3600
assert resource['actual_directory_bytes']<=CFG['budget']['new_bytes']
write('results/resource_summary.json',resource)
subprocess.run([sys.executable,str(R/'code/disaggregate_failures.py')],check=True)
subprocess.run([sys.executable,str(R/'code/build_report.py')],check=True)
with (R/'results/research_report.md').open('a') as f:
 f.write(f"\n## 最终核验与实测成本\n\n全部3072帧、24576条确认记录通过核验；所有被选候选均有实际姿态官方质量，未计算候选保留未知，未填失败。主指标与独立按场景/头重新求均值一致。\n\nC2阶段墙钟约{resource['C2_wall_seconds']/60:.1f}分钟；GPU导出{export['wall_seconds']/60:.1f}分钟。含保留的旧诊断尝试和360秒未精确计时失败/辅助任务保守预留，结算{resource['conservative_accounted_worker_hours']:.3f}worker小时；预留不是实测值。单任务峰值RSS {resource['max_single_worker_RSS_GiB']:.2f}GiB，Torch峰值GPU分配{export['peak_torch_gpu_bytes']/1024**2:.1f}MiB（不等于整卡/驱动显存）。本卡目录{resource['actual_directory_bytes']/1024**3:.3f}GiB。实际观测上限13进程，0新训练。\n")
print('FINAL_VALIDATION_PASS',json.dumps(decision),flush=True)

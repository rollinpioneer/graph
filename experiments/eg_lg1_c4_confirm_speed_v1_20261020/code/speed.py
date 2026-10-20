from common import *
import geometry_core as ref
from geometry_fast import *
import resource
assert (R/'results/C4_result_freeze.json').exists();cfg=json.load(open(R/'frozen_config.json'));lock=json.load(open(R/'method_lock.json'));start=time.time();cpu=time.process_time();os.nice(10)
def export(scene,frame):return P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
heads={v['seed']:GeoScorer(v['path'],v['sha256']) for v in lock['new_heads']}
import torch
torch.set_num_threads(1)
def outputs(feat,action,raw,scene,frame):
 result=[]
 for seed,head in heads.items():
  norm=geom_input(raw,head.ck['geom_norm']);q=head.quality(feat,action,raw,scene,frame);score,_=fused_score(pred[:,0],q);rank=np.lexsort((np.arange(1024),-score));result.append((norm,q,score,rank))
 return result
def check_output(a,b):return all(np.array_equal(x,y,equal_nan=True) for aa,bb in zip(a,b) for x,y in zip(aa,bb))
variants=['index_query','mask','cov'];prototype=[];eligible={v:True for v in variants}
for scene,frame in cfg['speed']['prototype']:
 with np.load(export(scene,frame)) as z:pred=z['pred'];feat=z['feature320'];action=z['original_action'];hyp=z['hypotheses']
 base,cloud,_=extract(scene,frame,pred,'baseline');base_output=outputs(feat,action,base,scene,frame);tree=cKDTree(cloud);cached=CachedTree(tree,pred)
 query_exact=all(np.array_equal(tree.query_ball_point(p[13:16],.16),cached.query_ball_point(p[13:16],.16)) for p in pred);assert query_exact
 # All six supervised poses checked using existing head sample IDs, never new quality labels.
 with np.load(P/f'B/labels/{scene:04d}_{frame:04d}.npz') as z:ids=z['ids']
 poses=hyp[ids].reshape(-1,17);gold=np.array([ref.geom(cloud,tree,p) for p in poses],np.float32);cached6=CachedTree(tree,poses)
 for variant in variants:
  t=time.perf_counter();raw,cc,_=extract(scene,frame,pred,variant);cold=time.perf_counter()-t
  fn={'index_query':ref.geom,'mask':geom_mask,'cov':geom_cov}[variant];six=np.array([fn(cc,cached6,p) for p in poses],np.float32)
  exact=bool(np.array_equal(cc,cloud) and np.array_equal(base,raw,equal_nan=True) and np.array_equal(six,gold,equal_nan=True) and check_output(base_output,outputs(feat,action,raw,scene,frame)))
  eligible[variant]&=exact;timings=[]
  if exact:
   for k in range(2):
    t=time.perf_counter();extract(scene,frame,pred,variant);timings.append(time.perf_counter()-t)
  prototype.append(dict(scene=scene,frame=frame,variant=variant,exact=exact,query_order_exact=query_exact,six_pose_features_exact=bool(np.array_equal(six,gold,equal_nan=True)),cold_seconds=cold,warm_seconds=timings))
write(R/'results/speed_prototype.json',prototype)
valid=[v for v in variants if eligible[v]];assert valid,'No exactly equivalent prototype'
selected=min(valid,key=lambda v:np.median([x for r in prototype if r['variant']==v for x in r['warm_seconds']]))
write(R/'results/speed_implementation_lock.json',dict(selected=selected,selected_before_heldout_benchmark=True,eligible=eligible,selection='lowest prototype median among exact variants',source_sha256=sha(R/'code/geometry_fast.py'),binary_sha256=sha(R/'code/masks.so'),prototypes=cfg['speed']['prototype'],validation_scenes=cfg['speed']['validation_scenes'],validation_frames=cfg['speed']['validation_frames'],time=time.time()))
rows=[]
for i,(scene,frame) in enumerate((s,f) for s in cfg['speed']['validation_scenes'] for f in cfg['speed']['validation_frames']):
 with np.load(export(scene,frame)) as z:pred=z['pred'];feat=z['feature320'];action=z['original_action']
 data={};timing={};cput={}
 for variant in (['baseline',selected] if i%2==0 else [selected,'baseline']):
  t=time.perf_counter();c=time.process_time();data[variant]=extract(scene,frame,pred,variant);cput[variant]=time.process_time()-c;timing[variant]=time.perf_counter()-t
 b,bc,_=data['baseline'];raw,cc,_=data[selected]
 with np.load(G/f'eg_lg1_local_geometry_v1_20261020/geometry/C3/{scene:04d}/{frame:04d}.npz') as z:gold=z['raw'][:,0]
 exact=dict(cloud=np.array_equal(bc,cc),baseline_cache=np.array_equal(b,gold,equal_nan=True),raw=np.array_equal(b,raw,equal_nan=True),missing=np.array_equal(~np.isfinite(b),~np.isfinite(raw)),head_normalized_Q_F05_full_rank=check_output(outputs(feat,action,b,scene,frame),outputs(feat,action,raw,scene,frame)))
 assert all(exact.values()),(scene,frame,exact)
 rows.append(dict(scene=scene,frame=frame,baseline_seconds=timing['baseline'],accelerated_seconds=timing[selected],baseline_cpu=cput['baseline'],accelerated_cpu=cput[selected],speedup=timing['baseline']/timing[selected],raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),**exact))
 if len(rows)%8==0:print('SPEED_VALIDATION',len(rows),'/96',selected,flush=True)
 assert time.time()-start<2*3600,'Speed CPU worker budget reached'
assert len(rows)==96;csv_write(R/'results/speed_per_frame.csv',rows)
b=np.array([v['baseline_seconds'] for v in rows]);v=np.array([v['accelerated_seconds'] for v in rows]);ratio=float(np.median(b)/np.median(v));summary=dict(status='SIGNIFICANT_SPEEDUP_EXACT' if ratio>=2 else 'EXACT_BUT_BELOW_2X_TARGET',selected=selected,frames=96,candidates=98304,three_head_predictions_checked=294912,all_features_and_outputs_exact=True,baseline_median=float(np.median(b)),accelerated_median=float(np.median(v)),baseline_p95=float(np.quantile(b,.95)),accelerated_p95=float(np.quantile(v,.95)),median_speedup=ratio,baseline_total=float(b.sum()),accelerated_total=float(v.sum()),CPU_process_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,single_CPU_worker=True,GPU_used=False,prototypes_included=True,C4_predictions_not_used_for_speed_selection=True,C4_result_freeze_sha256=sha(R/'results/C4_result_freeze.json'),implementation_lock_sha256=sha(R/'results/speed_implementation_lock.json'))
write(R/'results/speed_summary.json',summary)
lines=['# 几何计算加速报告','',f"状态：{summary['status']}。实现{selected}，96个固定C3帧，每帧全部1024候选；3个G头的输入/Q/F05/完整排序精确相同。",'', '| 单CPU端到端 | 原实现 | 等价实现 |','|---|---:|---:|',f"| 中位数(秒) | {summary['baseline_median']:.6f} | {summary['accelerated_median']:.6f} |",f"| p95(秒) | {summary['baseline_p95']:.6f} | {summary['accelerated_p95']:.6f} |",f"| 96帧合计(秒) | {b.sum():.3f} | {v.sum():.3f} |",'',f"中位数加速{ratio:.3f}倍；预先固定2倍门槛{'达到' if ratio>=2 else '未达到'}。这是同机交错顺序、各1遍96帧的单CPU测量，与历史3帧1.393秒不是完全相同的负载/场景。并不等于整个EG推理加速，也不声称实时部署。",'', '## 实现及数值约束','', 'index_query：等价一维整数编码体素键，保留原lexicographic体素顺序/逐点累加次序；批量KD邻域查询return_sorted=False，重复中心复用邻域。mask：进一步用无fast-math的C++单遍计算原五个布尔区域和inner索引，保留原点顺序及NumPy矩阵变换、均值、PCA/法向。cov：进一步展开现有无权重float64 np.cov的同一运算顺序，省去通用参数处理。未改变深度采样、半径、体素尺寸、任何区域阈值或模型参数。', '三个固定原型帧逐项检查全部1024原动作及已有128×6姿态；只有精确相同者可进入选择。按原型计时冻结最快实现，然后首次运行固定96帧独立数值/计时检查。所有float32原始33特征、NaN位置、缺失指示、归一化66输入及三个头的输出/融合/全排序精确相等，无容差放宽。独立96帧原版计算亦与上一轮原始几何缓存一致。', '验证为当前库版本和样本范围的精确等价证据；部署到不同库/CPU或未验证场景仍应使用同一一致性检查。原实现和冻结权重保留只读，C4结果在编写/执行加速前已冻结，未用C4挑选实现。', '', '## 成本','', f"原型与独立验证合计CPU进程时间{summary['CPU_process_seconds']:.3f}秒，墙钟{summary['wall_seconds']:.3f}秒；峰值RSS {summary['peak_rss_kib']/1024:.1f}MiB，GPU使用0。编译成本另见fast_build.json。未安装或修改共享环境。",'', '完整逐帧时间与一致性在speed_per_frame.csv；实现选择在speed_implementation_lock.json，编译/源码散列在fast_build.json。']
(R/'SPEED_REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary),flush=True)

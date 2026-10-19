"""Build a truthful stage report from saved decisions; does not select a method."""
import os,json,csv,datetime
from pathlib import Path
import numpy as np

R=Path(os.environ['LQR_ROOT']);cfg=json.loads((R/'frozen_config.json').read_text())
def read(name,default=None):
 p=R/name;return json.loads(p.read_text()) if p.exists() else default
p1=read('results/P1_decision.json');lock=read('method_lock.json');p2=read('results/C1_decision.json');audit=read('results/C1_exposure_audit.json');latency=read('results/latency.json');state=read('results/C1_status.json');dedup=read('results/C1_duplicate_sensitivity.json');preflight=read('results/preflight_complete.json')
lines=['# EG-LQR-BUILD-CONFIRM-V3 执行报告','',f'生成时间：{datetime.datetime.now().astimezone().isoformat()}。执行节点gpu04（aita_xushijie@172.19.164.161），同一共享盘，无数据集重复复制。','']
if p2:
 x=p2['primary'];lines += [f'**确认结论：{p2["status"]}。** 锁定族{p2["family"]}相对R0+CD的平均ΔAP={x["delta"]:+.3f}，分层场景95%CI=[{x["ci95"][0]:+.3f}, {x["ci95"][1]:+.3f}]。三头总体差分别为'+', '.join(f'{v:+.3f}' for v in x['seed_delta'])+f'；正向场景{x["positive_scenes"]}/12。','']
else:lines += [f'**确认未完成：{state["status"] if state else "NOT_OPENED"}。** 不输出未完成确认的科学结论。','']
lines += ['## 方法与工程检查','','固定原EG1024候选的所有几何分量，原动作Q直接替换score。三个既有第1000步头分别评价，无新训练、无EG更新、无新增训练标签、无头集成或新检查点选择。输入group256/score token64/原动作4维，归一化复用各头fit统计。评分不读取真值/物体模型/SDF。原EG数据预处理仍使用seg/meta等基准文件。','',f'P0：{preflight["status"] if preflight else "未完成"}。四个固定帧、三头线上/缓存、保存加载等价；56条原生路径候选、质量、碰撞和AP一致，R0/S历史指标相同。原共享环境未修改，独立Open3D0.19适配GLIBC2.34，NumPy/SciPy/PyTorch复用原版本。','']
if p1:
 lines += ['## P1 已观察迁移筛查','','12场景×32帧，仅是开发迁移。两族都通过预算闸门时按S优先，不以谁更高决定确认族。','', '|族|ΔAP+CD|描述性95%CI|三个头差值|正向场景|','|---|---:|---|---|---:|']
 for family,x in p1['gates'].items():lines.append(f'|{family}|{x["delta"]:+.3f}|[{x["ci95"][0]:+.3f}, {x["ci95"][1]:+.3f}]|'+', '.join(f'{v:+.3f}' for v in x['seed_delta'])+f'|{x["positive_scenes"]}/12|')
 lines += ['',f'锁定族：{p1["selected_family"]}。P1正向不等于独立确认。method_lock在C1性能读取之前写入；暴露审计状态{audit["status"] if audit else "尚未审计"}，检查{len(audit["checked_performance_files"]) if audit else 0}份已知记录。审计不能保证未知外部历史未暴露。','']
if p2:
 family=p2['family'];comparisons=read('results/C1_comparisons.json');selected=next(x for x in comparisons if x['family']==family and x['use_cd']);lines += ['## C1 完整选定场景确认','','12场景×256帧，R0±CD和锁定族三头±CD，共24576帧配置记录。只代表所选12场景，不是完整90场景GraspNet AP。','', '|指标|平均差值|场景95%CI|','|---|---:|---|']
 for metric in ['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','selected_top50_collision_rate','cd_rejection_rate']:
  x=selected['metrics'][metric];lines.append(f'|{metric}|{x["delta"]:+.3f}|[{x["ci95"][0]:+.3f}, {x["ci95"][1]:+.3f}]|')
 lines += ['', '率的差值单位为百分点，AP差值单位为AP百分点。三头先场景内平均，再按Seen/Similar/Novel内场景等权、三个划分等权，按划分分层bootstrap10000次，种子20261019。种子×场景和相邻帧不当作独立样本。','', '|划分|AP差值|Top1合格率差（百分点）|','|---|---:|---:|']
 for split in ['seen','similar','novel']:lines.append(f'|{split}|{selected["metrics"]["ap"]["split_delta"][split]:+.3f}|{selected["metrics"]["top1_valid"]["split_delta"][split]:+.3f}|')
 with open(R/'results/C1_per_scene.csv',newline='') as file:absolute_rows=list(csv.DictReader(file))
 lines += ['', '|组别|CD|所选Seen AP|所选Similar AP|所选Novel AP|三划分均值AP|Top1合格率%|Top1碰撞率%|','|---|---|---:|---:|---:|---:|---:|---:|']
 for cd in [False,True]:
  suffix='_CD' if cd else ''
  for method in ['R0',family]:
   names=['R0'+suffix] if method=='R0' else [f'{family}_{h["seed"]}{suffix}' for h in cfg['heads']]
   values={split:[x for x in absolute_rows if x['split']==split and x['configuration'] in names] for split in ['seen','similar','novel']}
   aps=[float(np.mean([float(x['ap']) for x in values[s]])) for s in ['seen','similar','novel']];top=float(np.mean([np.mean([float(x['top1_valid']) for x in v]) for v in values.values()]))*100;coll=float(np.mean([np.mean([float(x['top1_collision']) for x in v]) for v in values.values()]))*100
   lines.append(f'|{method}|{"有" if cd else "无"}|'+ '|'.join(f'{v:.3f}' for v in aps)+f'|{np.mean(aps):.3f}|{top:.3f}|{coll:.3f}|')
 top=selected['metrics']['top1_valid'];risk='Top1独立描述性区间支持提升' if top['ci95'][0]>0 else 'Top1区间支持下降，存在首选可靠性权衡' if top['ci95'][1]<0 else 'Top1改变尚未确认，不能声称更可靠'
 if top['delta']<0:risk+='；合格率均值下降，即使AP改善也需披露这一权衡'
 lines += ['',risk+'。Top1变化来自选中了另一个候选，不是同ID姿态修复；AP不等于机械臂实际成功率。次要区间为描述性。完整失败→成功/成功→失败数见C1_top1_transitions.csv。','', '## 去重敏感性','']
 if dedup:
  for scope,result in dedup['modes'].items():
   text=f'{scope}：{result["status"]}，保留{result["retained_frames"]}/3072帧'
   if 'metrics' in result:
    x=result['metrics']['ap'];text+=f'，ΔAP={x["delta"]:+.3f}，95%CI=[{x["ci95"][0]:+.3f}, {x["ci95"][1]:+.3f}]'
   lines.append(text+'。')
  lines += ['', '按解码RGB/深度像素和相机内参K识别，固定保留scene/frame字典序首个，不按预测筛选。原完整256帧主协议保留；去重后的选帧指标仅为敏感性，不能冒充完整官方帧协议。','']
 else:lines += ['原始观测去重敏感性仍待生成；主协议不删帧。','']
if latency:
 lines += ['## 同硬件开销','','固定单模型20261018，batch1，5次预热/20次测量，同机GPU+CPU头，GPU同步。已加载输入，排除磁盘I/O、NMS/官方评价和机器人执行；不做跨硬件速度倍数推断。','', '|路径|中位数ms|P95 ms|','|---|---:|---:|']
 for name,value in latency['summary_ms'].items():lines.append(f'|{name}|{value["median"]:.3f}|{value["p95"]:.3f}|')
lines += ['', '## 范围、资源与归档','','执行峰值上限8个计算进程：导出期间GPU导出1+CPU评价6+控制1，导出结束CPU评价7+控制1。单线程、低优先级；未终止其他任务。GPU最多2进程小时，P2最多24 worker小时，总卡40 worker小时，新增产物10GiB。实际资源见阶段status/COMPLETE、environment_gpu04及资源汇总。','', '首次P0评价写回发生重复scene/frame关键字错误，已修正，原算法/评价不变，保留失败日志；未精确计时的该次尝试计入0.1 worker小时保守预留。环境适配、代码身份和异常见对应回执。','', '分支codex/eg-lqr-build-confirm-v3-20261019；登记1c3ad4361cc15b168a87ee68305ffb1846a0d6be，父结果f92c05b7261eaa34d8d9380b28890b4ad004075f。大权重、输入、完整候选缓存和日志留共享盘本卡目录，轻量结果与相关实现归档。','', '后续不自动追加训练、改系数、换种子或测试未锁定族；当前结果只评价这一固定方法配方。']
if preflight:
 p1_state=read('results/P1_status.json',{});export_state=read('results/C1_export_status.json',{});eval_wall=sum(x['worker_wall_seconds'] for x in (state or {}).get('done',[]));total=preflight['P0_successful_worker_seconds']+p1_state.get('worker_wall_seconds',0)+eval_wall+export_state.get('wall_seconds',0);reserve=preflight['failed_attempt_worker_hours_reserve']*3600+300
 complete_accounting=bool(p2 and len((state or {}).get('done',[]))==12 and export_state.get('status')=='PASS')
 resources=dict(accounting_status='COMPLETE_WORKER_WALL' if complete_accounting else 'LOWER_BOUND_COMPLETED_TASKS_ONLY_ACTIVE_TASKS_NOT_SETTLED',P0_successful_worker_seconds=preflight['P0_successful_worker_seconds'],P1_worker_seconds=p1_state.get('worker_wall_seconds'),C1_completed_evaluation_worker_seconds=eval_wall,C1_export_worker_seconds=export_state.get('wall_seconds'),conservative_prior_helpers_and_failed_attempt_reserve_seconds=reserve,conservative_accounted_worker_hours=(total+reserve)/3600,conservative_gpu_process_hours=(export_state.get('wall_seconds',0)+300)/3600,prior_gpu_reserve_is_not_measured=True,P1_CPU_seconds=p1_state.get('child_cpu_seconds'),C1_CPU_seconds=(state or {}).get('child_cpu_seconds'),new_trainings=0,backbone_updates=0,head_ensemble=False)
 (R/'results/resource_summary.json').write_text(json.dumps(resources,indent=2)+'\n');lines += ['',f'已结算成功worker墙钟{total/3600:.3f}小时，含旧尝试/辅助任务保守预留后为{(total+reserve)/3600:.3f}小时；GPU导出加前期300秒保守预留计{resources["conservative_gpu_process_hours"]:.3f}进程小时。预留不是测量值。'+('当前仍有活动任务未结算，阶段计数只是下界。' if not complete_accounting else '')]
(R/'results/final_report.md').write_text('\n'.join(lines)+'\n');print('REPORT_BUILT',p2['status'] if p2 else 'CONFIRMATION_PENDING',flush=True)

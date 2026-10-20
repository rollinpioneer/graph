from common import *
import subprocess

assert json.load(open(R/'results/computation_complete.json'))['status']=='PASS'
cfg=json.load(open(R/'frozen_config.json'));res=json.load(open(R/'results/resource_report.json'));s=json.load(open(R/'results/C3_summary.json'));speed=json.load(open(R/'results/speed_summary.json'));timing={v['family']:v for v in speed.get('measurements',[])}
def comparison(name):return next(v['metrics'] for v in s['comparisons'] if v['comparison']==name and v['use_cd'])
def performance_a(m):
 a=m['AP'];t=m['Top1_pct'];return (a['delta']>=.75 and a['CI95'][0]>0 and t['delta']>=-.5 and t['CI95'][0]>=-1 and a['positive_seeds']>=2) or (t['delta']>=1.5 and t['CI95'][0]>0 and a['delta']>=-.25 and a['CI95'][0]>=-.5 and t['positive_seeds']>=2)
def noninferior(m):
 a=m['AP'];t=m['Top1_pct'];return a['delta']>=-.25 and a['CI95'][0]>=-.5 and t['delta']>=-.5 and t['CI95'][0]>=-1
decisions={}
for family in ['C','S']:
 m=comparison(family+'-G');tt=timing.get(family);a=performance_a(m) and bool(tt) and tt['median_ratio_to_G']<=1.25 and tt['p95_ratio_to_G']<=1.5;b=noninferior(m) and bool(tt) and tt['median_ratio_to_G']<=.8 and tt['p95_ratio_to_G']<=1
 decisions[family]=dict(classification='A' if a else 'B' if b else 'NO_FROZEN_ADVANTAGE',performance_A=performance_a(m),noninferiority_B=noninferior(m),efficiency_measured=bool(tt),median_ratio=tt['median_ratio_to_G'] if tt else None,p95_ratio=tt['p95_ratio_to_G'] if tt else None)
rank={'A':2,'B':1,'NO_FROZEN_ADVANTAGE':0};c=rank[decisions['C']['classification']];z=rank[decisions['S']['classification']]
choice='G'
if c>0 and c>=z:choice='C'
if z>c:choice='S'
if c==z==2 and performance_a(comparison('S-C')):choice='S'
if c==z==1 and noninferior(comparison('S-C')) and timing['S']['total']['median']<=.8*timing['C']['total']['median']:choice='S'
sm=comparison('S-G');sa=sm['AP']['delta'];st=sm['Top1_pct']['delta'];distill=bool(timing.get('S')) and timing['S']['median_ratio_to_G']<=.8 and timing['S']['p95_ratio_to_G']<=1 and sa>=-1 and st>=-2 and (sa<0 or st<0) and z==0
write(R/'results/development_decision.json',dict(status='COMPLETE',classifications=decisions,preferred=choice,distillation_future_separate_card_suggested=distill,C3_development_only=True,independent_confirmation=False,C4_run=False,new_training_started=False))
primary=[v for v in s['metrics'] if v['use_cd'] and v['split']=='all'];lines=['# EG-GEOMETRY-COMPACT-V1 最终报告','','状态：P0–P3完成。用户批准P1–P3，不授权远端推送。C3开发结果，不是独立确认；本轮C4预测读取/评价0次。开跑前EXPERIMENT_CARD/frozen_config保持冻结版本，批准和实际状态见approval_P1_P3.json及本报告。','','## 主要结果：CD，三种子先在场景内平均','','| 方法 | AP | AP0.4 | AP0.8 | Top-1合格% | 非空碰撞% | 空抓% |','|---|---:|---:|---:|---:|---:|---:|']
for family in ['O','G','C','S']:
 v=next(x for x in primary if x['family']==family);lines.append(f"| {family} | {v['AP']:.3f} | {v['AP04']:.3f} | {v['AP08']:.3f} | {v['Top1_pct']:.3f} | {v['Top1_nonempty_collision_pct']:.3f} | {v['Top1_empty_pct']:.3f} |")
lines+=['','全部12场景×256帧=3072帧；24配置(CD/noCD×四方法×三个头)，73728条帧配置结果。O/G逐帧缓存散列及指标核验后复用。新C/S从同一1024原动作解码，只改Q/F05排序；姿态、位置和宽度不变。','', '| 配对比较 | AP差 [95%场景区间] | AP正向场景 | Top-1百分点差 [95%场景区间] | Top-1正向场景 |','|---|---:|---:|---:|---:|']
for name in ['C-G','S-G','C-O','S-O']:
 m=comparison(name);a=m['AP'];t=m['Top1_pct'];lines.append(f"| {name} | {a['delta']:+.3f} [{a['CI95'][0]:+.3f}, {a['CI95'][1]:+.3f}] | {a['positive_scenes']}/12 | {t['delta']:+.3f} [{t['CI95'][0]:+.3f}, {t['CI95'][1]:+.3f}] | {t['positive_scenes']}/12 |")
lines+=['','预注册统计：先三头场景内平均，再Seen/Similar/Novel各4场景分层配对bootstrap10000次，seed20261020。帧和头种子不作独立场景；区间为开发比较描述，未进行族错误率校正。完整逐头/逐场景、三划分和noCD指标见results/*.csv及C3_summary.json。','','## 计算成本','','| 方法 | 完整评分中位秒 | p95秒 | 中位/G | p95/G | 峰值GPU MiB |','|---|---:|---:|---:|---:|---:|']
for family in ['G','C','S']:
 if family not in timing:lines.append(f'| {family} | 未达到冻结测速准入 | — | — | — | — |');continue
 t=timing[family];lines.append(f"| {family} | {t['total']['median']:.6f} | {t['total']['p95']:.6f} | {t['median_ratio_to_G']:.3f} | {t['p95_ratio_to_G']:.3f} | {t['peak_gpu_bytes']/1024**2:.2f} |")
lines+=['','速度为同进程同GPU、固定96帧交替顺序、FP32、种子20261018单头；每项完整包含深度云预处理、邻域、几何/采样、归一化、传输、编码评分和F05。G为先前精确等价cov加速版。全部计时帧G与旧缓存对应输入精确相等；若C参加则其保留列亦逐项相等。只有S参加时才执行96帧缩球点/mask与完整.16邻域核验；未入选S没有正式P3速度或96帧等价结果。各阶段明细及同期共享负载见speed_summary.json和speed_per_frame.csv。显存为所有被测模型常驻下该方法的峰值分配；三头共享输入成本只作网络成本外推，非另一次实测。不把新增评分耗时等同完整EG端到端延迟。' if speed['status']=='PASS' else '没有新方法达到冻结的P3准入，未运行正式测速；P0少量帧时间仅供可行性参考。','','## 冻结判据与决策','']
for family in ['C','S']:
 lines.append(f"- {family}：{decisions[family]['classification']}。性能A部分{'满足' if decisions[family]['performance_A'] else '不满足'}，性能非劣B部分{'满足' if decisions[family]['noninferiority_B'] else '不满足'}；成本亦按冻结中位/p95门槛判断。")
lines+=['',f'首选：保留/发展{choice}。该选择严格采用开跑前A优先B、并列优先C及S-C例外规则。',('建议下一张独立实验卡单独测试G→S蒸馏，原因是S明显更快但当前小幅负性能差未通过A/B；本轮未执行。' if distill else '本轮证据不支持把G→S蒸馏列为必做下一步；不自动增加蒸馏或其他模块。'),'','三个问题的回答：',f"1. 删除PCA/法向后的C相对G：AP {comparison('C-G')['AP']['delta']:+.3f}，Top-1 {comparison('C-G')['Top1_pct']['delta']:+.3f}pp；是否足够接近取决于上述冻结区间和成本判据。",f"2. 小PointNet S相对G：AP {sa:+.3f}，Top-1 {st:+.3f}pp；不能仅凭点估计或代理MSE宣称超过G。",f'3. 按本轮完整开发证据首选{choice}；独立新数据确认需另卡批准，C4已使用不能再作为新方法独立确认。','','## 训练、身份与资源','']
for family in ['C','S']:
 for seed in SEEDS:
  v=json.load(open(R/f'results/train_{family}_{seed}.json'));lines.append(f"- {family}/{seed}：step1000固定，参数{v['parameters']}，模型sha256 `{v['checkpoint_sha256']}`；model-val MSE {v['diagnostics']['model_val']['MSE']:.6f}，gate-cal MSE {v['diagnostics']['gate_cal']['MSE']:.6f}。未按其选模。")
lines+=['',f"累计GPU {res['GPU_process_hours_upper']:.4f}进程小时、CPU {res['CPU_worker_hours_upper']:.3f}worker小时，新增逻辑存储{res['new_logical_bytes']/1024**3:.3f}GiB。GPU进程时间包含其CPU取点/等待，CPU按worker墙钟保守记账；共享GPU同期负载记录在资源回执中。实际峰值CPU准备72、GPU训练6、GPU评分24、CPU评价72、GPU交替计时1。",'正式训练仅6头×1000步；保护O/G和EG训练0，EG前向0，新训练GT0。训练Top64+随机64候选身份与旧O相同，原24576组跨帧混合索引和scene/frame/candidate流哈希三种子全一致，所有六姿态分别取几何。step250/500/750只是恢复快照，只有step1000进入指标。','主要训练/评价日志、权重、逐帧缓存留服务器本轮独立目录；Git仅代码、冻结配置、摘要CSV/JSON和报告，本轮不推送远端。','','## 证据局限','','C3已开发使用，本轮比较不是独立泛化证明；12场景区间有限。C/S同时改变几何表示、维度与少量参数量，不能将差异完全解释为单一PCA信息的因果效应。S固定128点/五区域配额会截断密度信息，1000步固定训练的结果只适用于本配方，未证明所有PointNet设计上限。只使用推理可见深度，未读GT模型/分割/接触作为输入；官方完整模型仅用于测试质量评价。共享硬件计时有负载噪声，轮换同环境降低混淆，仍不等同真实机器人成功率。','']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines));write(R/'results/final_result_receipt.json',dict(status='PASS',config_sha256=sha(R/'frozen_config.json'),final_report_sha256=sha(R/'FINAL_REPORT.md'),preferred=choice,decisions=decisions,distill_recommendation_only=distill,new_experiment_started=False,remote_push=False))
print('FINAL_RESEARCH_DECISION',choice,json.dumps(decisions),flush=True)

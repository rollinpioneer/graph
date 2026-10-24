import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;S=json.loads((R/'statistics.json').read_text());A=np.load(R/'selected_angles.npz')
def get(g):return next(x for x in S['summaries'] if x['group']==g)
def pair(a,b,m):return next(x for x in S['paired'] if x['first']==a and x['second']==b and x['metric']==m)
def pp(r):return f"{r['mean']*100:+.2f} [{r['low']*100:+.2f}, {r['high']*100:+.2f}]"
def num(v):return str(int(round(v))) if abs(v-round(v))<1e-8 else f'{v:.1f}'
eff=list(csv.DictReader((R/'effective_set_sizes.csv').open()));overlap=[]
for k in [3,5]:
 for filt in ['none','EG_cd']:
  p=f'P{k}_{filt}';l=f'L{k}_{filt}';pm=A[p+'_mask'];lm=A[l+'_mask'];pa=A[p];la=A[l];nom=A['original_angle']
  dist=lambda angle:np.minimum((angle.astype(int)-nom)%12,(nom-angle.astype(int))%12)
  overlap.append(dict(k=k,filter=filt,identical_sets=int((pm==lm).all(1).sum()),identical_choices=int((pa==la).sum()),P_mean_distance_degrees=float(dist(pa).mean()*15),L_mean_distance_degrees=float(dist(la).mean()*15),P_mean_eligible=float(np.mean([int(r['eligible']) for r in eff if r['group']==p])),L_mean_eligible=float(np.mean([int(r['eligible']) for r in eff if r['group']==l]))))
(R/'set_overlap_diagnostics.json').write_text(json.dumps(overlap,indent=2))
# Counts are paired fixed-panel outcomes; random bars are Monte Carlo ranges, not scene CIs.
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,filt in zip(axes,['none','EG_cd']):
 for k in [3,5]:
  for family,color in [('P','#3568ad'),('L','#d97a24'),('R','#757575')]:
   g=f'{family}{k}_{filt}';x=get(g);ax.scatter(x['collision'],x['recovery'],marker='o' if k==3 else '^',color=color,s=70)
   ax.annotate(family+str(k),(x['collision'],x['recovery']),xytext=(5,6 if family!='L' else -12),textcoords='offset points')
   if family=='R':
    cr=x['monte_carlo_95_range']['collision'];rr=x['monte_carlo_95_range']['recovery']
    ax.errorbar(x['collision'],x['recovery'],xerr=[[x['collision']-cr[0]],[cr[1]-x['collision']]],yerr=[[x['recovery']-rr[0]],[rr[1]-x['recovery']]],fmt='none',color=color,alpha=.5)
 ax.set_title('No collision filter' if filt=='none' else 'Existing EG collision filter');ax.set_xlabel('New nonempty collisions (lower is better)');ax.set_ylabel('Weak to feasible recoveries');ax.grid(alpha=.2)
fig.suptitle('Z0 fixed-angle-set comparison | random bars: 95% Monte Carlo range',fontsize=11);fig.tight_layout();fig.savefig(R/'Z0_tradeoff.png',dpi=170);plt.close(fig)
rows=list(csv.DictReader((R/'distance_rank_tertiles.csv').open()));fig,ax=plt.subplots(figsize=(7,4))
for b,label,color in [(1,'Higher logits','#3568ad'),(2,'Middle','#909090'),(3,'Lower logits','#d97a24')]:
 rr=[r for r in rows if int(r['tertile'])==b];x=np.array([int(r['degrees']) for r in rr]);y=np.array([float(r['mean']) for r in rr]);lo=np.array([float(r['low']) for r in rr]);hi=np.array([float(r['high']) for r in rr]);ax.errorbar(x+(b-2)*1.5,y*100,yerr=np.stack([y-lo,hi-y])*100,fmt='o-',color=color,label=label,capsize=3)
ax.set_xticks([15,30,45,60,75,90]);ax.set_xlabel('Cyclic distance from original angle (degrees)');ax.set_ylabel('Scene-equal nonempty collision rate (%)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'distance_rank_collision.png',dpi=170);plt.close(fig)
pr=get('P3_none');lo=get('L3_none');ci=pair('P3_none','L3_none','collision')
lines=['# Z0：检测器角度先验与局部修正零训练诊断','日期：2026-10-24。**主结论：未支持PRIOR；结果落在LOCALITY的±30%区间，P₃的碰撞点估计反而略差。**',
'只执行用户批准的Z0。复用本地已验证的服务器缓存，没有SSH访问、新推理、训练、风险标签预计算或新增力闭合评价；C4/C5未访问。',
'## 1. 主比较与判定',
f"k=3、无过滤：P₃新增碰撞 **96**，L₃为 **87**，比值 **{S['primary_collision_ratio']:.3f}**（P₃多10.3%）；弱→可行恢复 **142 / 141**，比值 **{S['primary_recovery_ratio']:.3f}**。场景等权碰撞率差P₃−L₃为 **{pp(ci)} 个百分点**。",
'PRIOR的三项要求中，仅“恢复≥局部法90%”通过；“碰撞≤局部法70%”和“碰撞差CI上界<0”均未通过。因此不能写成“logit先验比局部约束更能抑制碰撞”。',
'判定文字存在重叠：原表把±30%差异称为LOCALITY，同时把“P₃比L₃差”称为NO-PRIOR。执行前的`protocol.json`明确先按量化±30%区间归类，再单独披露较差点估计。本轮据此记为LOCALITY；若将NO-PRIOR解释为任何较差点估计，则也满足该文字条件。**不受命名影响的结论是：没有达到PRIOR，且没有证据证明P₃优于L₃。** CI跨零也不能证明两者严格等效。',
'## 2. 输入与必要核验',
'面板为12场景×4帧、1,229个原干净候选、14,748个已直接解析评价角度姿态；原可行641、原弱力闭合588。48份角度文件和48份EG_cd文件均与历史结果SHA一致，候选ID/原角度逐项对齐。',
'原角度与前12类角度输出argmax不一致数为 **0/1,229**，没有触发强制补入。第13类未参与。P/L每个原始集合严格含k个类并包含原角度；缺失I1依原规则排除，过滤版另排除EG_cd为碰撞的角度。向量化choose与原标量规则逐候选核对通过。',
'1,000次随机抽样使用PCG64、rng=20261024，每次保留原角度并均匀无放回抽其他类；k=3、k=5和两种过滤共享随机排列以便配对。场景bootstrap另以rng=20261024在seen/similar/novel内分层抽样10,000次。随机组先对每个候选的1,000次结果求期望，再做场景配对区间；另单列固定面板的随机抽样范围，二者不混用。',
'## 3. 全部固定组结果',
'可行：无碰撞、非空且0<μ≤0.4；弱力闭合：0.4<μ≤1.2。碰撞、空夹、无力闭合解分别计数。下表为1,229候选的总数，R为1,000次抽样均值；所有组空夹均为0。',
'| k | EG过滤 | 集合 | 可行数 | 弱→可行 | 原可行受损 | 新增碰撞 | 无力闭合解 |','|---|---|---|---:|---:|---:|---:|---:|']
for k in [3,5]:
 for filt in ['none','EG_cd']:
  for family in ['P','L','R']:
   x=get(f'{family}{k}_{filt}');lines.append(f"| {k} | {'无' if filt=='none' else '有'} | {family} | "+' | '.join(num(x[m]) for m in ['feasible','recovery','harm','collision','no_FC'])+' |')
lines += ['P=logit前k；L=循环局部±⌊k/2⌋；R=原角度加随机k−1类。加入EG过滤后P₃/L₃新增碰撞为66/60；k=5无过滤为147/144，过滤后94/86。次要比较也没有出现P的碰撞优势。随机组加过滤后的碰撞可能更少，但恢复明显不足，不能只按碰撞数量排优劣。',
'| 随机组 | 碰撞均值 [95%随机范围] | 恢复均值 [95%随机范围] |','|---|---:|---:|']
for k in [3,5]:
 for filt in ['none','EG_cd']:
  x=get(f'R{k}_{filt}');a=x['monte_carlo_95_range']['collision'];b=x['monte_carlo_95_range']['recovery'];lines.append(f"| R{k} / {filt} | {x['collision']:.1f} [{a[0]:.1f}, {a[1]:.1f}] | {x['recovery']:.1f} [{b[0]:.1f}, {b[1]:.1f}] |")
lines += ['上述范围只表达随机角度集合的波动，不是泛化到新场景的置信区间。完整场景配对区间见`paired_intervals.csv`。',
'| P−L，场景等权差（百分点） | 可行率 | 碰撞率 |','|---|---:|---:|']
for k in [3,5]:
 for filt in ['none','EG_cd']:lines.append(f"| k={k}, {filt} | {pp(pair(f'P{k}_{filt}',f'L{k}_{filt}','feasible'))} | {pp(pair(f'P{k}_{filt}',f'L{k}_{filt}','collision'))} |")
lines += ['![Z0恢复与碰撞权衡](Z0_tradeoff.png)','## 4. 机制统计：排序并非毫无信息，但没有带来主比较优势',
'去掉原角度后共有13,519个变体；其中1个空夹单独排除，机制AUROC使用13,518个，未知标签0。I1缺失的变体仍纳入碰撞知识统计，避免只看能恢复接触的姿态。',
'| 碰撞风险预测量 | 场景等权AUROC [95%区间] | 合并候选AUROC（描述性） |','|---|---:|---:|']
for x in S['mechanism']['AUROC']:
 e=x['scene_equal'];lines.append(f"| {x['predictor']} | {e['mean']:.4f} [{e['low']:.4f}, {e['high']:.4f}] | {x['pooled_AUROC']:.4f} |")
d=S['mechanism']['paired_AUROC_difference'];lines += [f"logit排名−循环距离的AUROC差为 **{d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]**，没有整体判别优势。这里logit排名越靠后、循环距离越大都定义为更高碰撞风险，未根据结果翻转方向。",
'| 相同循环距离 | 较高logit三分位碰撞率 | 中间 | 较低logit | 低−高差及95%区间（百分点） |','|---|---:|---:|---:|---:|']
for dist in range(1,7):
 rr=[r for r in rows if int(r['distance'])==dist];c=next(x for x in S['mechanism']['tertile_contrasts'] if x['distance']==dist);lines.append(f"| {dist*15}° | "+' | '.join(f"{float(r['mean'])*100:.2f}%" for r in rr)+f" | {pp(c)} |")
lines += ['同距分层仍显示较高logit与较低碰撞率的关联，例如15°为12.71%对24.55%，30°为31.33%对52.94%。但各层不是都严格单调，45°的高低差区间跨零；三分位保留相同排名的并列值，实际组大小不相等，完整切点和样本数已保存。',
'这些分层仅控制角度距离，未消除跨候选、场景内物体或几何难度等混杂。可以保留“角度输出包含碰撞相关信号”的描述，不能由此推翻预注册主比较，也不能断言碰撞知识由某种训练机制因果产生。',
'![同距logit三分位碰撞率](distance_rank_collision.png)','## 5. 局部性、有效预算与H3登记']
for x in overlap:
 if x['k']==3 and x['filter']=='none':
  lines.append(f"主比较中P₃/L₃有{x['identical_sets']}/1,229个集合完全相同，最终选择相同{x['identical_choices']}个。实际选角平均偏移为{x['P_mean_distance_degrees']:.2f}° / {x['L_mean_distance_degrees']:.2f}°；排除I1缺失后平均可选角度数为{x['P_mean_eligible']:.3f} / {x['L_mean_eligible']:.3f}。名义候选预算相同不意味着有效集合及选角距离完全相同；这是解释限制，不是事后改变集合或指标。")
lines += ['**论文叙事：** 当前证据更适合“局部角度修正的摩擦—碰撞权衡”，不支持把“检测器先验优于局部性”作为已经成立的机制。也不能反过来声称EG的logit完全不含碰撞信息。',
'**H3后续预测登记：** 若未来另行授权T轮，不因Z0而预期EG先验会可靠抵消T2的碰撞增幅；保留原H3的待检验预测，即不协调摩擦与碰撞的T2比T1更容易增加碰撞。该预测尤其针对训练将角度推离原局部范围的情况，是否成立必须由预注册T1/T2对照判断，Z0不提供定量训练效果预测。',
'Z0不决定是否启动训练。本轮没有执行风险标签预计算、λ标定、T0/T1/T2训练或全C3 Level B。',
'## 6. 可信度、边界与归档',
f"CPU核心计算约{S['wall_seconds']:.2f}秒；输入核验与统计均在本地完成。未创建SSH会话、没有GPU任务。L₃的87个碰撞复现原±15°结果。原始输入保持只读，代码、协议、逐候选选择、1,000次随机选择索引与抽样统计、场景级区间和图表保存于独立目录。",
'只使用12个既有C3开发场景、官方完整EG及既定干净候选，结论不等于全流水线Top-1/AP提升、跨模型普遍性或真实机器人成功率。选择使用GT辅助I1，是离线诊断，不能称为可部署推理策略。',
'交付：`statistics.json`、`selected_candidates.csv`、`selected_angles.npz`、`random_draw_counts.csv`、`scene_statistics.csv`、`paired_intervals.csv`、`AUROC_by_scene.csv`、`distance_rank_tertiles*.csv`、`input_manifest.json`、`protocol.json`及复现脚本。']
text=lines[0]
for old,new in zip(lines,lines[1:]):text+=('\n' if old.startswith('|') and new.startswith('|') else '\n\n')+new
(R/'report_zh.md').write_text(text+'\n',encoding='utf-8')
(R/'H3_prediction.json').write_text(json.dumps(dict(Z0='LOCALITY; PRIOR not supported; P3 point estimate slightly worse',prediction='Do not assume EG logit prior will reliably offset T2 collision growth. Retain the preregistered T2>T1 collision prediction, especially if angles move beyond the original local neighborhood; must be tested by authorized T1/T2 controls.',training_authorized_by_Z0=False,risk_label_generation_executed=False),indent=2))
assert get('L3_none')['collision']==87
print(json.dumps(dict(overlap=overlap,report='report_zh.md'),indent=2))

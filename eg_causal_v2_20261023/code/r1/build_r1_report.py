"""Report existing R1 results only; no inference/evaluation/training."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]/'outputs/eg_causal_v2_20261023'
DATA=ROOT/'data/r1';OUT=ROOT/'r1';OUT.mkdir(exist_ok=True)
def readcsv(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
rows=readcsv(DATA/'label_semantics_trace.csv');ties=readcsv(DATA/'tie_rule_gradients.csv');wrows=readcsv(ROOT/'width_diagnostics.csv')
summary=json.loads((DATA/'r1_semantic_findings.json').read_text());training=json.loads((ROOT/'training_summary.json').read_text())
zero=[r for r in rows if float(r['q_ref'])==0]
boundary=[r for r in zero if r['selected_raw_width_over100mm']=='True' and float(r['w_ref_m'])<=float(np.float32(.1))]
assert len(boundary)==5
cases=[];raw_sources={}
for r in boundary:
    p=DATA/f"frames/{r['frame_id']}_raw_grids.npz";a=np.load(p);j=int(np.flatnonzero(a['candidate_id']==int(r['candidate_id']))[0]);ww=a['width'][j,0,0]
    assert float(ww)==float(r['raw_reference_width_m']) and np.asarray(ww*np.float32(1000)).astype(np.uint8)==100
    assert abs(float(np.float32(100)/np.float32(1000))-float(r['w_ref_m']))<1e-9
    cases.append({k:r[k] for k in ['frame_id','candidate_id','object_id','raw_point_index','source_view','raw_reference_width_m','w_ref_m','raw_reference_mu','raw_reference_collision']})
    raw_sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
examples=[]
groups=[('raw_has_no_positive_quality',lambda r:int(r['raw_positive_cells'])==0),('collision_removes_all_positive_mu',lambda r:int(r['raw_positive_cells'])>0 and int(r['post_collision_positive_cells'])==0),('width_removes_remaining_positive_mu',lambda r:int(r['post_collision_positive_cells'])>0)]
for name,fun in groups:
    r=next(r for r in zero if fun(r));p=DATA/f"frames/{r['frame_id']}_raw_grids.npz";a=np.load(p);j=int(np.flatnonzero(a['candidate_id']==int(r['candidate_id']))[0])
    assert np.all(a['quality'][j]==0)
    tr=[t for t in ties if t['frame_id']==r['frame_id'] and t['candidate_id']==r['candidate_id']]
    for t in tr:
        aa,dd=int(t['angle_class']),int(t['depth_class']);assert float(a['width'][j,aa,dd])==float(t['raw_width_m'])
    examples.append(dict(description=name,trace=r,tie_rows=tr,raw_grid_source=str(p),raw_grid_candidate_row=j,
        mu=a['mu'][j].tolist(),width_m=a['width'][j].tolist(),collision=a['collision'][j].tolist(),quality=a['quality'][j].tolist()))
    raw_sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
checks=dict(boundary_cases=cases,three_examples=examples,raw_grid_sha256=raw_sources,
    example_selection='First query by file/candidate order within each described sequential label-elimination stratum; illustrative, not representative effect estimates.',
    boundary_count=5,boundary_unique_object_point_view=len(set((r['object_id'],r['raw_point_index'],r['source_view']) for r in boundary)),checks_passed=True)
(OUT/'r1_examples_and_quantization.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(1,3,figsize=(12.8,4.0),gridspec_kw={'width_ratios':[1.05,1.2,1]});fig.subplots_adjust(left=.065,right=.99,top=.77,bottom=.27,wspace=.43)
blue='#2C6E9C';orange='#D17A32';gray='#667788'
vals=[80,311,513];labels=['No raw $\\mu>0$\nin all 48 cells','Collision\non positive-$\\mu$\ncells','Width >100 mm\non positive-$\\mu$\ncells']
x=np.arange(3);bars=ax[0].bar(x,100*np.array(vals)/634,color=[gray,blue,orange],width=.65)
ax[0].set_xticks(x,labels);ax[0].set_ylim(0,100);ax[0].set_ylabel('Fraction of zero-quality queries (%)');ax[0].set_title('A  Sources overlap (n=634)',loc='left',fontweight='bold')
for b,v in zip(bars,vals):ax[0].text(b.get_x()+b.get_width()/2,b.get_height()+3,str(v),ha='center')
x=np.arange(3);cy=np.array([634/634,0,791/3838])*100;rev=np.array([1,0,1])*100
b1=ax[1].bar(x-.17,cy,.32,color=blue,label='Cycle by 17');b2=ax[1].bar(x+.17,rev,.32,color=orange,label='Reverse')
ax[1].set_xticks(x,['Zero quality\nn=634','Positive, unique\nn=4,260','Positive, tied\nn=3,838']);ax[1].set_ylim(0,125);ax[1].set_ylabel('Output gradient changed (%)');ax[1].set_title('B  Same grids, different tie order',loc='left',fontweight='bold');ax[1].legend(loc='upper center',ncol=2,fontsize=8,frameon=False,bbox_to_anchor=(.5,1.04))
for bs in [b1,b2]:
    for b in bs:ax[1].text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{b.get_height():.0f}' if b.get_height()!=cy[2] else '20.6',ha='center',fontsize=8)
x=np.arange(2);sat=np.array([359/634,1134/8098])*100;push=np.array([183/634,474/8098])*100
b1=ax[2].bar(x-.17,sat,.32,color=blue,label='At upper clip');b2=ax[2].bar(x+.17,push,.32,color=orange,label='Still pushed up')
ax[2].set_xticks(x,['Zero quality\nn=634','Positive quality\nn=8,098']);ax[2].set_ylim(0,75);ax[2].set_ylabel('Fraction of each query group (%)');ax[2].set_title('C  Executable width saturation',loc='left',fontweight='bold');ax[2].legend(loc='upper center',frameon=False,fontsize=8,bbox_to_anchor=(.5,1.04))
for bs in [b1,b2]:
    for b in bs:ax[2].text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{b.get_height():.1f}',ha='center',fontsize=8)
for a in ax:a.spines[['top','right']].set_visible(False);a.set_axisbelow(True);a.yaxis.grid(True,alpha=.18)
fig.suptitle('R1: label semantics, tie sensitivity, and width saturation',x=.065,ha='left',y=.98,fontweight='bold',fontsize=13)
fig.text(.065,.88,'Frozen inference queries in 16 training-set frames / 8 scenes. No parameter updates.',fontsize=10,color='#445566')
fig.text(.065,.065,'A: collision and width flags overlap in 270/634 queries; all 634 final grids are zero. Source flags are not exclusive causal attribution.',fontsize=8,color='#445566')
fig.text(.065,.02,'B: derivatives at cached outputs, not harmful parameter updates.  C: fixed inference queries, not historical or sampled training exposure.',fontsize=8,color='#445566')
fig.savefig(OUT/'r1_mechanism_summary.png',dpi=180);fig.savefig(OUT/'r1_mechanism_summary.pdf');plt.close(fig)

bt='\n'.join(f"| {r['frame_id']} / {r['candidate_id']} | {r['object_id']} / {r['raw_point_index']} / {r['source_view']} | {float(r['raw_reference_width_m'])*1000:.6f} | 100 |" for r in boundary)
example_text=[]
titles=['原始方向没有正质量标签','场景碰撞清除了全部正摩擦标签','宽度约束清除了碰撞后剩余标签']
for ex,title in zip(examples,titles):
    r=ex['trace'];ts=ex['tie_rows'];widths=' / '.join(f"{float(t['quantized_width_m'])*1000:.0f}" for t in ts);gr=' / '.join(f"{float(t['width_raw_gradient']):+.4f}" for t in ts)
    example_text.append(f"- **{title}：{r['frame_id']}，candidate {r['candidate_id']}**；物体 {r['object_id']}、原始点 {r['raw_point_index']}、方向 {r['source_view']}。48格中原始 μ>0 共 {r['raw_positive_cells']} 格，其中碰撞标记 {r['collision_positive_cells']} 格，碰撞后 μ>0 剩 {r['post_collision_positive_cells']} 格，再因宽度超限排除 {r['width_excludes_noncollision_positive_cells']} 格。三种遍历顺序的宽度目标为 **{widths} mm**，对应原始宽度输出梯度为 **{gr}**。")
g=training['gradients'];grtable='\n'.join(f"| {label} | {g[k]['weighted_norms']['width_zero']['median']:.3f} | {g[k]['weighted_norms']['width_positive']['median']:.3f} | {g[k]['cosine_q0_positive']['median']:+.3f} | {g[k]['cosine_q0_positive']['negative_batches']}/16 |" for k,label in [('grasp_head.conv_width','宽度末层'),('grasp_head.conv_width_feature','宽度特征'),('head_attention','头部 attention'),('backbone','骨干')])
report=f'''# R1：零质量姿态监督的语义与梯度适用域

**机制判断：支持“全零方向的代表 ADW 受平局约定支配，且该监督在真实训练路径中仍生效”；对“因此造成可行抓法学习受损”仍证据不足。方法开发：HOLD。** 下一步只考虑已设计的 R1.4 成对微更新，须额外参数更新授权；本轮没有训练或更新。

![R1诊断图](r1_mechanism_summary.png)

## 1. 标签精确溯源：零质量不是单一失败原因

8个训练场景、每场景帧0/128，共8,732个固定推理有效查询全部精确溯源，unknown=0。匹配关系保持冻结，按物体ID、原始点坐标的精确身份、方向与存储slot回溯，没有用最近离散GT替代预测姿态评价。

634个 q_ref=0 查询的完整48格**全部确为零**，不是微小正质量量化为零。554个方向原来有 μ>0 标签；其中468个在场景碰撞过滤后仍有 μ>0 格，随后全部被100mm宽度约束排除。其余86个方向的正μ格均有碰撞标记，80个方向原来就没有正μ标签。这是按既定过滤顺序的描述，不是互斥因果归因，μ>0也不等于满足主评价 μ≤0.4。

允许重叠的来源标记为：无原始正μ **80/634**、碰撞排除过正μ格 **311/634**、正μ格宽度超限 **513/634**，碰撞与超宽同时出现 **270/634**。原始μ≤0不足以继续区分无接触、摩擦不足等未记录原因。

全零后 `max(depth) → max(angle) → gather(width)` 固定选第0角度／第0深度，并携带该格的宽度。448个查询该格记录了非零宽度；**有几何宽度记录不等于该宽度代表一个可行抓法**。

## 2. 存储顺序干预与正质量控制

原序、左移17格、反序映射回原物理网格，分别在全零网格选格0、17、47，即AD索引(0,0)、(4,1)、(11,3)。点、方向、48格质量和物理动作集合均不改变；仅平局优先顺序变化。

| 群体 | n | 循环：AD/宽度/输出梯度改变 | 反序：AD/宽度/输出梯度改变 |
|---|---:|---|---|
| 全零质量 | 634 | 634 / 586 / 634 | 634 / 480 / 634 |
| 正质量、唯一最大值 | 4,260 | 0 / 0 / 0 | 0 / 0 / 0 |
| 正质量、最大值平局 | 3,838 | 791 / 720 / 791 | 3,838 / 2,590 / 3,838 |

AD一栏表示角度或深度任一改变。全零组循环干预与原始加权输出梯度的余弦中位数为−0.623；反序为+0.757。正质量唯一最大值控制全部不变；正质量平局单列，因为它们是在正质量目标支持的多个并列姿态之间选择，不能混作零质量语义缺陷。上述是固定输出上的梯度，不是参数更新，也不是损害证明。

## 3. 宽度：记录目标、原始输出与最终执行宽度不能混淆

原损失为 `SmoothL1(raw_width, 10*w_ref)`，单样本原始输出导数为 `clip(raw_width−10*w_ref, −1, 1)`，按原分母乘 `10/N_valid`。执行宽度为 `clip(1.2*raw_width/10, 0, 0.1)`。

| 固定推理有效查询 | q_ref=0，n=634 | q_ref>0，n=8,098 |
|---|---:|---:|
| 预测已到100mm上界 | 359（56.6%） | 1,134（14.0%） |
| 已到上界，梯度仍推动raw增大 | 183（28.9%） | 474（5.85%） |
| 平均绝对raw梯度 | 0.406 | 0.0783 |

零质量冻结宽度目标为：0mm 186个、>100mm 293个、其余155个；其中只有86个进入SmoothL1线性区。大损失不等于无限大梯度。场景等权上界饱和率差为+29.34个百分点，8场景bootstrap区间[16.06,41.44]；7/8场景方向相同，scene76略相反。scene31占零质量样本332/634，因此不能把634个候选当独立统计单位。

**298与293的差异已逐候选核验。** 原始被选宽度超过100mm有298个；其中以下5个查询的原始宽度位于100–101mm，保存时 `float32(width*1000) → uint8 → /1000` 截断为100mm，故冻结目标超限为293。5次查询来自3个不同物体—点—方向，均原始第0格μ=−1且无碰撞；不是5个额外的边界可行格被错删。

| 帧 / candidate ID | 物体 / 原始点 / 方向 | 原始宽度(mm) | 保存宽度(mm) |
|---|---|---:|---:|
{bt}

这5例已从逐帧原始48格重新读取并复算量化；100mm的float32表示为0.10000000149m，分类时用同精度阈值，未把该浮点表示误当超限。

## 4. 真实训练路径：疑似监督存在，但构成与固定推理查询不同

当前权重下，16个预登记增强batch使用官方训练方向采样，`net.is_training/view.is_training/PyTorch train=True`，FP32，batch size=1，只求导。8,676个有效查询中零质量3,456（39.83%）；排除重复方向映射歧义后仍为3,058/7,675（39.84%）。全部零质量目标的角度／深度为(0,0)，8场景均出现。参数保持不变、BN缓冲恢复，无优化器。

但训练零质量目标中**2,696/3,456（78.0%）为零宽度**，超过100mm只有324；预测上界饱和294（8.5%），下界饱和793（22.9%）。因此固定推理查询的56.6%上界饱和**不能外推为训练时的主导暴露**，也不能把不同方向采样造成的39.8%对7.26%当成成对因果效应。

| 参数位置 | 零质量width梯度范数中位数 | 正质量width梯度范数中位数 | 两者余弦中位数 | 负余弦batch数 |
|---|---:|---:|---:|---:|
{grtable}

这些梯度确实抵达共享表示，并常与正质量width方向相反；范数采用原有效样本分母和width权重10。但与其他损失的共享表示梯度余弦靠近零，且已汇总梯度包含方向映射歧义样本。负余弦不等于留出场景物理表现受损，更不是实际Adam更新。宽度末层没有其他损失的结构路径，数值消减残差不解释为额外梯度路径。

首次重放仅1个有效mask、3个ADWQ标签不同，均位于预记录的重复方向映射歧义集合；查询与采样方向一致。train/eval的下游模式对照也会改变输出，不能混为整网模式因果试验。上述是当前权重的16个batch核验，不能追溯官方每轮历史训练。

## 5. 可复核的候选实例

以下为三种过滤过程各按文件／候选顺序选择的首例，仅解释机制，不估计效果：

{chr(10).join(example_text)}

三例的原始μ、宽度、碰撞、过滤后48格以及对应输出梯度完整保存于 [案例与量化核验](r1_examples_and_quantization.json)，直接对应 `data/r1/frames/train_0002_0000_raw_grids.npz` 中candidate 2/47/73。

## 6. 决策与证据边界

- **支持**：全零方向的代表ADW含任意平局约定；有符号的width梯度会进入共享表示；这些目标并非仅固定推理标签探针的人造现象。
- **尚不支持**：零质量监督已被证明造成EG实际失败，或删除这些监督一定更好；饱和与梯度冲突不足以完成该推断。
- **方法HOLD；R1.4值得作为唯一下一步**：原损失、仅去零质量width梯度、保留末层但阻断共享表示、等数量随机去width，并补实际步长控制。每臂从同一参数/BN/优化器/RNG出发，固定分母，仅用未参与更新样本的重新解析结果判断；需额外授权，当前未执行。

证据文件：[逐候选语义](../data/r1/label_semantics_trace.csv)、[存储顺序与梯度](../data/r1/tie_rule_gradients.csv)、[原始48格](../data/r1/frames/)、[宽度统计](../width_diagnostics_summary.json)、[真实训练路径](../training_summary.json)。代码路径为官方 `dataset/generate_economic.py` 的过滤与max/gather、`utils/label_generation.py` 的点/方向有效匹配、`models/loss_economicgrasp.py` 的有效掩码ADWQ损失；sidecar只读，未改原始标签或EG/G。
'''
(OUT/'R1_summary.md').write_text(report,encoding='utf-8')
print(json.dumps({'report':str(OUT/'R1_summary.md'),'figure':str(OUT/'r1_mechanism_summary.png'),'boundary_cases_verified':len(boundary),'examples':[(e['trace']['frame_id'],e['trace']['candidate_id']) for e in examples]},ensure_ascii=False))

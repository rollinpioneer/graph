import os
from pathlib import Path
import json,hashlib,base64,html,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

O=Path(os.environ.get('EG_POSE_OUTPUT_DIR', './outputs/EG_pose_quality_20261022')).resolve()
W=Path(__file__).parent;D=O/'data';F=O/'figures';F.mkdir(exist_ok=True)
s=json.loads((O/'EG_pose_quality_statistics.json').read_text());A=s['A'];B=s['B'];S=B['supplementary']
df=pd.read_csv(O/'EG_pose_quality_evidence.csv',low_memory=False);t=df[df.split.eq('train')].copy();v=t.valid_match.eq(True);h=t.final_rank.gt(0);bad=~t.P0_good
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'figure.facecolor':'white','axes.titleweight':'bold','savefig.dpi':160})
blue='#235789';teal='#0d8279';amber='#b26b13';red='#bd514e';grey='#b1bac5'
def pc(x):return f'{100*x:.1f}%'
def ci(x,pct=True):
 if 'aggregate' in x:x=x['aggregate']
 fn=pc if pct else lambda z:f'{z:.3f}'
 return f"{fn(x['mean'])} [{fn(x['ci95'][0])}, {fn(x['ci95'][1])}]"
def nr(x):return f"{x['numerator']:,}/{x['denominator']:,}"
def tab(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def save(fig,name):fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)
def img(name,caption):return '<figure><img src="data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>'
def u(q):
 q=np.asarray(q);return ((q[:,None]>0)&(q[:,None]<=np.array([.2,.4,.6,.8,1,1.2])[None,:]+1e-7)).mean(1)

# All valid pairs, deterministic visual jitter only; numerical statistics use original values.
g=t[v];x=u(g.R_labelpoint_mu);y=u(g.P0_mu);rg=np.random.default_rng(20261022)
fig,ax=plt.subplots(figsize=(7.6,5.4));w=g.reference_good_actual_bad.eq(True).to_numpy()
ax.scatter(x[~w]+rg.uniform(-.017,.017,(~w).sum()),y[~w]+rg.uniform(-.017,.017,(~w).sum()),s=5,alpha=.10,color=blue,rasterized=True,label='其他有效配对')
ax.scatter(x[w]+rg.uniform(-.017,.017,w.sum()),y[w]+rg.uniform(-.017,.017,w.sum()),s=7,alpha=.15,color=red,rasterized=True,label='参考可行、实际失败（1,730）')
ax.plot([0,1],[0,1],color=grey,lw=1,ls='--');ax.set(xlim=(-.06,1.06),ylim=(-.06,1.06),xlabel='参考姿态重新评价的质量效用',ylabel='实际预测姿态的质量效用',title='配对质量：参考标签高分并不保证实际姿态可行')
ax.legend(loc='upper left',frameon=False,markerscale=3);fig.tight_layout();save(fig,'paired_quality')

fig,ax=plt.subplots(figsize=(9,4.4));labels=['全部高排名失败','可配对的失败','参考可行、实际失败','且物理 P5 恢复'];known=[315,149,94,49]
for j,n in enumerate(known):
 ax.barh(j,n,color=[grey,blue,teal,amber][j]);ax.text(n+3,j,str(n),va='center');
 if j>0:ax.barh(j,315-n,left=n,color='#edf0f4')
ax.set(yticks=range(4),yticklabels=labels,xlim=(0,338),xlabel='候选数（8 个训练场景；固定最终候选池）',title='分母必须保留：49/315 是已证实覆盖下界');ax.invert_yaxis();ax.text(.99,.02,'166 个失败缺少有效参考；其机制归因仍未知',transform=ax.transAxes,ha='right',color=red);fig.tight_layout();save(fig,'denominators')

keys=['P1_aligned','P2','P3','P4_aligned','P5_aligned'];labels=['物理角度','深度','宽度','物理角度+深度','物理角度+深度+宽度']
fig,ax=plt.subplots(figsize=(10,4.5));xx=np.arange(5)
for offset,event,color,lab in [(-.19,'recovery_high_failures',teal,'失败→可行（149 个失败）'),(.19,'harm_high_success',red,'可行→失败（311 个可行）')]:
 vals=[B['counterfactuals'][k][event]['aggregate'] for k in keys];m=np.array([z['mean'] for z in vals])*100;lo=np.array([z['ci95'][0] for z in vals])*100;hi=np.array([z['ci95'][1] for z in vals])*100
 ax.bar(xx+offset,m,.36,color=color,label=lab,yerr=[m-lo,hi-m],capsize=3)
 for i,k in enumerate(keys):ax.text(i+offset,hi[i]+2,nr(B['counterfactuals'][k][event]),ha='center',fontsize=8)
ax.set(xticks=xx,xticklabels=labels,ylabel='场景等权比例（%）及 95% 区间',ylim=(0,73),title='反事实存在恢复，也存在损害；GT 替换不是部署策略');ax.legend(frameon=False,loc='upper left',ncol=2);fig.tight_layout();save(fig,'counterfactual')

fig,axs=plt.subplots(1,2,figsize=(12,4.4));sc=s['configuration']['C3_scenes'];xx=np.arange(len(sc));av=A['top1']['scene_values'];bv=A['EG_B_top1']['scene_values']
axs[0].plot(xx,np.array(av)*100,'o-',color=blue,label='官方完整 EG');axs[0].plot(xx,np.array(bv)*100,'s--',color=grey,label='历史 EG-B');axs[0].set(xticks=xx,xticklabels=[str(x) for x in sc],ylim=(-5,105),xlabel='C3 场景',ylabel='Top-1 可行率（每场景 4 帧，%）',title='A｜完整权重未消除错误');axs[0].tick_params(axis='x',labelrotation=45);axs[0].legend(frameon=False)
rows=B['by_scene'];xx=np.arange(8);vals=[r['H1_aligned_witness']/r['high_fail']*100 for r in rows]
axs[1].bar(xx,vals,color=teal)
for i,r in enumerate(rows):axs[1].text(i,vals[i]+.7,f"{r['H1_aligned_witness']}/{r['high_fail']}",ha='center',fontsize=8)
axs[1].set(xticks=xx,xticklabels=[str(r['scene']) for r in rows],ylim=(0,30),xlabel='训练场景',ylabel='已证实失配占全部高排名失败（%）',title='B/C｜物理对齐后 8/8 场景出现见证');fig.tight_layout();save(fig,'by_scene')

# Complete provenance, keeping the public manifest free of local user/server paths.
m=json.loads((D/'asset_manifest.json').read_text());m['runtime']['model_is_training']=False;m['runtime'].pop('inference_mode',None)
m['runtime']['precision_note']='Network inference retains TF32. Only read-only official label probe uses FP32 matrix multiplication; no prediction was changed.'
m['historical_weight']={'path':'$WORK/eg_sub25_b_f05_repl_v1_20261019/runs/B/train/economicgrasp_epoch10.tar','sha256':'533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98','training':'50 scenes, 6400 frames, 10 epochs, 16000 updates; history only; no candidate ID correspondence'}
m['source_caveat']='Inference/label-matching/loss sources are official. Existing label generator includes operational resume/atomic-save patches; compression logic checked against raw labels. Official release epoch=10 is not independent verification of training logs.'
m['evaluation_preflight']=json.loads((D/'evaluation_preflight.json').read_text());m['raw_label_preflight']=json.loads((D/'labels_preflight_all.json').read_text())
m['runtime']['evaluation_cpu_seconds']=sum(json.loads(p.read_text()).get('cpu_seconds',0)+json.loads(p.read_text()).get('physical_controls_cpu_seconds',0) for p in (D/'evaluation').glob('*.json'))
m['runtime']['recorded_inference_wall_seconds']=sum(json.loads(p.read_text())['seconds'] for p in (D/'cache').glob('*.json'))
m['runtime']['evaluation_pose_calls']=2780*2+16384*2+8732*10
m['runtime']['compute_note']='Pose calls include duplicate controls; CPU time sums completed final evaluators only, excluding preflight/failed pilot/asset hashing. Inference receipt timing does not include all setup/reprobe overhead.'
m['cache_provenance']=[dict(name=p.stem,**json.loads(p.read_text())) for p in (D/'cache').glob('*.json')]
m['evaluation_receipts']=[dict(name=p.stem,**json.loads(p.read_text())) for p in (D/'evaluation').glob('*.json')]
m['local_light_cache_note']='Local cache NPZ strips input point arrays; original server cache hashes in receipts refer to complete cache. Prediction geometry SHA256 separately verified. Exact 20k input was used for contact support before transfer.'
m['light_cache_hashes']=[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (D/'cache').glob('*.npz')]
m['diagnostic_scripts']=[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in W.glob('*.py') if p.name in ['infer.py','evaluate.py','aligned_controls.py','labels_check.py','asset_manifest.py','analyze.py','report.py']]
m['literature_source_manifest']=json.loads((W/'literature/source_manifest.json').read_text())
m['bounded_amendment']='After finding class-index/physical-frame ambiguity, added P1_aligned/P4_aligned on the same fixed 8732 pairs, CPU only. P5_aligned was already evaluated as D3 control. Original class-index results retained; no threshold or sample changes.'
m['cleanup']={'server_compute_jobs_remaining':0,'SSH_sessions_closed':True,'saved_credentials_retained':True}
def sanitize(x):
 if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
 if isinstance(x,list):return [sanitize(v) for v in x]
 if isinstance(x,str):return x.replace('/aita_data/home/aita_xushijie/work/grasp','$WORK').replace('/aita_data/home/aita_xushijie','$HOME')
 return x
m=sanitize(m);(O/'asset_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')

decision='HOLD'
core='EG 的代表性质量目标确实不直接重评最终预测姿态，且存在跨场景、可纠正的姿态—质量失配；但本轮尚不能把它与普通物理姿态误差及训练／推理坐标差异分离为独立的质量监督机制，结论为 HOLD。'
unique_gap='在统一物理坐标、保持普通姿态监督与监督预算一致后，最终姿态质量对齐是否仍有额外作用。'
nextstep='停止扩大零训练诊断。若继续，只进行一次经授权的同预算最小对照，隔离“普通姿态监督／坐标一致性”与“最终姿态质量目标对齐”的增量；不增加重排序器或新模块。本轮没有启动该训练。'
s['decision']={'status':decision,'core':core,'unique_gap':unique_gap,'next_step':nextstep,'method_innovation_proven':False,'causality_proven':False}
s['bounded_amendment']=m['bounded_amendment'];s['resource_accounting']=m['runtime'];s['evidence_schema']={'rows':len(df),'C3_rows':2780,'training_rows':16384,'C3_scope':'Union of original final pools with and without CD; other raw candidates retained in prediction cache, quality unknown','invalid_reference':'valid_match=false; target and counterfactual numeric fields NaN; boolean witness flags false mean not established, not disproven','mu':'-1 is evaluated analytic failure, not missing; missing numeric field is unknown','quality_utility':'mean of pass(mu<=f) for f=.2,.4,.6,.8,1,1.2; collision/empty/noFC utility zero','angle_periodicity':'parallel-jaw pi symmetry; class-index mismatch is not physical roll mismatch','physical_controls':'P1_aligned/P4_aligned/P5_aligned preserve predicted center and approach, project transformed GT closing axis into prediction normal plane','reference':'R_labelpoint re-evaluates actual label center and full label rotation; R_predpoint changes approach but keeps predicted center; neither is counted as fixed-point/fixed-approach C recovery','H1_witness':'R_labelpoint feasible & P0 failed & specified P5 feasible; sufficient descriptive witness, not causal attribution'}
(O/'EG_pose_quality_statistics.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

decision_text=f'''# EG 姿态—质量一致性：{decision}

{core}

## 已完成
官方 Kinect EG（epoch 10）正常推理 C3 48 帧及预先固定的训练集 8 场景×2 帧；解析重评实际姿态与所有有效反事实。无训练、无 EG/G 修改、无 C4/C5 访问。小范围物理角度对照已完成。

## 关键证据
- C3 Top-1 可行 **20/48，{ci(A['top1'])}**；28 次失败均有最终池内可行替代，20 次有同物体替代。同物体排序反转 **{ci(A['inversion'])}**。完整权重未消除现象；与 EG-B 的配对差异区间跨零。
- 训练候选 **8,732/16,384** 有精确规则匹配；高排名失败 **149/315** 可配对。参考姿态重评可行、实际失败 **94/149**。
- 固定位置和方向后，物理角度＋深度恢复 **73/149**；再换代表宽度恢复 **67/149**。其中 **49** 同时满足“参考重评可行”，覆盖 **8 场景、22 物体**；占全部高排名失败的已证实场景等权下界 **{ci(B['rates']['all_high_failures']['H1_P5_aligned_witness'])}**。另 **166** 个失败归因未知，不能算作无失配。
- 49 个见证中 **31** 的质量分类已匹配高分参考目标，说明“类别目标正确而实际姿态失败”真实存在；不证明训练时该推理查询的监督历史，也不证明因果。

## 为什么还不是 GO
- **D1/D3：** 类别编号不是物理角度；原样替换 P5 仅恢复 18 个，对齐后 67 个。物理角度单独恢复 63 个，普通姿态误差／分组坐标差异仍是强竞争解释。
- **反证：** 训练配对的分数—实际质量相关性 **0.357**，高于分数—参考质量 **0.163**（同一批对象组）；不支持“分数主要跟随参考、忽略实际姿态”的强版本。物理 P5 同时破坏 **137/311** 个原可行姿态；GT 全量替换不能当方法收益。
- **D2/D4：** C3 去共同偏移、softmax、取消宽度放大分别为 **21/48、18/48、8/48**，不能简单解释或修复全部错误。
- **覆盖／观测：** 高排名失败只有 47.3% 可配对；训练 Top-1 仅 3 次失败，其中 1 次可配对，不能声称已解释 C3 Top-1 的失配比例。67 个物理 P5 恢复中，23 个至少一个接触距输入点超过 10 mm，单视角可学习性未证实。
- **创新边界：** GraspGen 已有 on-generator 实际抓法质量学习；Contact-GraspNet 有预测姿态 ADD-S 约束；已有接触 SDF 物理监督。不能把生成后补质量标签或增加物理约束本身当新原理。第二检测器仅源码对照，未经实验证实。

## 唯一关键缺口与下一步
**{unique_gap}**

{nextstep}

所有置信区间均为场景分层 bootstrap 95% 区间，描述这批固定开发场景；不代表机器人成功率。完整报告、统计及逐候选证据见同目录文件。
'''
(O/'decision.md').write_text(decision_text,encoding='utf-8')

source=lambda url,label:f'<a href="{url}">{label}</a>'
egbase='https://github.com/iSEE-Laboratory/EconomicGrasp/blob/4119bdcd6bf5d3712a110f78ca87504dd359eec0/'
mechanism=f'''<p>官方生成器先用场景碰撞网格过滤，再在每个点—方向的 12×4 角度—深度格中取代表最大质量，保存对应角度、深度、宽度；本次资产保留 300 个方向。宽度按毫米整数存储。完整标签保留的多姿态结构被压缩，但本轮不研究新压缩方案。质量变换为正标签 <code>1.1−μ</code>，六类目标为 <code>floor(5q)</code>；分数越大越好，解析 μ 越小越好。来源：{source(egbase+'dataset/generate_economic.py','标签生成')}、{source(egbase+'models/loss_economicgrasp.py','官方损失')}。</p>
<p>匹配要求最近标签点距离 &lt;5 mm，且预测方向存在映射；无效位置／方向不参与角度、深度、宽度、质量损失。有效匹配中的零质量负例仍受监督，因此不能称 EG“没有负例”。质量交叉熵取代表目标，未用最终解码 ADW 重新解析评价；但各头共享并交互特征，姿态头也有显式损失，不能称“完全无约束”。</p>
<p>训练分组使用物体变换后的标签旋转，推理使用预测方向的规范旋转。直接比较类别编号会混入零角基准差异；夹爪对称按 π 处理。本轮以正常推理得到唯一 P0，随后只读调用同一 <code>process_grasp_labels</code>，恢复当前查询的官方规则参考目标；这不是其历史训练曝光记录，也不是训练模式前向分数。来源：{source(egbase+'utils/label_generation.py','标签匹配')}、{source(egbase+'models/economicgrasp.py','训练／推理及解码')}。</p>'''

arows=[['完整 EG：Top-1 可行','20/48',ci(A['top1'])],['历史 EG-B：Top-1 可行','21/48',ci(A['EG_B_top1'])],['完整 EG − EG-B','−1/48',ci(A['full_minus_EGB'])],['同物体失败—可行配对排序反转',f"{A['mixed_object_frames']} 个混合对象帧",ci(A['inversion'])],['分数—解析质量效用 Spearman ρ','对象帧内相关，再场景等权',ci(A['score_quality_rho'],False)],['Top-1 失败但最终池有可行替代','28/28','其中同物体 20/28'],['碰撞／空夹／无力闭合／质量不足','8 / 1 / 6 / 13','分母 48；空夹从碰撞中单列']]
brows=[['原始训练候选','16,384','全部 P0 实际评价完成'],['有效参考匹配','8,732/16,384',ci(B['coverage'])],['无效原因','6,140 / 766 / 746','点距超限 / 方向未匹配 / 两者'],['高排名候选有效匹配','460/779',ci(S['coverage_high_rank'])],['高排名失败有效匹配','149/315',ci(S['coverage_high_failures'])],['高排名成功有效匹配','311/464',ci(S['coverage_high_success'])],['参考重评可行、实际失败','94/149',ci(B['rates']['valid_high_failures']['reference_good_actual_bad'])],['训练 Top-1：可行／失败／失败可配对','13 / 3 / 1','不把 Top-50 候选比例外推为 Top-1 归因']]
cfrows=[]
for label,k in [('P1：角度（物理对齐）','P1_aligned'),('P2：深度','P2'),('P3：宽度','P3'),('P4：角度＋深度（物理对齐）','P4_aligned'),('P5：角度＋深度＋宽度（物理对齐）','P5_aligned')]:
 z=B['counterfactuals'][k];cfrows.append([label,nr(z['recovery_high_failures']),ci(z['recovery_high_failures']),nr(z['harm_high_success']),ci(z['harm_high_success'])])
classrows=[]
for k in ['P1','P2','P3','P4','P5']:
 z=B['counterfactuals'][k];classrows.append([k,nr(z['recovery_high_failures']),ci(z['recovery_high_failures']),nr(z['harm_high_success'])])
scrows=[[r['scene'],r['valid'],r['high_fail'],r['valid_high_fail'],r['reference_good_actual_bad'],r['P5_rescues'],r['aligned_rescues'],r['H1_aligned_witness']] for r in B['by_scene']]
controls=[['官方 logit 加权','20/48','基准'],['去除各候选 logits 共同偏移','21/48',ci(A['controls']['centered']['delta'])],['softmax 概率期望','18/48',ci(A['controls']['softmax']['delta'])],['固定原 Top-1，取消宽度 ×1.2','8/48',ci(A['controls']['width_no_scale']['delta'])],['不做模型自由碰撞过滤','21/48','单独敏感性；不替换主分析']]
litrows=[
 ['EG',source('https://github.com/iSEE-Laboratory/EconomicGrasp','官方源码'),'每点—方向代表 ADW 的质量；分离预测、特征交互','质量损失不重评最终姿态；本轮有实证，但混入分组基准与普通姿态误差'],
 ['GraspNet baseline',source('https://github.com/graspnet/graspnet-baseline/blob/main/models/loss.py','官方损失'),'每个深度以 GT 最佳角度选择质量监督；推理取预测角度质量','错误角度对应项可能缺少同等监督；类似可能性，仅源码推断'],
 ['GSNet 参考实现',source('https://github.com/graspnet/graspness_unofficial/blob/main/models/loss.py','graspness_unofficial'),'密集角度×深度质量回归，取最高分格的 AD','AD 与质量更明确绑定；不能直接推广 EG 的代表姿态假设。该仓库名义上为 unofficial'],
 ['Contact-GraspNet',source('https://github.com/NVlabs/contact_graspnet/blob/main/contact_graspnet/contact_graspnet.py','官方代码'),'置信度与附近成功接触相关；方向、宽度分离预测','已有 confidence-weighted ADD-S 姿态几何约束；不能说没有姿态—置信度耦合'],
 ['GraRe',source('https://github.com/Minakanmi-Yuki/grare','官方代码'),'冻结候选，用候选实际姿态解析质量、碰撞、空夹标签训练重排序','已覆盖实际候选质量学习；不改变原检测器代表监督。本轮不回到 G 式重排序'],
 ['GraspGen',source('https://arxiv.org/html/2507.13097v1','论文 §3.2')+' · '+source('https://github.com/NVlabs/GraspGen','官方代码'),'on-generator：生成器实际输出经仿真标记，再训练姿态条件化判别器','生成分布与离线标签分布失配已经被研究；生成后重新标注质量不是新思想'],
 ['Generalizing-Grasp',source('https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.html','CVPR 2024')+' · '+source('https://github.com/mahaoxiang822/Generalizing-Grasp/blob/main/utils/contact_point_loss.py','接触损失'),'直接使用预测接触的 SDF、表面与法向约束','已有姿态相关物理监督与推理接触／分数优化；不能把加物理一致性约束当新原理']]
cases=[]
for kind in ['physical_rescued','physical_unrescued','physical_harmed']:
 for r in s['cases'][kind]:
  cases.append([{'physical_rescued':'恢复','physical_unrescued':'未恢复','physical_harmed':'损害'}[kind],f"{r['scene']}/{r['frame']}/{r['candidate_id']}",r['object_id'],r['final_rank'],f"{r['original_score']:.2f}",r['P0_mu'],r['R_labelpoint_mu'],r['P5_aligned_mu'],f"{r['match_distance_m']*1000:.2f}",f"{r['physical_roll_error_deg']:.1f}"])

body=f'''<header><div class="eyebrow">GRASP · 完整、有边界的零训练机制诊断 · 2026-10-22</div><h1>EG 姿态—质量一致性</h1><p class="lead">{core}</p><div class="tags"><b>HOLD</b><span>官方 Kinect 权重</span><span>C3 48 帧 + 训练 16 帧</span><span>0 次训练更新</span></div></header>
<nav><a href="#decision">结论</a><a href="#A">排序现象</a><a href="#B">精确配对</a><a href="#C">反事实</a><a href="#D">竞争解释</a><a href="#E">已有工作</a><a href="#audit">复核与资产</a></nav>
<main><section id="decision"><h2>结论：现象成立，独立监督机制尚未成立</h2><div class="callout"><b>唯一关键缺口</b><p>{unique_gap}</p><p>{nextstep}</p></div>
<p>这不是资产不足导致的 HOLD。本轮 A–E 均已执行，必要几何、标签和解析资产无缺失。描述性见证要求“参考姿态实际重评可行、原姿态失败、固定点与方向的物理 P5 恢复”。共有 <b>49 个</b>，覆盖 <b>8/8 场景、22 个物体类别</b>；其中 31 个质量头分类已符合高分参考目标。但其余解释仍足以动摇“独立质量监督缺陷”的判断。</p>
<div class="cards"><div><strong>20/48</strong><span>完整 EG 的 C3 Top-1 可行</span></div><div><strong>8,732/16,384</strong><span>训练候选有效匹配</span></div><div><strong>49/315</strong><span>全部高排名失败中的已证实见证</span></div><div><strong>73/149</strong><span>物理角度＋深度可恢复的有效失败</span></div></div></section>
<section><h2>范围、定义与证据层级</h2><p>运行前固定 C3 场景 {s['configuration']['C3_scenes']}，每场景帧 0/64/128/192；训练场景 {s['configuration']['training_scenes']}，每场景帧 0/128，覆盖 38 个物体类别。训练场景按编号四分层、物体数量高低与新物体覆盖选取，未根据预测结果挑样本；有效样本充足，未触发扩展。只用一张 RTX 4090 推理，CPU 并行解析评价。</p>
<p>可行定义始终是无碰撞、非空夹且 0&lt;μ≤0.4。高排名指原始固定后处理池中的候选：碰撞过滤阈值 0.01 → NMS 3 cm/30° → 每物体 10 → 最终至多 50。对指定姿态的解析评价本身绕过这些截断，保留原始 ID。C3 评价两种最终池的并集 2,780 个实际姿态，训练集评价全部 16,384 个 P0；未把未知原始 C3 候选当失败。</p>
<p>主要估计按场景等权，10,000 次分层场景 bootstrap；C3 按 seen/similar/novel，训练按场景编号四分层。候选分子／分母只说明覆盖量，不把候选当独立样本。区间反映这批固定开发场景中的不确定性，不代表整套基准或真实机器人成功率。观察到差异 → 重评质量下降 → GT 反事实恢复，均不自动证明监督机制因果。</p></section>
<section><h2>源码核实：目标代表什么，不能代表什么</h2>{mechanism}</section>
<section id="A"><h2>A｜官方完整 EG 仍然存在选优错误</h2>{tab(['指标','原始分母／范围','场景等权估计 [95% CI]'],arows)}
<p>经过碰撞过滤后，NMS／每物体截断／Top-50 均未改变最高分候选，28 次失败的可行替代仍在最终池中；因此该批 Top-1 问题不能解释成后处理把所有好候选删光。完整 EG 与 EG-B 的差异为 −2.1 个百分点，区间 [−12.5, 6.3] 跨零。两者是相同图像，历史点采样与本轮候选不同，不能将差异仅归因于训练量；本轮没有复用 EG-B 的候选质量。</p>{img('by_scene','左：同帧场景级历史对照；右：训练配对的固定位置、方向反事实见证。两个分析集的比例不可互相替代。')}</section>
<section id="B"><h2>B｜精确匹配与参考姿态重评</h2>{tab(['项目','分子／分母','估计或解释'],brows)}
<p>有效配对明显偏向较容易的候选：原始有效匹配候选可行率场景均值 55.8%，无效匹配为 38.4%；高排名成功的匹配覆盖 65.7%，失败为 48.2%。因此 149 个可配对失败不代表全部 315 个失败。无效样本参考质量未知，没有伪造标签。</p>
<p>有效配对的物理滚转偏差中位数 12.36°，同类别编号角度差中位数却是 30°；深度差中位数近 0 mm，四分位上界 10 mm；宽度差中位数 13.99 mm；接近方向差中位数 4.51°；最近参考点距离中位数 3.28 mm。模型最终宽度含 ×1.2，不能把此宽度差都算回归误差。</p>
<p>名义高分目标（q≥0.7）中的参考姿态，重新评价仅 <b>4,812/7,438</b> 可行；高排名子集为 <b>297/434</b>。完整几何评价与离散标签生成、毫米宽度量化及碰撞判据并非无条件等价，不能直接拿名义 GT 高分证明可行。49 个主要见证已要求参考重评通过。参考与预测最近物体归属冲突为 0。</p>{img('paired_quality','全部 8,732 个有效配对，固定种子微抖动仅为展示重合点。效用为六个摩擦阈值下通过率的平均；碰撞、空夹、无力闭合为 0。')}{img('denominators','49/315 是已证实的描述性下界，166 个无有效参考的高排名失败仍未知；不是将未知记作“无失配”。')}</section>
<section id="C"><h2>C｜固定预测点和方向的反事实</h2><p>将物体坐标下的参考夹爪闭合轴变换到相机坐标，再投影到预测接近方向的法平面，得到同一物理基准下的角度。P1/P4 的物理对齐对照在发现类别基准歧义后补算，样本和阈值完全不变，属于解释性补充；P5 的物理对齐已作为 D3 对照保留。原始类别替换结果完整列于下方，不以新结果覆盖旧结果。</p>
{tab(['替换参数','恢复 / 原失败','场景恢复率 [CI]','损害 / 原可行','场景损害率 [CI]'],cfrows)}{img('counterfactual','所有 460 个高排名有效配对都保留：149 个原失败与 311 个原可行。误差线为场景区间；分母不同，不能直接相减百分比。')}
<p>物理 P5 的 67 个恢复中，53 个也可仅通过物理角度＋深度恢复，46 个可仅换物理角度恢复；只有 6 个也能仅换代表宽度恢复，3 个能靠去掉 ×1.2 恢复。49 个严格见证包含 32 个质量不足、12 个碰撞、3 个空夹、2 个无力闭合。联合物理 P1–P5 任一种能恢复 90/149，是多种 GT 干预的 Oracle 并集，不能当可部署模型收益。</p>
<p>物理 P5 见证对全部高排名失败的场景等权覆盖下界为 <b>{ci(B['rates']['all_high_failures']['H1_P5_aligned_witness'])}</b>；在有效匹配失败内为 {ci(B['rates']['valid_high_failures']['H1_P5_aligned_witness'])}。其中 31 个质量分类符合高分参考目标，提供直接目标—实际结果不一致的见证，但不能证明该目标造成了该预测错误。</p>
<details><summary>保留的原始类别编号替换 P1–P5（非物理对齐）</summary>{tab(['编号','恢复 / 149','场景恢复率 [CI]','损害 / 311'],classrows)}<p>P5 原样类别替换：18 个恢复，230 个损害。其明显劣化显示类别编号不能直接充当世界坐标中的角度误差。</p></details>
{tab(['训练场景','有效原始配对','全部高排名失败','有效配对失败','参考好／实际坏','类别 P5 恢复','物理 P5 恢复','物理 P5 严格见证'],scrows)}
<h3>代表案例：恢复、未恢复和损害同时展示</h3>{tab(['类型','场景/帧/原ID','物体','原排名','原分数','P0 μ','参考 μ','物理 P5 μ','点距 mm','滚转差 °'],cases)}<p>按各类型最低原排名排序，每场景至多一例；不是随机样本，也不用于统计。μ=−1 表示已评价失败（碰撞／空夹／无力闭合细分见 CSV），不是缺失。所有候选原始分数并非概率。</p></section>
<section id="D"><h2>D｜竞争解释与反证</h2><h3>D1：普通姿态预测误差尚未排除</h3><p>物理角度越偏，失败越多：高排名有效配对按滚转差 0–7.5°、7.5–15°、15–30°、30–90° 分组，场景失败率约 23.0%、27.9%、43.8%、51.0%（解释性分组，未据此调阈值）。这与普通姿态误差一致。按场景／帧／物体／精确名义质量分层，直接类别 AD 错误组相对正确组的失败差反而为 −4.1 个百分点；但类别比较受坐标基准污染，不能作为物理姿态“预测正确”的证明。</p>
<p><b>重要反证：</b>同一批 100 个可比较对象帧组中，分数与实际质量效用的场景相关性 {ci(B['score_target_vs_actual']['actual_rho'],False)}，与重评参考质量 {ci(B['score_target_vs_actual']['reference_rho'],False)}；参考减实际为 {ci(B['score_target_vs_actual']['delta'],False)}。因此强版本“质量头只学代表分数，忽略实际姿态”不受数据支持。表征共享、训练局部几何和样本选择可能使分数已经反映部分实际可行性。</p>
<h3>D2：分数解码有问题空间，但解释不了全部错误</h3><p>官方解码直接对六类 logits 加权；交叉熵对共同偏移不敏感，而解码分数随共同偏移 c 改变 3c。固定最终候选池做控制，未训练重排序器，也未重跑筛选以改变池。</p>{tab(['控制','C3 Top-1','相对原方案的场景差 [CI]'],controls)}
<h3>D3：训练／推理分组差异是真实竞争机制</h3><p>同类编号的物理角度不同、接近方向量化、最近点偏移均存在。仅修正物理角度恢复 63 个失败；AD 类别都“正确”的 70 个高排名候选中仍有 23 个失败，物理 P5 可恢复其中 12 个，证明类别正确不等于姿态一致。对接近方向误差≤5°的有效高排名失败，仍有 32/84 个严格物理 P5 见证，故也不只是接近方向偏差。</p><p>允许同时更改接近方向的 R_predpoint 恢复 71/149；再改到标签点的 R_labelpoint 恢复 94/149。这两项改变了固定条件，仅用于分解，未并入固定点／方向恢复率。单纯“把 GT 放回去”不是 H1 的充分证明。</p>
<h3>D4：宽度放大总体有益，不能简单删除</h3><p>C3 固定原 Top-1 去掉 ×1.2 后从 20/48 降至 8/48。训练高排名有效失败仅 5/149 因去掉放大恢复，同时损害 97/311 个原可行抓法。代表宽度替换损害 122/311，物理 P5 损害 137/311；原始标签宽度的量化与评价几何差异仍需保留为解释，不能宣称 GT 全量参数替换必然改善。</p>
<h3>D5：单视角可学习性未知</h3><p>使用推理实际输入的 20,000 点检查解析接触邻近性。67 个物理 P5 恢复中，两个接触都在 5 mm 内为 14 个、10 mm 内为 44 个；23 个至少一侧超过 10 mm。距离小也可能来自其他物体，距离大也可能是采样稀疏；这只是观测支持代理，不是遮挡真值或可学习性证明。没有新增遮挡模型或声称模型能得到 GT Oracle 收益。</p></section>
<section id="E"><h2>E｜跨方法适用性与创新边界</h2>{tab(['方法','一手来源','质量与姿态对应方式','对本假设的限制'],litrows)}
<p>理论可迁移所需最小结构条件：质量标签附着于代表／邻近参考，姿态分支可独立偏离，且质量监督未覆盖该偏离后的实际几何。满足这些条件只是研究线索；密集 AD 评分、显式姿态条件化判别器及现有物理约束都构成重要区别。没有在第二检测器上推理或训练，不能声称通用机制已经实证成立。任何继续方案都必须显示相对普通姿态约束、on-generator 质量学习的增量，否则可能只是 EG 专用修补。</p></section>
<section id="audit"><h2>必要复核、资源与可复现证据</h2><ul><li>官方权重 strict load 成功；Kinect 配置一致。权重 SHA256：<code>a47b717ad41870e9b34e1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a</code>（完整准确值见下方资产摘要）。</li><li>8 个历史已知候选与 50 个本轮官方 eval_grasp 候选质量／碰撞一致，后处理 ID 集一致；全 64 帧几何 SHA 与预测记录一致。</li><li>8 个训练场景、38,400 个抽样点—方向目标与完整原始标签及碰撞过滤后的压缩规则一致；所有有效反事实均有解析结果，无效参考保持未知。</li><li>只读探针曾受 TF32 旋转舍入影响，最大正交误差 9.33×10⁻⁴，首个试算被 Dex-Net 拒绝。仅标签探针改为 FP32 矩阵乘法，重跑匹配；24 个有效掩码与 41 个分数目标变化，匹配点索引变化 0，正交误差降至 3.58×10⁻⁷。模型推理 TF32 与原输出完全保留，失败试算未入统计。</li><li>完整训练历史来自官方发布声明、README 配置与检查点 epoch=10；没有独立训练日志，不能进一步证明其收敛充分性。源码提交 <code>4119bdcd6bf5d3712a110f78ca87504dd359eec0</code>。</li><li>完成姿态评价调用 {m['runtime']['evaluation_pose_calls']:,} 次（含重复对照），累计最终评价 CPU 时间 {m['runtime']['evaluation_cpu_seconds']/3600:.2f} 核时；未计准备／失败试算开销。0 次训练更新，最多 1 GPU；本轮计算任务及 SSH 已清理，保留获授权保存的连接配置。</li></ul>
<p>统计文件保留全部场景级分母与区间，CSV 共 {len(df):,} 行：16,384 个训练原始候选和 2,780 个 C3 已评价候选。C3 其余原始候选仍在完整预测缓存中，未擅自补算质量。Git 只归档独立诊断代码、小报告与压缩逐候选证据，不含点云、权重或凭证。</p>
<details><summary>权重、配置与输出摘要</summary><pre>{html.escape(json.dumps({'weight_sha256':'a47b717ad41870e9b34e85b1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a','official_source':'https://github.com/iSEE-Laboratory/EconomicGrasp/releases/download/v1/economicgrasp_kinect.tar','training_config':m['official_training_config'],'missing_assets':m['missing'],'verification':s['verification']},ensure_ascii=False,indent=2))}</pre></details>
<div class="downloads"><a href="EG_pose_quality_statistics.json">全部统计 JSON</a><a href="EG_pose_quality_evidence.csv">逐候选 CSV</a><a href="EG_pose_quality_evidence.csv.gz">压缩证据</a><a href="asset_manifest.json">资产清单</a><a href="decision.md">一页结论</a></div>
</section></main><footer>本轮完成到零训练机制诊断为止。HOLD 指独立机制与创新判断，绝非遗漏已获授权的诊断步骤。</footer>'''
# Avoid accidental shortened weight hash in the narrative.
body=body.replace('a47b717ad41870e9b34e1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a','a47b717ad41870e9b34e85b1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a')
style='''*{box-sizing:border-box}body{margin:0;color:#203044;background:#f4f6f9;font:16px/1.75 "Microsoft YaHei",Arial,sans-serif}header{background:#15364e;color:white;padding:48px max(5vw,24px)}header h1{font-size:40px;margin:8px 0}header .lead{max-width:1100px;font-size:20px}.eyebrow{color:#b4d6e6;font-size:14px;letter-spacing:1px}.tags{display:flex;gap:12px;flex-wrap:wrap}.tags>*{padding:5px 13px;border-radius:4px;background:#2d5068}.tags b{background:#e7b75d;color:#263241}nav{padding:13px 5vw;background:white;display:flex;gap:25px;flex-wrap:wrap;border-bottom:1px solid #ddd}a{color:#136a91;text-decoration:none}a:hover{text-decoration:underline}main{max-width:1300px;margin:auto;padding:24px}section{padding:24px 28px;margin-bottom:22px;background:white;border:1px solid #e3e8ee;border-radius:8px}h2{font-size:25px;line-height:1.4;margin:0 0 18px}h3{font-size:19px;margin-top:26px}.callout{background:#fff6e3;border-left:5px solid #b26b13;padding:15px 20px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:24px 0}.cards div{padding:20px;background:#edf4f8;border-radius:6px}.cards strong{display:block;font-size:28px;color:#15506b}.cards span{font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th{background:#eaf0f5;text-align:left;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #dbe2e8;vertical-align:top}tr:nth-child(even){background:#f8fafc}figure{margin:28px 0}figure img{display:block;max-width:100%;margin:auto}figcaption{font-size:13px;color:#566779;margin-top:10px}code,pre{font-family:Consolas,monospace;overflow-wrap:anywhere;background:#eff3f7}pre{white-space:pre-wrap;font-size:12px;padding:16px}details{border:1px solid #dae2e9;border-radius:5px;padding:12px;margin:18px 0}summary{cursor:pointer;color:#235789;font-weight:bold}.downloads{display:flex;gap:20px;flex-wrap:wrap;padding:20px;background:#eef4f7}footer{max-width:1300px;margin:auto;padding:0 50px 40px;color:#637487;font-size:14px}@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}section{padding:20px 15px}main{padding:12px}header h1{font-size:30px}table{min-width:680px}}@media print{body{background:white;font-size:12px}nav{display:none}section{break-inside:auto;border:0}figure{break-inside:avoid}.cards strong{font-size:20px}}'''
(O/'EG_pose_quality_final_diagnosis.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EG 姿态—质量一致性完整诊断 · HOLD</title><style>'+style+'</style>'+body+'</html>',encoding='utf-8')
print(json.dumps({'decision':decision,'rows':len(df),'CPU_core_hours':m['runtime']['evaluation_cpu_seconds']/3600,'pose_calls':m['runtime']['evaluation_pose_calls'],'report_bytes':(O/'EG_pose_quality_final_diagnosis.html').stat().st_size},ensure_ascii=False))

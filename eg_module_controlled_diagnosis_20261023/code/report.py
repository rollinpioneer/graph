import json,base64,html
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(r'C:\Users\jackx\Documents\Codex\2026-10-21\ssh-aita-xushijie-172-19-164\outputs\eg_module_controlled_diagnosis_20261023')
S=json.loads((O/'statistics.json').read_text(encoding='utf-8'));C=json.loads((O/'counterfactual_composition.json').read_text(encoding='utf-8'));N=(O/'narrative.html').read_text(encoding='utf-8')
def delta(d):
 if d is None:return '未定义'
 return f"{100*d['scene_mean']:+.1f} [{100*d['CI95'][0]:+.1f}, {100*d['CI95'][1]:+.1f}]"
def table(rows):return pd.DataFrame(rows).to_html(index=False,escape=False,border=0)
arms=['H0','H1_half','H1_off','H2_half','H2_off','H3','H4','N'];names={'H0':'原模型','H1_half':'姿态→质量半强度','H1_off':'姿态→质量关闭','H2_half':'质量→姿态半强度','H2_off':'质量→姿态关闭','H3':'仅分支自读','H4':'头部残差+LN','N':'邻域残差+LN','E1':'GT辅助局部基底'}
rows=[]
for split in ['train','C3']:
 for v in arms+(['E1'] if split=='train' else []):
  d=S['interventions'][split+'/'+v];c=C[f'{split}/{v}/new/new'];rows.append({'分区':split,'干预':names[v],'可行/候选':f"{d['good']}/{d['n']}",'恢复/损害':f"{d['recovery']}/{d['harm']}",'可行率变化 pp [95%区间]':delta(d['scene_delta']),'固定池Top-1':f"{c['top1']}/{c['frames']}",'Top-1变化 pp [95%区间]':delta(c['top1_scene_delta'])})
ct=[]
for v in arms:
 x={'干预':names[v]}
 for geo,sc,label in [('original','original','原姿态+原分数'),('new','original','新姿态+原分数'),('original','new','原姿态+新分数'),('new','new','新姿态+新分数')]:x[label]=str(C[f'C3/{v}/{geo}/{sc}']['top1'])+'/48'
 x['同物体排序反转变化 pp']=delta(C[f'C3/{v}/new/new']['inversion_scene_delta']);ct.append(x)
qr=pd.read_csv(O/'quality_target_groups.csv');qt=[]
for group in ['valid','qref_zero','high_fail','original_feasible']:
 for target in ['Q0','Q1','Q2']:
  d=qr[(qr.group==group)&(qr.target==target)];w=d.n.to_numpy();qt.append({'子组':group,'目标':target,'样本':int(w.sum()),'CE（描述性）':f'{np.average(d.CE,weights=w):.3f}','与Q0类别不同':f'{100*np.average(d.target_disagree_Q0,weights=w):.1f}%','与Q1类别不同':f'{100*np.average(d.target_disagree_Q1,weights=w):.1f}%'})
lp=[]
for l,d in S['zero_loss'].items():lp.append({'损失':l,'零质量样本/有效':f"{d['zero_samples']}/{d['valid_samples']}",'零质量贡献（官方加权）':f"{d['zero_contribution_mean']:.5f}",'正质量贡献（官方加权）':f"{d['positive_contribution_mean']:.5f}"})
routes=[]
for l in ['angle','depth','width','score','view','graspness','objectness']:
 x={'损失':l}
 for b,label in [('backbone','骨干'),('neighborhood','邻域'),('grasp_head.conv_score_feature','质量特征层'),('grasp_head.conv_score','最终质量层')]:
  r=next(r for r in S['gradient_routes'] if r['loss']==l and r['block']==b);x[label]=f"{r['median_weighted_norm']:.3f} ({r['nonzero']}/16)"
 routes.append(x)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig,axs=plt.subplots(1,2,figsize=(11,4.3),gridspec_kw={'width_ratios':[1.45,1]})
v=arms[1:];x=np.arange(len(v));ds=[C[f'C3/{a}/new/new']['top1_scene_delta'] for a in v];y=np.array([100*d['scene_mean'] for d in ds]);lo=np.array([100*d['CI95'][0] for d in ds]);hi=np.array([100*d['CI95'][1] for d in ds]);axs[0].bar(x,y,color=['#347da8' if t>=0 else '#b26256' for t in y]);axs[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',color='#333',capsize=3);axs[0].axhline(0,color='#555',lw=.8);axs[0].set_xticks(x,v,rotation=35);axs[0].set_ylabel('C3 固定池 Top-1 变化（百分点）');axs[0].set_title('场景配对 bootstrap 95% 区间（12 场景）');axs[0].spines[['top','right']].set_visible(False)
g=S['grid'];vals=[g['baseline_good'],g['top2_any'],g['grid_any']];bars=axs[1].bar(['原 argmax','Top-2×2\nOracle','全 48 格\nOracle'],vals,color=['#777','#4c93aa','#82b2ac']);axs[1].bar_label(bars,labels=[f'{t}/{g["queries"]}' for t in vals],padding=4);axs[1].set_ylim(0,g['queries']*1.1);axs[1].set_ylabel('至少一个可行组合的查询数');axs[1].set_title('固定位置、方向和宽度（8 场景）');axs[1].spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(O/'mechanism_summary.png',dpi=170);plt.close(fig)
pic=base64.b64encode((O/'mechanism_summary.png').read_bytes()).decode();r=S['replay'];res=S['resources'];extra=S['extra_classes']
doc=f'''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>EG 模块级受控机制诊断 · 2026-10-23</title><style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#20303a;max-width:1200px;margin:36px auto;padding:0 24px}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0 30px}}th,td{{padding:8px 10px;border-bottom:1px solid #d9e1e5;text-align:left}}th{{background:#edf3f5}}.note{{color:#586772;font-size:14px}}.lead{{background:#edf3f5;padding:16px 20px;border-left:4px solid #347da8}}img{{width:100%}}a{{color:#176d8c}}@media print{{body{{font-size:12px}}table{{font-size:10px}}}}</style><body>
<h1>EG 模块级受控机制诊断</h1><p class="note">2026-10-23｜官方发布权重｜16 个训练帧（8 场景）+48 个 C3 帧（12 场景）｜E0–E4｜参数更新 0</p>
{N}
<img src="data:image/png;base64,{pic}" alt="C3 干预区间及局部 AD 网格可达性">
<p><strong>完整预设干预结果。</strong>训练池为每帧固定 128 个候选，含原最终池及原分数分层候选；C3 为原最终池共 2,274 个候选。恢复/损害均逐候选配对，不因新碰撞删分母。区间按场景等权、10,000 次配对 bootstrap 计算；未作多重比较校正。E1 全池数仅作审计，机制解释以有效匹配子集为准。</p>
{table(rows)}
<p><strong>固定 C3 候选池的四组合。</strong>混合组合用于归因；此处不是重新后处理后的 AP。排序反转指同物体内失败候选高于可行候选的比例，先物体等权、再帧与场景等权；仅有两类候选的物体贡献此指标。</p>{table(ct)}
<p><strong>实际梯度路径。</strong>表中为 16 帧官方加权 L2 范数中位数，括号内为非零帧数；范数受参数块大小影响，完整每参数 RMS 见 CSV。共享骨干还接受上游 view、graspness 和 objectness 损失。</p>{table(routes)}
<p><strong>零质量监督台账。</strong>保留原有效样本分母。屏蔽零质量 ADW 后，对应行贡献变为 0，正质量贡献及质量损失保持原值；这是损失账本，不是训练效果。有效目标中角度第13类、深度第5类分别为 {extra['valid_angle13']}、{extra['valid_depth5']}；原输出全类别 argmax 落入它们分别 {extra['angle13_argmax']}、{extra['depth5_argmax']} 次，推理按官方实现截去它们。</p>{table(lp)}
<p><strong>质量监督对象对照。</strong>Q0 为原代表标签，Q1 为同评价器重评参考姿态，Q2 为实际姿态。六类转换和共有有效交集固定。不同目标的 CE 不可直接解释为哪个目标更优，下面仅描述分布及标签差异。</p>{table(qt)}
<p><strong>一致性与证据限制。</strong>64 帧历史姿态逐元素一致；最大分数误差 {r['max_score_error']:.3g}，全开复现、H1 姿态不变、H2 分数不变均通过。最大几何往返误差 {r['max_roundtrip_error']:.3g}；FP32 梯度求和相对误差 {r['max_gradient_sum_error']:.3g}，未放宽预设容差。官方视图映射重复索引写入导致重放时 {r['mask_replay_differences']} 个掩码和 {r['target_replay_differences']} 个质量目标不同；位置匹配一致，所有差异均核实属于多方向映射歧义，本轮保留原匹配和目标。不能把重放差异当成训练收益或论文机制。</p>
<p>已直接评价 {res['analytic_calls']:,} 个新姿态（上限 50,000），未知 0；其中全 AD 网格 12,288 个，256 个原单元均复现旧标签。解析评价 CPU 合计 {res['analytic_CPU_hours']:.3f} 小时。单 GPU 探针完成后释放，CPU 两机各 4 进程。E3-2/E5/E6 未执行；无优化器更新、无 EG/G 源码修改、无 Git 提交、无 C4/C5 访问。官方权重的历史训练收敛程度无法由本轮确认；场景已用于开发分析。开关可能分布外，梯度负夹角不证明有害学习，Oracle 可达性不等于可部署收益，解析可行性不等同机器人成功率。</p>
<p><strong>可核验文件：</strong> <a href="statistics.json">统计</a> · <a href="baseline_replay.json">重放记录</a> · <a href="gradient_routes.csv">梯度路径</a> · <a href="gradient_cosines.csv">梯度夹角</a> · <a href="loss_population.csv">损失台账</a> · <a href="intervention_results.csv">逐候选干预</a> · <a href="counterfactual_composition.json">四组合统计</a> · <a href="local_AD_grid.csv">AD网格</a> · <a href="training_receipts.json">零更新凭据</a> · <a href="source_map.md">源码映射</a> · <a href="protocol.json">冻结协议</a></p></body></html>'''
(O/'mechanism_report.html').write_text(doc,encoding='utf-8');print('report',O/'mechanism_report.html')

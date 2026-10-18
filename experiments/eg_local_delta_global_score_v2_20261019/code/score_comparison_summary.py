from common_v2 import *
import csv
rows=list(csv.DictReader(open(R/'results/development_per_scene.csv')));table={(int(x['scene']),x['configuration']):x for x in rows};scenes=CFG['development'];rng=np.random.default_rng(20261019);ix=rng.integers(0,len(scenes),(10000,len(scenes)));out=[]
for method,reference in [('G0','R0'),('Gabs','R0'),('Gdelta','R0'),('Gabs','G0'),('Gdelta','G0')]:
 diff=[];top=[];absolute=[];modified=[]
 for seed in SEEDS:
  a=[table[s,f'{method}_{seed}_CD'] for s in scenes];b=[table[s,'R0_CD' if reference=='R0' else f'{reference}_{seed}_CD'] for s in scenes];absolute.append([float(x['ap']) for x in a]);diff.append([float(x['ap'])-float(y['ap']) for x,y in zip(a,b)]);top.append([float(x['top1_valid'])-float(y['top1_valid']) for x,y in zip(a,b)]);modified.append(sum(int(x['modified']) for x in a)/sum(int(x['candidates']) for x in a))
 diff=np.array(diff);top=np.array(top);absolute=np.array(absolute);out.append(dict(method=method,reference=reference,AP_mean=float(absolute.mean()),AP_by_seed=absolute.mean(1).tolist(),difference_pp=float(diff.mean()),difference_by_seed_pp=diff.mean(1).tolist(),scene_ci95_pp=np.quantile(diff.mean(0)[ix].mean(1),[.025,.975]).tolist(),positive_scenes=int((diff.mean(0)>0).sum()),top1_difference_pp=float(top.mean()*100),modified_fraction_by_seed=modified,secondary=method!='Gdelta' or reference!='R0'))
write(R/'results/score_comparison_summary.json',dict(scope='Prespecified score contrasts; secondary intervals descriptive and unadjusted, observed development scenes only; not new evaluation or model selection',rows=out))
gabs=next(x for x in out if x['method']=='Gabs' and x['reference']=='R0');delta=next(x for x in out if x['method']=='Gdelta' and x['reference']=='G0')
p=R/'results/final_report.md';text=p.read_text();text+='\n## 次要评分信号与本轮决策\n\n'
text+=f"Gabs+CD 相对R0+CD为 **{gabs['difference_pp']:+.3f} AP pp**，逐种子{gabs['difference_by_seed_pp']}，场景聚类描述性95%CI={gabs['scene_ci95_pp']}，正向场景{gabs['positive_scenes']}/8。它是预先列出的次要对照，区间未做多重比较修正，不能取代失败的主Gdelta。\n\n"
text+=f"Gdelta 对相同姿态的G0，AP变化{delta['difference_pp']:+.3f} pp，完整流程Top-1变化{delta['top1_difference_pp']:+.3f}个百分点；AP差的95%CI={delta['scene_ci95_pp']}。均值不支持锚定增量评分带来收益，区间也不支持确定的普遍损害结论。\n\n"
text+='固定原Top-1只在第三种子救回1次，前两个种子Net为0。Gabs的正向信号应归类为当前开发条件下的系统/排序收益，未支持可靠姿态改进。全部评分对照的物理姿态相同，所以它们之间的应用差异来自评分引起的排序/NMS选择变化；这仍不能量化上一张卡失败的原因占比。\n\n'
text+='研究决策：主方法未通过，停止本卡，旧迁移和新确认不运行。若另立下一卡，最小的区分实验是固定原EG姿态，仅用Q(original)重排，检验Gabs收益是否需要姿态变化；本轮未执行该新对照。H1/H2未训练，不能据此声称差值损失或FiLM无效。\n';p.write_text(text);print(out)

import json,csv,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 result=json.loads((ROOT/'receipts/admission_result.json').read_text())
 events=[e for e in json.loads((ROOT/'receipts/reference_events.json').read_text()) if e['updates']<=500]
 exitinfo=json.loads((ROOT/'receipts/admission_exit.json').read_text());assert exitinfo['exit_code']==0
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 rows=[]
 for e in events:
  row=dict(updates=e['updates'],common_radius=e['rho_common'],radius_fraction=e['rho_fraction'],global_relative_difference=e['relative_diff_global_start_outside'][0],start_relative_difference=e['relative_diff_global_start_outside'][1],outside_relative_difference=e['relative_diff_global_start_outside'][2],global_match=e['global_match'],regional_match=e['regional_match'],continuous=e['continuous'])
  for arm in ('A','B'):
   a=e['arms'][arm];r=a['final'];row.update({arm+'_rms':r['rms'],arm+'_start_rms':r['regional'][0],arm+'_outside_rms':r['regional'][1],arm+'_follow_loss':r['follow_loss'],arm+'_candidate_count':len(a['candidates'])})
  row['B_start_density_change'],row['B_outside_density_change']=e['arms']['B']['final']['density_change']
  rows.append(row)
 with open(out/'admission_events.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,axs=plt.subplots(1,3,figsize=(13,3.6));x=[r['updates'] for r in rows]
 axs[0].plot(x,[r['radius_fraction'] for r in rows],'o-');axs[0].axhline(.5,color='r',ls='--');axs[0].set(title='Actual common radius / calibrated radius',xlabel='GFP updates per arm',ylabel='Fraction')
 for k,label in [('global','Global'),('start','Start'),('outside','Outside')]:axs[1].plot(x,[r[k+'_relative_difference']*100 for r in rows],'o-',label=label)
 axs[1].axhline(20,color='r',ls='--');axs[1].axhline(5,color='grey',ls=':');axs[1].legend();axs[1].set(title='Paired actual movement mismatch',xlabel='GFP updates per arm',ylabel='Relative difference (%)')
 for k in ('start','outside'):axs[2].plot(x,[r['B_'+k+'_density_change'] for r in rows],'o-',label=k)
 axs[2].axhline(1e-5,color='r',ls='--');axs[2].legend();axs[2].set(title='B: check-set density loss change',xlabel='GFP updates per arm',ylabel='NLL / action dimension change')
 fig.tight_layout();fig.savefig(out/'admission_curves.png',dpi=160);plt.close(fig)
 checks=[]
 for arm in ('A','B'):
  for step in (0,4,500):
   p=ROOT/'runs'/arm/f'checkpoint_{step}.json';d=json.loads(p.read_text());checks.append(dict(arm=arm,path=str(p.with_suffix('.pkl')),**d))
 (out/'checkpoint_index.json').write_text(json.dumps(checks,indent=2))
 assert all(c['full_restore']=='PASS' and c['step']==350000+c['updates'] for c in checks)
 assert all(checks[i]['agent_rng']==checks[i+3]['agent_rng'] for i in range(3))
 pre=ROOT/'receipts/preflight_attempt1';failed_seconds=json.loads((pre/'execution_error.json').read_text())['time']-json.loads((pre/'admission_started.json').read_text())['time']
 summary=dict(result,initial_preflight_failed_seconds=failed_seconds,initial_preflight_gfp_updates=0,actual_training_process_exit=exitinfo['exit_code'],at_admission_end_evaluations_completed=0,continued_after_admission=(ROOT/'receipts/continuation_started.json').exists())
 (out/'admission_summary.json').write_text(json.dumps(summary,indent=2))
 if not result['passed']:
  table='\n'.join(f"| {r['updates']} | {r['radius_fraction']:.3f} | {100*r['global_relative_difference']:.2f}% | {100*r['start_relative_difference']:.2f}% | {100*r['outside_relative_difference']:.2f}% |" for r in rows)
  text=f'''# A/B 参考更新比较：准入结果

**结论：STOPPED_MATCHING_ADMISSION / 无法裁决数据支持是否提供额外保护。** 按预先冻结的500次更新准入停止两臂，没有继续50k，没有开展环境评价，没有调参或重复训练。

两臂各实际500次GFP更新（总1000）；各5次参考更新尝试（总10）。模型训练从原seed3全局350k推进至350500。密度模型更新0；新增环境步、control calls和physics substeps均为0。

## 实测结果

总体动作匹配通过 {result['global_matches']}/5 次（要求5/5）；两区域匹配通过 {result['regional_matches']}/5 次（要求至少4/5）；持续更新门槛通过 {result['continuous_events']}/5 次（要求至少4/5）。持续更新要求共同半径至少达到初始校准半径的一半，不能靠近乎冻结通过。

| 每臂GFP更新 | 共同半径/初始半径 | 总体差异 | 起点差异 | 非起点差异 |
|---|---:|---:|---:|---:|
{table}

总体初始RMS上限 {result['calibration']['rho0']:.9g}；分区上限（起点、非起点）为 {result['calibration']['regional_start_outside']}。校准使用既定等效100次tau=5e-5 EMA规则，没有按结果改变阈值。详细候选、方向投影、真实密度变化和拒绝结果保存于reference_events.json。

![准入曲线](admission_curves.png)

## 证据边界

这说明当前更新规则及固定门槛未能形成可解释的持续、幅度匹配A/B对照；并没有证明数据支持信息无效，也没有比较出A/B的成功率或防崩塌能力。500次更新不用于评判原50k窗口的价值异常、行为崩塌或收益保留。不能把没有运行的400回合评价写成0%成功率。

零更新时TD目标逐项一致，损失梯度最大绝对差为0。首四次真实fused更新后在线Actor、VaBC、Critic与Target Critic均改变，内部参考EMA保持关闭。0/4/500次完整Agent、Adam计数与RNG保存恢复均通过；两臂训练RNG相同。来源检查点和两个行为模型的SHA256未变化。

最初校准准备因JAX采样shape传入整数而退出1，发生在任何GFP或参考参数更新之前。失败日志及成本保留；改为一元素tuple后继续零更新准备，未变更算法、样本预算或超参数。实际500更新准入进程正常退出0，停止来自预定义研究门槛。

## 成本与后续

有效准入进程墙钟 {result['seconds']:.1f} 秒、CPU {result['cpu_seconds']:.1f} 秒、峰值RSS {result['max_rss_kib']/1024/1024:.2f} GiB。前次零更新准备另耗 {failed_seconds:.1f} 秒；前次CPU/峰值显存未完整计量，不补造。使用GPU0；墙钟包含编译、保存和同步，不能当作纯GPU计算时长。资源快照单独保留。

本卡到此停止。最值得先核查的是：限制更新的实际因素是留出检查集上的密度非恶化条件，还是两个区域的动作幅度匹配？可先读取已保存候选与投影记录，不能因准入失败自动放宽约束并继续训练。本轮不支持把数据支持方案升级为已验证方法。
'''
  (out/'RESULTS.md').write_text(text)
 print(json.dumps(summary))
if __name__=='__main__':main()

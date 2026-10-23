"""Small Chinese report from the frozen R2 summary."""
import json,sys,html,csv
from pathlib import Path
R=Path(sys.argv[1]);s=json.loads((R/'basis_summary.json').read_text(encoding='utf-8'))
tab=[];scenes=[]
for k in ['0','1','3','6']:
    x=s['conditions'][k]['all'];ci=x['paired_scene_delta']['ci95']
    tab.append(f'<tr><td>{int(k)*15}°</td><td>{x["new_feasible"]}/256</td><td>{x["rescue"]}/{x["harm"]}</td><td>{100*x["paired_scene_delta"]["mean"]:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]</td><td>{x["pose_changed"]}/256</td><td>{x["mean_angle_prob_L1"]:.3f}</td></tr>')
    if k!='0':
        for scene,val in x['paired_scene_delta']['per_scene'].items():scenes.append(dict(k=int(k),phi_deg=int(k)*15,scene=int(scene),n=32,net_feasible_change=int(round(val*32)),net_percentage_points=100*val))
with (R/'basis_scene_effects.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(scenes[0]));w.writeheader();w.writerows(scenes)
body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>R2 无GT基底重编码</title>
<style>body{font:16px/1.65 system-ui,"Microsoft YaHei";max-width:1080px;margin:30px auto;padding:0 20px;color:#202b39}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd4df;padding:8px;text-align:left}th{background:#f1f4f8}p{margin:16px 0}.key{background:#edf3fa;padding:15px;border-left:4px solid #416d9d}a{color:#245d95}</style>
<h1>R2：无GT坐标重编码</h1>
<p class="key"><b>机制判断：</b>支持“预测对局部零角表达敏感”；对“原始训练／推理基底差异导致有害性能影响”仍证据不足。<b>方法开发：HOLD。</b>所有预登记非零旋转的场景配对可行率差区间均跨零，没有可据此选定的稳定增益。</p>
<p>范围：原8个训练场景、16帧、预登记256个查询（16原始Top-1、112分数分层、128随机）。保持点云、点位、物理接近方向、邻居和上游特征不变，按0/15/45/90°改变分组局部xyz，同步改变解码基底。冻结原宽度、保持原12×4物理动作集合，仅查表复用既有直接解析结果；没有新增解析评价、更新参数或访问C3/C4/C5。</p>
<table><tr><th>零角变化</th><th>解析可行</th><th>恢复／损害</th><th>净变化pp [场景95%CI]</th><th>物理AD改变</th><th>角度分布L1</th></tr>'''+''.join(tab)+'''</table>
<p>可行标准保持无碰撞、非空夹且0&lt;μ≤0.4。角度分布先对正式12类条件归一化，再循环回原物理角度；额外第13角度／第5深度类质量另列在逐查询表。置信区间以8个场景为单位进行20,000次配对bootstrap，不把旋转版本、候选或网格格子作为独立样本。场景数很少，区间仅为本样本的不确定性描述。</p>
<p>15/45/90°下，深度类别分别改变14/28/32个，重新预测的宽度平均绝对变化3.22/7.38/9.48mm；这些新宽度<b>没有</b>使用旧网格强行赋值，也没有新解析评价。Q分布和原始解码分数也会变化；官方分数为logit加权量，不能当作校准成功概率。</p>
<p>16个原始Top-1固定查询原有12个可行，变换后分别11、12、11个。同一预选16查询池内，G0S0/G1S0/G0S1/G1S1的帧级可行数分别为：15°时12/11/12/13，45°时12/12/13/13，90°时12/11/13/12，分母均16。该池不是原1024候选全池，未运行碰撞过滤、NMS、每物体截断与最终Top-50，不能报告为完整流程提升。</p>
<h2>代表案例与异质性</h2>
<p>同为15°纯坐标重编码：scene0002/frame0000/query858（原raw Top-1）由物理角度6→7，深度3不变，μ由0.4→0.6，发生损害；同场景frame0128/query880由角度0→1，深度2不变，μ由-1→0.4，发生恢复。两者均保持原位置、方向与宽度，说明敏感性可以向两个方向改变决策。全256样本中，15°对scene0002净增5/32，对scene0076净减6/32。不能用平均小幅增加掩盖场景差异。</p>
<h2>实际训练／推理基底审查</h2>
<p>源码训练分支使用process_grasp_labels返回的对象基底经物体pose旋转结果；推理分支使用ViewNet按世界接近方向生成的零角基底。对既有固定推理查询的有效匹配8732个，二者投影roll绝对值中位35.77°（四分位14.13–64.65°），6433/8732（73.67%）超过15°；距离最近15°格的残差中位3.64°。同时接近方向本身的差异中位4.51°，因此真实基底差异<b>并非纯roll重编号</b>。</p>
<p>此分布来自冻结推理查询与其训练标签基底，不是历史训练批次轨迹，也不证明官方权重曾受到有害更新。R2隔离了整数roll坐标表达，排除了E1的GT零角注入及可表达物理网格变化；它仍不能证明EG本应对任意基底等变，或者当前原始约定有错。没有按结果选择φ，没有开展新训练支线。</p>
<h2>必要验证与证据边界</h2>
<p>16帧跨同型号GPU历史H0输出完全重放；k=0完全恒等；缓存输出只重编号控制的概率、物理输出和可行标签不变；48个物理动作集合在夹爪π对称下最大矩阵元素误差4.03×10⁻¹⁶；邻居、上游非坐标特征、参数和BN缓冲区保持不变。仅执行协议预登记0/1/3/6格；负φ网络前向未额外扩展，逆变换仅用于恒等接线控制。</p>
<p><b>结论：</b>坐标表示敏感性已被干净隔离，但“表示差异损害学习或决策”的机制证据尚不足；不应根据45°样本净增11个就选取该φ或开发基底修复方法。R2保留为有边界的竞争解释，方法开发HOLD。只有保持物理语义的训练表示干预与留出收益一致，才能继续向有害根因推进，本轮未执行该步骤。</p>
<p>数据：<a href="basis_reencoding_results.csv">逐查询结果</a> · <a href="basis_scene_effects.csv">逐场景配对变化</a> · <a href="basis_fixed_pool_compositions.csv">固定池动作／分数分解</a> · <a href="basis_training_inference_audit.csv">基底审查</a> · <a href="basis_summary.json">统计与不变量</a></p></html>'''
(R/'r2_summary.html').write_text(body,encoding='utf-8')
print('R2 report written',R/'r2_summary.html')

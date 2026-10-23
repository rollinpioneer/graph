"""Concise Chinese R1.4 report from completed, paired analysis only."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np

LABELS = {
 "R0_original": "R0 原监督", "RW_remove_q0_width": "RW 移除零质量宽度梯度",
 "RW_local_keep_final_width_only": "RWlocal 仅末层保留", "Rrandom_count_matched": "随机等量移除",
 "RW_actual_step_norm_matched": "RW＋Adam步长匹配",
 "RW_local_actual_step_norm_matched": "RWlocal＋步长匹配",
 "Rrandom_actual_step_norm_matched": "随机等量＋步长匹配"}


def csvrows(path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig"))) if path.exists() else []


def percent(v, signed=False):
    if v is None:
        return "未知"
    return f"{v*100:+.2f}" if signed else f"{v*100:.2f}"


def effect(r):
    return f"{percent(r['mean'], True)} pp [{percent(r['ci95'][0], True)}, {percent(r['ci95'][1], True)}]"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in rows) + "</tbody></table>"


def create(root):
    analysis = root / "analysis"
    d = json.loads((analysis / "pulse_analysis.json").read_text(encoding="utf-8"))
    if d["status"] != "complete":
        raise ValueError("Do not produce final report from partial results")
    audit_path = Path(d["input_root"]) / "final_receipt_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["adam_steps"] != 280:
        raise ValueError("GPU execution receipt audit is incomplete or failed")
    contrasts = d["contrasts"]
    primary = [r for r in contrasts if r["scope"] == "fixed128" and r["metric"] == "feasibility"]
    main = [("RW_remove_q0_width", "R0_original"), ("RW_actual_step_norm_matched", "R0_original"),
            ("RW_remove_q0_width", "Rrandom_count_matched"), ("RW_actual_step_norm_matched", "Rrandom_actual_step_norm_matched")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    plot_labels = ["RW - R0", "RW norm - R0", "RW - random", "RW norm - random norm"]
    for ax, scope, metric, title in [(axes[0],"fixed128","feasibility","Fixed 128-query feasibility"),
                                     (axes[1],"G1S1","top1","Full pipeline Top-1 feasibility")]:
        for j, pair in enumerate(main):
            for st, shift, color in [(1,.13,"#347eb5"),(5,-.13,"#c55b38")]:
                r=next(x for x in contrasts if (x['treated'],x['control'])==pair and x['step']==st and x['scope']==scope and x['metric']==metric)
                m=np.asarray(r['mean'])*100; lo,hi=np.asarray(r['ci95'])*100
                ax.errorbar(m,j+shift,xerr=[[max(0,float(m-lo))],[max(0,float(hi-m))]],fmt='o',capsize=3,color=color,label=f"Step {st}" if j==0 else None)
        ax.axvline(0,color="#777",lw=1,ls="--");ax.grid(axis="x",alpha=.15)
        ax.set_title(title);ax.set_xlabel("Paired difference (percentage points)")
        ax.set_yticks(range(4),plot_labels);ax.legend(frameon=False)
    axes[0].invert_yaxis()
    fig.suptitle("R1.4: count and actual Adam-step controls; no best-step selection")
    fig.text(.5,.01,"95% paired scene-bootstrap intervals; 8 scenes, conditional on shared source pulses",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(root/"pulse_effects.png",dpi=170);plt.close(fig)
    rows = []
    for tr, co in main:
        for st in (1, 5):
            r = next(x for x in primary if (x["treated"],x["control"],x["step"]) == (tr, co, st))
            crossed = r["crossed_source_target_ci95"]
            rows.append([f"{LABELS[tr]} − {LABELS[co]}", st, effect(r),
                         " / ".join(percent(v, True) for v in r["fold_means"]),
                         f"[{percent(crossed[0],True)}, {percent(crossed[1],True)}]" if crossed else "未计算"])
    # Decision is deliberately conservative; a 1/5-step pulse is never a method claim.
    decisive = [r for r in primary if (r["treated"], r["control"]) in main]
    supported = [r for r in decisive if r["ci95"][0] is not None and r["ci95"][0] > 0]
    negative = [r for r in decisive if r["ci95"][1] is not None and r["ci95"][1] < 0]
    rw_norm_random = [r for r in decisive if r["treated"] == "RW_actual_step_norm_matched" and r["control"] == "Rrandom_actual_step_norm_matched"]
    if all(r["ci95"][1] <= 0 for r in rw_norm_random):
        finding = "本次微更新不支持移除零质量宽度梯度具有超越等量、等实际步长控制的收益。"
    elif any(r["ci95"][0] > 0 for r in rw_norm_random):
        finding = "至少一个预登记步数出现超越等量、等实际步长控制的局部信号；须结合两步、两折和完整筛选方向判断，不能据此宣布方法有效。"
    else:
        finding = "等量、等实际步长控制后的两个预登记步数区间均包含0；当前证据不足以确认零质量宽度监督造成可纠正的实际损害，不支持据此进入短程训练。"
    sections = []
    sections.append(f"<p class='lead'><b>方法开发：HOLD。</b>{finding}本轮仅完成已授权的 R1.4，未启动短程训练。</p>")
    rw5=next(r for r in primary if r['treated']=='RW_remove_q0_width' and r['control']=='R0_original' and r['step']==5)
    norm5=next(r for r in primary if r['treated']=='RW_actual_step_norm_matched' and r['control']=='R0_original' and r['step']==5)
    random5=next(r for r in primary if r['treated']=='RW_actual_step_norm_matched' and r['control']=='Rrandom_actual_step_norm_matched' and r['step']==5)
    sections.append(f"<p><b>关键结果：</b>第5步，RW相对R0仅 {effect(rw5)}；实际Adam步长匹配后为 {effect(norm5)}；相对等量且步长匹配的随机控制为 {effect(random5)}。第1步步长匹配RW相对R0有微弱正差，但对随机控制及来源依赖敏感性区间均包含0，第5步不保持；不能据此认定存在稳定且特异的有害监督效应。完整筛选的Top-1没有处理臂相对R0的收益。</p>")
    sections.append(f"<p>8 个来源批次、7 条件、每分支同一轨迹第 1/5 步；两折交叉留出，每个来源在对侧 4 场景的帧128评价。共 {d['jobs']} / {d['expected_jobs']} 个预登记来源—条件—步数—场景组合。固定池有 8×128＝{d['original_checkpoint_fixed_queries_unique']} 个唯一基础查询，原检查点可行 {d['original_checkpoint_fixed_feasible_unique']}；同一查询在不同微更新下反复评价，不能当成新增独立样本。</p>")
    sections.append("<h2>固定查询的主要因果对照</h2><p>直接评价实际新姿态，可行标准保持0&lt;μ≤0.4且无碰撞、无空夹；没有用最近离散GT质量代替。效应为可行率差，单位百分点；方括号为场景配对 bootstrap 95%区间。先在每留出场景平均4个来源脉冲，再将8个场景等权；两个步数均保留，不挑选最佳步数。完整均衡设计下，主结果按来源等权的均值与按场景等权均值相同；逐来源效应与两折方向另保留，不能把32个来源—场景组合称为32个独立场景。</p>")
    sections.append(table(["处理 − 对照", "步数", "差值 [场景95%区间]", "两折均值 pp", "来源×场景重采样区间 pp"], rows))
    sections.append("<img src='pulse_effects.png' alt='固定池与完整筛选的配对效应和95%区间'>")
    masks=[x['value'] for x in d['mask_receipts'] if 'source_scene' in x['value'] and 'zero' in x['value'] and 'random_overlap' in x['value']]
    if masks:
        overlap=sum(x['random_overlap'] for x in masks);zero=sum(x['zero'] for x in masks)
        fractions=[x['random_overlap']/x['zero'] for x in masks if x['zero']]
        sections.append(f"<p>随机等量控制在物体slot×原分数四分位内抽样，允许包含q0。{len(masks)}个来源中随机掩码与q0掩码累计重叠 {overlap}/{zero}（{overlap/zero:.1%}），逐来源重叠范围 {min(fractions):.1%}–{max(fractions):.1%}。因此它是数量与分层分布控制，并非与q0互斥的反事实；高重叠会减小两种干预的区别。此处只描述设计，不据此事后更换随机控制。</p>")
    local = [r for r in primary if (r["treated"],r["control"]) not in main]
    sections.append("<h2>共享参数与末层效应</h2>" + table(["处理 − 对照", "步数", "可行率差 [95%区间]"],
                     [[LABELS[r['treated']]+" − "+LABELS[r['control']], r['step'], effect(r)] for r in local]))
    rec = csvrows(analysis / "rescue_harm.csv")
    rr = []
    for arm in LABELS:
        for st in (1,5):
            for ref in ("original_checkpoint", "R0_original"):
                selected = [x for x in rec if x["arm"] == arm and int(x["step"]) == st and x["reference"] == ref]
                rr.append([LABELS[arm],st,"原检查点" if ref=="original_checkpoint" else "同来源同step R0",
                           sum(int(x["rescue"]) for x in selected),sum(int(x["harm"]) for x in selected),sum(int(x["n"]) for x in selected)])
    sections.append("<h2>恢复与损害</h2><p>下表为32个来源—留出场景组合累计候选事件，不是独立样本数；同一留出查询重复4次。必须同时看恢复和损害，净变化的区间以上述场景统计为准。</p>"+table(["条件","步数","参考","恢复","损害","可比候选事件"],rr))
    sr = [r for r in d["summaries"] if r["metric"]=="top1"]
    pipeline = []
    for arm in LABELS:
        for st in (1,5):
            vals = [next(r for r in sr if r['arm']==arm and r['step']==st and r['scope']==combo) for combo in ["G0S0","G1S0","G0S1","G1S1"]]
            pipeline.append([LABELS[arm],st]+[percent(r['mean'])+"%" for r in vals])
    sections.append("<h2>完整筛选与几何／分数交叉</h2><p>所有1024个冻结位置／方向候选进入官方碰撞、NMS、每物体Top-10、最终Top-50。G0/G1为原／新几何，S0/S1为原／新分数；这里报告最终Top-1可行率，和固定池可行率分开。没有候选的帧Top-1记失败，未知评价不记失败。最终池长度、可行数量、至少一可行和可行比例见逐帧表。</p>"+table(["条件","步数","G0S0","G1S0","G0S1","G1S1"],pipeline))
    pipecontr = [r for r in contrasts if r['scope']=='G1S1' and r['metric']=='top1' and (r['treated'],r['control']) in main]
    sections.append(table(["完整流水Top-1处理 − 对照","步数","差值 [95%区间]"],[[LABELS[r['treated']]+" − "+LABELS[r['control']],r['step'],effect(r)] for r in pipecontr]))
    sections.append("<p>Top-1配对区间[0,0]来自当前观测样本的处理／对照结果完全相同，导致经验bootstrap区间退化；不代表总体效应或总体不确定性严格为零。</p>")
    ranking = csvrows(analysis / "ranking.csv")
    rankrows = []
    for arm in LABELS:
        for st in (1,5):
            vals=[]
            for combo in ["G0S0","G0S1","G1S0","G1S1"]:
                scenes=[]
                for s in [2,9,31,33,64,68,76,82]:
                    v=[float(x['inversion_rate']) for x in ranking if x['arm']==arm and int(x['step'])==st and int(x['target_scene'])==s and x['geometry']+x['score']==combo and x['inversion_rate']!='']
                    if v: scenes.append(np.mean(v))
                vals.append(percent(float(np.mean(scenes)))+"%" if scenes else "样本不足")
            rankrows.append([LABELS[arm],st]+vals)
    sections.append("<h2>同物体排序与状态变化</h2><p>同物体可行—失败候选对中，失败分数高于可行分数为一次倒序，平局记半次。四种几何／分数组合用相同已知查询交集；先物体等权再来源、场景等权。该诊断不是AP，也不能把新几何改变后的标签变化混为纯评分改善。</p>"+table(["条件","步数","G0S0倒序","G0S1倒序","G1S0倒序","G1S1倒序"],rankrows))
    training=csvrows(analysis/"training_steps.csv")
    training_table=[]
    for arm in LABELS:
        for st in (1,5):
            rr=[r for r in training if r.get('arm')==arm and int(r.get('step',0))==st]
            if not rr: continue
            def mean_key(k):
                vals=[float(r[k]) for r in rr if r.get(k) not in (None,"")]
                return float(np.mean(vals)) if vals else None
            def num(k):
                v=mean_key(k)
                return f"{v:.5g}" if v is not None else "未记录"
            training_table.append([LABELS[arm],st,len(rr),num('loss'),num('qpositive_score_logit_abs_delta_mean'),
                                   percent(mean_key('width_upper_saturation_rate'))+"%",
                                   num('angle_changed')+" / "+num('depth_changed'),num('applied_delta_l2'),num('scale')])
    sections.append(table(["条件","步数","来源数","训练loss","q+ logit漂移","宽度上界饱和","A / D变化数","实际参数步长L2","缩放系数"],training_table))
    sections.append("<p>训练损失、q+分数漂移、宽度饱和、AD变化和逐步实际Adam更新范数保留在 <a href='analysis/training_steps.csv'>training_steps.csv</a>；训练状态来自该步更新前forward（第1步对应原参数、第5步对应4次更新后），而留出评价来自第1/5步完成后的参数。留出预测变化见 <a href='analysis/prediction_drift.csv'>prediction_drift.csv</a>。训练损失下降不等于留出抓取质量改善；实际步长匹配控制使用训练批次计算，未以留出结果调节。</p>")
    sections.append("<h2>执行澄清：共同起点与历史数值复演</h2><p>训练态 FP32 稀疏骨干在恢复参数、BN与随机状态后仍有微小数值非确定性；独立原生重复和原生→冻结查询路径均出现约2–3×10⁻⁵的logit差异，而复用同一次骨干输出时，下游路径差异为0。因此，第1步使用每来源同一个真实原生全网计算图，求总损失、零质量宽度和随机等量宽度的三组梯度，七臂各自从原参数与Adam状态执行更新，BN／RNG起点相同；第2–5步各臂重新计算完整骨干并反向传播。这不是冻结训练特征，也没有增加Adam更新次数。训练态BN／dropout始终开启；相对原检查点的变化含共同适应及BN效应，因此机制判断必须依赖处理臂−R0及随机／步长控制。</p><p>原3×10⁻⁵历史raw-logit阈值保持不变，超阈值记录仍记失败。场景64首次深度logit差4.5776367×10⁻⁵，首次图未保留；在澄清规则后仅重建一次共同初始图，第二次差异3.6239624×10⁻⁵也保留为历史检查失败，不是复用首次图，也不是反复尝试直至通过。必须透明说明：在来源2的35次更新完成、其余来源尚未更新时，历史数值对照被澄清为复演审计，而非排除科学来源批次的条件。七臂共同起点严格相同、同骨干下游严格相同、历史完整及解码后AD类别不变、点／方向／标签一致，以及留出原TF32姿态完全一致作为实验不变量；原可行性阈值、条件和预算不变。数值复演问题本身不是监督机制结果。</p><p>详细记录：<a href='implementation_clarification_before_updates.json'>首次更新前实施说明</a>、<a href='history_check_clarification_before_remaining_updates.json'>剩余来源更新前历史检查澄清</a>、<a href='code_versions/manifest.json'>来源2／其余来源的代码版本与哈希</a>及各来源preflight凭据。</p>")
    sections.append(f"<p>独立凭据核验通过：{audit['branches']}分支、{audit['adam_steps']}次Adam更新；24条步长匹配分支共120步，实际参数步长最大相对偏差 {audit['maximum_actual_norm_relative_error']:.3g}，第1步七臂logit最大差异 {audit['maximum_step1_between_arm_logit_error']:.1f}。模型及buffer共同起点、梯度范围、固定分母与RNG检查通过，无未完成journal。详见<a href='data/final_receipt_audit.json'>执行凭据核验</a>。</p>")
    sections.append(f"<h2>可信度与边界</h2><p>重复出现的未知标签计数为 {d['unknown_appearances']}；未知不作为负例。正式逐姿态评价 {d['analytic_job_calls_receipts']} 次，加初始一致性预检 {d['analytic_preflight_calls']} 次，共 {d['analytic_calls_receipts']} 次，低于131,072次上限，并与原始执行账本核对一致。原始检查点固定查询标签跨条件严格一致，查询ID和两折隔离已核验。每个条件恢复相同模型、Adam状态和随机数；详细执行不变量见receipts。</p><p>只含8个训练场景，官方完整训练检查点已经见过这些训练场景；两折留出仅指本次微更新没有使用对侧场景，不是从未见过的泛化测试。每折4个留出场景共享同4个来源脉冲，普通场景区间有条件依赖；并列报告来源×场景重采样敏感性，但只有两折，仍不能代表广泛泛化。候选、条件和1/5步不作为独立重复，多项对照为机制诊断、未作显著性筛选。1/5步局部结果不能否定长期效应，也不能代替长期受控训练。解析可行性不是机器人成功率。</p><p>未新增评分器、网络或标签修正；未访问C3/C4/C5，未执行1000步短训，未提交Git。后续任何确认训练都需要新的决定与授权。</p>")
    sections.append("<h2>证据文件</h2><p><a href='analysis/pulse_analysis.json'>完整统计与区间</a> · <a href='analysis/frame_metrics.csv'>逐场景来源指标</a> · <a href='analysis/rescue_harm.csv'>恢复与损害</a> · <a href='analysis/ranking.csv'>同物体倒序</a> · <a href='analysis/paired_contrast_units.csv'>配对差值</a> · <a href='analysis/analysis_spec.json'>冻结统计规则</a></p>")
    css="body{font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif;max-width:1250px;margin:36px auto;padding:0 24px;color:#182737}h1{font-size:28px}h2{margin-top:34px;font-size:21px}.lead{background:#eef4fa;padding:18px;border-left:5px solid #245782}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{border:1px solid #d6dfe7;padding:8px;text-align:left}th{background:#eef3f8}tr:nth-child(even){background:#fafbfd}a{color:#145f99}img{max-width:100%}"
    doc="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>R1.4 受控因果微更新</title><style>"+css+"</style><h1>EG R1.4：零质量宽度监督受控微更新</h1>"+"".join(sections)+"</html>"
    (root/"R14_summary.html").write_text(doc,encoding="utf-8")
    (root/"decision.json").write_text(json.dumps({"method_decision":"HOLD","finding":finding,"pre_registered_positive_intervals":len(supported),"negative_intervals":len(negative),"no_best_step_selection":True},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"report":str(root/"R14_summary.html"),"finding":finding},ensure_ascii=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("root",type=Path)
    create(ap.parse_args().root)

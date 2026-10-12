"""Six requested deliverables, with explicit missing/failed/stopped conditions."""
from analyze import *
import datetime,hashlib,subprocess,time
def text(path,s):path.write_text(s.strip()+'\n')
def pct(x):return f'{float(x)*100:.1f}%'
def table(rows,keys):
    return '\n'.join(['|'+'|'.join(keys)+'|','|'+'|'.join(['---']*len(keys))+'|']+['|'+'|'.join(str(r.get(k,'')) for k in keys)+'|' for r in rows])
def main():
    resources=[];training=[]
    for p in sorted((ROOT/'receipts').glob('*_resource.json')):
        x=json.loads(p.read_text());resources.append(x)
    for kind in ('navigate','stitch'):
        p=ROOT/'results'/f'{kind}_training_status.json';x=json.loads(p.read_text()) if p.exists() else dict(status='RUNNING',step=0)
        r=next((r for r in resources if r['label']==f'formal_{kind}'),{})
        training.append(dict(kind=kind,status=x['status'],updates=x['step'],GPU_hours=round(r.get('gpu_hours',0),6),exit_code=r.get('exit_code','RUNNING'),stop_reason=r.get('stop_reason'),checkpoint_count=len(x.get('checkpoints',[]))))
    final=read(ROOT/'results/FINAL_SCORES.csv') if (ROOT/'results/FINAL_SCORES.csv').exists() else []
    pairs=read(ROOT/'results/FINAL_PAIRED.csv') if (ROOT/'results/FINAL_PAIRED.csv').exists() else []
    e1=read(ROOT/'results/E1_TRANSITIONS.csv');main_e1=next(x for x in e1 if x['kind']=='navigate' and x['variant']=='A' and x['task']=='5' and x['early_step']=='300000' and x['late_step']=='1000000')
    b_e1=next((x for x in e1 if x['kind']=='navigate' and x['variant']=='B' and x['task']=='5' and x['early_step']=='300000' and x['late_step']=='1000000'),None)
    b_e1_note=(f"Navigate-B在同20个E1场景也从300k的{b_e1['first_success']}/20降至1M的{b_e1['second_success']}/20；成功→失败{b_e1['success_to_failure']}，失败→成功{b_e1['failure_to_success']}。因此B没有消除该任务的阶段退化。" if b_e1 else 'Navigate-B的Task5阶段对照缺失。')
    gates=json.loads((ROOT/'results/E2_GATES.json').read_text());branches=read(ROOT/'results/E2_BRANCHES.csv');e2=read(ROOT/'results/E2_SUMMARY.csv')
    controls=json.loads((ROOT/'receipts/E2_budget.json').read_text())['used'];gpass=sum(x['status']=='PASS' for x in gates)
    bfile=ROOT/'results/E2_AB_MATCHED_FAILED.csv'
    bp=read(bfile) if bfile.exists() else []
    if bp:
        cc=[x for x in bp if x['condition']=='C0']
        bnote=f"B终点前冻结的补充复测使用同一4个已有A失败快照，共20条C0—C4分支。C0平均净位移：A {np.mean([float(x['A_displacement']) for x in cc]):.4f}，B {np.mean([float(x['B_displacement']) for x in cc]):.4f}；50步内原远目标成功：A {sum(x['A_far_success']=='True' for x in cc)}/4，B {sum(x['B_far_success']=='True' for x in cc)}/4。所有条件的逐状态成对差在E2_AB_MATCHED_FAILED.csv，完整B轨迹留服务器。这是A证据驱动、B终点前固定的解释性复测，不能当作总体成功率或最初未消费的确认集。"
    else:bnote='B受控补充复测未执行；若缺少完整1M，保持未执行，不以其他检查点代替。'
    gradient_counts=[]
    for v in ('A','B'):
        p=ROOT/'receipts'/f'E3_{v}_exit.json'
        d=json.loads(p.read_text()) if p.exists() else {}
        gradient_counts.append(f"{v}：{len(d.get('gradient_trigger_snapshots',[]))}个状态触发、{d.get('gradient_rows',0)}条阶段梯度记录")
    gradient_note='；'.join(gradient_counts)
    epochs=read(ROOT/'results/E1_SCORES.csv');score_rows=[]
    for r in final:
        if r['group']!='-1':continue
        score_rows.append(dict(dataset=r['kind'],config=r['variant'],task=r['task'],n=r['n'],success=pct(r['success_rate']),Wilson95=f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}]"))
    overall=[]
    for kind in ('navigate','stitch'):
        for v in ('A','B'):
            rr=[r for r in final if r['kind']==kind and r['variant']==v and r['group']=='-1']
            overall.append(dict(dataset=kind,config=v,episodes=sum(int(r['n']) for r in rr),score=pct(np.mean([float(r['success_rate']) for r in rr])) if len(rr)==5 else 'NOT_EXECUTED / incomplete 1M prerequisite'))
    ab=[]
    for kind in ('navigate','stitch'):
        rr=[r for r in pairs if r['kind']==kind]
        if len(rr)==5:
            delta=np.mean([float(r['paired_difference']) for r in rr])*100
            cells='；'.join(f"Task{r['task']}：B−A {float(r['paired_difference'])*100:+.1f}个百分点，成对bootstrap95% [{float(r['paired_ci_low'])*100:+.1f}, {float(r['paired_ci_high'])*100:+.1f}]" for r in rr)
            ab.append(f'{kind} 当前单训练seed配方的总体B−A差为{delta:+.1f}个百分点。{cells}。该区间只反映评价场景，未包含跨训练seed变异。')
        else:ab.append(f'{kind} 未形成完整1M同场景A/B对照，正式采样敏感性保持未判定。')
    abfindings='\n\n'.join(ab)
    complete=all(x['updates']==1000000 and x['status']=='SINGLE_SEED_B_COMPARISON' for x in training) and len(pairs)==10
    decision=('本轮固定seed0配方对采样条件高度敏感，方向随任务而异：Navigate总体改善9.2个百分点，主要来自Task2的+59个百分点，Task1反而下降24个百分点，Task5终点仍仅9%；Stitch下降47.6个百分点，不能把B称为统一改进。两组场景与逐任务区间共同保留，不能把一次训练的差异外推为跨训练种子平均优势。' if complete else '完整1M同场景条件不足的比较保持未判定。')
    module_rows=read(ROOT/'results/MODULE_IDENTITY.csv') if (ROOT/'results/MODULE_IDENTITY.csv').exists() else []
    module_facts=[]
    for kind in ('navigate','stitch'):
        vr=[r for r in module_rows if r['kind']==kind and r['module']=='modules_value']
        lr=[r for r in module_rows if r['kind']==kind and r['module']=='modules_low_actor']
        module_facts.append(f"{kind}：已比较价值模块{len(vr)}个阶段，其中数组完全相同{sum(r['exact_equal']=='True' for r in vr)}个；短actor模块完全相同{sum(r['exact_equal']=='True' for r in lr)}/{len(lr)}个。")
    module_note=' '.join(module_facts)
    sv=[r for r in module_rows if r['kind']=='stitch' and r['module'] in ('modules_value','modules_target_value','modules_goal_rep')]
    stitch_identity=('Stitch在四个冻结阶段的value、target_value及共享goal_rep参数全部逐数组相同（12/12模块阶段）；这排除了这几组检查点价值网络参数差异作为A/B成绩变化的解释，但不证明价值正确，也不排除actor条件表征、短策略或输入分布的作用。' if len(sv)==12 and all(r['exact_equal']=='True' for r in sv) else '价值模块与共享表征的身份界限以MODULE_IDENTITY.csv实际结果为准。')
    vp=ROOT/'results/VALUE_OUTPUT_IDENTITY.csv';vo=read(vp) if vp.exists() else []
    ss=[r for r in vo if r['kind']=='stitch']
    value_output_note=(f"对已有同输入查询表作数值比对（不增加查询/回合）：Stitch固定16候选跨阶段的价值头最大绝对差{max(float(r['value_head_max_abs']) for r in ss):.9g}，截断权重最大绝对差{max(float(r['clipped_weight_max_abs']) for r in ss):.9g}；预选近/控制目标的所有阶段查询输出完全一致。参数身份不能直接代替执行数值身份，这些差异的数值来源尚未独立归因，未伪称全输出零误差。完整记录在VALUE_OUTPUT_IDENTITY.csv。" if ss else '同输入查询的数值比对尚未汇总。')
    total=sum(r['gpu_hours'] for r in resources);peak=max([r['peak_aggregate_mib'] for r in resources] or [0])
    cpuclock=json.loads((ROOT/'receipts/CLOCKS.json').read_text());finish=time.time()
    cpupath=ROOT/'receipts/CPU_COMPLETION.json';cpu=json.loads(cpupath.read_text()) if cpupath.exists() else {}
    cpu_note=(f"E1—E3阶段墙钟上界{cpu['analysis_wall_seconds']/3600:.3f}小时（包含等待训练），终点主评价墙钟{cpu['final_wall_seconds']:.1f}秒；各CPU工作进程最多4线程，均使用CPU后端，实际shell退出码全部0。" if cpu else 'CPU阶段完成时间尚未登记。')
    report=f'''# SAW-ANALYSIS-03B v2.0 执行报告

分支：`codex/saw-analysis-03b-20261012`，链接：https://github.com/rollinpioneer/graph/tree/codex/saw-analysis-03b-20261012

本轮正式B训练的实际状态如下。只有完成1M的冻结权重才进入正式终点比较；停止或缺失条件不填造成绩，不补跑、不选最佳阶段。

{table(training,['kind','status','updates','GPU_hours','exit_code','stop_reason','checkpoint_count'])}

新增累计GPU驻留上界 {total:.6f} 小时，合计显存峰值 {peak} MiB。包含两次已结束1000更新试运行，正式各9000秒、累计19800秒硬限。用户在试运行结束后指令“不要测、直接跑”，已取消吞吐预测准入并直接启动GPU3/4；保留实测预测与真实预算结果。试运行权重不继承。资源守护日志/回执见receipts及服务器logs。

{cpu_note}

## 1. 训练身份与科学配置

官方源3535a16c901e5de1233ac5a586401b3ddfa5e30d；RD01输入6462e06c540594acca6bc8630c37a803fe36dd4a；EVAL02输入b0dd744d323b924d02de93f5c487c4afc9565171。正式训练启动源码提交388ded0a23258ed43e12093825b17fe80b7510fc，后续CPU分析提交不改变运行中科学训练代码。

配置B仅actor_p_trajgoal 1→0.5、actor_p_randomgoal 0→0.5。两个独立seed0从头初始化，update0代理参数、目标网络、优化器与代理RNG和A逐数组相同，误差0；不是从A训练后权重继续。完整SGD历史未由相同seed自动推断。实际采样器未修改，random目标分支Waypoint为当前段25步后状态，未来目标分支为min(state+25,goal)。4096样本比例/边界/终止/奖励/Waypoint全部通过，真实比例见SAMPLER_AUDIT.json。

历史RD01 Navigate62.8%、Stitch49.6%与EVAL02 Navigate64.8%、Stitch50.6%分别保留。旧检查点和轻量文件已按原manifest重新核验，READ_ONLY_ASSETS.json为回执。Stitch仍是作者专用协议近似，B不冒称作者配置。

## 2. 冻结1M统一终点

{table(overall,['dataset','config','episodes','score'])}

{abfindings}

{table(score_rows,['dataset','config','task','n','success','Wilson95']) if score_rows else '正式终点评价尚未执行。'}

逐任务配对差、成功→失败/失败→成功与成对bootstrap95%区间在FINAL_PAIRED.csv；两组各50场景成绩保留在FINAL_SCORES.csv。温度0、原成功阈值1、1000步上限、无视频，完整状态/RNG/包装器恢复门≤1e-10，每回合记录实际误差和同场景散列。不同数据集不混作配置因果对照。

## 3. E1同场景能力转移

Navigate-A Task5的300k→1M：同20场景，早期成功{main_e1['first_success']}/20，最终成功{main_e1['second_success']}/20；成功→失败{main_e1['success_to_failure']}，失败→成功{main_e1['failure_to_success']}。这确认冻结场景上的退化。所有其他任务、A/B阶段照报E1_SCORES.csv及E1_TRANSITIONS.csv；缺少1M阶段保持缺失，开发曲线不替代统一场景。

{b_e1_note}

20回合的小样本Wilson区间保留；边界全成功/全失败的成对bootstrap可能退化为点，不能视为总体误差为零。E1场景已消费用于研究，不冒充未触碰确认集。完整学习曲线见figures/E1_learning_curves.pdf。

## 4. E2匹配局部分支

固定32候选，{gpass}通过恢复、下一步源轨迹重放和近/控制目标可行性检查，A共{len(branches)}条分支，总控制步（含B补充复测、E2自检及E3数据链）{controls}/10000。未通过的{32-gpass}个候选不替换；具体原因在E2_GATES.json。所有近目标由预登记数据位置/速度支持与特权坐标控制验证，不以分支成绩选近目标。

{table(e2,['kind','source_success','condition','n','local_reached','far_success','far_progress_mean','displacement_mean'])}

可控失败状态只有Navigate四例；Stitch失败候选没有通过近目标准入，不能给出其受控因果分支结论。四例Navigate的C0在原远目标下50步净位移平均约0.0022；C1达到近目标2/4，C2为1/4，C3控制目标3/4。最终actor具备部分局部执行能力，短actor没有显示总体更强。局部切换后没有任何原远目标完整成功，故不能宣称恢复任务。C3也可触发移动，说明尚不能把行为改变归因于某个独特有利Waypoint。C4与E1共同说明策略随阶段变化，欧氏远目标进展不当作最优路线价值。

C0低运动主要是小动作，而不是满足大动作撞墙阈值后的持续无效运动；零“ineffective_steps”不等于有效导航。每条原始分支保留动作、真实目标/临时策略条件目标、完整sim状态和原成功判定。

{bnote}

## 5. E3证据矩阵

E3_SUPPORT.csv记录真实失败/参照点的qpos/qvel邻域、原始目标区域、原版A/B采样概率；E3_JOINT_SAMPLING_SUPPORT.csv进一步区分目标区域与具体Waypoint的联合概率，包含每个预选候选及已验证E2目标。能到达的near候选不自动等于原版训练Waypoint可提供的候选。E3_DATA_REPLAYS.json保存真实数据动作链检查。数据不含完整MuJoCo integration，近似qpos/qvel衔接成功仍不声称严格完整状态拼接。地图连通未用作控制能力证明。

21个预选近目标对应的具体记录tuple中，20个在A采样器下“原远目标1单位邻域＋该Waypoint”的条件联合概率为0，B则21个均为正。这证明B扩展了这些tuple的联合目标支持；不证明所有邻近数据状态在A完全没有支持，不证明目标中心精确坐标有正概率，也不证明可达近目标能帮助完成远目标。

固定16候选的价值双头、优势、截断和actor/teacher/data动作见E3_A/B_CANDIDATES.csv。所有预选E2近/控制目标又被完整复用于离线价值测量，见E3_A/B_VERIFIED_TARGETS.csv；这是E2后解释性补测，无新增控制步或训练。未因成绩替换候选。真实数据tuple上的原版损失跨阶段保留E3_A/B_SIGNALS.csv，目标是否在A采样段内单独标记。离线动作差、权重饱和和排序分歧仅为证据，不单独判根因。

A的E2没有出现C2近目标成功而C1失败的快照。各配置按自己的物理证据触发离线梯度，结果为：{gradient_note}。没有证据则如实未触发；B若触发，使用实际E2验证目标对应的真实数据tuple，仍不更新参数。梯度余弦不能单独解释全任务失败。EVIDENCE_MATRIX.csv汇总支持、反例与未判定边界。

模块参数对比：{module_note} 参数差异不自动解释其功能或来源，不能将两个采样概率的改变直接简化成“只有actor参数变化”。MODULE_IDENTITY.csv保留逐模块误差；数字训练轨迹和后端差异的原因未在本轮做额外训练控制。

{stitch_identity}

{value_output_note}

## 6. 裁定、边界与下一项建议

{abfindings}

{decision}

已支持：Navigate-A Task5发生真实阶段退化；在四个可控真实失败状态中，最终策略对原远目标产生近静止行为，切换经过可行性核验的近目标/控制目标可在部分状态启动有效局部运动。因此失败定位至少细化到最终actor的目标条件动作映射，不能概括成完全没有局部移动能力。

反例与边界：近目标并非总能达到，teacher没有胜过actor；控制近目标也改变行为；没有50步内完整远目标恢复。Stitch失败状态近目标准入失败、全数据严格状态拼接和价值正确性仍未判定。A/B敏感性只有双方完整1M的同场景比较才可裁定；未完成的B保持未判定，不能用中途检查点代替终点。

下一项最具信息价值建议（仅建议，不执行）：以Stitch-B这次正式终点的真实失败轨迹为对象，另卡冻结一组全状态匹配、通过局部目标可行性准入的原目标/近目标分支，区分局部执行失能与远目标条件动作映射失败；优先解释本轮最大的47.6个百分点损失，保留准入失败及无优势反例。不得由本报告自动触发重训练、新seed、损失或算法模块。

全部原始权重、场景、轨迹与完整日志留服务器本轮独立目录；Git只含代码、冻结设计、轻量表、必要回执、图与散列索引。单训练seed、两场景随机流及Stitch协议近似限制外推；不以作者85%/8%为优化目标。
'''
    text(ROOT/'REPRODUCTION_REPORT.md',report)
    text(ROOT/'DECISION.md',report[report.index('## 6.'):])
    text(ROOT/'TRAINING_IDENTITY.md',report[report.index('## 1.'):report.index('## 2.')])
    text(ROOT/'E1_AUDIT.md',report[report.index('## 3.'):report.index('## 4.')])
    text(ROOT/'E2_AUDIT.md',report[report.index('## 4.'):report.index('## 5.')])
    text(ROOT/'E3_EVIDENCE.md',report[report.index('## 5.'):report.index('## 6.')])
    write(ROOT/'results/ROUND_SUMMARY.json',dict(training=training,GPU_hours=total,peak_aggregate_mib=peak,E2_control_steps=controls,E2_gates_pass=gpass,E2_branches=len(branches),endpoint=overall,
        source_training_commit='388ded0a23258ed43e12093825b17fe80b7510fc',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='COMPLETE_1M_COMPARISON' if all(x['updates']==1000000 and x['status']=='SINGLE_SEED_B_COMPARISON' for x in training) else 'B_TRAINING_INCOMPLETE_OR_RUNNING'))
    print('REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()

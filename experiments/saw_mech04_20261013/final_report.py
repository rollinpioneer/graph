"""Evidence-bounded final documents; execution status remains explicit."""
from summarize import *
def main():
    cs=json.loads((P/'results/C_SUMMARY.json').read_text());ts=json.loads((P/'results/T_SUMMARY.json').read_text());ra=json.loads((P/'results/R_SUMMARY.json').read_text())
    resources=[json.loads(p.read_text()) for p in (P/'receipts').glob('*_resource.json')];budget={g:sum(r['residency_seconds'] for r in resources if r['group']==g)/3600 for g in ['C-S','C-N','R','C-EVAL','T','T-EVAL']}
    failures=[dict(label=r['label'],exit_code=r['exit_code'],stop_reason=r['stop_reason']) for r in resources if r['exit_code']!=0 or r['stop_reason']]
    write(P/'results/RESOURCE_SUMMARY.json',dict(residency_hours=budget,total_GPU_hours=sum(budget[g] for g in ['C-S','C-N','T']),peak_aggregate_own_GPU_mib=max((r['peak_aggregate_mib'] for r in resources),default=0),failures_or_stops=failures,unit='sum of each own job wall residency, including compile/development evaluation; not elapsed parallel wall time'))
    statuses=[]
    for kind,v,seed in [(k,v,s) for k,vs in [('stitch',['A','B']),('navigate',['A'])] for v in vs for s in [1,2]]:
        p=P/'results'/f'C_{kind}_{v}_s{seed}_training_status.json';s=json.loads(p.read_text()) if p.exists() else dict(status='NOT_EXECUTED',step=0);statuses.append(dict(kind=kind,variant=v,seed=seed,status=s['status'],actual_updates=s['step']))
    table(P/'results/TRAINING_EXECUTION_STATUS.csv',statuses)
    lines=['# SAW-MECH-04 v1.1 实验报告','', '用户“建分支执行”授权 R 与 C-S/C-N。T 依卡中单独执行门办理；C-NB、C-M 未执行。原卡及冻结附表保留，先固定条件再读取正式成绩，不以成绩决定种子、检查点或运行长度。','', '## 六条完整训练与同场景确认','', '| 数据集 | 配置 | seed | 状态 | 实际更新 |','|---|---|---:|---|---:|']
    lines += [f'|{x["kind"]}|{x["variant"]}|{x["seed"]}|{x["status"]}|{x["actual_updates"]}|' for x in statuses]
    endpoints=read(P/'results/C_TASK_SCORES.csv')
    lines += ['', '| 1M模型 | seed | 五任务平均 | Task1 | Task2 | Task3 | Task4 | Task5 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=sorted({(x['kind'],x['variant'],x['training_seed']) for x in endpoints if x['scope']=='full' and x['task']=='0' and x['valid_full_endpoint']=='True'})
    for kind,v,seed in keys:
        values=[]
        for task in range(6):
            row=next(x for x in endpoints if x['kind']==kind and x['variant']==v and x['training_seed']==seed and x['task']==str(task) and x['scope']=='full');values.append(f"{float(row['success_rate'])*100:.1f}%")
        lines.append(f'|{kind}-{v}|{seed}|'+'|'.join(values)+'|')
    lines += ['', '正式1M只在完整计数、退出码0、参数有限、模型散列吻合后计算。缺失完整终点记未执行，不能作为低分反例；全部阶段与失败保留。', '', '| Stitch训练seed | A新场景 | B新场景 | A−B（百分点） |','|---:|---:|---:|---:|']
    lines += [f'|{x["training_seed"]}|{x["A"]:.1f}%|{x["B"]:.1f}%|{x["delta_pp"]:+.1f}|' for x in cs['C_S']['all_seed_scores']]
    lines += ['',f'C-S预登记裁定：**{cs["C_S"]["status"]}**。新seed差的均值/样本标准差/范围：{cs["C_S"]["delta_mean_pp"]} / {cs["C_S"]["delta_sample_sd_pp"]} / {cs["C_S"]["delta_range_pp"]}（百分点）。', '', '| Navigate-A训练seed | Task5 300k | Task5 1M | 下降（百分点） | 满足门 |','|---:|---:|---:|---:|---|']
    lines += [f'|{x["training_seed"]}|{x["Task5_300k"]:.1f}%|{x["Task5_1M"]:.1f}%|{x["drop_pp"]:+.1f}|{x["criterion"]}|' for x in cs['C_N']['confirmation']]
    lines += ['', '100k/500k仅为预登记的次要时序描述，不改主要比较：']
    lines += [f'- seed{x["training_seed"]}：100k={x.get("Task5_100k_descriptive")}%，300k={x["Task5_300k"]}%，500k={x.get("Task5_500k_descriptive")}%，1M={x["Task5_1M"]}%；{x.get("stage_description","")}。' for x in cs['C_N']['confirmation']]
    lines += ['', '若100k已成功而300k已失败，应描述为能力在主窗口之前已丢失，不能称从未学会；若300k→1M维持或改善，也不能用另一个早期阶段替换预登记主结论。']
    lines += ['',f'C-N预登记裁定：**{cs["C_N"]["status"]}**。判据固定300k成功≥30%并在1M下降≥20百分点；不事后替换为最好的早期阶段。', '', '全部任务、两流、Wilson区间、同场景成对bootstrap及成功→失败/反向转移见 C_TASK_SCORES.csv、C_STREAM_TASK_SCORES.csv / C_PAIRED_CONTRASTS.csv。跨训练seed的均值、样本标准差、范围及每个原始值见 C_TRAINING_SEED_SUMMARY.csv 与 C_N_DROP_SEED_SUMMARY.csv；新seed1/2确认与含seed0的观察性汇总分开。bootstrap按任务/流分层；它只描述固定权重的场景误差。真正独立训练确认单位仅seed1/2，seed0参与发现与设计，不用500回合冒称500独立模型。旧03B同场景 Navigate A/B64.0%/73.2%、Stitch51.2%/3.6%保持原身份；本轮新场景seed0分数另列，不覆盖旧值。','', '## R 行为与监督追踪','', '完整详情见 R_AUDIT.md 及全部索引、负例、tuple/权重、梯度和物理可行性表。Navigate新保留10场景A300k9/10→A1M0/10。所有83锚点完整状态重放通过；Navigate近邻支持49/54、Stitch29/29。真实远目标不换，局部目标成功不等于原任务成功。','', 'R1在seed0同状态晚期行为损失上有条件支持；R2的损失来源根因仍未判定。价值/权重饱和、梯度余弦或动作差只能作解释线索，不能单独证明因果。','', '## T 有界干预','', f'T执行状态：**{ts["status"]}**。准备使用同Stitch-A500k起点、继承actor Adam计数500000，冻结参照模块与完整参照优化器；学生专用优化器独立保存。CPU准入只算梯度、0更新。候选流1024slot逐字节锁定，mask与两项损失分母相同；T0全未来，T1半未来/1024，T2半未来/512，T3半未来+半随机/1024。','']
    if ts['contrasts']:
        lines += ['| 终点对比 | Task | 差（百分点） | 成对95%区间 | 两流差 | 预定短效应门 |','|---|---:|---:|---|---|---|']
        lines += [f'|T{x["first_arm"]}−T{x["second_arm"]}|{x["task"]}|{x["difference_pp"]:+.1f}|[{x["ci95_low_pp"]:.1f},{x["ci95_high_pp"]:.1f}]|{x["group0_difference_pp"]:+.1f}/{x["group1_difference_pp"]:+.1f}|{x["meaningful_short_effect"]}|' for x in ts['contrasts'] if x['task'] in (1,3)]
    lines += ['', '即使T完成，仍只适用于共同seed0起点和冻结价值/教师下的100k干预，不等于完整联合A/B因果，也不触发重训练或新算法。25k/50k用于描述，不挑最优点。若未获单独批准，T按未执行归档，H1/H2未知。','', '## 身份、资源与复现边界','', f'累计驻留小时：{budget}；GPU合计{sum(budget[g] for g in ["C-S","C-N","T"]):.6f}，本轮自有GPU进程合计显存峰值{max((r["peak_aggregate_mib"] for r in resources),default=0)}MiB。预算C-S11、C-N5.5、T2.5分开计账；R和C评价各CPU4小时，T评价CPU独立列账。实际结束与源资产核验见 FINAL_AUDIT.json。','', '原官方代码3535a16c901e5de1233ac5a586401b3ddfa5e30d与数据只读；正式训练启动d77ca7d4c68d809058f273f6fdd4e23ea5e92703，TRAINING_SOURCE.json中四文件字节在结束重核。原配置回执sampler字符串残留“NumPy seed0”，实际源码init_agent(...,seed)使用1/2；以training_seed、update0完整RNG与源码为准，这是记录文字问题，不改运行权重/采样。','', 'Stitch依然是原作者协议近似，不能声称恢复原始专用训练协议或解释论文8%；本轮无新算法、不据结果择种子。原前置REASSESSMENT本地未发现，冻结事实已用03B原始材料核验。大型权重、轨迹、完整日志与索引流留服务器，RAW_ARTIFACT_INDEX.json登记路径/SHA；轻量代码、配置、全部分母/成绩、资源与负例归档当前分支，不合并、不强推。']
    (P/'REPRODUCTION_REPORT.md').write_text('\n'.join(lines)+'\n')
    meaningful=[x for x in ts['contrasts'] if x['first_arm']==3 and x['second_arm']==1 and x['task'] in (1,3) and x['meaningful_short_effect'] and x['difference_pp']<0]
    h2='SUPPORTED_CONDITIONALLY' if meaningful else 'UNKNOWN'
    verdict=['# 预登记假设裁定','', '| 假设 | 裁定 | 证据与边界 |','|---|---|---|', '|R1|SUPPORTED_CONDITIONALLY|seed0同状态动作分叉、50步行为损失及新10场景9→0；C跨seed阶段现象另表，不等于根因|', '|R2|UNKNOWN|native tuple、权重与梯度已测，相关性不足以指定AWR/Waypoint或权重的因果作用|', '|H1|UNKNOWN|未来曝光/剂量的纯因果解释不可由完整A/B单独判定；T全部对照如实列示|', f'|H2|{h2}|T3−T1主任务的预登记终点、两流、15pp及区间门；限一个共同起点，负向证据须结合R局部链，不能外推所有随机监督|', '|H3|UNKNOWN|局部动作/链与监督关联不等于特定失效根因；不同时扩展多个算法模块|','', f'C-S：{cs["C_S"]["status"]}；C-N：{cs["C_N"]["status"]}。C验证基础现象，不能代替H1/H2机制干预。全部反例见原表，未完成条件不当低分反例。']
    (P/'HYPOTHESIS_VERDICT.md').write_text('\n'.join(verdict)+'\n')
    stitch_stable=cs['C_S']['status']=='LARGE_DIFFERENCE_REPLICATED_TWO_SEEDS';decline_stable=cs['C_N']['status']=='DECLINE_REPLICATED_TWO_SEEDS'
    if cs['C_S']['status']=='PENDING' or cs['C_N']['status']=='PENDING':
        nexttext='正式权重或配对评价条件不完整，跨seed稳定性为UNKNOWN；优先问题仅保留技术可比性的恢复，不能将缺失条件当作低分或种子敏感的反例。'
        minimal='先审查明确的技术失败与已有检查点，恢复/重跑必须独立批准并保持科学条件。不追加算法或挑选新的更好seed。'
    elif stitch_stable:
        nexttext='优先解释仅保留：冻结参照下新增随机目标监督的净作用（仅在T达门时有条件支持）。' if meaningful else 'Stitch配置差异按两个新seed得到较强确认；目前没有单一损失根因获得因果确认，优先问题仅保留同未来监督条件下随机目标信号的净作用。'
        minimal='T未获批准时只待原四臂执行门，不追加算法或训练种子。T完成仍未形成明确效应则按UNKNOWN停止，不加训至出现差异。若形成预定短程负效应，最小后续是另卡核验同一干预在一个独立、已保存500k起点是否成立；须独立批准，不扩大为1M新方法。'
    elif decline_stable:
        nexttext='Stitch的大幅采样差异尚不稳定，优先问题仅保留已学动作映射的训练阶段保持。不能用Stitch特定seed失败作为通用采样机制的基础。'
        minimal='下一步最多提出一个有界因果对照：从已经成功的固定早期Navigate起点，在同场景下比较冻结价值/教师的actor续训与未续训参照。另立卡和预算，不在本轮自动运行；不同时增加多个损失模块。'
    else:
        nexttext='两个基础现象未同时形成预登记的有限跨seed稳定性，优先问题仅保留训练seed对任务能力及学会时点的影响，不能把seed0现象写成普遍遗忘或随机监督有害。'
        minimal='本轮按现有证据停止；最小下一步只规划对已保存阶段、逐任务同场景转移的解释复核，不追加训练直到出现期待差异。暂不据不稳定现象开发新算法；另有训练请求须独立批准。'
    (P/'NEXT_DECISION.md').write_text('# 下一步科研决策\n\n'+nexttext+'\n\n'+minimal+'\n\n按原卡§5.6矩阵选择一项优先问题；C全部种子方向与阶段门约束解释。当前不自动执行C-NB或C-M，不把弱B作为唯一算法基线。\n')
    print('FINAL_REPORT_WRITTEN')
if __name__=='__main__':main()

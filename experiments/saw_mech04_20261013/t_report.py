"""Interpret the frozen four-arm endpoints without adding decision criteria."""
from summarize import *
def main():
    summary=json.loads((P/"results/T_SUMMARY.json").read_text());scores=read(P/"results/T_TASK_SCORES.csv");contrasts=summary["contrasts"]
    names={-1:"共同起点：原A500k",0:"T0 F100",1:"T1 F50-mask /1024",2:"T2 F50-renorm /512",3:"T3 F50+R50 /1024"}
    lines=["# T 四臂终点与竞争解释","",f"执行状态：{summary['status']}。四臂共同起点为Stitch-A seed0 500k；只更新最终actor和专用目标编码器，全部四臂按100k固定终点判读。","", "| 模型 | 五任务平均 | Task1 | Task2 | Task3 | Task4 | Task5 |","|---|---:|---:|---:|---:|---:|---:|"]
    for arm in (-1,0,1,2,3):
        z=[r for r in scores if int(r["arm"])==arm and int(r["student_step"])==(0 if arm==-1 else 100000)]
        if len(z)!=6:continue
        lines.append("|"+names[arm]+"|"+"|".join(f"{float(next(r for r in z if int(r['task'])==t)['success_rate'])*100:.1f}%" for t in range(6))+"|")
    lines += ["","| 固定100k对比 | 主任务 | 差（百分点） | 场景配对95%区间 | 两流差 | 达15pp/两流/区间门 |","|---|---:|---:|---|---|---|"]
    decision=[]
    for first,second,label,scope in [(1,0,"减少未来总贡献","曝光数量与梯度尺度的组合效应"),(2,0,"恢复平均尺度后减少样本","独立样本数与噪声改变"),(2,1,"同未来样本恢复尺度","在相同512未来slot下的归一化尺度效应"),(3,1,"新增随机目标项","相同未来slot下随机项的短程净效应"),(3,0,"短程A/B类比","冻结参照类比，非完整联合训练")]:
        for task in (1,3):
            z=[x for x in contrasts if x["first_arm"]==first and x["second_arm"]==second and x["task"]==task]
            if not z:continue
            x=z[0];valid=summary["status"]=="FOUR_FINALS_COMPLETE";effect="MEANINGFUL_NEGATIVE_SHORT_EFFECT" if valid and x["meaningful_short_effect"] and x["difference_pp"]<0 else "MEANINGFUL_POSITIVE_SHORT_EFFECT" if valid and x["meaningful_short_effect"] and x["difference_pp"]>0 else "BELOW_PRESET_EFFECT_GATE_OR_INCOMPLETE"
            decision.append(dict(first_arm=first,second_arm=second,task=task,question=label,interpretation_scope=scope,classification=effect,**{k:x[k] for k in ("difference_pp","ci95_low_pp","ci95_high_pp","group0_difference_pp","group1_difference_pp","meaningful_short_effect")}))
            lines.append(f"|T{first}−T{second}|{task}|{x['difference_pp']:+.1f}|[{x['ci95_low_pp']:.1f}, {x['ci95_high_pp']:.1f}]|{x['group0_difference_pp']:+.1f} / {x['group1_difference_pp']:+.1f}|{x['meaningful_short_effect']}|")
    table(P/"results/T_INTERPRETATION.csv",decision)
    lines += ["","## 按预登记竞争解释判读",""]
    harm=[x for x in decision if x["first_arm"]==3 and x["second_arm"]==1 and x["classification"]=="MEANINGFUL_NEGATIVE_SHORT_EFFECT"]
    gain=[x for x in decision if x["first_arm"]==3 and x["second_arm"]==1 and x["classification"]=="MEANINGFUL_POSITIVE_SHORT_EFFECT"]
    reduced=[x for x in decision if x["first_arm"]==1 and x["second_arm"]==0 and x["classification"]=="MEANINGFUL_NEGATIVE_SHORT_EFFECT"]
    lines.append("T3−T1在主任务"+",".join(str(x["task"]) for x in harm)+"形成预设负效应门：限共同起点、冻结价值/教师和100k更新，支持新增随机目标项的短程净害；仍不指定AWR/Waypoint或数据动作错误这一根因。" if harm else "T3−T1没有在主任务形成预设负效应门。本轮不能据此确认H2，也不能因未达门而宣称两臂等价或推翻原完整1M A/B差异。")
    if gain:lines += ["","T3−T1在主任务"+",".join(str(x["task"]) for x in gain)+"形成预设正效应门；这属于该任务/冻结起点下的补偿或收益证据，应同时保留，不能称随机目标训练一律有害。"]
    lines += ["","T1−T0在主任务"+",".join(str(x["task"]) for x in reduced)+"显示减少未来总贡献的短程负效应；这是数量/尺度组合，T2相关对照另列，不等于纯曝光数量的唯一因果证明。" if reduced else "T1−T0未在主任务形成预设负效应门；当前不支持用这一短程起点证明未来监督减半足以造成原完整训练失败。"]
    lines += ["","T2−T0与T2−T1用于分开描述样本数量/噪声和归一化尺度影响；Adam继承的历史与梯度尺度并非可直接等同的更新剂量。各任务原始值、Wilson区间、成对成功/失败转移和两条流结果完整保留。", "", "主要对比为Task1/3的T3−T1，15百分点、两流同方向且成对bootstrap95%排除0；其余已登记对照完整报告。区间为未作多项调整的描述性区间，不能用多个尝试中一项显著当作跨seed普遍机制。25k/50k曲线仅作次要时序，不挑选最优检查点。", "", "只有一个共同训练seed0起点，价值/教师完全冻结；结论不覆盖原1M联合更新或跨训练种子。本轮不自动追加训练、改变损失或启动C-NB/C-M。"]
    base4=next((float(x["success_rate"])*100 for x in scores if x["arm"]=="-1" and x["task"]=="4"),None)
    end4={a:next((float(x["success_rate"])*100 for x in scores if x["arm"]==str(a) and x["task"]=="4" and x["student_step"]=="100000"),None) for a in range(4)}
    if base4 is not None and all(x is not None for x in end4.values()):
        lines += ["",f"辅助Task4共同起点{base4:.1f}%，T0/T1/T2/T3终点分别为"+"/".join(f"{end4[a]:.1f}%" for a in range(4))+"。四臂均发生较大下降，属于共同设定下的辅助任务现象，不能专门归因T3随机目标项，也不能替换Task1/3的主判据。", "", "Task4各臂与起点的同场景转移及区间见T_PAIRED_CONTRASTS.csv；冻结参照、只更新actor和完整联合训练之间仍有复现边界。中间阶段使用每任务20场景，终点100场景，曲线仅描述已登记时序，不以最差或最好中间阶段作主要结论。"]
    (P/"T_INTERPRETATION.md").write_text("\n".join(lines)+"\n")
    write(P/"results/T_INTERPRETATION.json",dict(status=summary["status"],H2_short_harm_main_tasks=[x["task"] for x in harm],random_addition_short_gain_main_tasks=[x["task"] for x in gain],reduced_future_composite_harm_main_tasks=[x["task"] for x in reduced],rule="prespecified >=15pp, two-stream same direction, paired95 excludes0; same seed0 source only",decision=decision))
    print("T_INTERPRETATION_WRITTEN")
if __name__=="__main__":main()

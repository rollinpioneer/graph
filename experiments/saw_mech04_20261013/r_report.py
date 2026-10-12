"""Read-only R evidence summary; all selected and negative cases retained."""
import csv,json,pathlib,numpy as np
P=pathlib.Path(__file__).resolve().parent
def read(n):return list(csv.DictReader(open(P/'results'/n)))
def main():
    summaries=[]
    for kind in ['navigate','stitch']:
        div=read(f'R_EARLIEST_DIVERGENCE_{kind}.csv');support=read(f'R_SUPPORT_{kind}.csv');branch=read(f'R_BRANCHES_{kind}.csv');phy=read(f'R_TUPLE_PHYSICAL_{kind}.csv');grad=read(f'R_GRADIENTS_{kind}.csv')
        matched=[]
        for x in branch:
            if x['variant']!='A' or int(x['checkpoint_step'])!=(300000 if kind=='navigate' else 1000000):continue
            if kind=='navigate' and x['task']!='5':continue
            v='A' if kind=='navigate' else 'B'
            y=next(y for y in branch if y['case_id']==x['case_id'] and y['anchor']==x['anchor'] and y['variant']==v and int(y['checkpoint_step'])==1000000)
            matched.append(float(x['far_euclidean_progress'])-float(y['far_euclidean_progress']))
        probes=[]
        for typ in ['F','R']:
            for policy in ['data_chain','privileged','teacher','actor']:
                for variant in (['A','B'] if policy in ['teacher','actor'] else ['']):
                    rows=[x for x in phy if x['goal_type']==typ and x['policy']==policy and x['variant']==variant and float(x['target_initial_distance'])>1.05 and x['measurement_status']=='MEASURED']
                    probes.append(dict(goal_type=typ,policy=policy,variant=variant,eligible_nontrivial=len(rows),local_target_reached=sum(x['target_reached']=='True' for x in rows)))
        summaries.append(dict(kind=kind,cases=len(div),divergence_triggered=sum(x['triggered']=='True' for x in div),anchors=len(support),replay_pass=sum(x['replay_status']=='PASS' for x in support),data_support=sum(x['status']=='PASS' for x in support),matched_branches=len(matched),median_early_or_A_minus_late_or_B_far_progress=float(np.median(matched)),branch_positive_delta_count=sum(x>0 for x in matched),physical_probes=probes,gradient_rows=len(grad),negative_F_R_gradient_cosines=sum(float(x['F_R_total_cosine'])<0 for x in grad),zero_step_probe_rows=sum(x['measurement_status']=='ZERO_STEPS' for x in phy)))
    confirm=[read(f'Reval_navigate_A_s0_{s}_Rconfirm_episodes.csv') for s in [300000,1000000]]
    c=dict(scenes=len(confirm[0]),early_success=sum(int(x['success']) for x in confirm[0]),late_success=sum(int(x['success']) for x in confirm[1]))
    report=dict(R=summaries,new_scene_confirmation=c,causal_root='UNKNOWN',training_updates=0)
    (P/'results/R_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# R 逐状态行为与监督审计','',f'新保留 Navigate Task5 的 10 个完整场景：A300k {c["early_success"]}/10，A1M {c["late_success"]}/10。仍属 seed0 模型的场景确认，不是独立训练种子。','']
    for x in summaries:
        lines += [f'## {x["kind"]}', '',f'入选 {x["cases"]} 条轨迹，全部 {x["divergence_triggered"]} 条触发预登记动作差门；{x["replay_pass"]}/{x["anchors"]} 锚点通过完整状态、RNG、包装器及源动作回放门。数据近邻支持 {x["data_support"]}/{x["anchors"]}；缺失支持保留，不放宽容差。', '',f'同初态 50 步分支 {x["matched_branches"]} 对，早期/A 减晚期/B 的远目标欧氏推进中位差为 {x["median_early_or_A_minus_late_or_B_far_progress"]:.3f}，正差 {x["branch_positive_delta_count"]}/{x["matched_branches"]}。该距离不等于障碍绕行的最优进度，重复锚点不是独立样本。','',f'固定主锚点的原损失梯度记录 {x["gradient_rows"]} 行，F/R 总梯度余弦负值 {x["negative_F_R_gradient_cosines"]} 行；只描述局部方向，不证明造成退化。', '', '| 目标 | 控制器 | 模型 | 非平凡局部目标成功/分母 |','|---|---|---|---:|']
        lines += [f'|{p["goal_type"]}|{p["policy"]}|{p["variant"]}|{p["local_target_reached"]}/{p["eligible_nontrivial"]}|' for p in x['physical_probes']]
        lines += ['',f'零步探针 {x["zero_step_probe_rows"]} 行按未测量保留；其默认 far_success=False 不作为失败统计。局部可行性要求初始目标距离>1.05，成功阈值仍为1；原环境远目标从未更换。原始数据链只做位置≤0.15/速度≤0.5的近邻连接，不伪称完整数据仿真初态完全一致。','']
    lines += ['## 裁定与 T 准入','', 'R1：SUPPORTED_CONDITIONALLY。在 seed0 已成功源轨迹与新增小确认集，后期同状态动作改变并造成短程行为损失；C 新训练种子尚待确认。R2/H3：UNKNOWN。已测真实 F/R tuple、AWR/Waypoint 权重及梯度，相关性不足以指定一个损失根因。H1/H2：UNKNOWN，须等待四臂有界干预。','', 'T 科学准入：Stitch 动作分叉有完整状态重放和非平凡局部可行数据链；共同 A500k 旧开发评价 Task1=100%、Task3=94%（仅说明已学能力，不能充当 T 新场景成绩）。value、target、共享表征及短 actor 均保持只读，拟只更新最终 actor 与独立目标编码器，继承其 Adam 状态。T CPU 梯度代数准入结果另见 T_CPU_ADMISSION.json；它不执行参数更新。','', '独立执行批准仍待用户决定。不得由 R 证据或 C 训练的独立批准自动启动 T。']
    (P/'R_AUDIT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()

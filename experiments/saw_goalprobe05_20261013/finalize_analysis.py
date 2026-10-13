"""Finalize reporting after JSON-null export failure; never recompute or overwrite estimates/figures."""
from core import *
import pandas as pd

def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list):return [safe(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def main():
    guard();assert not (ROOT/'receipts/ANALYSIS_EXIT.json').exists()
    before={str(p.relative_to(ROOT)):sha(p) for p in list(ROOT.glob('*.csv'))+list((ROOT/'results').glob('*.csv'))+list((ROOT/'figures').glob('*.png'))+list((ROOT/'results').glob('*.json'))}
    r=pd.read_csv(ROOT/'BRANCH_ROLLOUTS.csv');s=pd.read_csv(ROOT/'GOAL_SENSITIVITY.csv');contrasts=list(csv.DictReader(open(ROOT/'PAIRED_CONTRASTS.csv')));verdict=json.loads((ROOT/'results/VERDICT.json').read_text());gates=verdict['gates']
    for row in contrasts:
        for k in ('seed','clusters','states','goal_tests_per_model','training_seeds'):row[k]=int(row[k])
        for k in ('A','B','A_minus_B','CI95_low','CI95_high'):row[k]=float(row[k])
    choices=[];candidates=r[r.group=='P1'].sort_values(['seed','state_id','goal_index','variant'])
    for category in ('correct','wrong_channel','not_left_region','no_motion'):
        matches=candidates[candidates.category==category]
        if len(matches):choices.append(safe(matches.iloc[0].to_dict()))
    write(ROOT/'results/REPRESENTATIVE_SELECTION.json',dict(rule='first lexicographic (seed,state_id,goal_index,variant) per P1 outcome class',selected=choices,repair='CSV blank fields represented as JSON null, not NaN; figures and scientific estimates unchanged'))
    roles=[]
    for seed in (0,1,2):
        for gi in (0,1):
            for task in (1,3):
                for v in ('A','B'):
                    sub=r[(r.seed==seed)&(r.group=='P3')&(r.goal_index==gi)&(r.variant==v)&r.state_id.str.startswith(f'P3_T{task}_')]
                    roles.append(dict(seed=seed,task=task,goal_index=gi,role='original_official_goal' if gi==0 else 'diagnostic_goal_intervention',variant=v,tests=len(sub),correct=int(sub.correct.sum()),CBR=float(sub.correct.mean()),wrong_channel=int((sub.category=='wrong_channel').sum()),not_left_region=int((sub.category=='not_left_region').sum()),no_motion=int((sub.category=='no_motion').sum()),mean_action_norm=float(sub.mean_action_norm.mean()),mean_displacement=float(sub.displacement.mean()),scope='auxiliary transparent source/goal-role decomposition; not an added H0 criterion'))
    rows(ROOT/'results/P3_GOAL_ROLE.csv',roles)
    offline=s[s.offline_alignment_status!='UNMEASURABLE'];lines=['# SAW-GOALPROBE-05 机制裁定','', '**H0 未获支持；本轮可行局部分岔中的强形式预测受到削弱。**','', f'机器可读裁定：{verdict["verdict"]}。24 个 P1 状态来自 6 个路口，每路口 4 个；24 个 P2 和 16 个 P3 均通过独立物理准入。总计 768 次、每次最多 25 步。局部通道成功不代表完成远目标任务。','', '| 训练 seed | P1 A | P1 B | A−B 百分点［路口聚类95%区间］ | P2 A−B 百分点 | 判据1/2/3 |','|---|---:|---:|---:|---:|---|']
    for g in gates:
        z=next(x for x in contrasts if x['seed']==g['seed'] and x['group']=='P1' and x['metric']=='correct')
        lines.append(f'| {g["seed"]} | {100*z["A"]:.2f}% | {100*z["B"]:.2f}% | {100*z["A_minus_B"]:.2f}［{100*z["CI95_low"]:.2f}, {100*z["CI95_high"]:.2f}］ | {g["P2_CBR_delta_pp"]:.2f} | {g["gate1"]}/{g["gate2"]}/{g["gate3"]} |')
    lines.extend(['', 'seed 1、2 的 P1 通道差均未达到原 15 百分点门槛。B 的平均目标对动作 L2 均略小于 A，并在少量 A 成功/B 失败样本上对应未离开区域，满足补充操作化的判据2；P2 两者均100%，满足判据3。但三个支持判据必须同时满足，不能以动作差异或少量反例代替主通道门槛。', '', 'P1 的 B 已分别正确进入 46/48、46/48、45/48 个目标对应通道。P1 的 B 错通道均为0；未成功主要是没有在25步内离开初始区域，其中 seed0 有1次最大位移不超过0.1的无运动。A在seed1/2各有1次错通道。因此，B普遍忽略目标、普遍动作幅度塌缩、或在这些近处分岔普遍选错方向，均不能作为本轮结果的充分解释。', '', 'P3 中原官方初始状态与原远目标下仍有大差异，详见 results/P3_GOAL_ROLE.csv；目标替换后的诊断回合另列，不与原官方目标混为同一测试分布。这里没有新增完整1000步评价，不能把25步未离开区域当作重新测得的远任务成功率。P1/P2 使用较近的目标，因此不能以其阳性局部能力排除远目标条件下的动作不足、状态支持或后续路线执行问题。H-D只是一项后续候选，尚未被本轮因果确认。', '', '全部P1配对及反例见 results/ALL_P1_PAIRED_TESTS.csv；每种子的得分区间见 results/SCORE_UNCERTAINTY.csv；不同来源单列 results/SOURCE_STRATIFIED_RESULTS.csv；相同路线负对照与低运动分类见 NEGATIVE_CONTROLS.csv。代表轨迹按预登记字典序选取每个结果类的第一条，未按最好/最坏表现选取。', '',f'E0-C：{offline.state_id.nunique()} 个严格匹配离线观察/qpos/qvel的记录状态，共{len(offline)}个状态-目标-模型动作对；其余状态不可测。离线记录原始完整integration不可得，使用完整合法重建仿真状态，明确标作 offline_record_reconstructed_full_physics；记录动作是真实数据动作，但不是真正新目标下的最优动作标签。','', '补充操作化在查询前冻结：判据2使用平均裁剪动作 L2，并要求在 A正确/B不正确样本上 B平均L2更小；判据3保守要求P2的A−B小于15百分点，并同时展示进展和运动幅度。这些是技术登记标准，不是实验后根据成绩挑选的阈值。', '', '复现边界：训练种子只有3个；seed0探索，seed1/2既有训练结果曾被查看，不能声称完全盲确认。按路口/来源簇重采样的区间描述固定模型上的状态不确定性，不是跨训练seed显著性。P1/P2是几何构造或离线记录重建状态；P3状态来自官方评价，但第二目标是特权几何干预。BFS仅定义参考初通道，不提供最优Actor动作真值。没有新增训练、梯度更新或GPU分配；原权重、优化器、配置、数据和旧成绩保持只读。', '', '裁定范围：未支持预登记的“在这些可行局部分岔中出现跨种子至少15百分点有效目标区分下降”的解释；不能据此排除全部随机目标机制、长距离目标条件敏感度或AWR/Waypoint的影响。'])
    (ROOT/'HYPOTHESIS_VERDICT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'NEXT_DECISION.md').write_text('# 下一阶段决策\n\n本轮H0未获一致支持，停止依据“随机目标使学生忽略目标”设计专用修复模块。\n\n保留远目标条件下的局部动作质量、离线数据支持和后续路线执行作为候选。P3分解可用于提出新诊断设计，但不自动追加状态、修改目标、延长窗口或重训练。若要区分目标距离/目标身份造成的行为变化，需要独立预登记的匹配距离实验卡；若要测试A/B采样敏感性是否跨方法存在，需要另行批准GCIVL等跨方法训练。\n\n本轮没有支持H0，仍是一项有效诊断；阴性结果不排除全部随机目标机制，也不恢复作者原始训练/评价协议。\n')
    for name,h in before.items():assert sha(ROOT/name)==h,name
    write(ROOT/'receipts/ANALYSIS_EXIT.json',dict(exit_code=0,epoch=time.time(),verdict=verdict['verdict'],scientific_estimates_reused_unchanged=True,figures_reused_unchanged=True,model_queries_added=0,physical_steps_added=0,repair='representative JSON NaN -> null; finish reports only',protected_output_sha256=before,new_training_updates=0,GPU_hours=0));guard();print('ANALYSIS_REPORT_FINALIZED',verdict['verdict'],flush=True)
if __name__=='__main__':main()

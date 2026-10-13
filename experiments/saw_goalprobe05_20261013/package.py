"""Final scientific/resource inventory and explicit light-file allowlist."""
from core import *
import subprocess, importlib.metadata as md, pandas as pd

def main():
    assert not (ROOT/'receipts/COMPLETION.json').exists()
    final=json.loads((ROOT/'receipts/FINAL_AUDIT.json').read_text());assert final['status']=='PASS'
    for m in json.loads((ROOT/'MODEL_IDENTITY.json').read_text())['models']:
        x=json.loads((ROOT/'receipts'/f'{m["variant"]}_s{m["seed"]}_EXIT.json').read_text());assert x['network_counter_before']==x['network_counter_after']==1000001 and x['new_updates']==0 and x['GPU_MiB']==0
    assert json.loads((ROOT/'receipts/AUDIT_PROCESS_RESOURCE.json').read_text())['exit_code']==0
    start=json.loads((ROOT/'receipts/RESOURCE_START.json').read_text());finish=time.time();wall=finish-start['epoch'];assert wall<14400
    budgets=json.loads((ROOT/'receipts/CONTROL_BUDGET.json').read_text());assert budgets['used']==21249<=25000
    resources=[json.loads(p.read_text()) for p in (ROOT/'receipts').glob('*PROCESS_RESOURCE.json')]
    assert all(not r['own_GPU_process_matches'] for r in resources)
    write(ROOT/'receipts/RESOURCE_SUMMARY.json',dict(epoch_start=start['epoch'],epoch_diagnosis_complete=finish,CPU_diagnosis_wall_seconds=wall,CPU_diagnosis_wall_hours=wall/3600,CPU_wall_limit_seconds=14400,CPU_allowed_cores=[72,73,74,75],GPU_hours=0,GPU_MiB=0,new_training_updates=0,control_steps=budgets,phase_resources=resources,preparation_CPU_seconds='not instrumented; included in full diagnosis wall window',dependency_install='isolated target, exit0; also included in diagnosis wall window',observed_CPU_seconds_instrumented_sum=sum(x['observed_CPU_seconds'] for x in resources),own_GPU_process_observations=sum(x['resource_samples'] for x in resources),formal_rollouts=768,full_1000_step_evaluations=0))
    versions={n:md.version(n) for n in ['numpy','jax','jaxlib','flax','mujoco','gymnasium','matplotlib','pandas','pytz','tzdata']}
    write(ROOT/'receipts/RUNTIME_IDENTITY.json',dict(python=sys.version,executable=sys.executable,packages=versions,CPU_devices=[str(x) for x in jax.devices()],affinity=sorted(os.sched_getaffinity(0)),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],JAX_PLATFORMS=os.environ['JAX_PLATFORMS'],core_source_hashes=final['source_sha256']))
    (ROOT/'requirements_analysis.txt').write_text('pandas==2.2.3\npytz==2024.2\ntzdata==2024.2\n')
    # Audit ambiguous P3 paths transparently; original sensitivity CSV keeps its exact bytes.
    bank={x['state_id']:x for x in pickle.load(open(ROOT/'runs/FROZEN_BANK.pkl','rb'))};s=pd.read_csv(ROOT/'GOAL_SENSITIVITY.csv');ambiguity=[]
    for rr in s.itertuples(index=False):
        item=bank[rr.state_id];gi=int(rr.goal_index);a=np.asarray(json.loads(rr.clipped));cell=item['cell'];angles=[]
        for neighbor in item['firsts'][gi]:
            direction=np.asarray([4*(neighbor[1]-cell[1]),4*(neighbor[0]-cell[0])]);angles.append(between(a,direction))
        ambiguity.append(dict(seed=int(rr.seed),variant=rr.variant,state_id=rr.state_id,group=rr.group,goal_index=gi,valid_first_channels=json.dumps(item['firsts'][gi],default=int),valid_first_count=len(angles),canonical_first_angle_error=rr.reference_angle_error,min_valid_first_angle_error=min(x for x in angles if x is not None) if any(x is not None for x in angles) else None,interpretation='canonical angle is descriptive; any shortest-path first channel accepted for CBR'))
    rows(ROOT/'results/ROUTE_AMBIGUITY_AUDIT.csv',ambiguity)
    card=ROOT/'EXPERIMENT_CARD.md';card.write_text(card.read_text()+'\n## 14. 完成登记（条件与判据不变）\n\n64状态全部准入，6模型768次CPU局部测试完成；物理控制步21249，训练更新0、GPU0。H0未获支持；来源、负对照、反例、区间与复现边界见 HYPOTHESIS_VERDICT.md 和 REPRODUCTION_REPORT.md。完整轨迹留服务器。\n')
    verdict=ROOT/'HYPOTHESIS_VERDICT.md';verdict.write_text(verdict.read_text()+'\n不确定性补充：seed2的P1差值95%区间为[-6.25,18.75]百分点，仍包含15百分点；本轮不属于A/B等价检验，也没有排除全部潜在局部效应。部分聚类区间退化为一点，源于已观察的少数簇结果相同，不代表总体成功率已知或零不确定性。P3存在并列最短首通道；GOAL_SENSITIVITY.csv中的reference_angle_error仅针对字典序参考通道，results/ROUTE_AMBIGUITY_AUDIT.csv另列对全部合法首通道的最小夹角。CBR一直接受全部合法首通道，主P1/P2均无首通道歧义。\n')
    write(ROOT/'receipts/TECHNICAL_FAILURE_LEDGER.json',dict(scientific_training_failures=0,formal_model_rollout_failures=0,model_history_mismatches=0,errors=[dict(stage='prepare',exit_code=1,error='NumPy int64 JSON export after all admission tests',resolution='reuse frozen bank and all admission traces; repeat restore selfchecks, counted128 steps'),dict(stage='recovery staging',error='missing partial CSV path; report exporter not installed yet',resolution='guarded extraction and separate receipts; no model queries'),dict(stage='execute guard preflight',exit_code=1,error='psutil unavailable',resolution='standard-library procfs monitor, no formal execution before repair'),dict(stage='analysis import',exit_code=1,error='pandas absent',resolution='isolated pinned analysis dependency target; original environment untouched'),dict(stage='analysis representative metadata',exit_code=1,error='CSV empty first-exit step read as NaN; strict JSON refused',resolution='NaN to null; all completed estimate CSVs and figure SHA retained; report-only finalize'),dict(stage='report recovery preflight',error='unmatched parenthesis caught before reporting script execution',resolution='syntax repair, no model queries or physical steps')],no_sample_goal_threshold_window_changes=True,no_formal_model_reruns=True))
    c=pd.read_csv(ROOT/'PAIRED_CONTRASTS.csv');counts=pd.read_csv(ROOT/'SCENE_AND_GOAL_BANK.csv').drop_duplicates('state_id').groupby(['group','source']).size().reset_index(name='states');counts.to_csv(ROOT/'results/STATE_SOURCE_COUNTS.csv',index=False)
    lines=['# SAW-GOALPROBE-05 复现报告','', '**执行完成；H0未获预登记支持。**','', '分支：https://github.com/rollinpioneer/graph/tree/codex/saw-goalprobe-05-20261013','', '来源冻结提交：176ad8d17ec9d71eaca9ed55d87a8ee0bf837a28；正式CPU查询源码：'+json.loads((ROOT/'receipts/EXECUTION_CODE_VERSION.json').read_text())['commit']+'。最终提交与远端一致性记录在独立发布回执，避免自引用提交哈希。','', '| Seed | P1 A/B | P1差百分点及95%区间 | P2 A/B |','|---|---|---|---|']
    for seed in (0,1,2):
        z=c[(c.seed==seed)&(c.group=='P1')&(c.metric=='correct')].iloc[0];lines.append(f'| {seed} | {100*z.A:.2f}% / {100*z.B:.2f}% | {100*z.A_minus_B:.2f} [{100*z.CI95_low:.2f}, {100*z.CI95_high:.2f}] | 100% / 100% |')
    lines.extend(['', 'P1是近处分岔、匹配BFS距离的不同通道目标；P2是同通道近目标对照。在这些场景中，B大多能区分并进入正确初通道。P3原官方远目标下，A均16/16进入初通道；B为seed0 3/16、seed1 3/16、seed2 0/16。Task1/3各8状态，替换后的目标单列。不能把P3两种目标合并后解释成新的官方成绩，也不能据P1/P2的高分断言只有后段路线执行失败。','', '结论与下一步：停止依据本轮证明“普遍忽略目标”的修复设计；目标身份/距离条件下的动作质量与状态支持仍值得设计新卡核验。没有追加状态、目标、窗口、完整评价或训练。seed2区间较宽，不能称A/B等价，也不能排除全部随机目标机制。','', '身份与一致性：6份1M权重SHA及来源索引核验；12项CPU历史动作最大绝对误差0；64状态完整integration、qpos/qvel、目标、所有记录RNG和wrapper恢复/重复一致；768条保存轨迹独立重算首次通道、裁剪和累计路径，最大路径误差8.89e-16。原MECH工作区仍干净、HEAD未变；原权重、数据、配置及旧成绩未覆盖。','', f'实际资源：诊断窗口{wall:.3f}秒（{wall/3600:.6f}小时），含准备、导出修复、依赖补全、统计与审计；允许CPU核心72–75，硬上限4小时。GPU驻留0、显存0、新训练更新0。所有物理控制步21249/25000，其中正式18024、准入2969、首次恢复128、导出修复重复恢复128。准备阶段CPU秒未单独测量，不能把守卫采样CPU秒冒充完整CPU消耗；完整墙钟窗口覆盖该成本。','', '技术状态：所有正式模型及历史检查通过。导出阶段的NumPy/NaN编码、缺失统计包和报告预检错误均保留失败代码/日志/退出回执，修复没有重新选择样本、重新查询模型或改变判据；唯一新增的128步恢复自检计入硬预算。统计/图已完成后的NaN导出故障使用只读报告恢复，原CSV和图SHA保持一致。','', '结果入口：HYPOTHESIS_VERDICT.md（机制与边界）、NEXT_DECISION.md（停止/后续卡）、PAIRED_CONTRASTS.csv（配对区间）、NEGATIVE_CONTROLS.csv（失败/运动）、results/SOURCE_STRATIFIED_RESULTS.csv（来源）、results/P3_GOAL_ROLE.csv（官方目标与干预）、results/OFFLINE_ALIGNMENT.csv（严格数据动作）、receipts/FINAL_AUDIT.json（身份/轨迹审计）。','', '四张结果图已逐张可视检查，布局和标注清晰。CBR_BY_SEED_AND_CONTROL.png的误差条是簇重采样区间；P3柱包含两个目标角色，只作诊断混合集描述。GOAL_ACTION_DIRECTION_DISTRIBUTION.png展示方向分布；GOAL_SENSITIVITY_CONTROLS.png展示目标对动作差；REPRESENTATIVE_TRACES.png展示每个类别按字典序选择的第一条轨迹。','', '完整初态、每步integration/qpos/qvel、裁剪前后动作、目标、参考路线与终止记录位于本目录runs；完整日志在logs。RAW_ARTIFACT_INDEX.json登记路径、大小、SHA。原大权重和数据路径见MODEL_IDENTITY.json与BANK_FROZEN.json；Git仅归档代码、卡片、配置、轻量表/回执/图。'])
    (ROOT/'REPRODUCTION_REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'README.md').write_text('# SAW-GOALPROBE-05\n\n完整结论见 REPRODUCTION_REPORT.md 与 HYPOTHESIS_VERDICT.md。冻结设计见 EXPERIMENT_CARD.md、PREREGISTRATION.json，原始用户设计稿见 CARD_SUPPLIED.md。\n\n运行阶段：CPU环境与4核affinity固定 → prepare.py（不加载策略、冻结场景和物理准入）→ execute.py（只读模型和历史输出检查）→ analyze.py（簇bootstrap及图）→ audit.py（只读轨迹重算）。本次prepare导出错误由recover_export.py复用已有准入；analysis的NaN导出由finalize_analysis.py完成报告，科学CSV/图不重算。run_phase.py只监控本轮自己的子进程。\n\n所有运行必须CUDA_VISIBLE_DEVICES空值、JAX_PLATFORMS=cpu、taskset -c 72-75；统计依赖通过uv pip install --python <原只读解释器> --target <本轮vendor/analysis_deps> --no-deps -r requirements_analysis.txt安装，并加入PYTHONPATH；不能安装覆盖原环境。首次运行创建独立输出目录；已有状态/输出拒绝直接重跑，不删除断言或覆盖旧结果。\n\n这是CPU诊断而非训练入口。没有自动追加、优化种子或GCIVL训练。\n')
    raw=[]
    for folder in ('runs','logs'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file():raw.append(dict(path=str(p),relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(ROOT/'RAW_ARTIFACT_INDEX.json',dict(files=raw,count=len(raw),weights=json.loads((ROOT/'MODEL_IDENTITY.json').read_text())['models'],dataset_inputs=json.loads((ROOT/'receipts/BANK_FROZEN.json').read_text())['source_inputs'],dependencies='isolated vendor/analysis_deps; pinned versions and metadata identity in receipts/ANALYSIS_DEPENDENCIES.json; omitted from Git'))
    write(ROOT/'receipts/COMPLETION.json',dict(status='COMPLETE_FROZEN_DIAGNOSTIC',epoch=finish,exit_code=0,formal_models=6,formal_tests=768,states=64,control_steps=21249,verdict='H0_NOT_SUPPORTED_BY_LOCAL_BRANCH_DEFICIT',new_updates=0,GPU_hours=0,GPU_MiB=0))
    allowed_root={'.gitignore','EXPERIMENT_CARD.md','CARD_SUPPLIED.md','PREREGISTRATION.json','MODEL_IDENTITY.json','SCENE_AND_GOAL_BANK.csv','ADMISSION_AUDIT.json','GOAL_SENSITIVITY.csv','BRANCH_ROLLOUTS.csv','PAIRED_CONTRASTS.csv','NEGATIVE_CONTROLS.csv','HYPOTHESIS_VERDICT.md','NEXT_DECISION.md','REPRODUCTION_REPORT.md','README.md','RAW_ARTIFACT_INDEX.json','requirements_analysis.txt','core.py','prepare.py','execute.py','analyze.py','audit.py','recover_export.py','run_phase.py','finalize_analysis.py','package.py'}
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if len(rel.parts)==1 and p.name in allowed_root:files.append(p)
        elif rel.parts[0] in ('results','receipts') and len(rel.parts)==2 and p.suffix in ('.json','.csv','.txt'):files.append(p)
        elif rel.parts[0]=='figures' and p.suffix=='.png':files.append(p)
    assert len(files)>40
    manifest=dict(files=[dict(relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],count=len(files),bytes=sum(p.stat().st_size for p in files),excludes=['runs','logs','vendor','weights','datasets','keys'],scope='this new experiment directory only')
    write(ROOT/'LIGHT_MANIFEST.json',manifest);files.append(ROOT/'LIGHT_MANIFEST.json');(ROOT/'receipts/STAGING_ALLOWLIST.txt').write_text('\n'.join(str(p.relative_to(ROOT)) for p in files)+'\n')
    print('PACKAGE_COMPLETE',len(files),manifest['bytes'],wall,flush=True)

if __name__=='__main__':main()

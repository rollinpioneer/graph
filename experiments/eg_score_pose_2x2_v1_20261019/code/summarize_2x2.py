import os, json, csv, hashlib, sys, time
from pathlib import Path
import numpy as np

R = Path(os.environ['P2_ROOT'])
V = Path(os.environ['PREV_ROOT'])
cfg = json.loads((R / 'frozen_config.json').read_text())
seeds, scenes, frames = cfg['seeds'], cfg['development'], cfg['frames']
records = []
complete = []
for scene in scenes:
    folder = R / f'evaluation/development/{scene:04d}'
    done = json.loads((folder / 'COMPLETE.json').read_text())
    assert done['status'] == 'PASS' and done['frames'] == frames
    assert done['S_geometry_all_exact'] and done['repeated_V2_metrics_all_exact']
    complete.append(done)
    for frame in frames:
        rows = json.loads((folder / f'{frame:04d}.json').read_text())
        assert len(rows) == 20
        records += rows
assert len(records) == 5120

def write(name, value):
    (R / 'results' / name).write_text(json.dumps(value, indent=2)+'\n')
def write_csv(name, rows):
    with open(R / 'results' / name, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
write_csv('per_frame_results.csv', records)
metrics = ['ap', 'ap04', 'ap08', 'top1_valid', 'top1_collision', 'top1_empty', 'cd_rejection_rate']
scene_rows = []
table = {}
for scene in scenes:
    for name in sorted(set(x['configuration'] for x in records)):
        rows = [x for x in records if x['scene'] == scene and x['configuration'] == name]
        assert len(rows) == len(frames)
        value = dict(scene=scene, configuration=name, method=rows[0]['method'], seed=rows[0]['seed'], use_cd=rows[0]['use_cd'], frames=len(rows))
        value.update({k: float(np.mean([x[k] for x in rows])) for k in metrics})
        for key in ['selected_top50_count', 'selected_top50_collisions', 'cd_rejected', 'modified', 'candidates']:
            value[key] = int(sum(x[key] for x in rows))
        value['selected_top50_collision_rate'] = value['selected_top50_collisions'] / value['selected_top50_count'] if value['selected_top50_count'] else None
        scene_rows.append(value)
        table[scene, name] = value
write_csv('per_scene_results.csv', scene_rows)
reported_metrics = metrics + ['selected_top50_collision_rate']
global_rows = []
for cd in [False, True]:
    suffix = '_CD' if cd else ''
    for method in ['R0', 'S', 'G0', 'Gabs']:
        selected_seeds = [0] if method == 'R0' else seeds
        for seed in selected_seeds:
            name = method + (f'_{seed}' if seed else '') + suffix
            global_rows.append(dict(method=method, seed=seed, use_cd=cd, **{m: float(np.mean([table[s, name][m] for s in scenes])) for m in reported_metrics}))
write_csv('global_by_seed.csv', global_rows)

rng = np.random.default_rng(cfg['bootstrap_seed'])
indices = rng.integers(0, len(scenes), (cfg['bootstrap_repetitions'], len(scenes)))
comparisons, differences = [], []
contrasts = {
    'S-R0': {'S': 1, 'R0': -1},
    'Gabs-S': {'Gabs': 1, 'S': -1},
    'G0-R0': {'G0': 1, 'R0': -1},
    'Gabs-G0': {'Gabs': 1, 'G0': -1},
    'Gabs-R0': {'Gabs': 1, 'R0': -1},
    'interaction': {'Gabs': 1, 'S': -1, 'G0': -1, 'R0': 1},
}
for cd in [False, True]:
    suffix = '_CD' if cd else ''
    for contrast, weights in contrasts.items():
        result = dict(contrast=contrast, use_cd=cd, metrics={})
        for metric in reported_metrics:
            per_seed = np.array([[sum(weight * table[s, method + (f'_{seed}' if method != 'R0' else '') + suffix][metric] for method, weight in weights.items()) for s in scenes] for seed in seeds])
            mean_scene = per_seed.mean(0)
            scale = 1 if metric.startswith('ap') else 100
            ci = np.quantile(mean_scene[indices].mean(1), [.025, .975]) * scale
            result['metrics'][metric] = dict(mean_difference=float(mean_scene.mean()*scale), ci95=ci.tolist(), seed_difference=(per_seed.mean(1)*scale).tolist(), scene_difference=dict(zip(map(str, scenes), (mean_scene*scale).tolist())), positive_scenes=int((mean_scene > 0).sum()), unit='AP percentage points' if metric.startswith('ap') else 'percentage points')
        comparisons.append(result)
        for j, scene in enumerate(scenes):
            differences.append(dict(contrast=contrast, use_cd=cd, scene=scene, **{metric: result['metrics'][metric]['scene_difference'][str(scene)] for metric in reported_metrics}))
write('paired_comparisons.json', comparisons)
write_csv('per_scene_differences.csv', differences)

frame_lookup = {(x['scene'], x['frame'], x['configuration']): x for x in records}
transitions = []
for cd in [False, True]:
    suffix = '_CD' if cd else ''
    for seed in seeds:
        for left, right in [('S','R0'), ('Gabs','S'), ('G0','R0')]:
            better = worse = same_top = 0
            for scene in scenes:
                for frame in frames:
                    a = frame_lookup[scene, frame, f'{left}_{seed}{suffix}']
                    b = frame_lookup[scene, frame, right + (f'_{seed}' if right != 'R0' else '') + suffix]
                    better += int(a['top1_valid'] and not b['top1_valid'])
                    worse += int(b['top1_valid'] and not a['top1_valid'])
                    same_top += int((a['top_candidate'],a['top_action']) == (b['top_candidate'],b['top_action']))
            transitions.append(dict(contrast=f'{left}-{right}', use_cd=cd, seed=seed, N=256, application_top1_failure_to_success=better, application_top1_success_to_failure=worse, identical_top1_action=same_top, note='System Top1 transitions can compare different candidate identities; not same-candidate local pose rescue/damage.'))
write('application_top1_transitions.json', transitions)

status = json.loads((R / 'results/pipeline_status.json').read_text())
resources = dict(development_wall_seconds=status['seconds'], child_cpu_seconds=status.get('child_cpu_seconds'), successful_worker_wall_seconds=sum(x['seconds'] for x in complete), new_native_calls=sum(x['new_native_calls'] for x in complete), reused_native_calls=sum(x['reused_native_calls'] for x in complete), max_scene_rss_kb=max(x['rss_kb'] for x in complete), max_evaluation_workers=7, controller=1, threads_per_worker=1, gpu_jobs=0, new_trainings=0, micro_seconds=json.loads((R / 'evaluation/micro/0037/COMPLETE.json').read_text())['seconds'])
write('resource_summary.json', resources)
verification = dict(status='PASS', frames=256, configurations_per_frame=20, frame_records=len(records), S_original_geometry_exact_all_frames=True, Q_cache_frozen_heads_exact_micro_all_three=True, all_source_exports_and_V2_caches_hash_verified=True, native_full_S_micro_checks=6, V2_repeated_AP_AP04_AP08_Top1_collision_empty_metrics_exact_all_frames=True, confirmation_opened=False)
write('verification.json', verification)

def mean_value(method, metric, cd=True):
    return float(np.mean([x[metric] for x in global_rows if x['method'] == method and x['use_cd'] == cd]))
def contrast(name, cd=True):
    return next(x for x in comparisons if x['contrast'] == name and x['use_cd'] == cd)
report = [
    '# EG 评分与姿态最小2×2开发对照', '',
    '**结论：观察到的AP收益主要可由只重评分实现，未显示姿态重选的稳定额外AP收益。** 带CD时，S为57.923 AP（比R0 +1.991，描述性场景95%CI [1.518, 2.520]，8/8场景正向）；Gabs仅比S增加0.059 AP（CI [-0.068, 0.199]，5/8场景正向，三头中一个为负）。G0比R0增加0.181 AP。', '',
    '状态：完成。三个既有轻量头，新增训练0，GPU任务0。结果仅来自已经观察的8个开发场景，每场景32帧，共256帧；不是独立泛化或官方完整场景评价。', '',
    'R0：原始姿态/原始分数；S：原始姿态/Q(original)；G0：锁定重选姿态/原始分数；Gabs：相同锁定重选姿态/Q(selected)。各组均从完整1024候选重新执行官方NMS及原生候选选择，CD沿用官方部署碰撞过滤参数。', '',
    '## 完整开发样本上的指标', '',
    '|方法（三头均值）|CD|AP|AP0.4|AP0.8|Top1合格率|Top1几何碰撞率|入选Top50碰撞率|候选CD拒绝率|',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|',
]
for cd in [False, True]:
    for method in ['R0','S','G0','Gabs']:
        values = [mean_value(method, m, cd) * (1 if m.startswith('ap') else 100) for m in ['ap','ap04','ap08','top1_valid','top1_collision','selected_top50_collision_rate','cd_rejection_rate']]
        report.append(f'|{method}|{"有" if cd else "无"}|'+'|'.join(f'{v:.3f}' for v in values)+'|')
report += ['', '## 带CD的预设比较', '', '|比较|AP差值|场景95%CI|三个头分别差值|正向场景|Top1合格率差（百分点）|', '|---|---:|---|---|---:|---:|']
for name in contrasts:
    x = contrast(name)
    a = x['metrics']['ap']
    report.append(f'|{name}|{a["mean_difference"]:+.3f}|[{a["ci95"][0]:+.3f}, {a["ci95"][1]:+.3f}]|'+', '.join(f'{v:+.3f}' for v in a['seed_difference'])+f'|{a["positive_scenes"]}/8|{x["metrics"]["top1_valid"]["mean_difference"]:+.3f}|')
report += ['', '## 结果解释与证据范围', '',
    '1. 只重评分已经获得与Gabs接近的AP，说明当前冻结特征、已有MSE头在原始动作质量排序上保留了有效开发信号；不能再把Gabs的整体AP提升直接作为局部姿态选择有效的证据。', '',
    '2. G0的姿态修改有小幅AP正信号；在轻量头评分条件下，其额外收益缩小且区间跨零。交互为-0.122 AP，CI [-0.256, 0.036]，不足以确定稳定的负交互。当前证据未支持为局部重选增加网络复杂度，也不能证明所有冻结特征缺少姿态信息或MSE损失必然失败。', '',
    '3. Top1需单独判断：S合格率均值增加2.604个百分点，但场景区间为[-5.990, 10.286]，场景55和93分别降低18.750、10.417个百分点。带CD时三个头的整帧Top1由失败变成功次数为50/52/46，同时成功变失败43/47/38；这是不同候选之间的排序交换，不是同一候选的局部姿态损坏。AP正向不能当作Top1安全或可靠性已经确认。', '',
    '4. S的Top1真实几何碰撞率从13.281%降至8.464%，入选Top50碰撞率从30.169%降至28.591%。因为全部几何候选不变，候选CD拒绝率仍完全相同；改善来自排序/NMS后保留了不同候选。', '',
    '研究建议：若另立下一卡，优先冻结S的权重与推理规则，在获批的未使用场景确认AP、Top1和碰撞表现；本轮不打开新场景或追加训练。是否需要更有姿态条件信息的特征或更适合局部选择的损失，仍需有针对性的另一组对照。', '',
    'S−R0测量原始1024几何候选不变时，轻量头评分引起的排序/NMS与最终候选变化。Gabs−S测量在轻量头评分条件下启用锁定姿态修改的额外系统收益，其中Q(selected)随姿态改变；不能把它解释为独立于评分的纯几何因果效应。G0−R0是原始EG评分条件下的姿态修改效果。', '',
    '交互=(Gabs−S)−(G0−R0)。配对统计先平均三个头，再以场景聚类bootstrap10000次（固定随机种子20261019）。区间未多重比较修正，开发场景已观察，不设事后成功门槛，不能作为独立确认。三个训练种子不当作三个独立场景样本。', '',
    'Top1合格按原生摩擦质量0<μ≤0.4；AP严格沿用六摩擦系数和Top1至50平均。几何碰撞来自真实物体模型/桌面原生评价，CD拒绝率来自部署点云碰撞过滤，二者含义不同。Top50碰撞率按每场景入选前50的总碰撞数/总入选数计算，再场景等权。逐帧指标中保留入选数及空抓标记。', '',
    '## 必要验证、资源与位置', '',
    'S全部几何分量逐元素保持原始1024候选不变。三头首帧Q与冻结权重重算精确一致；S三个头带/不带CD共6条路径均与官方完整eval_grasp的候选、质量、碰撞及AP精确一致。R0/G0/Gabs在全部256帧的既有指标与V2精确一致。源导出/原生评价代码/完整Q及逐姿态质量缓存均做身份核验，仅对新排序需要且原缓存未覆盖的姿态补算。', '',
    f'正式评价墙钟{resources["development_wall_seconds"]:.1f}秒；worker墙钟累计{resources["successful_worker_wall_seconds"]/3600:.3f}小时；新增原生质量评价{resources["new_native_calls"]}条，复用{resources["reused_native_calls"]}条。最多7个CPU单线程worker加1个控制进程，无GPU、新训练或确认评价。', '',
    '逐场景指标见per_scene_results.csv；逐场景差值见per_scene_differences.csv；逐头汇总见global_by_seed.csv；完整逐帧结果per_frame_results.csv和候选质量NPZ留服务器并记录哈希。', '',
    '分支：codex/eg-score-pose-2x2-v1-20261019。起点84268e4efc347a92da2b5f49958a6cce64e155ae，登记5b527d72f542cfa9dbc2c7bc5f729e3faa13dfb6。',
]
(R / 'results/final_report.md').write_text('\n'.join(report)+'\n')
status.update(status='COMPLETED', stage='summarized', new_trainings=0, confirmation_opened=False)
write('pipeline_status.json', status)
print(json.dumps(dict(status='PASS', summary={m:mean_value(m,'ap') for m in ['R0','S','G0','Gabs']}, resources=resources), indent=2))

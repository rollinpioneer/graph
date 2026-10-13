"""Summarize completed frozen FLOQ runs without loading models or training.

Inputs remain on the server. Only derived tables, plots and file identities are
written to the registered archive worktree. No process or Git control is used.
"""
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import shlex
import shutil
import statistics
import subprocess
import time

SOURCE = '3d60e638b42bbef018c56cb1199ff37c3470520d'
BRANCH = 'codex/floq-min-repro-v2-20261013'
TAG = '20261014_0530'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(path, records, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def aggregate(values):
    return dict(mean=statistics.mean(values), std_ddof0=statistics.pstdev(values),
                std_ddof1=statistics.stdev(values), seed_count=len(values))


def markdown_reports(root, archive, result, files):
    def save(name, text):
        (archive / name).write_text(text.strip() + '\n')
        if name not in files:
            files.append(name)
    identity = [
        '# FLOQ 复现身份与运行回执', '',
        '状态：两条正式训练均正常完成 2M；分类 B — FUNCTIONAL_REPRODUCED。', '',
        '官方源码 SHA：`' + SOURCE + '`；原 checkout 清洁，未修改算法。',
        '源码：[CMU-AIRe/floq](https://github.com/CMU-AIRe/floq/tree/' + SOURCE + ')。',
        '归档分支：[' + BRANCH + '](https://github.com/rollinpioneer/graph/tree/' + BRANCH + ')。',
        '服务器 ROOT：`' + str(root) + '`；最终归档 SHA 由 `archive_status.json` 的远端核验记录给出。', '',
        '## 环境、数据和随机数', '',
        'Python 3.8.19；JAX 0.4.13；jaxlib 0.4.12+cuda11.cudnn86；NumPy 1.22.4；Flax 0.7.2；Optax 0.1.3；'
        'OGBench 1.1.2；MuJoCo 3.1.6；dm-control 1.0.20；Gymnasium 1.0.0。完整包表见 '
        '`report/installed_environment.json`，与作者 YAML 的差异见 `report/dependency_differences.csv`。', '',
        '隔离 uv 环境使用 pip CUDA 11.8/cuDNN 8.9.2.26 库及官方 CUDA wheel，未宣称与作者 conda 二进制构建逐字一致。'
        '矩阵精度保持官方 JAX 默认 None，x64=false，参数 float32。A100 40GB 上两个种子分别使用 GPU 3/6、各四个 CPU 核；'
        '共享卡的总利用率包含其他用户，未干预其进程。', '',
        '命令任务为 `antmaze-large-navigate-singletask-v0`；OGBench 实际环境 spec 为 `antmaze-large-singletask-v0`。'
        '与显式 Task1 的初始 qpos/qvel/observation 一致性已在 P0 验证；见 `report/task_alias_check.json`。'
        '训练/验证数据分别 1M/100k transitions，observation 29 维、action 8 维，动作 [-1,1]，episode 上限 1000。'
        'Success 使用官方环境回合末 info 的 success，经官方 evaluate 聚合；return 为原版未折扣回合奖励总和。', '',
        '原始 NPZ 来自官方 OGBench 下载并通过只读链接复用。以下数据文件在收尾再次核验 SHA256：', '',
        '| 文件 | 字节 | SHA256 |', '|---|---:|---|']
    for entry in result['dataset_identity']['files']:
        identity.append('| %s | %s | `%s` |' % (pathlib.Path(entry['path']).name, entry['bytes'], entry['sha256']))
    identity += ['', '种子 0/1 从头独立初始化，配置 SHA 相同，初始网络参数 SHA 不同；seed 0 与 pristine P1 初始化一致，'
                 '未复用 P1 权重。见 `receipts/formal_config_validation.json`。各种子的初始 Python/NumPy/JAX 身份见 '
                 '`formal/formal_seed*/initial_state.json`。', '',
                 '官方 main 在设置全局 seed 前初始化环境 RNG；保留该顺序并记录真实状态。原生起点未与 FQL/GFP 匹配。'
                 'temperature=0 仍有 Gaussian action noise，不能称为确定性策略。每次评价的 policy key、环境 RNG '
                 '及逐回合初始观察哈希保留在服务器回执；2M 评价状态和 100 条逐回合轻量表另存 `final/`。', '',
                 '## 完整命令与参数', '',
                 '实际入口是工程包装器导入固定官方 main；仅切换 W&B offline、记录既有数值/RNG、每 100 更新检查有限性，'
                 '不增加算法项或随机采样。正式训练不运行 P1 探针。以下命令在 `ROOT/source` 执行；路径与环境覆盖原样记录。', '']
    for entry in result['identities']:
        command = ['env'] + [key + '=' + value for key, value in entry['environment_overrides'].items()] + entry['command']
        identity += ['Seed %d：' % entry['seed'], '', '```bash', 'cd ' + shlex.quote(str(root / 'source')),
                     ' '.join(shlex.quote(part) for part in command), '```', '',
                     '全部展开模型参数：`formal/formal_seed%d/actual_config.json`；全部实际 flags：'
                     '`formal/formal_seed%d/expanded_flags.json`。' % (entry['seed'], entry['seed'])]
    identity += ['', '配置 SHA256：`' + result['identities'][0]['config_sha256'] + '`。'
                 '参数来源表 `report/actual_parameter_origins.csv` 标明作者 get_config 默认、官方 main 派生、卡片/CLI 和工程路径字段；'
                 '种子及输出路径的实际不同值以逐种子 JSON 为准。α10、batch256、lr3e-4、γ.99、τ.005、actor flow10、'
                 'critic flow8、双 Flow/Scalar ensembles、noise8、coverage.1、mean Q，均保持冻结值。', '',
                 '## 完整性与可追溯产物', '',
                 'P0/P1 已通过，P1 独立 5k：初始化/前向/梯度、目标更新一拍延迟、全参数/优化器/JAX RNG 恢复身份及固定 key 动作一致性通过。'
                 '正式训练每条 exit0、2M updates、101×50 原生回合、400 行官方训练/验证指标。全记录未发现非有限数值；'
                 '未重启、延长、调参或挑选最好模型。', '',
                 '1M/2M 四份 checkpoint 文件大小与 SHA 核验通过，索引见 `formal/*/checkpoint_index.jsonl` '
                 '和 `final/artifact_index.json`。正式模型未另加恢复评价；保存/恢复工程验证来自 P1。', '',
                 '全 stdout、全部原生逐回合记录、资源采样、完整 RNG 和模型留服务器 `ROOT/runs/`；'
                 '`final/artifact_index.json` 给出路径、大小与 SHA。Git 保存全部原生曲线、轻量 loss 曲线、最终评价、逐回合指标及成本，'
                 '权重、数据、全日志未入库。之前明确禁止推送的 FQL/Q-Flow/GFP 分支未推送。']
    save('REPRO_IDENTITY.md', '\n'.join(identity))
    performance = ['# FLOQ 性能与复现判定', '',
                   '**B — FUNCTIONAL_REPRODUCED**。两条完整训练产生有效策略；目前不能标 A，也没有按事后阈值标 C。', '',
                   '按看到结果前冻结的协议：主结果只用 2M 时点原生 50 回合。', '',
                   '| Seed | Updates | 成功回合 | Success | Return | Mean length |',
                   '|---|---:|---:|---:|---:|---:|']
    for value in result['seeds']:
        performance.append('| %d | 2M | %d/50 | %.1f%% | %.2f | %.2f |' %
                           (value['seed'], value['successes'], value['success'] * 100, value['return_'], value['episode_length']))
    performance += ['', 'Success 等权均值 **%.1f%%**；总体标准差（ddof=0）**%.2f 个百分点**；样本标准差（ddof=1）'
                    '**%.2f 个百分点**。Return 均值 %.2f，ddof0 %.2f、ddof1 %.2f。不是置信区间，也不足以证明跨种子稳健性。' %
                    (100 * result['success']['mean'], 100 * result['success']['std_ddof0'], 100 * result['success']['std_ddof1'],
                     result['return_']['mean'], result['return_']['std_ddof0'], result['return_']['std_ddof1']), '',
                    '预先固定作者参考为 arXiv 2509.06863v2 Table 2 Task1 的 floq-default 94±4%，3 seeds。'
                    '[论文固定版本](https://arxiv.org/html/2509.06863v2)。作者检查点聚合窗口仍未核实，因此不能把本轮 96% '
                    '与其 94% 的差额称为提升、复现偏差检验或 Critic 效果。没有用五任务均值替代 Task1。', '',
                    '每条保留全部 101 点原生曲线、1M/2M 模型。`final/primary_results.csv` 是主表；'
                    '`final/primary_episodes.csv` 保留两种子共 100 个 2M 回合的轻量记录；全 10,100 回合留服务器并建散列索引。'
                    '训练中最高分和最后三个检查点平均均未替代本卡的 2M 主结果。', '',
                    '![Native evaluation curves](final/native_curves.png)', '',
                    '## 与 FQL 的描述性关系', '',
                    '旧 FQL 已完成每种子 1M，原生评价和所有十个 Matched 检查点均完整。FLOQ 每种子 2M；'
                    '旧 FQL 主表聚合最后三检查点，本卡只用 2M；FQL 原生每 100k、本卡每 20k；'
                    '旧环境/依赖与新卡作者环境存在差异。原始 NPZ 字节由只读链接复用，不能据此推断运行环境及评价 RNG 相同。', '',
                    '只读核验旧 FQL 实际参数：α10、batch256、lr3e-4、γ.99、τ.005、4×512 MLP、BC flow10、'
                    'actor layer norm=false、critic layer norm=true；这些与本卡相应 Actor 配方一致。'
                    'FQL 使用标量 TD Critic、Q=min，本卡增加 Flow/Target Flow/Scalar 蒸馏并使用 Q=mean；'
                    '更新预算、聚合、评价 RNG、依赖仍是混杂因素。不能把分数差异单独归因于 Critic。', '',
                    '本 FLOQ 卡没有新增 Matched Evaluation；原生场景与旧 FQL/GFP 未对齐。'
                    '旧卡的成绩和损失未复制到本卡 Git 分支，也未启动新的 FQL、GFP、Q-Flow 或模块替换训练。']
    save('PERFORMANCE_REPORT.md', '\n'.join(performance))
    cost = ['# FLOQ 训练成本', '',
            '| Seed | GPU | GPU 驻留 h | 进程墙钟 h | CPU h (user+system) | 显存峰值 MiB | 进程树 RAM峰值 GiB | 评价总秒 | 稳态日志窗口中位 ms/update |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for value in result['formal_costs']:
        cost.append('| %d | %d | %.4f | %.4f | %.4f | %d | %.3f | %.2f | %.3f |' %
                    (value['seed'], value['gpu'], value['gpu_residence_hours'], value['process_lifetime_hours'],
                     value['cpu_total_hours'], value['gpu_peak_mib'], value['process_tree_rss_peak_bytes'] / 2**30,
                     value['native_evaluation_seconds'], value['steady_logged_update_median_ms']))
    cost += ['', '正式两条 GPU 驻留合计 **%.4f h**，并行墙钟 **%.4f h**，CPU 合计 **%.4f h**。'
             '独立 P1 另计 %.5f GPU h；P1+正式合计 %.4f GPU h。P0 安装、数据核验、队列等待和本轮整理不计入正式训练墙钟，'
             '不把此数值冒充整个工程从开始到归档的总耗时。' %
             (result['formal_gpu_hours'], result['formal_wall_hours'], result['formal_cpu_hours'],
              result['p1_gpu_hours'], result['p1_plus_formal_gpu_hours']), '',
             'GPU 驻留以进程有 GPU 显存的采样起止计算，包含评价及停顿；正式资源轮询约 5s，边界有采样误差。'
             '共享 A100 卡的总利用率不等于本任务 GPU 利用率，该成本不是专用卡的内核活跃小时。'
             'RAM 为进程树轮询峰值；GNU time 单进程 MaxRSS 另保存在 cpu_time.txt。', '',
             '各进程绑定四核，实际 user+system 平均约 1.09 CPU cores；未改变 batch、精度或训练长度。'
             '每条 101 次评价，共 5,050 回合；评价耗时分别约 1.703/1.785 小时，已包含在 GPU 驻留和墙钟中，不能再相加。', '',
             '稳态统计取官方 5k 日志窗口，舍弃前 50k 和每次评价后的首个窗口（官方 main 在日志后评价），'
             '每种子剩 293 个窗口。中位约 8.07 ms/update，仍含数据、日志和验证开销，非纯 JAX kernel 计时。'
             '全进程墙钟减原生评价后折算约 122.14/122.15 updates/s，此口径还包含初始化、保存等开销。'
             'p10/p90、CPU user/system 原值和单次评价中位见 `final/costs.csv`/`final/summary.json`。']
    save('COST_REPORT.md', '\n'.join(cost))
    research = ['# FLOQ Baseline 研究准入', '',
                '| 判断 | 本轮结论 |', '|---|---|',
                '| 工程运行成功 | 通过：两条 2M、exit0、有限损失、101×50 原生评价、1M/2M文件身份核验；P1恢复测试通过 |',
                '| 论文成绩复现 H1 | 未确认：只执行两个种子、作者聚合窗口未核实，保持 B 标签 |',
                '| 模块拆分 H2 | 通过静态接口及P1探针准入；未执行替换训练 |',
                '| 新算法论文潜力 | 未验证；高分不能证明贡献、受控改动的有效性或新颖性 |', '',
                '建议把 floq 作为研究 Flow Critic 的**有条件 Baseline**。适合保留已验证的 FQL 风格 Actor，'
                '把 Flow Critic + Target Flow Critic + Distilled Scalar Critic 及其损失/状态作为整体受控替换单元。'
                '是否比 FQL/GFP 更适合长期研究，仍取决于要检验的机制，不按本次分数自动推荐。', '',
                '模块输入/输出、计算图和梯度关系详见 [CRITIC_INTERFACES.md](CRITIC_INTERFACES.md)。'
                '接口须保留 observation `[B,29]`、action `[B,8]`、Flow velocity `[2,B,1]`、Scalar Q `[2,B]` '
                '和 Scalar Q 对 action 的导数；当前 encoder=None，不把低维验证泛化到视觉 encoder。', '',
                'Flow 值的蒸馏目标固定，Actor Q loss 只经 action 回传，TD backup 不回传到 Actor/Target Flow。'
                '8 个 noise 的 Floq loss 先求和，不能换成全维均值；Target Flow 的 Polyak 读取更新前在线参数，'
                '保留原版一拍延迟。共同 ModuleDict/Adam 的参数树、优化器状态、bootstrap、噪声范围、ensemble聚合'
                '与 RNG 消费必须一并定义。单换 online Flow 而保留旧 target/读出会破坏状态一致性。', '',
                '后续容易受控验证的失败假设：Flow return 噪声方差及 Scalar 蒸馏误差，使 Actor 的 Q 梯度偏离 '
                'Flow 价值。当前只保留 `var_flow_returns`、蒸馏 loss、TD error、Actor loss 的观察，'
                '没有直接比较 dQ/da、做干预或验证因果；不能声称该假设已证实。', '',
                '![Raw loss diagnostics](final/loss_curves.png)', '',
                '已有 FLOQ 论文/代码已经包含 Flow Critic、Scalar 蒸馏、FQL BC/one-step Actor 组合，以及 t=0、'
                'Flow steps、noise coverage、time/return embedding、ResNet 等对照。直接复制这些改动不能作为原创贡献。'
                '与 Value Flows、GFP 或 FAN 的具体组合是否已有相同贡献，本卡没有完成文献差异核验，'
                '必须在下一研究决策中先核查，不能预先声称新颖。', '',
                '未新增第二任务、种子、时长、超参搜索、Matched评价或模块替换训练；Q-Flow 仍 '
                'PAPER_REPRODUCTION_BLOCKED。全部已登记训练/评价结束后暂停监控；不自动扩大实验。']
    save('RESEARCH_ADMISSION.md', '\n'.join(research))


def finalize(root):
    archive = root / 'archive_worktree/reports/floq_min_repro_v2_20261013'
    monitor = read(root / ('receipts/monitor/patrol_' + TAG + '.json'))
    assert not monitor['alerts'], monitor['alerts']
    assert monitor['queue']['status'] == 'ALL_COMPLETE'
    assert not monitor['own_processes'], monitor['own_processes']
    assert monitor['source_commit'] == SOURCE and monitor['source_worktree_clean']
    summary, inventory, episode_table, checkpoints, costs, identities, training = [], [], [], [], [], [], []
    files = read(root / ('receipts/patrol_archive_manifest_' + TAG + '.json'))
    def add_file(relative):
        if relative not in files:
            files.append(relative)
    def file_identity(path, role, seed=None):
        inventory.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), role=role, seed=seed))
    monitored = {job['job']: job for job in monitor['jobs']}
    metric_suffixes = ['critic/floq_loss', 'critic/distilled_critic_loss', 'critic/td_error',
                       'critic/var_flow_returns', 'critic/q', 'actor/bc_flow_loss', 'actor/distill_loss',
                       'actor/q_loss', 'actor/actor_loss']
    curve_fields = ['step'] + [prefix + '/' + key for prefix in ['training', 'validation'] for key in metric_suffixes]
    curve_fields += ['training/grad/norm', 'training/grad/max', 'training/grad/min', 'time/epoch_time', 'time/total_time']
    for seed in [0, 1]:
        run = root / 'runs' / ('formal_seed%d' % seed)
        rel = 'formal/formal_seed%d' % seed
        destination = archive / rel
        status, flags, state, config = [read(run / name) for name in
                                        ['status.json', 'expanded_flags.json', 'initial_state.json', 'actual_config.json']]
        assert status['status'] == 'COMPLETE' and status['returncode'] == 0
        assert status['completed_updates'] == 2000000 and status['source_commit'] == SOURCE
        assert flags['offline_steps'] == 2000000 and flags['online_steps'] == 0 and flags['seed'] == seed
        assert flags['eval_interval'] == 20000 and flags['eval_episodes'] == 50 and flags['save_interval'] == 1000000
        assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == state['config_sha256']
        evaluations, episodes = rows(run / 'native_evaluations.jsonl'), rows(run / 'native_episodes.jsonl')
        expected_steps = [1] + list(range(20000, 2000001, 20000))
        assert [value['step'] for value in evaluations] == expected_steps
        assert len(episodes) == 5050
        grouped = collections.defaultdict(list)
        for episode in episodes:
            grouped[episode['step']].append(episode)
        for evaluation in evaluations:
            batch = grouped[evaluation['step']]
            assert len(batch) == 50 and len({episode['episode'] for episode in batch}) == 50
            assert abs(statistics.mean(episode['success'] for episode in batch) - evaluation['metrics']['success']) < 1e-8
            assert abs(statistics.mean(episode['return_'] for episode in batch) - evaluation['metrics']['episode.return']) < 1e-8
        final = evaluations[-1]
        summary.append(dict(seed=seed, step=2000000, episodes=50,
                            successes=sum(episode['success'] for episode in grouped[2000000]),
                            success=final['metrics']['success'], return_=final['metrics']['episode.return'],
                            episode_length=final['metrics']['episode.length'], protocol='FROZEN_2M_NATIVE_50'))
        for episode in grouped[2000000]:
            episode_table.append(dict(seed=seed, step=2000000, episode=episode['episode'],
                                      success=episode['success'], return_=episode['return_'], length=episode['length'],
                                      initial_observation_sha256=episode['initial_observation_sha256'],
                                      policy_initial_key=json.dumps(episode['policy_initial_key'])))
        indexed = rows(run / 'checkpoint_index.jsonl')
        assert [checkpoint['step'] for checkpoint in indexed] == [1000000, 2000000]
        for checkpoint in indexed:
            path = pathlib.Path(checkpoint['path'])
            assert path.stat().st_size == checkpoint['bytes'] and digest(path) == checkpoint['sha256']
            checkpoints.append(dict(seed=seed, **checkpoint))
        csv_path = pathlib.Path(flags['save_dir']) / 'train.csv'
        with csv_path.open() as stream:
            values = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        assert len(values) == 400 and [value['step'] for value in values] == list(range(5000, 2000001, 5000))
        assert all(math.isfinite(value) for row in values for value in row.values())
        training.append(values)
        table(destination / 'loss_curve.csv', values, curve_fields)
        add_file(rel + '/loss_curve.csv')
        native_curve = []
        for evaluation in evaluations:
            native_curve.append(dict(step=evaluation['step'], success=evaluation['metrics']['success'],
                                     return_=evaluation['metrics']['episode.return'], episodes=50,
                                     duration_seconds=evaluation['duration_seconds'],
                                     result_status='FROZEN_PRIMARY' if evaluation['step'] == 2000000 else 'INTERMEDIATE_NOT_PRIMARY'))
        table(destination / 'native_curve.csv', native_curve,
              ['step', 'success', 'return_', 'episodes', 'duration_seconds', 'result_status'])
        write(destination / 'native_final_evaluation.json', final)
        add_file(rel + '/native_final_evaluation.json')
        for name in ['cpu_time.txt', 'entry_final_state.json', 'environment.json', 'initial_state.json',
                     'actual_config.json', 'expanded_flags.json', 'launch.json', 'status.json', 'training_progress.json',
                     'checkpoint_index.jsonl']:
            shutil.copy2(run / name, destination / name)
            add_file(rel + '/' + name)
        cpu_text = (run / 'cpu_time.txt').read_text()
        def time_value(label):
            return float(next(line.split(':', 1)[1] for line in cpu_text.splitlines() if line.strip().startswith(label)))
        user_seconds, system_seconds = time_value('User time (seconds)'), time_value('System time (seconds)')
        assert time_value('Exit status') == 0
        wall_seconds = status['end_epoch'] - status['start_epoch']
        eval_seconds = sum(evaluation['duration_seconds'] for evaluation in evaluations)
        # Official main.py logs before evaluation. The following 5k window contains that evaluation.
        steady = [value['time/epoch_time'] for value in values if value['step'] > 50000 and
                  value['step'] % 20000 != flags['log_interval']]
        assert all(value > 0 for value in steady)
        cost = dict(seed=seed, gpu=status['gpu'], start_epoch=status['start_epoch'], end_epoch=status['end_epoch'],
                    gpu_residence_hours=status['gpu_residence_hours'], process_lifetime_hours=wall_seconds / 3600,
                    cpu_user_seconds=user_seconds, cpu_system_seconds=system_seconds,
                    cpu_total_hours=(user_seconds + system_seconds) / 3600,
                    mean_cpu_cores=(user_seconds + system_seconds) / wall_seconds,
                    gpu_peak_mib=status['gpu_peak_mib'], process_tree_rss_peak_bytes=status['rss_peak_bytes'],
                    native_evaluation_count=101, native_episode_count=5050, native_evaluation_seconds=eval_seconds,
                    native_eval_median_seconds=statistics.median(e['duration_seconds'] for e in evaluations),
                    steady_logged_update_median_ms=statistics.median(steady) * 1000,
                    steady_logged_update_p10_ms=sorted(steady)[int((len(steady) - 1) * .1)] * 1000,
                    steady_logged_update_p90_ms=sorted(steady)[int((len(steady) - 1) * .9)] * 1000,
                    steady_window_count=len(steady),
                    aggregate_non_eval_updates_per_second=2000000 / (wall_seconds - eval_seconds))
        costs.append(cost)
        identity = dict(seed=seed, source_commit=SOURCE, config_sha256=state['config_sha256'],
                        initial_agent_tree_sha256=state['agent_tree_sha256'], initial_network_sha256=state['network_param_sha256'],
                        matrix_precision=state['matrix_precision'], x64=state['x64'],
                        command=status['command'], environment_overrides=status['environment_overrides'],
                        initial_jax_rng=state['jax_rng'], initial_numpy_state_sha256=state['numpy_state_sha256'],
                        initial_python_state_sha256=state['python_state_sha256'],
                        evaluation_environment=read(run / 'environment.json'), last_training_row=values[-1])
        identities.append(identity)
        for name in ['stdout.log', 'native_episodes.jsonl', 'native_evaluations.jsonl', 'loss_samples.jsonl',
                     'resource_samples.jsonl', 'supplemental_rng_1000000.pkl', 'supplemental_rng_2000000.pkl']:
            file_identity(run / name, name, seed)
        file_identity(csv_path, 'official_training_validation_metrics_csv', seed)
    assert identities[0]['config_sha256'] == identities[1]['config_sha256']
    assert identities[0]['initial_network_sha256'] != identities[1]['initial_network_sha256']
    data = read(root / 'report/dataset_identity.json')
    for entry in data['files']:
        path = pathlib.Path(entry['path'])
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
    formal_gpu_hours = sum(cost['gpu_residence_hours'] for cost in costs)
    formal_wall_hours = (max(cost['end_epoch'] for cost in costs) - min(cost['start_epoch'] for cost in costs)) / 3600
    p1 = read(root / 'runs/p1_seed0/status.json')
    result = dict(created_epoch=time.time(), source_commit=SOURCE, classification='B — FUNCTIONAL_REPRODUCED',
                  primary_protocol='FROZEN_2M_NATIVE_50', seeds=summary,
                  success=aggregate([value['success'] for value in summary]),
                  return_=aggregate([value['return_'] for value in summary]),
                  author_reference=dict(task='antmaze-large-task1(*)', success_percent_mean=94,
                                        success_percent_dispersion=4, seeds=3, aggregation='UNVERIFIED',
                                        source='https://arxiv.org/html/2509.06863v2'),
                  formal_costs=costs, formal_gpu_hours=formal_gpu_hours, formal_wall_hours=formal_wall_hours,
                  formal_cpu_hours=sum(cost['cpu_total_hours'] for cost in costs),
                  p1_gpu_hours=p1['gpu_residence_hours'],
                  p1_plus_formal_gpu_hours=p1['gpu_residence_hours'] + formal_gpu_hours,
                  parameter_origins='report/actual_parameter_origins.csv', dataset_identity=data,
                  identities=identities, anomalies=[], checkpoint_hashes_verified=True,
                  matched_evaluation='NOT_RUN_BY_THIS_FROZEN_CARD',
                  baseline_admission='Engineering and static Critic-interface admission; conditional research baseline',
                  H1='Not confirmed against author aggregation; descriptive two-seed endpoint result only',
                  H2='Static interfaces plus P1 probes verified; no replacement experiment',
                  old_scores_excluded_from_this_git_archive=True,
                  limits=['Two seeds instead of three; 50 episodes per endpoint', 'Author aggregation window unverified',
                          'Native evaluation scenes are not matched to prior FQL/GFP',
                          'GPU residency includes evaluation and shared-card time; not exclusive GPU kernel time',
                          'No causal Critic comparison or novel algorithm experiment'])
    output = archive / 'final'
    output.mkdir(exist_ok=True)
    write(output / 'summary.json', result)
    write(output / 'artifact_index.json', dict(source_commit=SOURCE, checkpoints=checkpoints, server_artifacts=inventory))
    table(output / 'primary_results.csv', summary, ['seed', 'step', 'episodes', 'successes', 'success', 'return_', 'episode_length', 'protocol'])
    table(output / 'primary_episodes.csv', episode_table,
          ['seed', 'step', 'episode', 'success', 'return_', 'length', 'initial_observation_sha256', 'policy_initial_key'])
    table(output / 'costs.csv', costs, list(costs[0]))
    for name in ['summary.json', 'artifact_index.json', 'primary_results.csv', 'primary_episodes.csv', 'costs.csv']:
        add_file('final/' + name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed in [0, 1]:
        curve = rows(root / 'runs' / ('formal_seed%d' % seed) / 'native_evaluations.jsonl')
        steps = [value['step'] / 1e6 for value in curve]
        axes[0].plot(steps, [100 * value['metrics']['success'] for value in curve], label='seed %d' % seed)
        axes[1].plot(steps, [value['metrics']['episode.return'] for value in curve], label='seed %d' % seed)
    axes[0].axhline(94, color='gray', linestyle='--', label='Author mean; aggregation unverified')
    axes[0].set(ylabel='Native success (%)', ylim=(-2, 102))
    axes[1].set(ylabel='Native mean return')
    for ax in axes:
        ax.set(xlabel='Offline updates (millions)', xlim=(0, 2.02))
        ax.axvline(1, color='gray', alpha=.3)
        ax.axvline(2, color='gray', alpha=.3)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('FLOQ Task 1: all native evaluations (50 episodes per point); frozen endpoint = 2M')
    fig.savefig(output / 'native_curves.png', dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, key in zip(axes.flat, metric_suffixes[:4] + ['actor/bc_flow_loss', 'actor/distill_loss']):
        for seed in [0, 1]:
            ax.plot([value['step'] / 1e6 for value in training[seed]],
                    [value['training/' + key] for value in training[seed]], label='seed %d' % seed, linewidth=1)
        ax.set(title=key, xlabel='Offline updates (millions)', xlim=(0, 2.02))
        if 'critic/' in key:
            ax.set_yscale('symlog', linthresh=.01)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Raw logged training diagnostics (5k interval; Critic panels use symlog; no smoothing)')
    fig.savefig(output / 'loss_curves.png', dpi=120)
    plt.close(fig)
    add_file('final/native_curves.png')
    add_file('final/loss_curves.png')
    markdown_reports(root, archive, result, files)
    shutil.copy2(pathlib.Path(__file__), archive / 'scripts/finalize_report.py')
    add_file('scripts/finalize_report.py')
    write(root / 'receipts/final_summary.json', result)
    write(root / ('receipts/patrol_archive_manifest_' + TAG + '.json'), files)
    print(json.dumps(dict(classification=result['classification'], seeds=summary, success=result['success'],
                          return_=result['return_'], formal_gpu_hours=formal_gpu_hours,
                          formal_wall_hours=formal_wall_hours, formal_cpu_hours=result['formal_cpu_hours'],
                          costs=costs, manifest_file_count=len(files))))


if __name__ == '__main__':
    finalize(pathlib.Path(__file__).resolve().parents[1])

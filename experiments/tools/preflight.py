#!/usr/bin/env python3
"""Read local metadata and save a BLOCKED inventory; never certify Stage 0A."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = {
    'torch': '2.7.1+cu126', 'torch-geometric': '2.6.1', 'numpy': '1.26.4',
    'pandas': '2.2.3', 'matplotlib': '3.9.4', 'jsonschema': '4.23.0',
    'PyYAML': '6.0.2', 'tensorboard': '2.19.0', 'pytest': '8.3.5',
    'gymnasium': '1.1.1', 'uv': '0.8.22', 'dashscope': '1.27.6',
}
BINDINGS = [
    ('runtime.target_host', '实际训练主机及资源授权'),
    ('runtime.repository', '当前v2.1实现worktree/commit及入口'),
    ('runtime.simulator_or_robot', '环境及版本'),
    ('runtime.task_assets', 'D0/T_A/T_C真实资产与初始化池'),
    ('runtime.controller', '冻结技能执行器及参数生成'),
    ('runtime.camera', '观测来源、标定和冻结感知'),
    ('runtime.verifier_thresholds', '谓词校准与独立成功评价'),
    ('runtime.skill_timeouts', '技能时限、任务deadline与安全许可'),
    ('runtime.interaction_time_unit', '时钟单位、d_ref和耗时计数'),
    ('vlm.account_region_endpoint', '实际账户、地区、endpoint、SDK调用验证'),
]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--context', default='conversation_sandbox')
    p.add_argument('--repository', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'part_0_validation/stage_0a'
    archive = output / 'preflight_history' / stamp
    archive.mkdir(parents=True, exist_ok=True)
    try:
        sources = json.loads((root / 'manifests/source_manifest.json').read_text(encoding='utf-8'))
        inventory = []
        for item in sources:
            path = root / item['relative_path']
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            inventory.append({**item, 'exists': path.is_file(), 'observed_sha256': digest,
                              'match': digest == item['sha256']})
        observed = {}
        for package in PACKAGES:
            try:
                observed[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                observed[package] = None
        git_hash = None
        git_note = '未绑定项目；没有访问远程仓库。'
        if a.repository is not None:
            result = subprocess.run(['git', '-C', str(a.repository.resolve()), 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, timeout=10, check=False)
            if result.returncode == 0:
                git_hash = result.stdout.strip()
                git_note = '仅读取显式指定本地目录的HEAD；未验证其方法语义或运行入口。'
            else:
                git_note = '指定目录无法读取HEAD；不推测有效commit。'
        probe = {
            'context': a.context, 'scope': 'LOCAL_METADATA_INVENTORY_ONLY',
            'timestamp_utc': now.isoformat(), 'python': platform.python_version(),
            'python_full': sys.version, 'platform': platform.platform(),
            'packages_metadata': observed, 'nvidia_smi_executable': shutil.which('nvidia-smi'),
            'git_hash': git_hash, 'git_note': git_note,
            'imports_executed': False, 'cuda_tensor_test_executed': False,
            'dependency_resolution_executed': False, 'production_tests_executed': False,
            'rl_jobs_executed': 0, 'vlm_requests_executed': 0, 'robot_actions_executed': 0,
            'user_training_host_verified': False, 'source_hashes_all_match': all(i['match'] for i in inventory),
        }
        for directory in [archive, output]:
            write_json(directory / 'environment_probe.json', probe)
            write_json(directory / 'source_inventory.json', inventory)
            with (directory / 'dependency_table.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['package', 'requested', 'observed', 'import_status'])
                writer.writerow(['Python', '3.11.13', platform.python_version(), 'CURRENT_PROCESS_ONLY'])
                for package, expected in PACKAGES.items():
                    writer.writerow([package, expected, observed[package] or 'NOT_INSTALLED', 'NOT_TESTED'])
            with (directory / 'binding_table.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['field', 'requirement', 'status', 'evidence'])
                for field, requirement in BINDINGS:
                    writer.writerow([field, requirement, 'MUST_BIND_OR_AGENT_VERIFY', 'No bound runtime manifest verified'])
        missing = ', '.join(name for name, version in observed.items() if version is None) or '无；但尚未导入核验'
        report = f'''# Stage 0A：本地资料预检\n\n**状态：BLOCKED；范围：{a.context}的元数据盘点，不是用户训练机验收。**\n\n已记录三份输入的SHA-256、当前Python与包metadata，并写入缺项表。没有安装依赖，没有进行生产模型前后向、完整单元测试、VLM请求、RL训练或机器人动作。\n\n当前Python为{platform.python_version()}；torch为{observed['torch'] or '未安装'}；nvidia-smi路径为{probe['nvidia_smi_executable'] or '未发现'}。缺失包metadata：{missing}。这些只描述当前进程所在环境，不能推出用户目标主机的情况。\n\n## 阻塞项\n\n'''
        report += '\n'.join(f'- `{field}`：{description}。' for field, description in BINDINGS)
        report += '\n\n## 下一步\n\n在实际项目环境读取已有绑定并完成0A剩余工作；证据齐全后由Agent更新状态。本预检工具从不自动给出PASS，也不执行0B。论文方法写作不受阻塞。\n'
        for directory in [archive, output]:
            (directory / 'stage_0a_summary.md').write_text(report, encoding='utf-8')
            (directory / 'binding_report.md').write_text(report, encoding='utf-8')
            (directory / 'implementation_inventory.md').write_text(
                '# Implementation inventory\n\n本包是执行计划、配置和资料预检工具，不是已完成的CP-DISR训练仓库。\n'
                'core_code=false（指本包没有生产模型/collector/executor）；用户项目是否已有实现尚未核验。\n'
                '训练入口、相机、控制器、任务资产、缓存、checkpoint均须在实际worktree定位或实现。\n', encoding='utf-8')
        status_path = root / 'stage_status/stage_0a.json'
        previous = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        write_json(archive / 'previous_stage_status.json', previous)
        current = {
            'stage': '0A', 'status': 'BLOCKED', 'method_version': 'CP-DISR-v2.1',
            'git_hash': git_hash, 'started_at': now.isoformat(), 'completed_at': None,
            'last_preflight_at': now.isoformat(), 'scope': 'local_inventory_only',
            'context': a.context, 'runs_completed': [], 'runs_failed': [],
            'selected_tasks': [], 'selected_config': None,
            'issues': [{'field': f, 'issue': d, 'state': 'MUST_BIND_OR_AGENT_VERIFY'} for f,d in BINDINGS],
            'tuning_changes': [], 'next_stage': '0B', 'skipped': False,
            'readiness': {'core_code': False, 'target_host': False, 'runtime': False, 'account': False},
            'production_tests_executed': False, 'rl_jobs_executed': 0,
            'note': '仅本地inventory；不代表完整0A已执行完成。',
        }
        # An inventory must not downgrade an already completed real Stage 0A.
        if previous.get('status') in {'PASS', 'PASS_WITH_NOTES'}:
            write_json(archive / 'proposed_status_not_applied.json', current)
            print('Inventory saved; existing completed Stage 0A status preserved.')
        else:
            write_json(status_path, current)
        print(f'Inventory saved: {output}\nStage 0A remains BLOCKED; no experiments launched.')
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'Preflight failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

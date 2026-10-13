"""Audit an existing monitor snapshot; write receipts and light archive files only.

Run monitor_all.py first and preserve its previous latest.json as
receipts/monitor/previous_<tag>.json. This script never controls training or Git.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import time

import psutil

SOURCE_COMMIT = '3d60e638b42bbef018c56cb1199ff37c3470520d'
BRANCH = 'codex/floq-min-repro-v2-20261013'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def phase(status, progress):
    if status.get('status') in ['TRAINING_RUNNING', 'NATIVE_EVALUATING']:
        return progress.get('phase', status['status'])
    return status.get('status')


def audit(root, tag, previous_close_tag):
    n = read(root / 'receipts/monitor/latest.json')
    b = read(root / ('receipts/monitor/previous_' + tag + '.json'))
    prior = {j['job']: j for j in b['jobs']}
    close = read(root / ('receipts/patrol_final_live_' + previous_close_tag + '.json'))
    close_jobs = {j['job']: j for j in close['jobs']}
    source = root / 'source'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    clean = not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    out, alerts = [], []
    if commit != SOURCE_COMMIT or not clean:
        alerts.append(dict(type='SOURCE_IDENTITY_MISMATCH'))
    for j in n['jobs']:
        p = pathlib.Path(j['path'])
        v, s = j['progress'], j['status']
        e = j['latest_native_evaluation'] or {}
        step = v.get('updates', v.get('step'))
        pv = prior.get(j['job'], {}).get('progress', {})
        before = pv.get('updates', pv.get('step'))
        x = dict(job=j['job'], status=s.get('status'), phase=phase(s, v), updates=step,
                 previous_monitor_updates=before,
                 delta_updates=step-before if before is not None else None,
                 native_last_step=e.get('step'), native_last_episode_count=e.get('episodes'),
                 checkpoint_count=j['checkpoint_index_count'],
                 checkpoint_errors=j['indexed_checkpoint_errors'],
                 nonfinite_loss_samples=j['nonfinite_loss_samples'],
                 last_csv_nonfinite=j['nonfinite_columns_in_latest_training_or_validation_row'],
                 failure=j['failure'], matched_completed_steps=[], invalid_matched_receipts=[],
                 native_episode_errors=[], training_csv_nonfinite_history=[])
        for f in p.glob('matched_*_final.json'):
            k, z = int(f.name.split('_')[1]), read(f)
            ep = p / 'matched' / str(k) / 'episodes.jsonl'
            count = len(ep.read_text().splitlines()) if ep.exists() else 0
            code = z.get('exit_code', z.get('returncode'))
            if code == 0 and count == 100:
                x['matched_completed_steps'].append(k)
            else:
                x['invalid_matched_receipts'].append(dict(step=k, exit_code=code, episodes=count))
        x['matched_completed_steps'].sort()
        flags = read(p / 'expanded_flags.json') if (p / 'expanded_flags.json').exists() else {}
        train_csv = pathlib.Path(flags.get('save_dir', v.get('checkpoint_directory', str(p)))) / 'train.csv'
        if train_csv.exists():
            with train_csv.open() as stream:
                for row in csv.DictReader(stream):
                    for key, value in row.items():
                        try:
                            if not math.isfinite(float(value)):
                                x['training_csv_nonfinite_history'].append(dict(step=row.get('step'), metric=key))
                        except (ValueError, TypeError):
                            pass
        if j['job'] in ['floq-0', 'floq-1']:
            old = close_jobs[p.name]
            x.update(gpu=s.get('gpu'), gpu_peak_mib=s.get('gpu_peak_mib'),
                     rss_peak_bytes=s.get('rss_peak_bytes'), gpu_residence_hours=s.get('gpu_residence_hours'),
                     last_metrics=v.get('last_metrics'), live_pids=[z['pid'] for z in j['live_processes']],
                     previous_close_updates=old['updates'], delta_since_previous_close=step-old['updates'])
            episodes = collections.defaultdict(list)
            for z in rows(p / 'native_episodes.jsonl'):
                episodes[z['step']].append(z)
            evaluations = rows(p / 'native_evaluations.jsonl')
            x.update(native_evaluation_count=len(evaluations),
                     native_raw_episode_count=sum(map(len, episodes.values())))
            for z in evaluations:
                eps = episodes[z['step']]
                if len(eps) != 50 or len({a['episode'] for a in eps}) != 50:
                    x['native_episode_errors'].append(dict(step=z['step'], error='EPISODE_COUNT'))
                elif (abs(sum(a['success'] for a in eps)/50-z['metrics']['success']) > 1e-8 or
                      abs(sum(a['return_'] for a in eps)/50-z['metrics']['episode.return']) > 1e-8):
                    x['native_episode_errors'].append(dict(step=z['step'], error='AGGREGATION'))
            state, cfg = read(p / 'initial_state.json'), read(p / 'actual_config.json')
            x['config_sha256'] = state['config_sha256']
            x['config_hash_matches'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == x['config_sha256']
            if not x['config_hash_matches']:
                alerts.append(dict(job=j['job'], type='CONFIG_HASH_MISMATCH'))
            if step > 1000000 and x['checkpoint_count'] == 0:
                alerts.append(dict(job=j['job'], type='DUE_1M_CHECKPOINT_MISSING'))
        for key in ['checkpoint_errors', 'nonfinite_loss_samples', 'last_csv_nonfinite', 'failure',
                    'invalid_matched_receipts', 'native_episode_errors', 'training_csv_nonfinite_history']:
            if x[key]:
                alerts.append(dict(job=j['job'], type=key))
        out.append(x)
    if n['floq_manager_failure']:
        alerts.append(dict(type='MANAGER_FAILURE'))
    queue = n['floq_queue']
    manager_pid = read(root / 'receipts/manager.json').get('pid', 2630246)
    manager_alive = psutil.pid_exists(manager_pid)
    if queue.get('status') == 'RUNNING' and not manager_alive:
        alerts.append(dict(type='RUNNING_QUEUE_MANAGER_ABSENT'))
    receipt = dict(epoch=n['epoch'], time_server=datetime.datetime.fromtimestamp(n['epoch']).isoformat(),
                   audit_epoch=time.time(), previous_monitor_epoch=b['epoch'], previous_close_epoch=close['epoch'],
                   source_commit=commit, source_worktree_clean=clean, branch=BRANCH, jobs=out,
                   own_processes=n['own_training_evaluation_and_gpu_processes'], gpu_summary=n['gpus'],
                   available_ram_bytes=n['available_ram_bytes'], disk_free_bytes=n['disk_free_bytes'],
                   queue=queue, manager_alive=manager_alive, manager_failure=n['floq_manager_failure'], alerts=alerts,
                   qflow_status=n['qflow_formal_status'], automation_id='lab', automation_status='ACTIVE',
                   next_action='Continue existing jobs; no restart or additional training' if not alerts else 'Report and diagnose preserved evidence',
                   old_scores_excluded_from_this_git_archive=True,
                   phase_note='Entry progress phase is current; manager status may lag its polling interval',
                   checkpoint_schedule_note='FLOQ formal checkpoints at 1M and 2M',
                   snapshot_sha256=hashlib.sha256((root / 'receipts/monitor/latest.json').read_bytes()).hexdigest(),
                   audit_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    write(root / ('receipts/monitor/patrol_' + tag + '.json'), receipt)
    destination = root / 'archive_worktree/reports/floq_min_repro_v2_20261013'
    files = ['patrols/' + tag + '.json', 'CURRENT_STATUS.md', 'RUNNING_REPORT.md',
             'receipts/queue.json', 'scripts/audit_patrol.py']
    write(destination / files[0], receipt)
    write(destination / 'receipts/queue.json', queue)
    shutil.copy2(pathlib.Path(__file__), destination / 'scripts/audit_patrol.py')
    by_job = {j['job']: j for j in n['jobs']}
    for seed in [0, 1]:
        p = root / 'runs' / ('formal_seed%d' % seed)
        rel = 'formal/formal_seed%d' % seed
        d = destination / rel
        shutil.copy2(p / 'status.json', d / 'status.json')
        write(d / 'training_progress.json', by_job['floq-%d' % seed]['progress'])
        with (d / 'native_curve.csv').open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['step', 'success', 'return', 'episodes', 'duration_seconds', 'result_status'])
            for z in rows(p / 'native_evaluations.jsonl'):
                writer.writerow([z['step'], z['metrics']['success'], z['metrics']['episode.return'],
                                 z['episodes'], z['duration_seconds'], 'INTERMEDIATE_NOT_PRIMARY'])
        files.extend(rel + '/' + f for f in ['status.json', 'training_progress.json', 'native_curve.csv'])
    text = ['# FLOQ monitoring handoff', '', 'Receipt snapshot (server time): ' + receipt['time_server'] + '.', '',
            '| Job | Updates | Delta from previous monitor | Phase | Matched checkpoints done |',
            '|---|---:|---:|---|---:|']
    for j in out:
        text.append('| %s | %s | %s | %s | %s |' % (j['job'], j['updates'], j['delta_updates'], j['phase'], len(j['matched_completed_steps'])))
    text.extend(['', 'Inspection alerts: ' + json.dumps(alerts) + '.', '',
                 'Manager alive: %s. Unregistered account training/evaluation/GPU processes: %s.' %
                 (manager_alive, len([z for z in receipt['own_processes'] if not z['associated_job']])), '',
                 'Entry progress gives the current phase; manager status may lag by its polling interval. Evaluation pauses are normal. Checkpoints are scheduled at 1M and 2M. Early curves are intermediate observations, not the frozen 2M primary result.', '',
                 'Available system RAM %.1f GiB; free disk %.1f GiB. Whole-card GPU utilization includes other users.' %
                 (receipt['available_ram_bytes']/2**30, receipt['disk_free_bytes']/2**30), '',
                 'The 30-minute heartbeat remains ACTIVE. Q-Flow remains PAPER_REPRODUCTION_BLOCKED. Prior FQL/GFP scores are excluded from this new-card Git archive.', '',
                 'Full trace: ROOT/receipts/monitor/latest.json; lightweight trace: ' + files[0] + '.'])
    (destination / 'CURRENT_STATUS.md').write_text('\n'.join(text) + '\n')
    report = destination / 'RUNNING_REPORT.md'
    s = report.read_text()
    start, end = s.index('Snapshot '), s.index('\n## P1')
    table = 'Snapshot ' + receipt['time_server'] + ' (server time).\n\n| Run | GPU | updates/2M | phase | own VRAM peak MiB |\n|---|---:|---:|---|---:|\n'
    for j in out:
        if j['job'] in ['floq-0', 'floq-1']:
            table += '| %s | %s | %s | %s | %s |\n' % (j['job'], j['gpu'], j['updates'], j['phase'], j['gpu_peak_mib'])
    report.write_text(s[:start] + table + s[end:])
    write(root / ('receipts/patrol_archive_manifest_' + tag + '.json'), files)
    print(json.dumps(dict(time=receipt['time_server'], alerts=alerts, files=len(files),
                          formal=[{k: j[k] for k in ['job', 'phase', 'updates', 'delta_since_previous_close',
                                  'native_evaluation_count', 'native_raw_episode_count', 'config_hash_matches']}
                                  for j in out if j['job'] in ['floq-0', 'floq-1']])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('previous_close_tag')
    args = parser.parse_args()
    for value in [args.tag, args.previous_close_tag]:
        if len(value) != 13 or value[8] != '_' or not (value[:8] + value[9:]).isdigit():
            raise ValueError('Expected YYYYMMDD_HHMM tag')
    audit(pathlib.Path(__file__).resolve().parents[1], args.tag, args.previous_close_tag)

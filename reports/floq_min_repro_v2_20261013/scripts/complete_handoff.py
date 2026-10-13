"""Record the externally confirmed monitoring pause in completed-run handoffs.

Does not control processes, automations, training, models, source checkout or Git.
The pause receipt must have been produced by the Codex automation API.
"""
import json
import pathlib
import shutil
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'archive_worktree/reports/floq_min_repro_v2_20261013'
TAG = '20261014_0530'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


metadata = read(ROOT / 'receipts/handoff_metadata_0530.json')
pause, schedule = json.loads(metadata['pause']), json.loads(metadata['schedule'])
assert pause['status'] == schedule['status'] == 'PAUSED'
assert pause['automation_id'] == schedule['id'] == 'lab'
proof = read(ROOT / 'receipts/all_jobs_complete_20261014_0530.json')
assert proof['queue_status'] == 'ALL_COMPLETE'
assert not proof['own_training_evaluation_gpu_processes']
assert all(job['status'] in ['COMPLETE', 'P1_PASSED'] for job in proof['jobs'])
manifest_path = ROOT / ('receipts/patrol_archive_manifest_' + TAG + '.json')
manifest = read(manifest_path)
for relative in ['MONITOR_SCHEDULE.json', 'monitor_pause_20261014_0530.json',
                 'final/all_jobs_complete.json', 'scripts/complete_handoff.py']:
    if relative not in manifest:
        manifest.append(relative)
write(DESTINATION / 'MONITOR_SCHEDULE.json', schedule)
write(DESTINATION / 'monitor_pause_20261014_0530.json', pause)
write(ROOT / 'receipts/monitor_pause_20261014_0530.json', pause)
write(DESTINATION / 'final/all_jobs_complete.json', proof)
shutil.copy2(pathlib.Path(__file__), DESTINATION / 'scripts/complete_handoff.py')
audit_path = ROOT / ('receipts/monitor/patrol_' + TAG + '.json')
audit = read(audit_path)
audit.update(automation_status_at_initial_snapshot='ACTIVE', automation_status='PAUSED',
             finalization_epoch=time.time(), finalized=True, closing_account_process_check_epoch=proof['epoch'],
             next_action='All jobs complete; monitor paused; no further experiments authorized')
write(audit_path, audit)
write(DESTINATION / ('patrols/' + TAG + '.json'), audit)
for path in [DESTINATION / 'final/summary.json', ROOT / 'receipts/final_summary.json']:
    summary = read(path)
    summary.update(automation_status='PAUSED', automation_id='lab', completion_inspection_epoch=proof['epoch'])
    write(path, summary)
success = summary['success']
values = summary['seeds']
scores = ' / '.join('seed%d %.1f%%' % (value['seed'], 100 * value['success']) for value in values)
text = ['# Final FLOQ monitoring handoff', '',
        'All registered training/evaluation tasks complete. Monitoring lab is PAUSED; the app update was verified.', '',
        '| Job | Updates | Final phase | Matched checkpoints complete |', '|---|---:|---|---:|']
for job in audit['jobs']:
    text.append('| %s | %s | %s | %s |' %
                (job['job'], job['updates'], job['phase'], len(job['matched_completed_steps'])))
text += ['', 'FLOQ: B - FUNCTIONAL_REPRODUCED. Frozen 2M native endpoints: %s; equal mean %.1f%%. '
         'Two seeds; author aggregation remains unverified.' % (scores, 100 * success['mean']), '',
         'All 1M/2M checkpoint hashes verified; no NaN/Inf, numerical/entry failure or missing native episodes. '
         'No unregistered account training/evaluation/GPU process was found at the closing inspection. '
         'Q-Flow remains PAPER_REPRODUCTION_BLOCKED.', '',
         'Formal GPU residence %.4f h; parallel wall time %.4f h. Full results: PERFORMANCE_REPORT.md; '
         'identity: REPRO_IDENTITY.md; costs: COST_REPORT.md; research conditions: RESEARCH_ADMISSION.md.' %
         (summary['formal_gpu_hours'], summary['formal_wall_hours']), '',
         'Final process inspection epoch: %s. No training process was stopped to close monitoring or SSH. '
         'Prior FQL/GFP scores and old branches were not pushed.' % proof['epoch']]
(DESTINATION / 'CURRENT_STATUS.md').write_text('\n'.join(text) + '\n')
report = DESTINATION / 'RUNNING_REPORT.md'
old = report.read_text()
p1_end = (old.index('## 评价与统计限制') if '## 评价与统计限制' in old else
          old.index('The earlier P1 cost projection is retained'))
p1 = old[old.index('## P1 工程证据'):p1_end]
head = ['# FLOQ-MIN-REPRO-V2 final run receipt', '',
        'Status: COMPLETE; B - FUNCTIONAL_REPRODUCED. Both seeds reached 2M with exit0, '
        '101 native evaluations and verified 1M/2M checkpoints.', '',
        'Official SHA: %s. Branch: %s. ROOT: %s.' % (audit['source_commit'], audit['branch'], ROOT), '',
        'Frozen native 2M success: %s; mean %.1f%%, population std %.2fpp, sample std %.2fpp. '
        'Author aggregation remains unverified; no claim of paper statistical reproduction.' %
        (scores, 100 * success['mean'], 100 * success['std_ddof0'], 100 * success['std_ddof1']), '',
        'See REPRO_IDENTITY.md, PERFORMANCE_REPORT.md, COST_REPORT.md and RESEARCH_ADMISSION.md. '
        'All original curves and artifact identities are retained.', '', p1.strip(), '',
        'The earlier P1 cost projection is retained as a historical estimate. '
        'Measured final costs are in COST_REPORT.md.', '',
        'All four prior FQL/GFP runs and Matched evaluations remain complete. '
        'Their old score artifacts/branch are not pushed here. Q-Flow remains blocked.', '',
        'Heartbeat lab is PAUSED after a fresh account-wide completion inspection. '
        'This does not authorize new seeds/tasks or module replacement training.']
report.write_text('\n'.join(head) + '\n')
write(manifest_path, manifest)
print('HANDOFF_FINALIZED', len(manifest), pause['status'])

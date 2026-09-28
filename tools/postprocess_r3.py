from pathlib import Path
import json, csv, hashlib, shutil

root = Path('/home/xushijie2/graph_cp_disr_v13_r3')
job = root / 'runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94'
rep = root / 'reports/v13_r3'
rep.mkdir(parents=True, exist_ok=True)

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def rel_ckpt(path):
    value = str(path)
    return str(Path(value).relative_to(root)) if value.startswith(str(root)) else value

split = json.loads((root / 'configs/splits/D0_stage_1a.json').read_text())

def relation_count(rec):
    path = root / rec['cache_dir'] / 'final_edges.json'
    if not path.exists():
        path = root / rec['cache_dir'] / 'accepted_relations.json'
    return len(json.loads(path.read_text()))

train = [{'case_id': r['case_id'], 'relation_count': relation_count(r)} for r in split['train']]
dev = [{'case_id': r['case_id'], 'relation_count': relation_count(r)} for r in split['dev']]
train_non = sorted(r['case_id'] for r in train if r['relation_count'] > 0)
dev_non = sorted(r['case_id'] for r in dev if r['relation_count'] > 0)

input_doc = {
    'status': 'FROZEN', 'account': 'xushijie2', 'host': 'gpu03',
    'python': '/home/xushijie2/envs/lerobotpi0-xfs/bin/python',
    'worktree': str(root),
    'source_commit': 'be5454b8165463ba92f8a2ac448fdf3cd89a69e0',
    'baseline_commit': '55a3b7ce35edebbbb8a587fe1e3a98ca1967db07',
    'configsha8': '0c414f94', 'stamp': '20260927T155520Z',
    'H': 23.09999999999752, 'd_ref': 4.5500000000015195,
    'Ncap': 8192, 'Tcap': 37273.60000001245, 'rollout': 1024,
    'ppo_epochs': 4, 'minibatch_target': 64,
    'eval_points_requested': [0, 2048, 4096, 8192],
    'S_train': train_non[:8], 'S_dev': dev_non,
    'train64_natural_nonempty_n': len(train_non), 'dev10_natural_nonempty_n': len(dev_non),
    'cache_total': 74, 'cache_missing': 0, 'cache_legal_empty': 69, 'cache_nonempty': 5,
    'cache_nonempty_train': train_non, 'cache_nonempty_dev': dev_non,
    'test_ids_created': 0, 'vlm_calls': 0,
    'source_input': 'configs/splits/D0_stage_1a.json; exact existing caches; no new cache/scene',
}
write(rep / 'input_audit.json', input_doc)
write(root / 'runs/v13_r3/input_audit_corrected.json', input_doc)

metrics = []
with (job / 'eval_metrics.csv').open() as f:
    for row in csv.DictReader(f):
        row['checkpoint'] = rel_ckpt(row['checkpoint'])
        metrics.append(row)
(rep / 'full_eval_metrics.csv').write_text(
    'update,skill_transitions,success_rate,success_n,mean_discounted_return,checkpoint\n' +
    '\n'.join(','.join(str(row[k]) for k in ['update', 'skill_transitions', 'success_rate', 'success_n', 'mean_discounted_return', 'checkpoint']) for row in metrics) + '\n'
)
for source, target in [('eval_n_000000.json', 'full_eval_n_000000.json'), ('eval_n_002048.json', 'full_eval_n_002048.json')]:
    doc = json.loads((job / source).read_text())
    doc['checkpoint'] = rel_ckpt(doc.get('checkpoint', ''))
    doc.pop('hashes', None)
    write(rep / target, doc)
shutil.copy2(job / 'train_metrics.csv', rep / 'full_train_metrics.csv')

checkpoints = []
for path in sorted((job / 'checkpoints').glob('*.pt')):
    checkpoints.append({'checkpoint': rel_ckpt(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'usable': True})
write(rep / 'checkpoint_index.json', {
    'endpoint_rule': 'last persisted valid checkpoint; no dev-based selection',
    'last_persisted_checkpoint': 'runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94/checkpoints/update_complete_2.pt',
    'last_persisted_N': 2048, 'last_persisted_complete_updates': 2,
    'observed_transition_log_N': 3072, 'checkpoints': checkpoints,
    'weights_not_committed': True,
})

error = json.loads((root / 'runs/v13_r3/hard_error_full_stall.json').read_text())
error['safe_persistence_boundary'] = {
    'N': 2048, 'T': 9771.400000000713, 'complete_updates': 2,
    'checkpoint': 'runs/v13_r3/D0/Full/seed_0/20260927T155520Z_0c414f94/checkpoints/update_complete_2.pt',
}
error['observed_transition_log_N'] = 3072
error['stall_point'] = 'PPO update attempt at N=3072; update_complete_3 was not persisted'
write(rep / 'hard_error_full_stall.json', error)
write(root / 'runs/v13_r3/hard_error_full_stall.json', error)

(rep / 'cost_ledger.csv').write_text(
    'item,authorized,actual,notes\n'
    'new_RL_jobs,2,1,Full started; B2 not started after hard stall\n'
    'new_PPO_updates,16,2 persisted + 1 attempted,attempted update at N=3072 did not persist\n'
    'new_optimizer_steps,NA,NA,not safely recoverable from persisted ledger\n'
    'new_VLM_calls,0,0,no VLM\n'
    'test_ID_episodes,0,0,no test IDs\n'
    'extra_seeds,0,0\n'
    'A_CAT_A_Q_A_B,0,0\n'
    'T_C,0,0\n'
    'observed_training_transitions,NA,3072,transition log; only 2048 persisted in checkpoint\n'
    'observed_training_T,NA,14758.050000000949,at observed N=3072; persisted T=9771.400000000713\n'
)
(rep / 'duration_audit.md').write_text(
    '# R3 duration audit\n\n'
    'The run did not record a complete task-duration interval from episode start to independent first success. '
    '`success_seconds` in the dev JSON is retained as the measured first-success clock value where present, '
    'but no full-task duration is inferred. Terminal-skill duration is not substituted for full-task duration; '
    'unreconstructable fields are `NA`.\n'
)
(rep / 'unresolved_items.md').write_text(
    '# R3 unresolved items\n\n'
    '- Full stalled during the PPO update attempt after the N=3072 transition log boundary; the last persisted checkpoint is N=2048/update 2.\n'
    '- B2 E8 was not started because the authorized run stopped on the Full hard stall; no automatic restart was performed.\n'
    '- Full evaluations at N=4096/N=8192 and all B2 evaluations are unavailable.\n'
    '- Full@Absent and S_train prior diagnostics were not run.\n'
    '- No R3 completion claim or comparative Full/B2 conclusion is made.\n'
)

status = {
    'status': 'FAILED_STALLED', 'phase': 'R3_D0_E8', 'stamp': '20260927T155520Z', 'configsha8': '0c414f94',
    'source_commit': 'be5454b8165463ba92f8a2ac448fdf3cd89a69e0',
    'baseline_commit': '55a3b7ce35edebbbb8a587fe1e3a98ca1967db07',
    'account': 'xushijie2', 'host': 'gpu03', 'python': '/home/xushijie2/envs/lerobotpi0-xfs/bin/python',
    'worktree': str(root), 'failed_method': 'Full', 'B2_started': False,
    'observed_transition_log_N': 3072, 'last_persisted_N': 2048, 'last_persisted_complete_updates': 2,
    'last_persisted_T': 9771.400000000713, 'eval_points_completed': [0, 2048],
    'eval_points_missing': [4096, 8192], 'S_train': train_non[:8], 'S_dev': dev_non,
    'vlm_calls': 0, 'test_episodes': 0, 'r2_b0_touched': False, 'auto_restart': False,
    'reports': 'reports/v13_r3',
}
write(root / 'status/v13_r3.json', status)

summary = (
    '# CP-DISR v1.3 R3 D0 prior comparison — execution summary\n\n'
    'Status: **FAILED_STALLED**.\n\n'
    f'- Execution identity: `xushijie2@gpu03`, Python `/home/xushijie2/envs/lerobotpi0-xfs/bin/python`, isolated worktree `{root}`.\n'
    '- Authorized jobs: `v13_R3_D0_Full_s0_E8` and `v13_R3_D0_B2_s0_E8`; only Full started. B2 was not started.\n'
    '- Full observed transition log: N=3072, T=14758.050000000949. Last persisted boundary: N=2048, T=9771.400000000713, 2 complete PPO updates.\n'
    '- Full stalled during the PPO update attempt at N=3072 for over 30 minutes without progress; process was stopped with evidence preserved and no restart.\n'
    '- Dev10 completed: N=0 = 0/10; N=2048 = 10/10, mean discounted return 0.6350014581. N=4096/N=8192 and all B2 points were not reached.\n'
    f'- Natural non-empty prior cases: S_train={train_non[:8]}; S_dev={dev_non}.\n'
    '- VLM/test-ID/extra seed/A_CAT/A_Q/A_B/T_C: 0. R2/B0 untouched.\n'
    '- No Full-vs-B2 or Full@Absent conclusion is made because the paired run was not completed.\n\n'
    'See `Results_v1.md`, `checkpoint_index.json`, `duration_audit.md`, `cost_ledger.csv`, and `unresolved_items.md`. '
    'Checkpoint binaries remain on the XFS host and are intentionally not committed.\n'
)
(rep / 'r3_summary.md').write_text(summary)
(rep / 'Results_v1.md').write_text(summary)
print('reports_written', rep)

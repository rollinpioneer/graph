"""Archive accounting only; no forward, search, or modifications to old assets."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

START = time.process_time()
ROOT = Path(sys.argv[1])
REG = json.loads((ROOT / 'preregistration.json').read_text())
RUN = json.loads((ROOT / 'run_receipt.json').read_text())
EVENTS = json.loads((ROOT / 'event_results.json').read_text())
REPEAT = json.loads((ROOT / 'repeatability_checks.json').read_text())
ANALYSIS = json.loads((ROOT / 'analysis_receipt.json').read_text())
INPUT = json.loads((ROOT / 'input_manifest.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def save(n, o):
    (ROOT / n).write_text(json.dumps(o, ensure_ascii=False, indent=2) + '\n')


def time_cpu(name):
    p = ROOT / name
    if not p.is_file():
        return None
    s = p.read_text()
    hits = re.findall(r'(?:User|System) time \(seconds\):\s*([0-9.]+)', s)
    return sum(float(v) for v in hits) if len(hits) == 2 else None


bad = [p for p, h in INPUT['remote_file_hashes'].items() if not Path(p).is_file() or sha(Path(p)) != h]
repo = REG['server_repository']
status = subprocess.check_output(['git', '-C', repo, 'status', '--porcelain=v1', '--untracked-files=all'])
head = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
processes = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid():
        continue
    try:
        cmd = (p / 'cmdline').read_bytes().decode(errors='replace').replace('\0', ' ')
        state = (p / 'stat').read_text().split(') ', 1)[1].split()[0]
        if str(ROOT / 'phase_c.py') in cmd:
            processes.append({'pid': int(p.name), 'state': state, 'command': cmd})
    except (OSError, IndexError):
        pass
finalcheck = {'old_input_mismatches': bad, 'server_HEAD': head,
              'server_status_sha256': hashlib.sha256(status).hexdigest(), 'server_status_lines': len(status.splitlines()),
              'active_experiment_runner_processes': [p for p in processes if p['state'] != 'Z'],
              'own_runner_zombies': [p for p in processes if p['state'] == 'Z'], 'checking_pid': os.getpid()}
save('server_final_check.json', finalcheck)
rawmanifest = {}
for p in ROOT.rglob('*'):
    if p.is_file() and p.name not in ['artifact_manifest.json', 'final_receipt.json', 'finalize.log', 'archive_time.txt']:
        rawmanifest[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
save('artifact_manifest.json', {'files': rawmanifest, 'scope': 'All saved artifacts except self/final receipt and volatile finalize.log/archive_time.txt bookkeeping; complete raw scores, all step logs and partial checkpoints included', 'files_count': len(rawmanifest)})
cpu_components = {'runner_OS_seconds': time_cpu('runner_time.txt'), 'analysis_OS_seconds': time_cpu('analysis_time.txt'),
                  'first_archive_OS_seconds': time_cpu('archive_time.txt'),
                  'preflight_seconds': json.loads((ROOT / 'preflight_identity.json').read_text())['cpu_seconds'],
                  'finalization_process_seconds': time.process_time()-START}
measured = sum(v for v in cpu_components.values() if v is not None)
valid = not bad and head == REG['expected_server_HEAD'] and hashlib.sha256(status).hexdigest() == REG['expected_status_sha256'] and not finalcheck['active_experiment_runner_processes']
valid = valid and RUN['counts']['slots_started'] == 8 and all(e['status'] != 'TECHNICALLY_INVALID' for e in EVENTS)
receipt = {'card': REG['card'], 'status': 'COMPLETE_WITH_BOUNDED_UNKNOWNS' if valid else 'COMPLETE_WITH_TECHNICAL_ISSUES',
           'phase_C_actual_runs': RUN['counts']['slots_started'], 'planned_slots': 8,
           'actual_model_state_evaluations': RUN['counts']['actual_model_state_evaluations'],
           'failed_or_interrupted_model_calls': RUN['counts']['failed_model_calls'], 'CPU_components': cpu_components,
           'CPU_measured_seconds': measured, 'CPU_ancillary_reserve_seconds': REG['CPU_ancillary_reserve_seconds'],
           'CPU_conservative_accounting_upper_seconds': measured+REG['CPU_ancillary_reserve_seconds'],
           'CPU_accounting_boundary': 'Measured runner/analysis/finalization plus 300s reserved for SSH/copy/archive/publication/local verification and remaining auxiliary processes; upper is conservative accounting, not observed CPU.',
           'GPU_device_seconds': 0, 'training': 0, 'fitting': 0, 'distance_certification': 0, 'full_graph_builds': 0,
           'precheck_forwards': 0, 'event_results': EVENTS, 'old_inputs_unchanged': not bad,
           'server_checkout_unchanged': head == REG['expected_server_HEAD'] and hashlib.sha256(status).hexdigest() == REG['expected_status_sha256'],
           'residual_runner_processes': finalcheck['active_experiment_runner_processes'],
           'H2_C': 'EVENT_SPECIFIC_ONLY; see event_results', 'H3': 'UNRESOLVED',
           'stop_reason': 'Eight fixed slots consumed; no replacement, extension or R4',
           'preregistered_commit': json.loads((ROOT / 'publication_preregistered.json').read_text())['commit'],
           'publication_branch': 'codex/cp-disr-q2-phase-c-single-pop-v1',
           'SSH_session': 'To close after final evidence transfer; local closure record accompanies publication'}
assert measured+REG['CPU_ancillary_reserve_seconds'] < 14400
save('final_receipt.json', receipt)
print(json.dumps(receipt, ensure_ascii=False))

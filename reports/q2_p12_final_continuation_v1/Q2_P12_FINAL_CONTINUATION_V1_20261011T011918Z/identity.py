"""Read-only identities and bounded resource availability; no model or search."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(2**20), b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    started = time.process_time()
    root = Path(sys.argv[1])
    reg = json.loads((root / 'preregistration.json').read_text())
    manifest = json.loads((root / 'input_manifest.json').read_text())
    failures = []
    for path, expected in manifest['remote_file_hashes'].items():
        if not Path(path).is_file() or sha(path) != expected:
            failures.append(path)
    for relative, expected in reg['code_sha256'].items():
        if sha(root / relative) != expected:
            failures.append('NEW_CODE:' + relative)
    for relative, expected in manifest['new_file_hashes'].items():
        if sha(root / relative) != expected:
            failures.append('NEW_INPUT:' + relative)
    repo = reg['server_repository']
    status = subprocess.check_output(['git', '-C', repo, 'status', '--porcelain=v1', '--untracked-files=all'])
    head = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
    checkout_ok = head == reg['expected_server_HEAD'] and hashlib.sha256(status).hexdigest() == reg['expected_status_sha256']
    output = {'mismatches': failures, 'HEAD': head, 'status_sha256': hashlib.sha256(status).hexdigest(),
              'status_lines': len(status.splitlines()), 'server_checkout_unchanged': checkout_ok,
              'old_inputs_unchanged': not failures, 'process_cpu_seconds': time.process_time()-started,
              'allowed_cpu_ids': sorted(os.sched_getaffinity(0)), 'own_pid': os.getpid(),
              'free_disk_bytes': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize}
    path = root / sys.argv[2]
    assert not path.exists()
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))
    if failures or not checkout_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

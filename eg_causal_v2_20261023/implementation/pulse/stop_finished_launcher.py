"""Stop only our superseded shell launcher; its current evaluator is untouched."""
import json, os, signal
from pathlib import Path

pid = 17616
proc = Path('/proc')/str(pid)
command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
assert 'r14/code/run_cpu_remaining.sh' in command, command
assert proc.stat().st_uid == os.getuid()
os.kill(pid, signal.SIGTERM)
print(json.dumps({'stopped_launcher_pid': pid, 'command': command,
                  'child_evaluators_signalled': False,
                  'reason': 'Source68 evaluator continues; dedicated tail launcher already owns remaining source64. Prevent duplicate scheduling.'}))

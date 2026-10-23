import base64, hashlib, json
from pathlib import Path

workspace = Path(__file__).resolve().parents[3]
queue = workspace/'work/ssh_queue'
out = workspace/'outputs/eg_causal_v2_20261023/r14/code_versions'
out.mkdir(parents=True, exist_ok=True)
rows = []
for number, name in [(213, 'pulse_train_source2.py'), (225, 'pulse_train_remaining.py')]:
    command = json.loads(next(queue.glob(f'{number}-*.request.json')).read_text())[0]['command']
    encoded = command.split("echo '", 1)[1].split("' | base64 -d", 1)[0]
    content = base64.b64decode(encoded)
    (out/name).write_bytes(content)
    rows.append({'file': name, 'request': number, 'sha256': hashlib.sha256(content).hexdigest()})
(out/'manifest.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))

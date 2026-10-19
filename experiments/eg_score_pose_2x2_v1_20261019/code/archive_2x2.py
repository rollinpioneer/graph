import os, json, hashlib, subprocess
from pathlib import Path

R = Path(os.environ['P2_ROOT'])
G = Path(os.environ['GRASP_ROOT'])
V = Path(os.environ['PREV_ROOT'])
P = G / 'eg_pose_select_pilot_v1_20261018'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')
locked = json.loads((V / 'results/locked_method.json').read_text())
identities = {}
for path in [V/'code/evaluate_v2.py', V/'results/artifact_manifest.json', V/'results/locked_method.json', P/'code/core.py', G/'eg_nqs_pilot_v2_20261018/code/common.py']:
    identities[str(path.relative_to(G))] = sha(path)
for head in locked['heads']:
    path = P / f'heads/{head["seed"]}/step{head["step"]}.pt'
    assert sha(path) == head['sha256']
    identities[str(path.relative_to(G))] = sha(path)
save(R/'results/source_identity.json', identities)
manifest = []
for folder in ['code', 'evaluation', 'logs', 'results']:
    for path in sorted((R/folder).rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in ['artifact_manifest.json']:
            continue
        manifest.append(dict(path=str(path.relative_to(R)), bytes=path.stat().st_size, sha256=sha(path)))
save(R/'results/artifact_manifest.json', manifest)

names = ['EXPERIMENT_CARD.md', 'frozen_config.json']
names += [str(p.relative_to(R)) for p in sorted((R/'code').glob('*.py'))]
names += [str(p.relative_to(R)) for p in sorted((R/'results').iterdir()) if p.is_file() and p.name != 'per_frame_results.csv']
names += [str(p.relative_to(R)) for p in sorted((R/'evaluation').rglob('COMPLETE.json'))]
payload = [{'path':'experiments/eg_score_pose_2x2_v1_20261019/'+name, 'mode':'100644', 'type':'blob', 'content':(R/name).read_text()} for name in names]
save(R/'archive_payload.json', payload)
parts = []
current = []
size = 0
for entry in payload:
    n = len(entry['content'].encode())
    if current and size+n > 60000:
        parts.append(current)
        current, size = [], 0
    current.append(entry)
    size += n
if current:
    parts.append(current)
for i, part in enumerate(parts):
    save(R/f'archive_payload_part_{i}.json', part)
save(R/'archive_payload_index.json', dict(files=len(payload), parts=len(parts), content_bytes=sum(len(x['content'].encode()) for x in payload), full_frame_metrics_server_only='results/per_frame_results.csv', full_cache_server_only='evaluation/*/*/*.npz'))

worktree = R/'archive_worktree'
assert subprocess.check_output(['git','-C',str(worktree),'branch','--show-current']).decode().strip() == 'codex/eg-score-pose-2x2-v1-20261019'
assert subprocess.check_output(['git','-C',str(worktree),'rev-parse','HEAD']).decode().strip() == '5b527d72f542cfa9dbc2c7bc5f729e3faa13dfb6'
assert not subprocess.check_output(['git','-C',str(worktree),'status','--porcelain']).strip()
for entry in payload:
    path = worktree/entry['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(entry['content'])
subprocess.run(['git','-C',str(worktree),'add','--sparse','--',*[x['path'] for x in payload]],check=True)
tree = subprocess.check_output(['git','-C',str(worktree),'write-tree']).decode().strip()
save(R/'archive_tree.json', dict(tree=tree, parent='5b527d72f542cfa9dbc2c7bc5f729e3faa13dfb6'))
print(json.dumps(dict(tree=tree, files=len(payload), parts=len(parts), bytes=sum(len(x['content'].encode()) for x in payload))))

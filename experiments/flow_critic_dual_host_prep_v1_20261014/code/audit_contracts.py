"""Source identity, dependency requirements and zero-update artifact audit."""
import hashlib,importlib.metadata as md,json,os,subprocess,sys
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

assets=json.loads((ROOT/'assets_manifest/asset_manifest.json').read_text())
source_checks=[]
for name,item in assets['sources'].items():
 for f in item['files']:
  p=Path(item['path'])/f['path'];assert sha(p)==f['sha256']
 source_checks.append({'source':name,'unchanged_files':len(item['files']),'commit':item['commit']})
for host in ['gfp','dfql']:
 upstream=ROOT/'upstream_snapshots'/host;workspace=ROOT/'workspaces'/f'{host}_floq'
 for p in upstream.rglob('*'):
  if p.is_file():assert sha(p)==sha(workspace/p.relative_to(upstream))
 assert sha(workspace/'floq_value_core/upstream_networks.py')==sha(ROOT/'upstream_snapshots/floq/utils/networks.py')
 # Named core shared byte-for-byte across two separate hosts.
 for p in (ROOT/'workspaces/gfp_floq/floq_value_core').glob('*.py'):
  assert sha(p)==sha(ROOT/'workspaces/dfql_floq/floq_value_core'/p.name)
checkpoint_checks=[]
for item in assets['user_assets']:
 if item.get('path','').endswith('.pkl'):
  assert sha(Path(item['path']))==item['sha256'];checkpoint_checks.append(item['path'])
old_receipts=[]
for file in ['existing_restore_gfp.json','existing_restore_fql.json','existing_restore_floq.json','source_value_mapping.json']:
 d=json.loads((ROOT/'reports'/file).read_text());assert d['status']=='PASS' and d['optimizer_steps']==0
 old_receipts.append(file)
own=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,etimes,pcpu,rss,args'],text=True)
originals=json.loads((ROOT/'receipts/start.json').read_text())
repo=Path('/home/xushijie3/work/graph_cp_disr')
for key,args in {'branch':['branch','--show-current'],'head':['rev-parse','HEAD'],'status':['status','--porcelain']}.items():
 assert subprocess.check_output(['git']+args,cwd=repo,text=True)==originals[key]
result=dict(status='PASS',optimizer_steps=0,source_checks=source_checks,
            checkpoints_unchanged=len(checkpoint_checks),original_restore_and_mapping_receipts=old_receipts,
            original_repository_branch_head_status_unchanged=True,own_processes_at_audit=own,
            namespace_network_sha256=sha(ROOT/'workspaces/gfp_floq/floq_value_core/upstream_networks.py'))
old=Path('/home/xushijie3/work/flow_baselines_preflight_20261013')
identity=json.loads((old/'receipts/source_file_identity.json').read_text())
if isinstance(identity,dict):
 for key in ['files','records']:
  if key in identity:identity=identity[key];break
count=0
for f in identity:
 if f['algorithm'] in ['gfp','fql']:
  p=old/'sources'/f['algorithm']/f['path'];assert sha(p)==f['sha256'];count+=1
result['original_baseline_source_files_unchanged']=count
(ROOT/'reports/zero_update_artifact_audit.json').write_text(json.dumps(result,indent=2))
print('ARTIFACT_AUDIT_PASS',len(checkpoint_checks),source_checks,flush=True)

import hashlib,json,os,platform,re,shutil,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
R=Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1')
E=R/'experiments'; O=E/'part_0_validation/stage_0a'
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=60)
 return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[]
for item in json.loads((E/'manifests/source_manifest.json').read_text()):
 p=E/item['relative_path']; sources.append({**item,'path_from_repository':str(p.relative_to(R)),'observed_sha256':sha(p),'match':sha(p)==item['sha256']})
for p in (R/'inputs').iterdir(): sources.append({'relative_path':str(p.relative_to(R)), 'observed_sha256':sha(p),'match':True,'check_type':'user_input_hash_record'})
(O/'source_inventory.json').write_text(json.dumps(sources,indent=2)+'\n')
probe={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),
 'ssh_target':'xushijie@172.19.164.160','platform':platform.platform(),'os_release':Path('/etc/os-release').read_text(),
 'kernel':platform.release(),'architecture':platform.machine(),'cpu_count':os.cpu_count(),
 'memory':Path('/proc/meminfo').read_text().splitlines()[:3],
 'disk':dict(zip(('total','used','free'),shutil.disk_usage(R))),
 'project_writable':os.access(R,os.W_OK),'git':run(['git','--version']),
 'git_hash':run(['git','-C',str(R),'rev-parse','HEAD'])['stdout'].strip(),
 'git_branch':run(['git','-C',str(R),'branch','--show-current'])['stdout'].strip(),
 'nvidia_smi':run(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,memory.used','--format=csv,noheader']),
 'credential_presence_only':{k:bool(os.environ.get(k)) for k in ['DASHSCOPE_API_KEY','MODEL_STUDIO_API_KEY','ALIBABA_API_KEY']},
 'credentials_scope':'Current non-login SSH process environment only; no secret file contents read',
 'system_python':sys.version,'system_python_executable':sys.executable,
 'container_digest':None,'container_note':'Host execution; no container launched or digest bound',
 'robot_actions':0,'vlm_requests':0,'real_skill_transitions':0,'ppo_updates':0,'performance_episodes':0,
 'source_hashes_all_match':all(s['match'] for s in sources)}
(O/'environment_probe.json').write_text(json.dumps(probe,indent=2)+'\n')
review=[]
repositories=['graph_github_upload','graph_pathgraph_p2c_rl_evaluation_repair_v1_worktree','graph_pathgraph_p1_v6gm1_worktree']
for name in repositories:
 repo=R.parent/name
 files=run(['git','-C',str(repo),'ls-files'])['stdout'].splitlines()
 files=[f for f in files if not f.startswith('.') and not f.endswith('.placeholder.md') and '/site-packages/' not in f]
 method_candidates=[f for f in files if re.search(r'cp.?disr|(^|/)m1/',f,re.I)]
 relevant=[f for f in files if f.endswith('.py') and ('gym_adapter.py' in f or 'tracking_controller.py' in f or 'p1_mainline_grasp_v6gm1/replay_bridge.py' in f or 'p2crl/backends.py' in f)]
 selected=[]
 for f in relevant:
  p=repo/f
  if not p.is_file() or p.stat().st_size>300000: continue
  text=p.read_text(errors='replace')
  lines=[{'line':i,'text':line} for i,line in enumerate(text.splitlines(),1) if re.search(r'shaped_training_reward|MaskablePPO|stable_baselines|class |CUPID|robosuite|libero|reward_v6|def step|def reset|observation_space|action_space',line)]
  selected.append({'path':str(p),'sha256':sha(p),'evidence_lines':lines[:35]})
 review.append({'repository':str(repo),'git_hash':run(['git','-C',str(repo),'rev-parse','HEAD'])['stdout'].strip(),
  'search_scope':'Tracked non-vendor, non-placeholder filenames; targeted source inspection; no legacy execution',
  'cp_disr_path_candidates':method_candidates,'inspected_sources':selected,
  'cp_disr_compatibility':'NOT_CERTIFIED; historical task/reward code is not a v2.1 asset binding'})
assets=[]
for path in ['/home/xushijie/CUPID','/home/__compress_data/xushijie/CUPID','/home/xushijie/LIBERO','/home/__compress_data/xushijie/LIBERO']:
 p=Path(path); assets.append({'path':path,'exists':p.exists(),'resolved_path':str(p.resolve()),'binding':'DISCOVERY_ONLY_NOT_BOUND'})
(O/'legacy_resource_review.json').write_text(json.dumps({'repositories':review,'asset_roots':assets},indent=2)+'\n')
print(json.dumps({'host':probe['hostname'],'git_hash':probe['git_hash'],'source_hashes_all_match':probe['source_hashes_all_match'],'review':review,'asset_roots':assets},indent=2))

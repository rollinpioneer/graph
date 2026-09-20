"""Read-only, bounded source discovery. Never import legacy code or read credentials."""
import ast,hashlib,json,os,re,subprocess
from pathlib import Path
R=Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1')
OUT=R/Path((R/'.remediation_revision').read_text().strip())
roots=[R.parent/'graph_github_upload',R.parent/'graph_pathgraph_p2c_rl_evaluation_repair_v1_worktree',R.parent/'graph_pathgraph_p1_v6gm1_worktree',R.parent/'CUPID/repo',R.parent/'LIBERO']
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=90,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}); return p.stdout.strip() if p.returncode==0 else None
categories={
 'environment_adapter':r'class .*Env|def (reset|step)\(',
 'skill_controller':r'class .*Controller|def (pick|place|open|move|execute_skill|execute)\(',
 'observation_camera':r'def .*observ|camera_names|camera_height|camera_width|camera_depth|agentview',
 'verifier_evaluator':r'def (_check_success|check_success|verify|evaluate_predicate)|class .*Predicate',
 'termination_clock':r'timeout|time_limit|horizon|control_freq|terminated|truncated',
 'ppo_rollout_gru_checkpoint':r'class .*PPO|GRU|MaskablePPO|rollout_buffer|def .*checkpoint|torch.save',
}
records=[]; repos=[]
for root in roots:
 if not root.exists(): continue
 head=command(['git','-C',str(root),'rev-parse','HEAD'])
 status=command(['git','-C',str(root),'status','--porcelain','--untracked-files=no'])
 repo={'path':str(root),'git_commit':head,'dirty_tracked':bool(status) if status is not None else None,'dirty_status_scope':'tracked files only; untracked inventory separate','dirty_status_sha256':hashlib.sha256((status or '').encode()).hexdigest()}
 repos.append(repo)
 files=[]
 for d,dirs,names in os.walk(root):
  rel=Path(d).relative_to(root)
  dirs[:]=[n for n in dirs if not n.startswith('.') and n not in ['node_modules','__pycache__','site-packages','data','datasets','results','logs','outputs','wandb','artifacts','downloads']]
  if len(rel.parts)>6: dirs[:]=[]
  for n in names:
   p=Path(d)/n
   if p.suffix=='.py' and p.stat().st_size<200000 and any(s in str(p.relative_to(root)).lower() for s in ['adapter','controller','executor','environment','envs/','predicate','verifier','perception','camera','ppo','rollout','recurrent','checkpoint','task_registry']): files.append(p)
 for p in sorted(files):
  text=p.read_text(errors='replace'); matches={k:[{'line':i,'text':line.strip()[:260]} for i,line in enumerate(text.splitlines(),1) if re.search(v,line,re.I)][:8] for k,v in categories.items()}; matches={k:v for k,v in matches.items() if v}
  if not matches: continue
  try:
   tree=ast.parse(text); funcs=[{'symbol':node.name,'line':node.lineno,'arguments':ast.unparse(node.args),'return_annotation':ast.unparse(node.returns) if node.returns else 'UNANNOTATED_REQUIRES_ADAPTER_REVIEW'} for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))][:25]
   classes=[node.name for node in ast.walk(tree) if isinstance(node,ast.ClassDef)]
  except SyntaxError: funcs=[];classes=[]
  reasons=[]
  if 'shaped_training_reward' in text or 'reward_v6' in text: reasons.append('Historical shaped reward is incompatible with v2.1 independent terminal-only reward')
  if 'MaskablePPO' in text or 'stable_baselines3' in text: reasons.append('SB3/MaskablePPO is not the frozen recurrent SMDP PPO route')
  if 'sim.data' in text or 'get_body_xpos' in text: reasons.append('Simulator truth must be restricted to independent evaluator/diagnostic, never hidden policy/controller geometry')
  if not reasons: reasons.append('No verified mapping to CP-DISR typed skills, measured observations, three-state evidence, duration and safe termination contracts')
  records.append({'absolute_path':str(p),'repository':repo,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'module':str(p.relative_to(root)).replace('/','.'),'classes':classes,'functions':funcs,'input_output_schema':funcs,'categories':list(matches),'direct_reuse':False,'adapter_required':True,'semantic_differences':reasons,'provenance':'Read source AST and matching lines only; no module import or robot/simulator execution','evidence':matches})
assets=[]
for root in [R.parent/'LIBERO',R.parent/'CUPID',R.parent/'graph_pathgraph_p1_v6gm1_data']:
 if not root.exists():continue
 for d,dirs,names in os.walk(root):
  depth=len(Path(d).relative_to(root).parts); dirs[:]=[n for n in dirs if not n.startswith('.') and n not in ['__pycache__','node_modules','site-packages','logs','wandb']]
  if depth>=6: dirs[:]=[]
  for n in names:
   p=Path(d)/n; low=str(p).lower()
   if p.suffix in ['.bddl','.xml','.pth','.pt','.ckpt'] or (p.suffix in ['.json','.yaml'] and any(x in low for x in ['split','calibration','checkpoint','camera'])):
    stat=p.stat(); assets.append({'absolute_path':str(p),'size_bytes':stat.st_size,'category':'checkpoint' if p.suffix in ['.pth','.pt','.ckpt'] else 'asset_or_split','sha256':hashlib.sha256(p.read_bytes()).hexdigest() if stat.st_size<1000000 else None,'hash_note':'Large checkpoint contents not loaded; not bound','direct_reuse':False,'adapter_required':True,'semantic_differences':'No proven D0/T_A/T_C role/skill/observation/split mapping','provenance':'File metadata only; small nonsecret asset files hashed'})
report={'scope':'Bounded named repositories; not an exhaustive claim that resources do not exist anywhere on server','repositories':repos,'code_candidates':records,'asset_candidates':assets,'secrets_read':False,'runtime_actions':0}
(OUT/'asset_discovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'repositories':len(repos),'code_candidates':len(records),'asset_candidates':len(assets),'output':str(OUT)},indent=2))

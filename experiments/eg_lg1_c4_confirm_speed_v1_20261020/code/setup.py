import os,json,hashlib,shutil,subprocess,datetime
from pathlib import Path
R=Path(os.environ['R']);G=R.parent;L=G/'eg_lg1_local_geometry_v1_20261020';P=G/'eg_sub25_b_f05_repl_v1_20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
for folder in ['results','logs','exports','geometry/C4']: (R/folder).mkdir(parents=True,exist_ok=True)
reserve=G/'eg_b_global_rank_pilot_v1_20261020/C4_reservation.json';rr=json.load(open(reserve));scenes=sum(rr['selected'].values(),[])
assert set(scenes)=={120,108,113,123,156,159,130,133,173,170,167,163}
# Inspect only paths in registered newer experiment output roots; never read C4 metrics before freezing.
roots=[P,L,G/'eg_light_head_d7_diag_v1_20261020',G/'eg_c3_b_top1_rank_diag_v1_20261020',G/'eg_b_global_rank_pilot_v1_20261020']
matches=[];checked=[]
for root in roots:
 for folder in ['exports','evaluation','B/exports','B/evaluation','geometry']:
  pp=root/folder
  if not pp.exists():continue
  checked.append(str(pp))
  for base,dirs,files in os.walk(pp):
   if 'archive_worktree' in dirs:dirs.remove('archive_worktree')
   for d in dirs:
    if d in {f'{s:04d}' for s in scenes}|{f'scene_{s:04d}' for s in scenes}:matches.append(str(Path(base)/d))
assert not matches,matches
write(R/'results/preopen_audit.json',dict(status='PASS_KNOWN_HISTORY_ONLY',reservation_sha256=sha(reserve),selected=rr['selected'],prior_scope=rr['scope'],incremental_paths_checked=checked,matches=matches,authorization='user explicit execute C4 2026-10-20',time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
old=json.load(open(L/'method_lock.json'));assert sha(L/'code/geometry_core.py')==old['geometry_core_sha256']
heads=[v for v in old['new_heads'] if v['family']=='G']
for v in heads+old['old_O_heads']:assert sha(v['path'])==v['sha256']
back=P/'runs/B/train/economicgrasp_epoch10.tar';assert sha(back)=='533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98'
for name in ['common.py','geometry_core.py']:shutil.copy2(L/'code'/name,R/'code'/name)
cfg=dict(experiment='EG-LG1-C4-CONFIRM-SPEED-V1',branch='codex/eg-lg1-c4-confirm-speed-v1-20261020',parent='f9b03f499cb03d453121d15b5e0f3dc983b2d436',C4=rr['selected'],frames=list(range(256)),seeds=[20261018,20261019,20261020],F05=.5,CD=.01,primary='CD',AP_noninferiority_CI95_lower_min=-.5,Top1_one_sided95_lower_exclusive=0,bootstrap=10000,bootstrap_seed=20261020,resources=dict(GPU_process_hours_max=2,CPU_worker_hours_max=40,CPU_workers_max=72,GPU_process_max=6,new_GiB_max=6),speed=dict(after_C4_only=True,prototype=[[57,0],[57,32],[10,0]],validation_scenes=[100,106,115,124,132,139,150,157,160,169,176,186],validation_frames=list(range(0,256,32)),max_implementations=3,CPU_worker_hours_max=2,new_GiB_max=.5,required_median_speedup=2,exact=True))
write(R/'frozen_config.json',cfg)
lock=dict(frozen_config_sha256=sha(R/'frozen_config.json'),geometry_core_sha256=old['geometry_core_sha256'],new_heads=heads,old_O_heads=old['old_O_heads'],F05=.5,CD=.01,backbone_path=str(back),backbone_sha256=sha(back),C4_authorized=True,C4_predictions_opened_at_lock=False,reservation_sha256=sha(reserve),prior_method_lock_sha256=sha(L/'method_lock.json'))
write(R/'method_lock.json',lock)
# Adapt only data location and reuse sources; scientific operations remain unchanged.
s=(L/'code/evaluate.py').read_text().replace("cfg['C3']","cfg['C4']").replace("section='engineering_validation' if a.profile else 'development'","section='engineering_validation' if a.profile else 'confirmation'").replace('len(vv)==20','len(vv)==14')
s=s.replace("source_hashes={v['path']:v['sha256'] for v in json.loads((P/'results/B_artifact_manifest.json').read_text())}","old_heads={v['seed']:FrozenScorer(v['path'],v['sha256']) for v in lock['old_O_heads']}")
start=s.index(" ep=P/f'B/exports/");end=s.index(' rows={',start)
s=s[:start]+''' ep=R/f'exports/scene_{a.scene:04d}/{frame:04d}.npz'
 with np.load(ep) as z:pred=z['pred'];feat=z['feature320'];action=z['original_action'];input_hash=str(z['input_sha256']);observation=str(z['raw_observation_sha256'])
 export_hash=sha(ep);assert pred.shape==(1024,17)
 mu=np.full(1024,np.nan);collision=np.zeros(1024,bool);empty=np.zeros(1024,bool)
 oq={s:head.quality(feat,action) for s,head in old_heads.items()}
 geometry_hash=hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()
 with np.load(R/f'geometry/C4/{a.scene:04d}/{frame:04d}.npz') as zz:raw_geometry=zz['raw'][:,0];assert str(zz['export_sha256'])==export_hash
'''+s[end:]
s=s.replace(" assert float(cd.mean())==old['R0']['cd_rejection_rate']\n",'')
start=s.index("   if name=='EG' or name.startswith('O_'):");end=s.index(' needed=',start);s=s[:start]+s[end:]
s=s.replace(",'Gshuffle_20261018_CD'",'')
s=s.replace("old['R0']['export_sha256']","export_hash").replace("old['R0']['backbone_sha256']","lock['backbone_sha256']")
start=s.index("  if family in ['EG','O']:");end=s.index('  values.append(v)',start);s=s[:start]+s[end:]
s=s.replace('len(values)==20','len(values)==14').replace('old_all_selection_and_metrics_exact=True','frozen_rules_unchanged=True')
(R/'code/evaluate.py').write_text(s)
# Same frozen geometry implementation and precise per-frame identity.
s=(L/'code/geometry.py').read_text().replace("ep=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'","ep=R/f'exports/scene_{scene:04d}/{frame:04d}.npz'").replace("'C3'","'C4'").replace("cfg['C3']","cfg['C4']")
(R/'code/geometry.py').write_text(s)
# New independent branch starts at the published LG1 tree.
w=R/'archive_worktree';subprocess.run(['git','-C',str(L/'archive_worktree'),'worktree','add','-b',cfg['branch'],str(w),cfg['parent']],check=True)
dest=w/'experiments'/R.name;dest.mkdir(parents=True);shutil.copy2(R/'EXPERIMENT_CARD.md',dest/'EXPERIMENT_CARD.md');subprocess.run(['git','-C',str(w),'add','--sparse','--',str(dest/'EXPERIMENT_CARD.md')],check=True)
m=json.load(open(R/'initial_commit.json'));tree=subprocess.check_output(['git','-C',str(w),'write-tree'],text=True).strip();assert tree==m['tree']['sha'],(tree,m['tree'])
for tz in ['+0800','+0000']:
 for suffix in ['', '\n']:
  ts=lambda a:int(datetime.datetime.fromisoformat(a['date'].replace('Z','+00:00')).timestamp())
  raw=('tree '+tree+'\n'+''.join('parent '+a['sha']+'\n' for a in m['parents'])+f"author {m['author']['name']} <{m['author']['email']}> {ts(m['author'])} {tz}\ncommitter {m['committer']['name']} <{m['committer']['email']}> {ts(m['committer'])} {tz}\n\n"+m['message']+suffix).encode()
  if hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==m['sha']:
   subprocess.run(['git','-C',str(w),'hash-object','-t','commit','-w','--stdin'],input=raw,check=True);subprocess.run(['git','-C',str(w),'update-ref','HEAD',m['sha'],cfg['parent']],check=True);break
 else:continue
 break
else:raise AssertionError('Cannot reproduce connector commit')
write(R/'results/initial_freeze.json',dict(status='PASS',method_lock_sha256=sha(R/'method_lock.json'),card_sha256=sha(R/'EXPERIMENT_CARD.md'),registration_commit=m['sha'],time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('FROZEN',m['sha'],sha(R/'method_lock.json'),flush=True)

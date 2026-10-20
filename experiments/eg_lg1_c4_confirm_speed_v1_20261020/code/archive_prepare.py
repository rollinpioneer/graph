from common import *
import shutil,subprocess
w=R/'archive_worktree';prefix=Path('experiments')/R.name;dest=w/prefix
assert subprocess.check_output(['git','-C',str(w),'branch','--show-current'],text=True).strip()=='codex/eg-lg1-c4-confirm-speed-v1-20261020'
files=['EXPERIMENT_CARD.md','RESOURCE_AMENDMENT.md','frozen_config.json','method_lock.json','C4_CONFIRMATION_REPORT.md','SPEED_REPORT.md','FINAL_REPORT.md','HANDOFF.md']
files+=['code/'+p.name for p in sorted((R/'code').glob('*')) if p.suffix in ['.py','.cpp']]
files+=['results/'+p.name for p in sorted((R/'results').glob('*.json')) if p.name not in ['server_artifact_index.json','geometry_C4.json','pipeline_status.json'] and not p.name.startswith('export_')]
files+=['results/per_scene_results.csv','results/per_scene_differences.csv','results/top1_category_transitions.csv','results/speed_per_frame.csv']
for relative in files:
 p=R/relative;assert p.exists(),relative;q=dest/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
subprocess.run(['git','-C',str(w),'add','--sparse','--',*[str(prefix/x) for x in files]],check=True)
staged=subprocess.check_output(['git','-C',str(w),'diff','--cached','--name-only'],text=True).splitlines();assert all(p.startswith(str(prefix)+'/') for p in staged)
parent=subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD'],text=True).strip();tree=subprocess.check_output(['git','-C',str(w),'write-tree'],text=True).strip();base=subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD^{tree}'],text=True).strip()
entries=[dict(path=str(prefix/x),mode='100644',type='blob',content=(R/x).read_bytes().decode('utf-8')) for x in files]
out=R/'archive_payload';out.mkdir(exist_ok=True);batches=[];group=[];size=0
for v in entries:
 n=len(v['content'].encode())
 if group and size+n>42000:batches.append(group);group=[];size=0
 group.append(v);size+=n
if group:batches.append(group)
for i,v in enumerate(batches):write(out/f'batch{i}.json',v)
write(out/'metadata.json',dict(parent=parent,base_tree=base,expected_tree=tree,batches=len(batches),files=files,staged=staged,bytes=sum(len(v['content'].encode()) for v in entries)))
print('ARCHIVE_PREPARED',parent,tree,len(files),len(batches),flush=True)

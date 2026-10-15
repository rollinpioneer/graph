import ast,csv,json,hashlib,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def save(f,d):(R/f).write_text(json.dumps(d,indent=2,ensure_ascii=False))
U=R/'sources/gfp';methods={'GFP':'gfp-actor','ReBRAC':'rebrac','FQL':'fql-actor','VaBC':'gfp-vabc','GFP-AWR':'gfp-advantage-actor'};tables={};raw=[]
for method,name in methods.items():
 data=list(csv.DictReader((U/'results/per_task'/f'{name}_per_task_results.csv').open()));tables[method]={}
 for x in data:
  vals=np.array(ast.literal_eval(x['all_values']),float);m=float(x['mean']);std=float(x['std']);assert abs(m-vals.mean())<1e-8
  ddof=next((i for i in [0,1] if abs(std-vals.std(ddof=i))<1e-7),None);assert ddof is not None
  row=dict(method=method,task=x['task_name'],benchmark=x['benchmark'],mean=m,std=std,n=len(vals),std_ddof=ddof,seed_values=vals.tolist());raw.append(row);tables[method][x['task_name']]=row
keys=sorted(set().union(*(t.keys() for t in tables.values())));wide=[]
for t in keys:
 row={'task':t}
 for m in methods:
  x=tables[m].get(t);row[m]=None if x is None else x['mean'];row[m+'_std']=None if x is None else x['std'];row[m+'_n']=None if x is None else x['n']
 row['best_other_minus_GFP']=max((row[m] for m in ['ReBRAC','FQL','VaBC','GFP-AWR'] if row[m] is not None),default=float('nan'))-row['GFP'] if row['GFP'] is not None else None;wide.append(row)
with (R/'reports/official_task_gaps.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(wide[0]),lineterminator='\n');w.writeheader();w.writerows(wide)
save('reports/official_task_values.json',raw)
s=(R/'sources/ogbench/locomaze/maze.py').read_text();tree=ast.parse(s);maze=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='MazeEnv');maps=[];tasks=[]
for n in ast.walk(maze):
 if isinstance(n,ast.If) and "self._maze_type == 'medium'"==ast.unparse(n.test):
  for v in n.body:
   if isinstance(v,ast.Assign) and isinstance(v.targets[0],ast.Name):
    if v.targets[0].id=='maze_map':maps=ast.literal_eval(v.value)
    if v.targets[0].id=='tasks':tasks=ast.literal_eval(v.value)
assert len(maps)==8 and len(tasks)==5;grid=np.array(maps)
def bfs(goal):
 d=np.full(grid.shape,-1,int);d[goal]=0;q=collections.deque([goal])
 while q:
  i,j=q.popleft()
  for a,b in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
   if 0<=a<8 and 0<=b<8 and grid[a,b]==0 and d[a,b]<0:d[a,b]=d[i,j]+1;q.append((a,b))
 return d
geo=[]
for k,(start,goal) in enumerate(tasks,1):
 d=bfs(goal);path=[start]
 while path[-1]!=goal:
  i,j=path[-1];path.append(next((a,b) for a,b in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)] if 0<=a<8 and 0<=b<8 and d[a,b]==d[i,j]-1))
 eu=float(np.linalg.norm(np.array(start)-goal));geo.append(dict(task=k,start_ij=start,goal_ij=goal,start_xy=(np.array(start)[::-1]*4-4).tolist(),goal_xy=(np.array(goal)[::-1]*4-4).tolist(),bfs_cells=int(d[start]),euclidean_cells=eu,detour_ratio=float(d[start]/eu),one_shortest_path=path,bfs_to_goal=d.tolist()))
save('reports/geometry.json',dict(maze_map=maps,maze_unit=4,offset=4,tasks=geo))
fig,axes=plt.subplots(1,2,figsize=(10,4.6))
for ax,k in zip(axes,[1,4]):
 g=geo[k-1];ax.imshow(grid,cmap='Greys',vmin=0,vmax=1,origin='lower');p=np.array(g['one_shortest_path']);ax.plot(p[:,1],p[:,0],'-o',color='#178b9a',lw=2,ms=4);a,b=np.array(g['start_ij']),np.array(g['goal_ij']);ax.plot([a[1],b[1]],[a[0],b[0]],'--',color='#e08437');ax.scatter([a[1]],[a[0]],s=90,color='green',label='Start');ax.scatter([b[1]],[b[0]],s=100,marker='*',color='red',label='Goal');ax.set_title(f"Task {k}: BFS={g['bfs_cells']}, straight={g['euclidean_cells']:.2f}\ndetour ratio={g['detour_ratio']:.2f}");ax.set_xlabel('Grid column (4m per cell)');ax.set_ylabel('Grid row');ax.legend(loc='upper right')
fig.tight_layout();fig.savefig(R/'reports/geometry.png',dpi=170);plt.close(fig)
print('FOCUS_ROWS')
for x in wide:
 if x['task'].startswith('humanoidmaze_medium'):print(x)
print('GEOMETRY',[(x['task'],x['bfs_cells'],x['euclidean_cells'],x['detour_ratio']) for x in geo])

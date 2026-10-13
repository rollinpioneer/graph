"""Design-time map enumeration only: no policy, simulator, training, or remote access."""
from collections import deque
from pathlib import Path
import json

MAP = [
[1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,0,0,0,1,0,0,0,0,0,1],
[1,0,1,1,0,1,0,1,0,1,0,1],
[1,0,0,0,0,0,0,1,0,0,0,1],
[1,0,1,1,1,1,0,1,1,1,0,1],
[1,0,0,1,0,1,0,0,0,0,0,1],
[1,1,0,1,0,1,0,1,0,1,1,1],
[1,0,0,1,0,0,0,1,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1],
]
VERTICES = [(i,j) for i,row in enumerate(MAP) for j,v in enumerate(row) if v==0]
VSET=set(VERTICES)
ADJ={s:sorted((s[0]+di,s[1]+dj) for di,dj in ((-1,0),(1,0),(0,-1),(0,1)) if (s[0]+di,s[1]+dj) in VSET) for s in VERTICES}
DIST={}
for s in VERTICES:
    d={s:0}; q=deque([s])
    while q:
        v=q.popleft()
        for w in ADJ[v]:
            if w not in d: d[w]=d[v]+1; q.append(w)
    DIST[s]=d
STARTS=[(1,1),(7,4),(1,8),(3,1),(3,6),(3,10),(5,6),(5,8)]
rows=[]
for s in STARTS:
    for distance in [2,4,6,8,10,12,14]:
        goals=[g for g in VERTICES if DIST[s][g]==distance]
        unique=[g for g in goals if sum(DIST[n][g]==distance-1 for n in ADJ[s])==1]
        rows.append({'start_ij':list(s),'bfs_edges':distance,'all_goal_cells':len(goals),'unique_shortest_first_exit_cells':len(unique),'multiple_shortest_first_exit_cells':len(goals)-len(unique)})
result={'purpose':'Design feasibility only; no model/simulation result.',
        'source_repo':'johnlyzhou/saw',
        'source_commit':'3535a16c901e5de1233ac5a586401b3ddfa5e30d',
        'source_file':'ogbench/locomaze/maze.py',
        'grid_coordinates':'zero-based (row, column); four-neighbour free-cell graph',
        'free_cells':len(VERTICES),'diameter_edges':max(max(x.values()) for x in DIST.values()),
        'rows':rows,
        'limit':'Counts exclude continuous geometry and physics feasibility. Unique means shortest-reference first exit, not the sole feasible route.'}
Path('/mnt/data/saw06_support/GEOMETRY_PREFLIGHT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('design preflight rows',len(rows))
print('Task1 unique counts:',[r['unique_shortest_first_exit_cells'] for r in rows if r['start_ij']==[1,1]])

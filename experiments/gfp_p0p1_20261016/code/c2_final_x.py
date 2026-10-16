"""Post-hoc check: where do task4 episodes that never leave the start cell end? (run in out/cdiag)

West wall of the start cell (6,5) is at x = 14; cell centre x = 16.
"""
import json, glob, numpy as np
res = {}
for f in glob.glob('c2_*.json'): res.update(json.load(open(f)))
for arm in ['A0','A1','A4','A6','A2','A5','A3','F0','F2','F3']:
    for grp,names in (('G012',['G_task4_s0','G_task4_s1','G_task4_s2']),('F0123',['F_task4_s0','F_task4_s1','F_task4_s2','F_task4_s3']),('G3',['G_task4_s3'])):
        eps=[e for k,v in res.items() if v['arm']==arm and v['name'] in names for e in v['episodes']]
        stay=[e for e in eps if e['min_bfs']>=e['start_bfs']]
        fx=np.array([e['final_xy'][0] for e in stay]) if stay else np.array([np.nan])
        print(arm,grp,'n',len(eps),'stay',len(stay),'stay_final_x median %.2f'%np.median(fx),'frac x<15 %.2f'%np.mean(fx<15), 'len %.0f'%np.mean([e['length'] for e in eps]))

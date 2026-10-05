import json,glob,collections
sp=json.load(open('configs/splits/struct_gen_v1_train_dev.json'))
cell={c['case_id']:c['cell'] for c in sp['dev']}
print('dev cells', collections.Counter(cell.values()))
spec=json.load(open('runs/final_master/2.1.1/structgen/prep/structural_generalization_spec.json'))
print(list(spec.keys()))
for k in ('cells','depth','depth_by_cell','dev','test','train','axes'):
    if k in spec: print(k, json.dumps(spec[k])[:1500])
res={}
reas=collections.defaultdict(collections.Counter)
for d in sorted(glob.glob('runs/final_master/2.1.1/T_B/*/seed_*/R-TB-SG-*')):
    run=d.split('R-TB-SG-')[1].split('-2026')[0]
    e=json.load(open(d+'/eval_final.json'))
    t=collections.defaultdict(lambda:[0,0])
    for r in e['rows']:
        c=cell[r['case_id']]; t[c][1]+=1; t[c][0]+=int(r['success'])
    res[run]={c:'%d/%d'%tuple(v) for c,v in sorted(t.items())}
    tt=json.load(open(d+'/eval_struct_test_final.json'))
    for r in tt['rows']:
        if not r['success']: reas[run][r['reason'].split(':')[0]]+=1
for k,v in res.items(): print(k,v)
for k,v in reas.items(): print('TESTFAIL',k,dict(v))

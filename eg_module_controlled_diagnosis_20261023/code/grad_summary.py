from pathlib import Path
import csv,json,collections,numpy as np
R=Path.home()/'work/grasp/eg_module_controlled_diagnosis_20261023';d=collections.defaultdict(list)
for p in (R/'gradient').glob('*_routes.csv'):
 for r in csv.DictReader(p.open()):d[(r['loss'],r['block'])].append(float(r['weighted_norm']))
for (l,b),v in d.items():
 if l in ['angle','depth','width','score','view','graspness','objectness'] and (b in ['backbone','neighborhood','grasp_head.conv_score','grasp_head.conv_score_feature']):print(l,b,'nonzero',sum(x>1e-12 for x in v),'median',float(np.median(v)))
print('replay_max',max(json.loads(p.read_text())['gradient']['gradient_sum_relative_L2'] for p in (R/'receipts').glob('train*.json')))

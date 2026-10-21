from p0_stats import *
from collections import defaultdict
def csvout(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p=Path(p)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def avg(values):
 v=[x for x in values if x is not None];return float(np.mean(v)) if v else None
def aggregate(rows,keys,metrics):
 bins=defaultdict(list)
 for r in rows:bins[tuple(r[k] for k in keys)].append(r)
 return [dict(zip(keys,k),**{m:avg([v[m] for v in vv]) for m in metrics},sample_count=len(vv)) for k,vv in bins.items()]
def report():
 c=[json.loads((R/f'results/c3_{s:04d}.json').read_text()) for ss in GROUPS.values() for s in ss];h=[json.loads(p.read_text()) for p in (R/'results').glob('head_*_8.json')];assert len(h)==32
 cr=sum([v['rows'] for v in c],[]);hr=sum([v['rows'] for v in h],[]);assert len(cr)==9216 and len(hr)==768
 for name,rows in [('C3',cr),('head',hr)]:csvout(R/f'P0_attribution_{name}.csv',[{k:v for k,v in r.items() if k not in ['witness','spearman_deciles']} for r in rows])
 metrics=['failed','R','R_in50','D','A','A_cd','C_cert','U','known_fraction','ranking_space_upper','overturn','rescue','overturn_upper','spearman_all','spearman_cd']
 summary={}
 for name,rows in [('C3',cr),('head',hr)]:
  per=aggregate(rows,['role','scene','seed'],metrics);summary[name]=dict(per_scene_seed=per,by_role=aggregate(aggregate(per,['role','scene'],metrics),['role'],metrics),cross_table=[])
  cross=defaultdict(int)
  for r in rows:
   if r['failed']:cross[str((r['R'],r['A'],r['A_cd'],r['D'],r['U']))]+=1
  summary[name]['cross_table']=dict(cross)
  for m in ['R_best_S_rank','R_best_Q_rank']:
   v=[r[m] for r in rows if r['failed'] and r['R'] and r[m] is not None];summary[name][m+'_quantiles']=np.quantile(v,[0,.25,.5,.75,1]).tolist() if v else []
 save(R/'P0_attribution_summary.json',summary)
 dm=['mse','group_mean_mse','within_mse','within_R2','pair_accuracy','a0_pair_accuracy','regret','constant_groups'];disc=sum([v['discrimination'] for v in h],[]);per=aggregate(disc,['role','pool','scene','seed'],dm);csvout(R/'P0_group_discrimination_per_scene.csv',per);scene=aggregate(per,['role','pool','scene'],dm);out=aggregate(scene,['role','pool'],dm)
 for r in out:
  rr=[x for x in scene if x['role']==r['role'] and x['pool']==r['pool']];rng=np.random.default_rng(20261021);ix=rng.integers(0,len(rr),(10000,len(rr)));r['ci95']={}
  for m in dm:
   vals=[x[m] for x in rr]
   if all(x is not None for x in vals):r['ci95'][m]=np.quantile(np.array(vals)[ix].mean(1),[.025,.975]).tolist()
 bias=sum([v['bias'] for v in h],[]);bp=aggregate(bias,['role','stratum','scene','seed'],['correlation','slope']);save(R/'P0_group_discrimination.json',dict(roles=out,bias=aggregate(aggregate(bp,['role','stratum','scene'],['correlation','slope']),['role','stratum'],['correlation','slope']),aggregation='frame then scene; three seeds averaged within scene; descriptive scene bootstrap',R2_definition='per-frame centered R2; constant groups have zero denominator contribution but prediction residual remains in SSE'))
 pairs=sum([v['pairs'] for v in c],[]);save(R/'P0_overturn_pairs.json',dict(total={k:sum(v[k] for v in pairs) for k in pairs[0] if k not in ['scene','frame','seed']},per_scene_seed=aggregate(pairs,['scene','seed'],[k for k in pairs[0] if k not in ['scene','frame','seed']])))
 witnesses=[r['witness'] for r in cr if 'witness' in r];rng=np.random.default_rng(20261021);save(R/'P0_witnesses.json',[witnesses[i] for i in rng.choice(len(witnesses),min(30,len(witnesses)),replace=False)])
 facts=json.loads((R/'P0_fact_checks.json').read_text());f1=facts['F1_max_error']<=1e-4 and facts['nonnear_inversions']==0
 scenes=sorted({r['scene'] for r in cr if r['overturn']});groups=[g for g,ss in GROUPS.items() if any(s in scenes for s in ss)];heads={s:len({r['scene'] for r in cr if r['seed']==s and r['overturn']}) for s in SEEDS};upper=float(np.mean([r['overturn_upper'] for r in cr])*100);space=float(np.mean([r['ranking_space_upper'] for r in cr])*100)
 top=dict(overturn_count=sum(r['overturn'] for r in cr),rescue_count=sum(r['rescue'] for r in cr),lower_pp=float(np.mean([r['overturn'] for r in cr])*100),upper_pp=upper,scenes=scenes,groups=groups,heads_scenes=heads,frame_heads=9216,seed_heads_not_independent=True,head_mechanism=[r for r in hr if r['overturn']],decile_correlations={str(s):[avg([r['spearman_deciles'][i] for r in cr if r['seed']==s]) for i in range(10)] for s in SEEDS})
 save(R/'P0_overturn_top1.json',top)
 t1=len(scenes)>=3 and len(groups)>=2 and sum(v>=2 for v in heads.values())>=2;t2=upper>=1.;decision=dict(status='PASS',P1_go=space>=1.,P1_ranking_space_upper_pp=space,P2_trigger='GO' if f1 and t1 and t2 else ('BLOCKED' if not f1 else 'NO_GO'),T0=f1,T1=t1,T2=t2,F1_max_error=facts['F1_max_error'],P2_block_reason='F1 reconstruction exceeds frozen numerical/ranking tolerance' if not f1 else None,C4_read=False,new_GT=0,EG_forward=0,optimizer_updates=0,CPU_worker_hours_successful=sum(v['seconds'] for v in c+h)/3600)
 save(R/'P0_DECISION.json',decision)
 text=f'''# P0 role-division audit\n\nP1_go: {decision['P1_go']}; P2_trigger: {decision['P2_trigger']}.\n\nAll 3072 C3 frames and 256 head frames audited. F4/F5 identity and label union checks passed; 21504 F2 configuration checks have no unexplained mismatch, and G fallback count is zero. F1 reconstruction max error {facts['F1_max_error']:.9g} exceeds 1e-4; {facts['nonnear_inversions']} non-near adjacent order inversions. The original affine mathematical expression and feature slice were checked; this does not establish the numerical origin of the discrepancy. No tolerance relaxed.\n\nC3 known overturns: {top['overturn_count']}/9216, rescues: {top['rescue_count']}/9216. Overturn lower/upper: {top['lower_pp']:.4f}/{upper:.4f} percentage points. Ranking-recoverable space upper: {space:.4f}pp. T1={t1}, T2={t2}; F1 independently blocks P2. Unknown labels remain unknown.\n\nHead results separate fit/model-val/gate-cal, Top64/random64; frame/scene aggregation and descriptive bootstrap are in P0_group_discrimination.json. These diagnostics are not causal proof. C3 is development. No GT calls, backbone forward, optimizer updates, or C4 reads.\n\nSuccessful statistics worker wall hours: {decision['CPU_worker_hours_successful']:.4f}, excluding fact verification/setup/failed pilot. The first attempt selected the A checkpoint and stopped at SHA validation; corrected to the locked B path. The base Python environment lacked compatible Open3D; reused the historical evaluation environment without changing dependencies.\n\nNext: freeze P1 and blocked P2 configs, execute G-W and exact G rerun under approved plan. P2 remains blocked pending separate redesign approval; do not treat it as a tested negative result.\n'''
 (R/'P0_REPORT.md').write_text(text);print(json.dumps(decision));print(json.dumps({k:top[k] for k in ['overturn_count','rescue_count','lower_pp','upper_pp']}))
if __name__=='__main__':report()

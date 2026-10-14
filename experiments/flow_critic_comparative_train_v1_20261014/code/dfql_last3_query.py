"""Read-only last-three DFQL comparison; never executes optimizer updates."""
import pathlib,json,hashlib,math,re,time,statistics,csv,sys
r=pathlib.Path(__file__).resolve().parents[1]
label=sys.argv[1] if len(sys.argv)>1 else '20261014_1413'
a=json.loads((r/'receipts/monitor/audit_latest.json').read_text())
assert not a['issues'],a['issues']
card={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['EXPERIMENT_CARD.md','reports/ALL_EIGHT_PARALLEL_AMENDMENT.md','reports/all_eight_launch_verification.json']}
assert card==json.loads((r/'receipts/monitor/card_identity_20261014_1358.json').read_text())
rows=[];summaries=[];validations=[]
for j in ['D0_seed0','D0_seed1','D1_seed0','D1_seed1']:
 p=r/'runs'/j;s=json.loads((p/'status.json').read_text());ex=json.loads((p/(s['phase']+'_exit.json')).read_text())
 assert s['status']=='COMPLETED' and s['updates']==1000000 and s['exit_code']==0 and ex['exit_code']==0 and ex['status']=='PASS'
 for protocol in ['native','matched_verified']:
  prs=[]
  for step in [800000,900000,1000000]:
   f=p/protocol/(str(step)+('_Actor.json' if protocol=='native' else '.json'))
   if not f.exists():continue
   d=json.loads(f.read_text());es=d['episodes'];count=50 if protocol=='native' else 100
   assert len(es)==count and all(math.isfinite(e['success']) and math.isfinite(e['return_']) for e in es)
   suc=d['stats']['success'] if protocol=='native' else d['success_rate']
   ret=d['stats']['episode.return'] if protocol=='native' else d['mean_return']
   assert math.isclose(suc,statistics.mean(e['success'] for e in es),abs_tol=1e-12) and math.isclose(ret,statistics.mean(e['return_'] for e in es),abs_tol=1e-10)
   row={'job':j,'seed':int(j[-1]),'model':'DFQL' if j.startswith('D0') else 'DFQL+floq','protocol':protocol,'step':step,'episodes':count,'success_rate':suc,'mean_return':ret,'path':str(f.relative_to(r)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
   prs.append(row);rows.append(row)
   if protocol=='matched_verified':
    assert d['status']=='PASS' and d['optimizer_updates']==0 and d['global_rng_control_verified']
    assert all(e['episode']==n and all(e[k]==20261014+n for k in ['reset_seed','numpy_seed','python_seed']) and e['policy_seed']==420000+n and re.fullmatch('[0-9a-f]{64}',e['initial_observation_sha256']) for n,e in enumerate(es))
    ids=list((p/'output').rglob('params_%d.identity.json'%step));assert len(ids)==1
    ident=json.loads(ids[0].read_text());assert ident['checkpoint_sha256']==d['checkpoint_sha256'] and ident['state_sha256']==d['state_sha256']
    paired=None
    if j.startswith('D1'):
     bd=json.loads((r/'runs'/('D0_seed'+j[-1])/'matched_verified'/(str(step)+'.json')).read_text())
     paired=[e['initial_observation_sha256'] for e in es]==[e['initial_observation_sha256'] for e in bd['episodes']];assert paired
    validations.append({'job':j,'step':step,'status':'PASS','optimizer_updates':0,'paired_initial_observations_match':paired,'checkpoint_state_identity':True,'raw_statistics_match':True})
  summaries.append({'job':j,'model':'DFQL' if j.startswith('D0') else 'DFQL+floq','seed':int(j[-1]),'protocol':protocol,'available_steps':[v['step'] for v in prs],'complete_last3':len(prs)==3,'last3_success_mean':statistics.mean(v['success_rate'] for v in prs) if len(prs)==3 else None,'last3_return_mean':statistics.mean(v['mean_return'] for v in prs) if len(prs)==3 else None})
aggregate={}
for protocol in ['native','matched_verified']:
 ag={}
 for model in ['DFQL','DFQL+floq']:
  ss=[v for v in summaries if v['protocol']==protocol and v['model']==model]
  if len(ss)==2 and all(v['complete_last3'] for v in ss):
   scores=[v['last3_success_mean'] for v in ss];returns=[v['last3_return_mean'] for v in ss]
   ag[model]={'two_seed_mean_success':statistics.mean(scores),'two_seed_sample_sd_success':statistics.stdev(scores),'two_seed_range_success':[min(scores),max(scores)],'two_seed_mean_return':statistics.mean(returns)}
  else:ag[model]={'status':'PENDING_COMPLETE_BOTH_SEEDS'}
 aggregate[protocol]=ag
matched_complete=all(v['complete_last3'] for v in summaries if v['protocol']=='matched_verified')
result={'epoch':time.time(),'task':'cube-double-play-singletask-task2-v0','status':'DFQL_LAST3_COMPLETE' if matched_complete else 'NATIVE_COMPLETE_MATCHED_PARTIAL','steps':[800000,900000,1000000],'averaging':'equal last3 checkpoint mean, then equal seed mean; no best checkpoint selection','optimizer_updates_this_query':0,'rows':rows,'summaries':summaries,'aggregates':aggregate,'matched_validations':validations,'frozen_documents':card,'progress':{j:{k:v.get(k) for k in ['updates','stage','gpu']} for j,v in a['runs'].items()},'limits':['two seeds, single task','native50episodes not controlled matched starts','matched100episodes per checkpoint; missing full sets remain pending','early inline matched excluded']}
(r/'reports'/('dfql_last3_query_'+label+'.json')).write_text(json.dumps(result,indent=2))
with (r/'reports'/('dfql_last3_query_'+label+'.csv')).open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=['# DFQL last-three checkpoint comparison: interim query','', 'Task cube-double-play-singletask-task2-v0; snapshot '+time.strftime('%Y-%m-%d %H:%M:%S CST',time.localtime(a['monitor_epoch']))+'. Both original and combination seeds trained exactly1M, native exit0/PASS. GFP combinations continue.','']
for protocol,title in [('native','Native: official serial50episodes'),('matched_verified','Matched:100episodes, common starts and policy RNG')]:
 report+=['## '+title,'','| Model | Seed |800k success|900k success|1M success|Last3 mean|Last3 Return|','|---|---:|---:|---:|---:|---:|---:|']
 for j in ['D0_seed0','D0_seed1','D1_seed0','D1_seed1']:
  rr=[v for v in rows if v['job']==j and v['protocol']==protocol];s=next(v for v in summaries if v['job']==j and v['protocol']==protocol);vals={v['step']:v for v in rr}
  cells=[('%.1f%%'%(100*vals[step]['success_rate'])) if step in vals else 'PENDING' for step in [800000,900000,1000000]]
  report.append('|%s|%d|%s|%s|%s|%s|%s|'%(s['model'],s['seed'],*cells,'%.2f%%'%(100*s['last3_success_mean']) if s['complete_last3'] else 'PENDING','%.2f'%s['last3_return_mean'] if s['complete_last3'] else 'PENDING'))
 report.append('')
report+=['Native two-seed last3 success mean original46.67% vs combination4.00%, delta-42.67pp. Native Return original-595.09 vs combination-886.00; less-negative is better. Native original evaluator uses50episodes and does not enforce common starts.', '', 'Complete seed1 matched last3 success original20.00% vs combination4.00%, delta-16.00pp; Return-760.94 vs-918.91.100 paired initial-observation hashes match at each available combination checkpoint; RNG, zero optimizer updates, raw statistics and native checkpoint/state identities are independently verified. Seed1 single1M success2% vs11% does not replace the frozen three-checkpoint average.', '', 'Missing seed0 matched checkpoint sets remain PENDING. Never average an incomplete set as a final result. Two native seeds show no benefit in this task; preserve poor-but-finite results and do not tune/restart. This does not establish general algorithm superiority or identify the cause.', '', 'Early inline DFQL matched diagnostics are excluded. No new experiment, optimizer update, stop/restart/migration or algorithm change. Q-Flow stays PAPER_REPRODUCTION_BLOCKED. Overall final report awaits remaining GFP combinations and all matched evaluations.']
(r/'reports'/('DFQL_LAST3_QUERY_'+label+'.md')).write_text('\n'.join(report)+'\n')
print('NATIVE_AGGREGATES',aggregate['native'])
print('MATCHED_STATUS',[(v['job'],v['complete_last3'],v['last3_success_mean']) for v in summaries if v['protocol']=='matched_verified'])
print('PAIRED_VALIDATION_PASS',len(validations))

import json,csv,pathlib,collections,itertools
r=pathlib.Path(__file__).resolve().parents[1];rows=json.load(open(r/'reports/branch_results.json'));group=collections.defaultdict(dict)
for x in rows:group[x['state']][x['candidate']]=x
summary=[]
for sid,g in group.items():
 a=g['original_actor'];b=g['resampled_actor'];v=g['vabc'];top=max(g,key=lambda x:g[x]['q_mean']);topt=max(g,key=lambda x:g[x]['target_q_mean'])
 summary.append(dict(state=sid,category=a['category'],seed=a['seed'],original_success=a['success'],resampled_success=b['success'],vabc_success=v['success'],online_top=top,target_top=topt,online_top_success=g[top]['success'],rescued=not a['success'] and bool(b['success'] or v['success'])))
with open(r/'reports/state_summary.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
fail=[x for x in summary if x['category'] in ('near_goal','pre_stall')]
out=dict(states=len(summary),failure_states=len(fail),rescued_states=sum(x['rescued'] for x in fail),actor_resample_success=sum(x['resampled_success'] for x in fail),vabc_success=sum(x['vabc_success'] for x in fail),rescues_not_ranked_first_by_online_critic=[x['state'] for x in fail if x['rescued'] and not x['online_top_success']],restoration_exact=all(all(v==0 for v in x['restore_max_errors'].values()) for x in rows if x['candidate']=='original_actor'),max_model_updates=0)
(r/'reports/probe_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
print('DISTRIBUTION')
for d in json.load(open(r/'reports/proposal_distributions.json')):
 print(d['state'],[(k,round(v['action_std_mean'],3),round(v['clipped_coordinate_fraction'],3),round(v['q_mean'],2)) for k,v in d['draws'].items()])

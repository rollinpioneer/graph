import pathlib,json,hashlib,collections
R=pathlib.Path.cwd(); P=pathlib.Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
src={(s,k):json.loads((P/f'runs/G0_seed{s}/matched_verified/{k}.json').read_text()) for s in (0,1) for k in (800000,900000,1000000)}
ep={(s,k):{e['episode']:e for e in d['episodes']} for (s,k),d in src.items()}
def rank(s,i):return hashlib.sha256(f'GFP-BADCASE-V1|{s}|{i}'.encode()).hexdigest()
def strat(s,ok,n,preferred=()):
 pools=collections.defaultdict(list)
 for i,e in ep[s,1000000].items():
  if bool(e['success'])==ok:pools[tuple(int(ep[s,k][i]['success']) for k in (800000,900000))].append(i)
 for p in pools.values():p.sort(key=lambda i:(i not in preferred,rank(s,i)))
 chosen=[]
 while len(chosen)<n:
  for key in sorted(pools):
   if pools[key] and len(chosen)<n:chosen.append(pools[key].pop(0))
 return chosen
fail={s:strat(s,False,12) for s in (0,1)}
success={s:strat(s,True,6,fail[1-s]) for s in (0,1)}
cases=[]
def add(s,i,k,kind,delta=0):
 ref=ep[s,k][i]
 c=dict(id=f's{s}_k{k}_e{i:03d}_{kind}',model_seed=s,episode=i,checkpoint=k,kind=kind,reference=ref,
 policy_seed=ref['policy_seed']+delta,history=[int(ep[s,j][i]['success']) for j in (800000,900000,1000000)],
 checkpoint_sha256=src[s,k]['checkpoint_sha256'],state_sha256=src[s,k]['state_sha256'])
 cases.append(c)
for s in (0,1):
 for i in fail[s]:add(s,i,1000000,'failure')
 for i in success[s]:add(s,i,1000000,'success')
for s in (0,1):
 eligible=[i for i in fail[s] if any(ep[s,k][i]['success'] for k in (800000,900000))]
 for j,i in enumerate(eligible[:3]):
  ks=[k for k in (800000,900000) if ep[s,k][i]['success']]
  add(s,i,ks[j%len(ks)],'earlier')
 for i in fail[s][:3]:add(s,i,1000000,'rng',1000000)
gate=[c['id'] for s in (0,1) for kind,ids in [('failure',fail[s][:2]),('success',success[s][:2])] for c in cases if c['model_seed']==s and c['kind']==kind and c['episode'] in ids]
assert len(gate)==8 and len(cases)<=48
out=dict(card='GFP-BADCASE-TRAJECTORY-V1',task='humanoidmaze-medium-navigate-singletask-task1-v0',parent_root=str(P),
 selection='800k/900k outcome-stratified round-robin; SHA256 fixed rank; success pools prioritize other-seed selected failures; earlier one successful checkpoint per scene; RNG +1000000',
 population={str(s):dict(n=100,failures=sum(not e['success'] for e in ep[s,1000000].values())) for s in (0,1)},
 gate_ids=gate,cases=cases,optimizer_updates_allowed=0,max_episodes=48)
(R/'configs/selection.json').write_text(json.dumps(out,indent=2))
print('SELECTION',len(cases),'GATE',gate)
print('COUNTS',dict(collections.Counter(c['kind'] for c in cases)))
for c in cases:
 if c['kind']=='earlier':print('EARLIER',c['id'],c['history'])


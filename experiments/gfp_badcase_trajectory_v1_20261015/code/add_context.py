import json,pathlib,collections,hashlib
R=pathlib.Path(__file__).resolve().parents[1];P=pathlib.Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
out=[]
for seed in (0,1):
 ds={k:json.loads((P/f'runs/G0_seed{seed}/matched_verified/{k}.json').read_text())['episodes'] for k in (800000,900000,1000000)}
 for k in (800000,900000):
  cnt=collections.Counter((int(a['success']),int(b['success'])) for a,b in zip(ds[k],ds[1000000]))
  same=all(a['episode']==b['episode'] and a['initial_observation_sha256']==b['initial_observation_sha256'] and a['policy_seed']==b['policy_seed'] for a,b in zip(ds[k],ds[1000000]))
  out.append(dict(seed=seed,earlier=k,n=100,same_initial_and_policy_rng=same,
   early_success_late_failure=cnt[1,0],early_failure_late_success=cnt[0,1],both_failure=cnt[0,0],both_success=cnt[1,1]))
(R/'reports/full_evaluation_transitions.json').write_text(json.dumps(out,indent=2))
rows=json.loads((R/'reports/case_metrics.json').read_text())
defs=dict(low_head='head height <0.8m for >=40 steps, after >=20 steps with head >1.2m and torso zz >0.6; posture proxy, not official fall',
 stall='For each trailing 200-step position window, bounding-box diagonal <1m; flag if this condition persists for >=100 consecutive window endpoints. Not a unique causal onset.',
 near_goal='Minimum horizontal distance <=2m; actual success tolerance is <=0.5m',
 final_low_head='At least 60% of final400 state samples have head height <0.8m',
 final_wall='At least25% of final400 control steps end with a wall contact; not substep-complete',
 reversal='At least4 immediate returns in raw grid-cell transition sequence; includes boundary jitter, not a verified wrong turn',
 path_separation='Euclidean separation >2m for >=40 steps between time-aligned trajectories; can reflect speed difference',
 counts='Overlapping descriptive labels within selected cases, never population prevalence')
(R/'reports/metric_definitions.json').write_text(json.dumps(defs,indent=2))
ident=json.loads((R/'reports/asset_identity.json').read_text())
for a in ident:a['config_sha256']=hashlib.sha256(json.dumps(a['config'],sort_keys=True).encode()).hexdigest()
(R/'reports/asset_identity.json').write_text(json.dumps(ident,indent=2))
files=['workspaces/gfp_floq/agents/gfp.py','workspaces/gfp_floq/agents/floq_adapter.py','workspaces/gfp_floq/training_wiring.py','code/matched_verified.py']
(R/'reports/source_identity.json').write_text(json.dumps({f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files},indent=2))


from p2_residual import *
cfg=json.loads((R/'frozen_config.json').read_text());d=dict(np.load(RP/'prepared.npz'));geo=dict(np.load(L/'prepared_geometry.npz'));assert sha(RP/'prepared.npz')==cfg['protected_prepared_sha256'];assert sha(L/'prepared_geometry.npz')==cfg['protected_geometry_sha256']
heads=gheads(cfg);first=heads[SEEDS[0]].ck
for h in heads.values():assert np.array_equal(h.ck['mean'],d['mean']) and np.array_equal(h.ck['std'],d['std'])
T=np.asarray([token(f,first) for f in d['feature']]);Y=utility(d['mu']).astype(np.float64);S=[];logs={s:[] for s in SEEDS};man={v['path']:v['sha256'] for v in json.loads((G/'eg_role_p0_audit_v1_20261021/P0_input_manifest.json').read_text())}
for i,(scene,frame) in enumerate(zip(d['scene'],d['frame'])):
 p=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz';assert sha(p)==man[str(p)]
 with np.load(p) as z:
  assert np.array_equal(z['feature320'][d['ids'][i]],d['feature'][i]);S.append(z['pred'][d['ids'][i],0].astype(float))
 for seed,h in heads.items():
  x=np.concatenate([(d['feature'][i]-h.ck['mean'])/h.ck['std'],d['action'][i,:,0],geo['geometry'][i,:,0]],axis=-1).astype(np.float32)
  with torch.no_grad():logs[seed].append(h.net[:-1](torch.from_numpy(x)).squeeze(-1).numpy())
S=np.asarray(S);fit=d['role']=='fit';assert fit.sum()==192;cal={}
for fam,target in [('Eact',Y[:,:,0]),('Epot',Y.max(-1))]:
 alpha=float(target[fit].std()/S[fit].std());beta=float(target[fit].mean()-alpha*S[fit].mean());assert alpha>0
 cal[fam]=dict(alpha=alpha,beta=beta,target_mean=float(target[fit].mean()),target_std=float(target[fit].std()),score_mean=float(S[fit].mean()),score_std=float(S[fit].std()),fit_groups=int(fit.sum()*128))
np.savez_compressed(R/'head_inputs.npz',T=T,S=S,Y=Y,role=d['role'],scene=d['scene'],frame=d['frame'],ids=d['ids'],collision=d['collision'],mu=d['mu'],**{f'logit_{s}':np.asarray(v) for s,v in logs.items()})
write(R/'results/readout_init.json',dict(calibration=cal,delta_w=[0.]*64,delta_b=0.,trainable_parameters=65,head_inputs_sha256=sha(R/'head_inputs.npz'),normalization_source=cfg['protected_G'],initial_fit_only=True,config_sha256=sha(R/'frozen_config.json')))
print(json.dumps(cal))

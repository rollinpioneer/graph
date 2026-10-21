import json,hashlib,difflib,shutil,time
from pathlib import Path
B=Path.home()/'work/grasp';P0=B/'eg_role_p0_audit_v1_20261021';L=B/'eg_lg1_local_geometry_v1_20261020';P=B/'eg_sub25_b_f05_repl_v1_20261019';R=B/'eg_gw_p1_v1_20261021';P2=B/'eg_l1_readout_p2_v1_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
decision=json.loads((P0/'P0_DECISION.json').read_text());assert decision['P1_go']
for r in [R,P2]:
 r.mkdir(exist_ok=False)
 for d in ['code','results','logs','heads','evaluation']: (r/d).mkdir()
 shutil.copyfile(P0/'EXPERIMENT_CARD.md',r/'EXPERIMENT_CARD.md')
# Complete the declared cache manifest with head geometry, code, source identities,
# and C3 geometry needed by the already-approved subsequent evaluation.
m={v['path']:v for v in json.loads((P0/'P0_input_manifest.json').read_text())}
paths=[P/'results/backbone_status.json',Path(json.loads((P/'results/backbone_status.json').read_text())['checkpoint']),L/'results/preparation.json',L/'results/training_summary.json',B/'eg_decision_relation_v1_20261021/results/result_identity.json']
paths+=list((L/'geometry/head').glob('*/*.npz'))+list((L/'geometry/C3').glob('*/*.npz'))
paths+=list((B/'eg_decision_relation_v1_20261021/results/scenes').glob('*.json'))
for root in [L/'code',B/'eg_lqr_build_confirm_v3_20261019/code',B/'eg_lg1_c4_confirm_speed_v1_20261020/code',B/'eg_decision_relation_v1_20261021/code']:
 paths+=list(root.glob('*.py'))
for p in paths:m[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
(P0/'P0_input_manifest.json').write_text(json.dumps(list(m.values()),indent=2))
src=(L/'code/train_head.py').read_text();gw=src.replace("choices=['G','Gshuffle']","choices=['GW','Gshuffle']").replace('loss=((pred-yf[ix].cuda())**2).mean()','loss=(((pred-yf[ix].cuda())**2)*pred.new_tensor([.5,.1,.1,.1,.1,.1])).sum(-1).mean()')
assert gw!=src
(R/'code/train_head.py').write_text(src);(R/'code/train_gw.py').write_text(gw);shutil.copyfile(L/'code/common.py',R/'code/common.py')
(R/'code/GW.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),gw.splitlines(True),fromfile='G',tofile='GW')))
for dest,source in [(R/'prepared_geometry.npz',L/'prepared_geometry.npz'),(R/'geometry',L/'geometry')]:dest.symlink_to(source)
shutil.copyfile(L/'results/preparation.json',R/'results/preparation.json')
cfg=dict(phase='P1',approval='User: 执行',approved_items=['A1','A2','A3','A4','A5','A6','A7'],freeze_time=time.time(),plan_sha256=sha(P0/'EXPERIMENT_CARD.md'),P0_input_manifest_sha256=sha(P0/'P0_input_manifest.json'),P0_decision_sha256=sha(P0/'P0_DECISION.json'),C3=dict(seen=[100,106,115,124],similar=[132,139,150,157],novel=[160,169,176,186]),seeds=[20261018,20261019,20261020],weights=[.5,.1,.1,.1,.1,.1],steps=1000,batch_groups=128,optimizer='AdamW',lr=.001,weight_decay=.0001,GPU='A100',F05=.5,CD=.01,bootstrap_seed=20261021,bootstrap_samples=10000,gate=dict(AP_delta=.5,Top1_delta_pp=1.,AP_CI_lower_min_other=-.2,Top1_CI_lower_min_other_pp=-.5,positive_seeds=2,collision_protection_pp=.5),G_rerun=dict(byte_exact_or_maxQ_le_1e5_excluded_from_C3=True,max_Q_tolerance=1e-5,otherwise_additional_route_CI_lower_gt_zero=True),C4_read=False,new_training_labels=0,new_evaluation_labels_allowed=True,storage_cap_GiB=4,train_G_source_sha256=sha(R/'code/train_head.py'),train_GW_sha256=sha(R/'code/train_gw.py'),difference_sha256=sha(R/'code/GW.diff'))
(R/'frozen_config.json').write_text(json.dumps(cfg,indent=2))
p2=dict(cfg,phase='P2',status='BLOCKED',block_reason=decision['P2_block_reason'],families=['E_act','E_pot','G-cap65'],parameters_each=65,CPU_only=True,initialization='positive affine original readout fit moments; G-cap zero',targets=['u0','max6','u0'],sequential_order=['E_act-G','E_act-E_pot','E_act-G-cap65'],only_E_ACT_FULL_counts=True,I1_max_F05_error=1e-9)
(P2/'frozen_config.json').write_text(json.dumps(p2,indent=2));(P2/'results/sequential_decision.json').write_text(json.dumps(dict(status='BLOCKED_F1',training_started=False,C3_evaluated=False,reason=decision['P2_block_reason']),indent=2))
print(json.dumps(dict(P1=str(R),P2=str(P2),manifest_files=len(m),P1_config_sha=sha(R/'frozen_config.json'),P2_config_sha=sha(P2/'frozen_config.json'))))

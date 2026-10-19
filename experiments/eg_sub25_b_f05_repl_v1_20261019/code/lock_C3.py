from repl_common import *
assert json.loads((R/'results/export_head_B.json').read_text())['status']=='PASS'
record=json.loads((R/'results/B_heads.json').read_text());assert record['status']=='PASS' and len(record['heads'])==3
locks={};allheads={};_,hb=backbone('B')
for bone in ['A','B']:
 root=R/bone;root.mkdir(exist_ok=True)
 if bone=='A':heads=[dict(seed=h['seed'],sha256=h['sha256'],path=str(G/f'eg_pose_select_pilot_v1_20261018/heads/{h["seed"]}/step1000.pt')) for h in P['A_heads']]
 else:heads=[{k:h[k] for k in ['seed','sha256','path']} for h in record['heads']]
 for h in heads:assert sha256(Path(h['path']))==h['sha256']
 _,bh=backbone(bone);allheads[bone]=heads
 config=dict(heads=heads,backbone_sha256=bh,budget=dict(new_bytes=P['budget']['new_bytes']),exploration={},protocol_sha256=sha256(R/'protocol.json'))
 write(root/'frozen_config.json',config);write(root/'C3_reservation.json',dict(selected=P['C3'],audit_sha256=sha256(R/'results/C3_exposure_audit.json')))
 lock=dict(selected=P['C3'],config_sha256=sha256(root/'frozen_config.json'),reservation_sha256=sha256(root/'C3_reservation.json'),method='F05 only',checkpoint_rule='epoch10 and step1000',C3_performance_opened=False)
 write(root/'C3_method_lock.json',lock);locks[bone]=sha256(root/'C3_method_lock.json')
for path,h in json.loads((R/'results/execution_code_manifest.json').read_text()).items():assert sha256(Path(path))==h,path
write(R/'C3_method_lock.json',dict(selected=P['C3'],protocol_sha256=sha256(R/'protocol.json'),backbones=dict(A=P['A_backbone_sha256'],B=hb),heads=allheads,locks=locks,execution_code_manifest_sha256=sha256(R/'results/execution_code_manifest.json'),C3_performance_opened=False,time=time.time()))
print('C3_METHOD_LOCKED_BEFORE_ANY_C3_PERFORMANCE',flush=True)

import json,hashlib,datetime
from pathlib import Path
W=Path(__file__).parent;O=W.parents[2]/'outputs/eg_causal_v2_20261023/r14'
src=O.parent/'pulse_preregistration.json';p=json.loads(src.read_text(encoding='utf-8'))
p['status']='authorized_execution_frozen'
p['authorization']={'scope':'exact R1.4 bounded pulse only','user_answer':'授权仅执行上述 R1.4 微更新（推荐）','parameter_updates_cap':280,'analytic_cap':131072,'short_confirmation':False,'C3_C4_C5':False,'git_write':False}
p['preregistration_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
p['execution_notes']={'GPU_hosts':['172.19.164.245','172.19.164.248'],'CPU_primary_host':'172.19.164.161','stages':['no-update wiring preflight','source2 registered pulse and first CPU job','remaining registered sources in parallel','paired scene statistics and cleanup'],'no_extra_pilot_update':'source2 pilot counts as one of8; reuse saved outputs and receipts; do not repeat updated branches','replay':'historical TF32 heldout baseline; FP32 train gradients/Adam updates; original Nvalid fixed','actual_step_norm_matching_relative_tolerance':0.001,'forward_invariance':'same fixed inputs/RNG/snapshot and raw per-sample losses across all7 arms before first step; no treatment modifies forward values','unknowns':'do not classify unknown as negative; stop job and record','max_evaluation_jobs':448,'new_pose_calls_per_job_cap':228,'evaluator_preflight_max_calls':16,'conservative_direct_call_cap_after_old_full_label_reuse':102160,'statistics':'scene primary; aggregate source pulses within scene, report shared-fold dependence; do not countsteps/arms as samples'}
O.mkdir(parents=True,exist_ok=True);(O/'protocol_R14_frozen.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
print(O)

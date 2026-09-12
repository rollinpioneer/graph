from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from .protocol import PulseLevel,pulse_levels
from .simulator import ControlledForcedDropTabletop
from .intervention import make_force_pulse,run_force_pulse
from .trace_recorder_v2 import snapshot
from .physical_reference import evaluate_loss_trace
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

def _prehold(sim, family, seed):
    z0=float(sim.object_xyz[2]); sim.perform('approach_object'); sim.perform('close_gripper'); sim.perform('lift')
    anchor=None; rows=[]
    for i in range(10):
        sim.physics_step(); row=snapshot(sim,step=i,phase='pre_hold',family_id=family,seed=seed,pre_hold_verified=True)
        if anchor is None: anchor=np.asarray(row['object_in_gripper_position'])
        row['relative_drift_m']=float(np.linalg.norm(np.asarray(row['object_in_gripper_position'])-anchor)); rows.append(row)
    rise=float(sim.object_xyz[2]-z0); drift=max(r['relative_drift_m'] for r in rows)
    ok=all(r['weld_active'] and r['numeric_health']['passed'] for r in rows) and rise>=.03 and drift<=.01
    sim.pre_hold_verified=ok
    return rows,{'pre_hold_verified':ok,'height_rise_m':rise,'max_relative_drift_m':drift,'sample_count':len(rows)}

def _run_instance(root:Path, family:str, seed:int, level:PulseLevel|None, label:str):
    root.mkdir(parents=True,exist_ok=True); spec=family_spec(family,'normal_pick_place',seed,seed,probe_variant='default'); sim=ControlledForcedDropTabletop(spec,seed)
    pre,pre_summary=_prehold(sim,family,seed); trace=list(pre); force_start=None
    if sim.pre_hold_verified:
        if level is None:
            sim.disable_weld_for_intervention(); sim.forward()
        else:
            body=int(sim.mujoco.mj_name2id(sim.model,sim.mujoco.mjtObj.mjOBJ_BODY,'gripper')); rot=sim.data.xmat[body].reshape(3,3); pulse=make_force_pulse(float(sim.model.body_mass[sim.mujoco.mj_name2id(sim.model,sim.mujoco.mjtObj.mjOBJ_BODY,'object')]),rot,level); force_start=float(sim.data.time); trace.extend([snapshot(sim,step=len(trace),phase='post_weld_off_pre_force',family_id=family,seed=seed)])
            trace.extend([snapshot(sim,step=len(trace),phase='force_pulse_started',family_id=family,seed=seed)])
            pulse_runner=run_force_pulse; pulse_rows=pulse_runner(sim,pulse); trace.extend([snapshot(sim,step=len(trace),phase='force_pulse_step',family_id=family,seed=seed,pre_hold_verified=sim.pre_hold_verified) for _ in pulse_rows])
        for i in range(75):
            sim.physics_step(); trace.append(snapshot(sim,step=len(trace),phase='observation',family_id=family,seed=seed,pre_hold_verified=sim.pre_hold_verified))
    rows=[dict(r, outside_capture=bool(r.get('outside_capture',False)), support_force_ratio_mg=float(r.get('support_force_ratio_mg',1.0))) for r in trace]
    loss_rows=rows[10:]; outcome=evaluate_loss_trace(loss_rows); numeric=all(r['numeric_health']['passed'] for r in rows)
    force_start_time=force_start; loss_confirmed_delay_from_force=None; result={'label':label,'family_id':family,'seed':seed,'level_id':level.level_id if level else None,'execution_valid':bool(sim.pre_hold_verified and numeric and len(rows)>=75),'pre_hold':pre_summary,'pre_hold_verified':sim.pre_hold_verified,'numeric_health_pass':numeric,'physical_loss_confirmed':bool(outcome.get('physical_loss_confirmed',False)),'outcome':outcome,'force_start_time':force_start_time,'loss_confirmed_delay_from_force':loss_confirmed_delay_from_force,'selected':False,'failure_reason':None if outcome.get('physical_loss_confirmed') else outcome.get('state'),'trace_rows':len(rows)}
    for name in ('physics_trace.jsonl','contact_trace.jsonl','capture_volume_trace.jsonl','force_trace.jsonl'):
        (root/name).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (root/'physical_reference.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def run_calibration_v2(output_root:Path, authorized:bool=True):
    if not authorized: raise PermissionError('R16 calibration grant required')
    output_root.mkdir(parents=True,exist_ok=False); rows=[]
    primary=output_root/'control_primary'; r=_run_instance(primary,'L2RAR2_FDROP_CAL_00_859900',86099000,None,'primary_control'); rows.append(r)
    control_valid=bool(r['execution_valid'] and not r['physical_loss_confirmed'])
    if not control_valid:
        rows.append(_run_instance(output_root/'control_backup','L2RAR2_FDROP_CAL_01_859901',86099100,None,'backup_control'))
        if not (rows[-1]['execution_valid'] and not rows[-1]['physical_loss_confirmed']):
            raise RuntimeError('STOPPED_CALIBRATION_CONTROL_INVALID')
    for level in pulse_levels():
        r=_run_instance(output_root/level.level_id,'L2RAR2_FDROP_CAL_00_859900',86099000,level,level.level_id); rows.append(r)
        r['loss_confirmed_delay_from_force_s']=None if not r['physical_loss_confirmed'] else r['outcome'].get('loss_confirmed_time_s',0.0)
        if r['execution_valid'] and r['physical_loss_confirmed'] and r['outcome'].get('loss_confirmed_time_s',99)<=.40: r['selected']=True; break
    selected=next((r['level_id'] for r in rows if r.get('selected')),None); status='CALIBRATION_PASS' if selected else 'STOPPED_CALIBRATION_FAILED'
    (output_root/'execution_ledger.csv').write_text('label,level_id,execution_valid,pre_hold_verified,numeric_health_pass,physical_loss_confirmed,selected,failure_reason\n'+'\n'.join(','.join(str(r.get(k,'')) for k in ('label','level_id','execution_valid','pre_hold_verified','numeric_health_pass','physical_loss_confirmed','selected','failure_reason')) for r in rows)+'\n')
    summary={'schema':'l2rar2_r16_calibration_v2_result','status':status,'physical_executions':len(rows),'selected_intervention_level':selected,'rows':rows}
    (output_root/'calibration_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (output_root/'numerical_sanity.json').write_text(json.dumps({'passed':all(r['numeric_health_pass'] for r in rows),'instances':len(rows)},indent=2)+'\n')
    if selected: (output_root/'selected_intervention.json').write_text(json.dumps({'level_id':selected,'protocol':'L2RAR2_R16_CONTROLLED_FORCED_DROP_V2'},indent=2)+'\n')
    return summary

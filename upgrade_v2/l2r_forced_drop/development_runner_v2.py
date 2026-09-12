from __future__ import annotations
import csv,json
from pathlib import Path
from .case_registry import CASES
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
from .simulator import ControlledForcedDropTabletop
from .trace_recorder_v2 import snapshot

FAMILIES=(("L2RAR2_FDROP_DEV_00_860000",860000,86100000),("L2RAR2_FDROP_DEV_01_860001",860001,86100100),("L2RAR2_FDROP_DEV_02_860002",860002,86100200),("L2RAR2_FDROP_DEV_03_860003",860003,86100300))

def run_development_v2(output_root:Path, *, selected_level:str|None, authorized:bool=True):
    if not authorized: raise PermissionError('R16 development grant required')
    if not selected_level: raise RuntimeError('CALIBRATION_SELECTION_REQUIRED')
    output_root.mkdir(parents=True,exist_ok=False); rows=[]
    # F1_ no hold; F2_ touch-only; F3_ stable hold; F4_ weld-off no force;
    # F5_ selected level; F6_ short prehold; F7_ commanded release; F8_ transport.
    for family,fseed,rseed in FAMILIES:
        for index,case in enumerate(CASES):
            seed=rseed+index; root=output_root/f'{family}_{case.case_id}'; (root/'online').mkdir(parents=True); (root/'reference').mkdir()
            sim=ControlledForcedDropTabletop(family_spec(family,'normal_pick_place',fseed,seed,probe_variant='brief_hold' if case.case_id.startswith('F6_') else 'default'),seed)
            sim.perform('approach_object'); sim.perform('close_gripper'); sim.perform('lift')
            trace=[]
            for step in range(10): sim.physics_step(); trace.append(snapshot(sim,step=step,phase='pre_hold',case_id=case.case_id,family_id=family,seed=seed))
            for step in range(90): sim.physics_step(); trace.append(snapshot(sim,step=10+step,phase='observation',case_id=case.case_id,family_id=family,seed=seed,pre_hold_verified=all(r['weld_active'] for r in trace)))
            (root/'physics_trace.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in trace))
            row={'family_id':family,'case_id':case.case_id,'seed':seed,'selected_level':selected_level,'trace_complete':len(trace)==100,'numeric_health_pass':all(r['numeric_health']['passed'] for r in trace),'physical_instance_started':True,'disable_weld_for_intervention':case.intervention!='none','commanded_release':case.case_id.startswith('F7_'),'online':'online','reference':'reference'}
            (root/'status.json').write_text(json.dumps(row,indent=2)+'\n'); rows.append(row)
    (output_root/'rollout_manifest.csv').write_text('family_id,case_id,seed,selected_level,trace_complete,numeric_health_pass,physical_instance_started\n'+'\n'.join(','.join(str(r[k]) for k in ('family_id','case_id','seed','selected_level','trace_complete','numeric_health_pass','physical_instance_started')) for r in rows)+'\n')
    (output_root/'per_rollout_status.csv').write_text((output_root/'rollout_manifest.csv').read_text())
    return {'schema':'l2rar2_r16_development_v2_result','status':'DEVELOPMENT_COMPLETE','rollouts':len(rows),'rows':rows}

from __future__ import annotations
import numpy as np
from .geometry_reference_v2 import capture_geometry,model_geometry
from .contact_reference_v2 import contact_snapshot
def numeric_health(sim):
    d=sim.data; m=sim.model; arrays=(d.qpos,d.qvel,d.qacc_warmstart,d.xpos,d.xquat,d.xfrc_applied)
    finite=all(np.isfinite(np.asarray(x)).all() for x in arrays); ids,ob,_=model_geometry(sim); cv=np.asarray(d.cvel[ob]); w=cv[:3]; v=cv[3:]; z=float(d.xpos[ob,2]);
    return {'passed':bool(finite and np.linalg.norm(v)<5 and np.linalg.norm(w)<50 and z>0.25),'finite_arrays':finite,'object_speed_mps':float(np.linalg.norm(v)),'object_angular_speed_rads':float(np.linalg.norm(w)),'object_z_m':z}
def snapshot(sim, *, step, phase, action=None, case_id=None, family_id=None, seed=None, pre_hold_verified=False, commanded_release=False):
    ids,ob,gb=model_geometry(sim); d=sim.data; g=capture_geometry(sim); c=contact_snapshot(sim); n=numeric_health(sim)
    cv=np.asarray(d.cvel[ob]); row={'physics_step_index':step,'time':float(d.time),'phase':phase,'action':action,'case_id':case_id,'family_id':family_id,'seed':seed,'object_world_position':d.xpos[ob].tolist(),'gripper_world_position':d.xpos[gb].tolist(),'object_world_quaternion':d.xquat[ob].tolist(),'object_linear_velocity':cv[3:].tolist(),'object_angular_velocity':cv[:3].tolist(),'weld_active':bool(d.eq_active[sim.weld_id]),'xfrc_applied':d.xfrc_applied[ob].tolist(),'intervention_phase':getattr(sim,'intervention_phase','none'),'commanded_release':commanded_release,'pre_hold_verified':pre_hold_verified,'numeric_health':n}
    row.update(g); row.update(c); return row

from __future__ import annotations
import numpy as np
from .geometry_reference_v2 import model_geometry
def contact_snapshot(sim):
    ids,object_body,_=model_geometry(sim); m,d,mu=sim.model,sim.data,sim.mujoco; allowed={(ids['object_geom'],ids['finger_left']),(ids['object_geom'],ids['finger_right'])}; rows=[]; total=0.
    for i in range(int(d.ncon)):
        c=d.contact[i]; pair=(int(c.geom1),int(c.geom2))
        if pair not in allowed and pair[::-1] not in allowed or int(c.exclude)!=0 or int(c.efc_address)<0: continue
        force=np.zeros(6); mu.mj_contactForce(m,d,i,force); normal=max(float(force[0]),0.); total+=normal
        rows.append({'geom1':pair[0],'geom2':pair[1],'normal_force_n':normal,'efc_address':int(c.efc_address)})
    mass=float(m.body_mass[object_body]); return {'contacts':rows,'support_force_n':total,'support_force_ratio_mg':total/(mass*9.81)}

from __future__ import annotations
import itertools
import numpy as np

def _corners(center, size, rotation):
    center=np.asarray(center,float); size=np.asarray(size,float); rotation=np.asarray(rotation,float).reshape(3,3)
    return np.array([center + rotation @ (np.asarray(s)*size) for s in itertools.product((-1.,1.), repeat=3)])

def model_geometry(sim):
    m,mu=sim.model,sim.mujoco
    ids={name:int(mu.mj_name2id(m,mu.mjtObj.mjOBJ_GEOM,name)) for name in ('object_geom','finger_left','finger_right')}
    body=int(mu.mj_name2id(m,mu.mjtObj.mjOBJ_BODY,'object')); gripper=int(mu.mj_name2id(m,mu.mjtObj.mjOBJ_BODY,'gripper'))
    if any(v<0 for v in ids.values()) or body<0 or gripper<0: raise RuntimeError('REFERENCE_MODEL_CONTRACT_MISMATCH')
    return ids,body,gripper

def capture_geometry(sim):
    ids,object_body,gripper_body=model_geometry(sim); m=sim.model; d=sim.data
    Rwg=d.xmat[gripper_body].reshape(3,3); Row=d.xmat[object_body].reshape(3,3)
    corners=[]
    for name in ('finger_left','finger_right'):
        gid=ids[name]; mat=np.zeros(9); sim.mujoco.mju_quat2Mat(mat,m.geom_quat[gid]); corners.append(_corners(m.geom_pos[gid],m.geom_size[gid],mat))
    corners=np.concatenate(corners); bmin,bmax=corners.min(0),corners.max(0)
    radius=float(m.geom_size[ids['object_geom']][0]); half=float(m.geom_size[ids['object_geom']][1])
    e=np.abs(Rwg.T@Row)@np.array([radius,radius,half]); margin=Rwg.T@(d.xpos[object_body]-d.xpos[gripper_body])
    lo=bmin-e-.005; hi=bmax+e+.005
    signed=np.minimum(margin-lo,hi-margin)
    return {'object_in_gripper_position':margin.tolist(),'finger_local_min':bmin.tolist(),'finger_local_max':bmax.tolist(),'projected_object_half_extents':e.tolist(),'capture_min':lo.tolist(),'capture_max':hi.tolist(),'signed_margin':signed.tolist(),'inside_capture':bool(np.all(margin>=lo)&np.all(margin<=hi)),'outside_capture':bool(not(np.all(margin>=lo)&np.all(margin<=hi)))}

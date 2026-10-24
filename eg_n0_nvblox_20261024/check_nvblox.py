"""Two training-frame gate. No C3, GT geometry, segmentation or collision labels."""
import json,time,hashlib,sys,platform,importlib.metadata
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from PIL import Image
import torch
from nvblox_torch.mapper import Mapper,QueryType
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.sensor import Sensor
from nvblox_torch import _version
from proj_sdf_ref import query,classify,gripper_samples
R=Path(__file__).resolve().parent
D=R.parent/'eg_sub25_v1_20261015/data/graspnet'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mapper(depth,K):
    params=MapperParams()
    pi=params.get_projective_integrator_params()
    pi.projective_integrator_truncation_distance_vox=4.
    pi.projective_integrator_max_integration_distance_m=7.
    params.set_projective_integrator_params(pi)
    v=params.get_view_calculator_params();v.raycast_subsampling_factor=1
    params.set_view_calculator_params(v)
    m=Mapper(.0025,mapper_parameters=params)
    camera=Sensor.from_camera(fu=float(K[0,0]),fv=float(K[1,1]),cu=float(K[0,2]),cv=float(K[1,2]),width=depth.shape[1],height=depth.shape[0])
    torch.cuda.synchronize();start=time.perf_counter()
    m.add_depth_frame(torch.tensor(depth,device='cuda'),torch.eye(4),camera)
    torch.cuda.synchronize()
    return m,time.perf_counter()-start
def run(scene,frame):
    out=R/'preflight';out.mkdir(exist_ok=True)
    p=D/f'scenes/scene_{scene:04d}/kinect';dp=p/f'depth/{frame:04d}.png';mp=p/f'meta/{frame:04d}.mat'
    meta=loadmat(mp,variable_names=['intrinsic_matrix','factor_depth']);K=meta['intrinsic_matrix'].astype(np.float32)
    depth=np.asarray(Image.open(dp),dtype=np.float32)/np.float32(meta['factor_depth'].item())
    cp=R/f'cache/train_{scene:04d}_{frame:04d}.npz';a=np.load(cp)
    ids=np.arange(0,1024,16);poses=[];keys=[]
    for i in ids:
        original=a['pred'][i].astype(float)
        for angle in range(12):
            da=(angle-int(a['angle_index'][i]))*np.pi/12;c=np.cos(da);s=np.sin(da)
            rx=np.array([[1,0,0],[0,c,-s],[0,s,c]])
            row=original.copy();row[4:13]=(row[4:13].reshape(3,3)@rx).ravel()
            if angle==a['angle_index'][i]:row=original.copy()
            poses.append(row);keys.append([int(i),angle])
    torch.cuda.reset_peak_memory_stats();m,integrate=mapper(depth,K)
    matrices=[np.zeros((3,3),np.int64),np.zeros((3,3),np.int64)]
    per=[];examples=[];query_secs=0.;same_phi=[]
    for row,(cid,angle) in zip(poses,keys):
        points=gripper_samples(row);q=torch.tensor(points,device='cuda')
        torch.cuda.synchronize();ts=time.perf_counter()
        nv=m.query_layer(QueryType.TSDF,q,mapper_id=0).cpu().numpy()
        query_secs+=time.perf_counter()-ts
        state=classify(nv[:,0],nv[:,1]);rec=dict(candidate_id=cid,angle=angle,points=len(points),nv_counts=np.bincount(state,minlength=3).tolist())
        for j,centers in enumerate([False,True]):
            phi,weight=query(points,depth,K,centers);ref=classify(phi,weight)
            matrices[j]+=np.bincount(state*3+ref,minlength=9).reshape(3,3)
            rec['agreement_'+str(centers)]=float(np.mean(state==ref))
            if centers:
                both=(nv[:,1]>0)&(weight>0)
                if both.any():same_phi.append(float(np.max(abs(phi[both]-nv[both,0]))))
                if len(examples)<100:
                    for ix in np.flatnonzero(state!=ref)[:5]:
                        examples.append(dict(candidate_id=cid,angle=angle,xyz=points[ix].tolist(),nv=nv[ix].tolist(),ref_phi=float(phi[ix]),ref_weight=float(weight[ix])))
        per.append(rec)
    result=dict(scene=scene,frame=frame,poses=len(poses),candidates=len(ids),sampling='IDs 0,16,...,1008; every 12 angle classes; complete gripper solids at <=2.5 mm spacing',
      states=['free','occupied','unknown'],confusion_nv_rows_reference_columns=[x.tolist() for x in matrices],
      agreement_continuous=float(np.trace(matrices[0])/matrices[0].sum()),agreement_voxel_center=float(np.trace(matrices[1])/matrices[1].sum()),
      points=int(matrices[0].sum()),max_same_voxel_tsdf_error=max(same_phi,default=None),integration_seconds=integrate,query_seconds=query_secs,
      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
      depth_sha256=sha(dp),meta_sha256=sha(mp),cache_sha256=sha(cp),examples=examples[:100],per_pose=per)
    (out/f'{scene:04d}_{frame:04d}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_pose','examples']}),flush=True)
    del m;torch.cuda.empty_cache()
    return result
if __name__=='__main__':
    torch.set_num_threads(2)
    plan=json.loads((R/'protocol.json').read_text())
    env=dict(torch=torch.__version__,nvblox=_version.__version__,nvblox_wheel_commit=_version.__git_sha__,gpu=torch.cuda.get_device_name(),platform=platform.platform(),python=sys.version,files={n:sha(R/n) for n in ['protocol.json','check_nvblox.py','proj_sdf_ref.py']})
    (R/'runtime.json').write_text(json.dumps(env,indent=2))
    plane=np.full((48,64),1.,np.float32);kp=np.array([[100,0,32],[0,100,24],[0,0,1]],np.float32)
    pm,_=mapper(plane,kp)
    pts=np.array([[.00125,.00125,z] for z in [.98125,.99875,1.00375,1.01875]],np.float32)
    values=pm.query_layer(QueryType.TSDF,torch.tensor(pts,device='cuda'),mapper_id=0).cpu().numpy()
    st=classify(values[:,0],values[:,1]);assert st.tolist()==[0,1,1,2],(values,st)
    (R/'plane_sanity.json').write_text(json.dumps(dict(points=pts.tolist(),values=values.tolist(),states=st.tolist(),expected=[0,1,1,2]),indent=2))
    del pm;torch.cuda.empty_cache()
    results=[run(s,f) for s,f in plan['initial_check']]
    summary=dict(frames=[{k:x[k] for k in ['scene','frame','points','agreement_continuous','agreement_voxel_center','max_same_voxel_tsdf_error']} for x in results],
      continuous_gate=all(x['agreement_continuous']>=.99 for x in results),voxel_gate=all(x['agreement_voxel_center']>=.99 for x in results),C3_formal_runs=0)
    (R/'preflight_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

"""CPU-only shared definitions. Geometric proxies never identify contacts."""
import pathlib,json,csv,hashlib,datetime,os,numpy as np
R=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
SCALE=pathlib.Path('/home/xushijie3/work/pc_scale02_20261011/experiments/pc_scale02_20261011')
PRIOR=pathlib.Path('/home/xushijie3/work/graph_pc_p123_card_20261011/experiments/pc_p123_run_20261011')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,x):p=R/'results'/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def table(name,rows):
 if not rows:return
 with (R/'results'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def first_run(mask,length=5):
 for t in range(length-1,len(mask)):
  if np.all(mask[t-length+1:t+1]):return int(t)
 return None
def bbox_margin(st):
 x,y,z,w=np.moveaxis(st[...,3:7],-1,0)
 rot=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(st.shape[:-1]+(3,3))
 half=np.abs(rot).sum(-1)*.0175
 return np.minimum(.25-np.abs(st[...,0])-half[...,0],.30-np.abs(st[...,1])-half[...,1])
def no_progress(st,dd):
 ans=[]
 for end in range(20,len(st)):
  lo=end-20;part=st[lo:end+1]
  if (part[...,2]<.975).any() or (dd[end]<.04).all():continue
  if np.linalg.norm(part[...,:3]-part[0,:,:3],axis=-1).max()<.001 and dd[lo].mean()-dd[lo:end+1].mean(-1).min()<.001:ans.append(end)
 return ans
def episode_data(e):
 z=np.load(e['trajectory']);i=e['slot']
 return z,z['states'][:,i],z['distances'][:,i],z['eef'][:,i]
def chain(z,i,start,end):
 st=z['states'][start:end+1,i];d=z['distances'][start:end+1,i];ee=z['eef'][start:end+1,i];a=z['actions'][start:end,i]
 out={'start':start,'end':end,'online_detection_step':end,'eef_max_displacement_m':float(np.linalg.norm(ee-ee[0],axis=-1).max()),
 'eef_path_m':float(np.linalg.norm(np.diff(ee,axis=0),axis=-1).sum()),'eef_z_range_m':np.ptp(ee[:,2]).item(),
 'cube_max_displacement_m':float(np.linalg.norm(st[...,:3]-st[0,:,:3],axis=-1).max()),'per_object_goal_improvement_m':(d[0]-d.min(0)).tolist(),
 'max_mean_goal_improvement_m':float(d[0].mean()-d.mean(-1).min()),'action_abs_mean':np.abs(a).mean(0).tolist(),
 'action_clip_fraction':float((np.abs(a)>=1).mean()),'eef_object_min_distance_m':float(np.linalg.norm(ee[:,None,:]-st[...,:3],axis=-1).min()),
 'contact_counterpart':None,'contact_effectiveness':None,'category':'no-progress / evidence insufficient; no abnormal-policy label'}
 if 'q' in z:out['q_max_change_rad']=float(np.abs(z['q'][start:end+1,i]-z['q'][start,i]).max())
 if 'controller_goal' in z:
  g=z['controller_goal'][start:end,i];expected=np.clip(ee[:-1]+np.clip(a,-1,1)*.125,[-.2675,-.3175,1.045],[.2675,.3175,1.06])
  out['target_transform_max_error_m']=float(np.abs(expected-g).max());out['target_eef_offset_mean_m']=float(np.linalg.norm(g-ee[:-1],axis=-1).mean())
 return out
def image_refs(e,z,steps):
 arr=z['image_steps'];rows=[]
 for step in steps:
  k=int(np.argmin(np.abs(arr-step)))
  rows.append({'requested_step':int(step),'image_step':int(arr[k]),'array_index':k,'slot':e['slot'],'path':e['trajectory'],'offset_steps':int(arr[k]-step),'precision':'image evidence only; not contact time'})
 return rows

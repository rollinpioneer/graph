"""CPU collision-box separation at recorded times; never identifies contacts."""
from common import *
import xml.etree.ElementTree as ET,itertools,time
begin=time.time();cases=json.loads((R/'results/representative_cases.json').read_text())
urdf=OLD/'EC-Diffuser/assets/urdf/franka_description/robots/franka_panda_gripper.urdf';root=ET.parse(urdf).getroot()
sizes={}
for link in root.findall('link'):
 box=link.find('collision/geometry/box')
 if link.attrib['name'] in ('panda_leftfinger_tip','panda_rightfinger_tip'):
  assert box is not None;sizes[link.attrib['name']]=np.array([float(x) for x in box.attrib['size'].split()])
corners=np.array(list(itertools.product([-1,1],repeat=3)),dtype=float)
def rot(q):
 x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def sat(a,asize,b,bsize):
 A=rot(a[3:7]);B=rot(b[3:7]);axes=[A[:,k] for k in range(3)]+[B[:,k] for k in range(3)]
 axes+=[np.cross(A[:,k],B[:,l]) for k in range(3) for l in range(3)]
 av=(corners*asize/2)@A.T+a[:3];bv=(corners*bsize/2)@B.T+b[:3];gaps=[]
 for ax in axes:
  norm=np.linalg.norm(ax)
  if norm<1e-8:continue
  ax=ax/norm;x=av@ax;y=bv@ax;gaps.append(max(y.min()-x.max(),x.min()-y.max()))
 return float(max(gaps))
rows=[]
for c in cases:
 e=c['episode'];folder=pathlib.Path(e['trajectory']).parent;f=folder/'physical.npz'
 if not f.exists():
  rows.append({'case_id':c['case_id'],'episode_id':e['episode_id'],'status':'unavailable: historical fields only eef center, no gripper pose; not reconstructed from unlogged state'});continue
 ph=np.load(f);meta=json.loads((folder/'contact_source.json').read_text());bm={x['name']:x['index_env'] for x in meta['body_map']}
 tips=[bm[k] for k in sizes if k in bm];tipnames=[k for k in sizes if k in bm];cubes=[x['index_env'] for x in meta['body_map'] if x['actor']>=3]
 if not tips:
  rows.append({'case_id':c['case_id'],'episode_id':e['episode_id'],'status':'unavailable: finger-tip body mapping absent'});continue
 first=e['first_no_progress'];lo=max(0,(first or 50)-40);hi=min(e['steps'],(first or 50)+40);inds=np.flatnonzero((ph['policy_step']>=lo+1)&(ph['policy_step']<=hi))
 samples=[]
 for k in inds:
  body=ph['body_state'][k,e['slot']];gaps=[[sat(body[tip],sizes[name],body[cube],np.full(3,.035)) for cube in cubes] for tip,name in zip(tips,tipnames)]
  samples.append({'time_s':float(ph['time'][k]),'policy_step':int(ph['policy_step'][k]),'tip_cube_signed_SAT_gap_m':gaps,
  'finger_tip_pose':body[tips,:7].tolist()})
 flat=[g for q in samples for x in q['tip_cube_signed_SAT_gap_m'] for g in x]
 rows.append({'case_id':c['case_id'],'episode_id':e['episode_id'],'status':'recorded collision-box geometry evaluated','source':str(f),'samples':samples,
 'min_tip_cube_SAT_gap_m':min(flat) if flat else None,'tip_boxes_separated_at_all_recorded_times':all(x>0 for x in flat) if flat else None,
 'limitation':'only two tip boxes; SAT gap is separation-axis lower bound, not exact Euclidean distance; finger meshes/other arm parts and inter-sample contacts not excluded',
 'physx_contact_offset_m':.005,'contact_envelope':'gap near0/.005 does not identify physical force or counterpart'})
 ph.close()
dump('P2_gripper_collision_geometry.json',rows)
dump('P2_geometry_receipt.json',{'utc':utc(),'urdf_path':str(urdf),'urdf_sha256':sha(urdf),'tip_box_sizes_m':{k:v.tolist() for k,v in sizes.items()},
 'cases':len(rows),'available':sum('evaluated' in q['status'] for q in rows),'CPU_wall_seconds':time.time()-begin,'contact_claim':False})
print('GEOMETRY_DONE',len(rows),'available',sum('evaluated' in q['status'] for q in rows),flush=True)

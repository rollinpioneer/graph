"""Render saved integration states only; no env.step, policy or optimizer calls."""
import os,json,pathlib,time,hashlib
import numpy as np,mujoco,ogbench,imageio.v2 as imageio
from PIL import Image,ImageDraw
R=pathlib.Path(__file__).resolve().parents[1]
meta=json.loads((R/'reports/environment.json').read_text());sel=json.loads((R/'configs/video_selection.json').read_text())
env=ogbench.make_env_and_datasets(meta['task'],env_only=True);u=env.unwrapped
u.model.vis.global_.offwidth=640;u.model.vis.global_.offheight=480
renderer=mujoco.Renderer(u.model,height=480,width=640);cam=mujoco.MjvCamera()
cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=6;cam.azimuth=135;cam.elevation=-90
maze=np.array(meta['maze_map']);stride=5;fps=1/(meta['dt']*stride);results=[];thumbs=[];start=time.time()
def panel(xy,goal,t,n,label,ok):
 im=Image.new('RGB',(224,480),'#f2f3f4');d=ImageDraw.Draw(im);scale=7
 for i in range(maze.shape[0]):
  for j in range(maze.shape[1]):
   if maze[i,j]:d.rectangle((j*28,224-(i+1)*28,(j+1)*28,224-i*28),fill='#555555')
 def pt(p):return ((p[0]+6)*7,224-(p[1]+6)*7)
 path=[pt(p) for p in xy[:t+1:3]]
 if len(path)>1:d.line(path,fill='#407bbf',width=2)
 x,y=pt(goal);d.ellipse((x-4,y-4,x+4,y+4),fill='#20a050')
 x,y=pt(xy[t]);d.ellipse((x-3,y-3,x+3,y+3),fill='#dc4545')
 for k,line in enumerate([label[:25],label[25:50],f'Step {t}/{n}',f'Success: {ok}',f'Simulation {t*meta["dt"]:.2f}s','Frozen weights; replay','No additional env steps']):
  d.text((8,242+27*k),line,fill='#111111')
 return im
try:
 for item in sel:
  cid=item['id'];a=np.load(R/'trajectories'/f'{cid}.npz');record=json.loads((R/'trajectories'/f'{cid}.json').read_text())
  out=R/'videos_topdown'/f'{cid}.mp4'
  if out.exists():raise FileExistsError(out)
  u.set_goal(goal_xy=a['goal_xy'][0]);frames=list(range(0,len(a['xy']),stride))
  if frames[-1]!=len(a['xy'])-1:frames.append(len(a['xy'])-1)
  max_error=0.;t0=time.time();keyframes={}
  with imageio.get_writer(out,fps=fps,codec='libx264',quality=7,macro_block_size=16) as writer:
   for t in frames:
    mujoco.mj_setState(u.model,u.data,a['integration'][t],mujoco.mjtState(meta['state_spec']))
    mujoco.mj_forward(u.model,u.data)
    err=float(np.max(np.abs(u.data.xpos-a['body_positions'][t])));max_error=max(max_error,err)
    if err>1e-8:raise AssertionError(f'Render restore mismatch {cid} step{t}: {err}')
    cam.lookat[:]=[a['xy'][t,0],a['xy'][t,1],.8]
    renderer.update_scene(u.data,camera=cam);frame=renderer.render()
    canvas=Image.new('RGB',(864,480));canvas.paste(Image.fromarray(frame),(0,0))
    canvas.paste(panel(a['xy'],a['goal_xy'][t],t,len(a['actions']),cid,record['observed']['success']),(640,0))
    writer.append_data(np.asarray(canvas))
    if t in (frames[len(frames)//2],frames[-1]):keyframes[t]=canvas.copy()
  for t,im in keyframes.items():im.resize((432,240)).save(R/'videos_topdown'/f'{cid}_step{t}.jpg',quality=85)
  thumb=keyframes[frames[-1]].resize((432,240));thumbs.append(thumb)
  results.append(dict(id=cid,reason=item['reason'],path=str(out),bytes=out.stat().st_size,
   sha256=hashlib.sha256(out.read_bytes()).hexdigest(),fps=fps,frames=len(frames),stride=stride,
   playback='approximately real simulation time; final frame appended',elapsed_seconds=time.time()-t0,
   maximum_restored_body_position_error=max_error,environment_steps=0,optimizer_updates=0))
  print('VIDEO',len(results),cid,'restore_error',max_error,flush=True)
 sheet=Image.new('RGB',(864,960),'white')
 for j,im in enumerate(thumbs):sheet.paste(im,((j%2)*432,(j//2)*240))
 sheet.save(R/'reports/video_contact_sheet.jpg',quality=88)
 (R/'reports/video_index.json').write_text(json.dumps(results,indent=2))
 (R/'receipts/render_result.json').write_text(json.dumps(dict(status='PASS',videos=len(results),environment_steps=0,
 optimizer_updates=0,elapsed_seconds=time.time()-start),indent=2))
finally:renderer.close();env.close()


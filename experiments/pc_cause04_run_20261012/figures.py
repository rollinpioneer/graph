"""Scientific case boards. Exact image step labels; screenshots do not locate contacts."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,time
begin=time.time();es=json.loads((R/'results/episode_index.json').read_text());by={e['episode_id']:e for e in es};boards=[];cache={}
def board(e,steps,name,title):
 if e['trajectory'] not in cache:
  z=np.load(e['trajectory']);cache[e['trajectory']]={k:z[k] for k in ('states','distances','eef','images','image_steps')}
 z=cache[e['trajectory']];i=e['slot'];ini=np.load(e['initial']);fig,ax=plt.subplots(2,len(steps)+1,figsize=(2.6*(len(steps)+1),5.8))
 for v in range(2):
  for col,requested in enumerate(steps):
   k=int(np.argmin(np.abs(z['image_steps']-requested)));actual=int(z['image_steps'][k]);img=z['images'][k,i,v];img=np.moveaxis(img,0,-1);ax[v,col].imshow(img);ax[v,col].set_title('requested%d / image%d'%(requested,actual),fontsize=9);ax[v,col].axis('off')
  ax[v,-1].imshow(np.moveaxis(ini['goal_image'][i,v],0,-1));ax[v,-1].set_title('goal view%d'%v);ax[v,-1].axis('off')
 fig.suptitle(title+'\n'+e['episode_id']+' | images are sparse observations',fontsize=10);fig.tight_layout();fig.savefig(R/'results'/name,dpi=130);plt.close(fig)
 boards.append({'file':name,'episode_id':e['episode_id'],'source':e['trajectory'],'steps':image_refs(e,z,steps),'goal_source':e['initial'],'title':title})
matches=json.loads((R/'results/P2_matches_and_chains.json').read_text());m=next(q for q in matches if q['control'])
e=by[m['case']['episode_id']];a,b=m['case']['start'],m['case']['end'];board(e,[a-20,a,b,b+20],'P2_case.png','First no-progress candidate; abnormality unestablished')
q=by[m['control']['episode_id']];a,b=m['control']['start'],m['control']['end'];board(q,[a-20,a,b,b+20],'P2_control_proposal.png','Geometric control proposal; fails frozen future normal confirmation')
for slot,caption in [(3,'success: reached objects may depart and recover'),(5,'failure: reached objects depart; counterpart unknown')]:
 e=next(e for e in es if '/P1_crowded8/' in e['episode_id'] and e['slot']==slot)
 board(e,[0,50,100,200],'P1_%s.png'%('success' if slot==3 else 'failure'),caption)
falls=json.loads((R/'results/P3_first_crossings_and_falls.json').read_text());r=next(r for r in falls if r['physical_evidence'] and r['net_force_valid'] and r['first_fall_detection'] is not None)
e=by[r['episode_id']];t=r['first_bbox_cross_control_step'];board(e,[max(0,t-20),max(0,t-10),t,min(e['steps'],t+10)],'P3_fall.png','First bbox crossing of object%d; direct/secondary cause unknown'%r['object_id'])
z=np.load(e['trajectory']);i=e['slot'];j=r['object_id'];lo=max(0,t-25);hi=min(e['steps'],t+10);xx=np.arange(lo,hi+1)
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
axes[0].plot(xx,z['distances'][lo:hi+1,i,j]);axes[0].axhline(.04,color='k',ls=':');axes[0].set_ylabel('XY goal error [m]')
axes[1].plot(xx,bbox_margin(z['states'][lo:hi+1,i,j]));axes[1].axhline(0,color='k',ls=':');axes[1].set_ylabel('bbox margin [m]')
axes[2].plot(np.arange(lo,t),np.linalg.norm(z['states'][lo:t,i,j,7:10],axis=-1),label='strictly before crossing');axes[2].set_ylabel('speed [m/s]');axes[2].legend()
for ax in axes:ax.axvline(t,color='r',ls='--');ax.grid(alpha=.2)
axes[-1].set_xlabel('control step (contact order finer than this is unknown)');fig.suptitle(e['episode_id']+' object%d'%j,fontsize=10);fig.tight_layout();fig.savefig(R/'results/P3_pre_cross_trace.png',dpi=140);plt.close(fig)
# Verify inferred goal color labels against independently frozen requested palette, where available.
vision={v['episode_id']:v for v in json.loads((R/'results/P1_vision_proxy.json').read_text())};checks=[]
for e in es:
 if e['cohort']=='historical':continue
 cfg=json.loads((pathlib.Path(e['trajectory']).parent/'config.json').read_text())
 if e['cohort']=='PC_SCALE02':
  src=SCALE/'results/scenes.json';sc=json.loads(src.read_text());q=next(q for q in sc if q['group']==e['scene_group'] and q['condition']==e['layout'])
 else:
  src=PRIOR/'results'/cfg['scene_file'];sc=json.loads(src.read_text())
  # All matching scenes have unique condition/group identity; arm copies share palette.
  q=next(q for q in sc if q['group']==e['scene_group'] and q['condition']==e['layout'])
 got=vision[e['episode_id']]['goal_assignment_per_view'][0]
 checks.append({'episode_id':e['episode_id'],'scene_source':str(src),'expected':q['colors'],'inferred_goal_color':got,'match':got==q['colors']})
dump('goal_palette_verification.json',checks)
dump('case_image_index.json',boards);dump('figures_receipt.json',{'utc':utc(),'boards':len(boards),'palette_checks':len(checks),'palette_matches':sum(q['match'] for q in checks),'cpu_wall_seconds':time.time()-begin})
print('FIGURES_DONE',len(boards),'palette match',sum(q['match'] for q in checks),'of',len(checks),flush=True)

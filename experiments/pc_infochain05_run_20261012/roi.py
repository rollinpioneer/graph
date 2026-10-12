"""Object ROI diagnostics, permutation-safe visual evidence, CPU only."""
from common import *
import itertools,time
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
start=time.time();prev=pathlib.Path(os.environ['PREV'])
src=(prev/'p1.py').read_text();exec(src[src.index('palette='):src.index('vision=[]')])
cases=json.loads((R/'results/representative_cases.json').read_text())
vision={x['episode_id']:x for x in json.loads((prev/'results/P1_vision_proxy.json').read_text())}
idx=json.loads((R/'runs/offline_formal/data/decomposition_index.json').read_text())
rows=[];summ=[];traj={};inits={};figs=R/'figures';figs.mkdir(exist_ok=True)
for c in cases:
 e=c['episode'];eid=e['episode_id'];vmeta=vision[eid];colors=vmeta['goal_assignment_per_view'][0]
 if e['trajectory'] not in traj:
  zz=np.load(e['trajectory']);traj[e['trajectory']]={k:zz[k] for k in ['states','distances','eef','goal','actions']}
 if e['initial'] not in inits:
  zz=np.load(e['initial']);inits[e['initial']]={k:zz[k] for k in ['initial_latent','goal_latent']}
 z=traj[e['trajectory']];i=e['slot'];packs=[x for x in idx if x['case_id']==c['case_id']]
 for q in packs:
  pack=np.load(q['raw_path']);t=q['step'];im=pack['image'];rec=pack['rec'];fg=pack['dec_objects_trans'];am=pack['alpha_masks'][:,:,0]
  reps=pack['policy_particles'];saved=(inits[e['initial']]['goal_latent' if t is None else 'initial_latent'][i] if t is None or t==0 else None)
  summ.append({'case_id':c['case_id'],'n':e['n'],'step':t,'initial_or_goal_saved_maxabs':float(np.abs(reps-saved).max()) if saved is not None else None,'raw_path':q['raw_path'],'raw_sha256':q['sha256']})
  st=np.zeros((e['n'],13)) if t is None else z['states'][t,i].copy()
  if t is None:st[:,:2]=z['goal'][i];st[:,2]=1.0425;st[:,6]=1
  for v in range(2):
   cl=classes(im[v]);cr=classes(rec[v]);cf=classes(fg[v])
   for j,point in enumerate(st):
    quat=point[3:7];x,y,zz,w=quat;mat=np.array([[1-2*(y*y+zz*zz),2*(x*y-zz*w),2*(x*zz+y*w)],[2*(x*y+zz*w),1-2*(x*x+zz*zz),2*(y*zz-x*w)],[2*(x*zz-y*w),2*(y*zz+x*w),1-2*(x*x+y*y)]])
    pp=project(corners@mat.T+point[:3],v);lo=np.floor(pp.min(0)-2).astype(int).clip(0,127);hi=np.ceil(pp.max(0)+2).astype(int).clip(0,127)
    sl=(slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1));mask=cl[sl]==colors[j];count=int(mask.sum())
    yy,xx=np.where(mask);centroid=[float(xx.mean()+lo[0]),float(yy.mean()+lo[1])] if count else None
    active=reps[v,:,-1]>.1;px=(reps[v,:,1]+1)*127/2;py=(reps[v,:,0]+1)*127/2
    dist=float(np.linalg.norm(np.c_[px[active],py[active]]-centroid,axis=-1).min()) if count and active.any() else None
    visible=bool(count>=5 and vmeta['credible_color_projection_proxy'])
    alpha=am[v,:,sl[0],sl[1]].max(0)
    rows.append({'case_id':c['case_id'],'episode_id':eid,'n':e['n'],'success':e['endpoints'][str(e['steps'])]['all_success'],'step':t,'view':v,'object_id':j,'pending':t is not None and bool(z['distances'][t,i,j]>=.04),'roi_xyxy':[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],'object_xyz':point[:3].tolist(),'color_id':colors[j],'color_identity_projection_credible':vmeta['credible_color_projection_proxy'],'visible_color_pixels':count,'visible_proxy_qualified':visible,'color_centroid_xy_px':centroid,'reconstruction_color_pixels':int((cr[sl]==colors[j]).sum()),'foreground_color_pixels':int((cf[sl]==colors[j]).sum()),'ROI_reconstruction_RGB_MSE':float(np.square(rec[v,:,sl[0],sl[1]]-im[v,:,sl[0],sl[1]]/255.).mean()),'max_particle_alpha_mean_at_visible_pixels':float(alpha[mask].mean()) if count else None,'visible_pixels_alpha_ge0p2_fraction':float((alpha[mask]>=.2).mean()) if count else None,'nearest_active_particle_to_visible_centroid_px':dist,'scope':'projection/color visibility proxy; alpha/centers not object identity; ROI includes background; no causal encoding-failure label'})
  pack.close()
 # Four fixed example panels: one success/failure per n; show original/rec for both views.
 if c['case_id'] in [0,3,6,9]:
  choose=min(packs,key=lambda q:abs((q['step'] or 0)-(e['first_no_progress'] or 50)))
  pack=np.load(choose['raw_path']);fig,axs=plt.subplots(2,3,figsize=(9,6))
  for v in range(2):
   for k,key in enumerate(['image','rec','dec_objects_trans']):
    arr=pack[key][v];axs[v,k].imshow(np.moveaxis(arr,0,-1).clip(0,255 if key=='image' else 1));axs[v,k].axis('off');axs[v,k].set_title(['saved input','decoded full','decoded foreground'][k]+' view'+str(v))
   rep=pack['policy_particles'][v];mask=rep[:,-1]>.1;axs[v,2].scatter((rep[mask,1]+1)*127/2,(rep[mask,0]+1)*127/2,s=15,facecolors='none',edgecolors='white')
  fig.suptitle('case%d %dC success=%s step%s (particles are not object IDs)'%(c['case_id'],e['n'],e['endpoints'][str(e['steps'])]['all_success'],choose['step']));fig.tight_layout();fig.savefig(figs/('DLP_case%02d.png'%c['case_id']),dpi=120);plt.close(fig)
 fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True);st=z['states'][:,i];ee=z['eef'][:,i];dd=z['distances'][:,i];tt=np.arange(len(dd))
 axs[0].plot(tt,dd);axs[0].axhline(.04,color='k',linestyle=':');axs[0].set_ylabel('object goal distance m')
 axs[1].plot(tt,ee[:,0],label='eef x');axs[1].plot(tt,ee[:,1],label='eef y');axs[1].plot(tt,ee[:,2]-1.025,label='height over table');axs[1].legend();axs[1].set_ylabel('EEF m')
 dist=np.linalg.norm(ee[:,None,:]-st[...,:3],axis=-1);axs[2].plot(tt,dist);axs[2].set_ylabel('EEF to each cube center m');axs[2].set_xlabel('control step')
 if e['first_no_progress'] is not None:
  for ax in axs:ax.axvspan(e['first_no_progress']-20,e['first_no_progress'],color='orange',alpha=.2)
 fig.suptitle('case%d %dC full history; fall=%s; center distance is not contact'%(c['case_id'],e['n'],e['first_fall']));fig.tight_layout();fig.savefig(figs/('P2_case%02d.png'%c['case_id']),dpi=110);plt.close(fig)
dump('P1_object_ROI_evidence.json',rows);dump('P1_input_identity_check.json',summ)
qual=[q for q in rows if q['visible_proxy_qualified'] and q['pending']]
agg=[]
for n in [3,6]:
 for success in [True,False]:
  a=[q for q in qual if q['n']==n and q['success']==success]
  agg.append({'n':n,'success':success,'qualified_pending_object_view_frames':len(a),'median_ROI_MSE':float(np.median([q['ROI_reconstruction_RGB_MSE'] for q in a])) if a else None,'median_alpha_coverage':float(np.median([q['visible_pixels_alpha_ge0p2_fraction'] for q in a])) if a else None,'low_reconstructed_color_rows':sum(q['reconstruction_color_pixels']<5 for q in a),'interpretation':'descriptive, repeated frames not independent trials; visibility/color mask can miss surfaces'})
dump('P1_ROI_summary.json',agg)
dump('ROI_analysis_receipt.json',{'utc':utc(),'CPU_wall_seconds':time.time()-start,'projection_source':str(prev/'p1.py'),'projection_source_sha256':sha(prev/'p1.py'),'color_masks':'same prior exploratory definitions, not groundtruth','raw_hashes_in_decomposition':True,'script_sha256':sha(__file__)})
print('ROI_DONE',len(rows),len(qual),agg,flush=True)


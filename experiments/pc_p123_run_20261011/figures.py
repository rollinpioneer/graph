import pathlib,json,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parent;out=R/'results';folder=out/'cases';folder.mkdir(exist_ok=True)
rows=json.loads((out/'episode_metrics.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,step in zip(axs,[100,200]):
 for g in range(8):
  a=[r for r in rows if r['group']==g and r['step']==step and r['condition']=='6_sparse'][0];b=[r for r in rows if r['group']==g and r['step']==step and r['condition']=='6_crowded'][0]
  ax.plot([0,1],[a['fraction'],b['fraction']],'-o',label='g%d block%d'%(g,g%2))
 ax.set_xticks([0,1]);ax.set_xticklabels(['sparse','crowded']);ax.set_ylim(-.05,1.05);ax.set_title('6 cubes, step %d, n=8 pairs'%step);ax.set_ylabel('identity-matched XY hit fraction')
axs[1].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/'P1_results.png',dpi=150);plt.close(fig)
F=R/'runs/validation_P3_8/data'
if (F/'completed.json').exists():
 ini=np.load(F/'initial.npz');z=np.load(F/'trajectory.npz');assignment=json.loads((out/'validation_assignment.json').read_text())
 fig,axs=plt.subplots(2,2,figsize=(11,7))
 for ax,g in zip(axs.ravel(),range(8,12)):
  for s in assignment:
   if s['group']!=g:continue
   i=s['slot'];ax.plot((z['distances'][:,i]<.04).mean(-1),label=s['arm']+(' scale0.5' if s['arm']=='B1' else ' original'))
   rr=[r for r in rows if r['group']==g and r.get('arm')==s['arm'] and r['step']==200][0];fall=rr['events']['first_fall']
   if fall is not None:ax.axvline(fall,color='red',ls=':',alpha=.6)
  ax.set_ylim(-.05,1.05);ax.set_title('new scene g%d block%d'%(g,g%2));ax.set_xlabel('control step (red: fall detection)');ax.set_ylabel('XY hit fraction');ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(out/'P3_results.png',dpi=150);plt.close(fig)
 for s in assignment:
  i=s['slot'];fig,axs=plt.subplots(2,3,figsize=(9,6))
  for view in range(2):
   imgs=[ini['initial_image'][i,view],ini['goal_image'][i,view],z['images'][-1,i,view]]
   for col,img in enumerate(imgs):
    if img.shape[0]==3:img=np.transpose(img,(1,2,0))
    axs[view,col].imshow(img);axs[view,col].axis('off');axs[view,col].set_title(['initial','goal','step200'][col])
  fig.suptitle('P3 '+s['arm']+' scene g%d'%s['group']);fig.tight_layout();fig.savefig(folder/('P3_%s_g%d.png'%(s['arm'],s['group'])),dpi=100);plt.close(fig)
print('FIGURES_COMPLETE')

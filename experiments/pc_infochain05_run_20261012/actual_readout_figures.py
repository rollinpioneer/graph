from common import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
idx=json.loads((R/'results/P1_actual_recordings_index.json').read_text());t=108
fig,axs=plt.subplots(2,4,figsize=(12,6))
for col,run in [(0,'validate6_blocked'),(2,'validate6_permuted')]:
 q=next(x for x in idx if x['run']==run and x['state_step']==t);z=np.load(q['raw_path'])
 for v in [0,1]:
  for k,key in enumerate(['image','dec_objects_trans']):axs[v,col+k].imshow(np.moveaxis(z[key][v],0,-1).clip(0,255 if key=='image' else 1));axs[v,col+k].axis('off');axs[v,col+k].set_title(run.replace('validate6_','')+' '+['actual RGB','stored particles foreground'][k]+'\nview'+str(v),fontsize=10)
 z.close()
fig.suptitle('6C state108: yellow pending ROI retains color in foreground reconstruction\npost-intervention observations differ; this is exploratory readout, not matched-state causal encoding test');fig.tight_layout();fig.savefig(R/'figures/P1_actual_6C_state108.png',dpi=120);plt.close(fig)

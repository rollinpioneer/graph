"""Publication-style static curves from existing logs; CPU rendering only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';F=R/'figures';F.mkdir(exist_ok=True)
native=json.loads((R/'native_learning_curves.json').read_text());points=json.loads((R/'final_checkpoint_results.json').read_text());loss=json.loads((R/'loss_learning_curves.json').read_text())
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':110})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for row,host in enumerate(['G','D']):
 for arm,color in [('0','#225ea8'),('1','#d95f0e')]:
  for seed,style in [(0,'-'),(1,'--')]:
   name=f'{host}{arm}_seed{seed}';xs=sorted([x for x in native if x['job']==name and x['component']=='Actor'],key=lambda x:x['step'])
   for col,key,scale in [(0,'success',100),(1,'return_mean',1)]:
    axes[row,col].plot([x['step']/1e6 for x in xs],[x[key]*scale for x in xs],color=color,linestyle=style,label=name)
    ms=[x for x in points if x['job']==name and x['protocol']=='matched_verified']
    axes[row,col].scatter([x['step']/1e6 for x in ms],[x[key]*scale for x in ms],color=color,marker='o' if seed==0 else 'x',s=35)
  
 for col,title in [(0,'Success (%)'),(1,'Return')]:
  axes[row,col].set_title(('HumanoidMaze-medium task1' if host=='G' else 'Cube-double task2')+' | '+title);axes[row,col].set_xlabel('Formal updates (millions)');axes[row,col].set_ylabel(title);axes[row,col].legend(fontsize=8)
 axes[row,0].set_ylim(-3,103)
fig.suptitle('Native Actor curves; markers: verified matched 100-episode evaluations\nTwo seeds, different tasks: compare within each row only',fontsize=12)
fig.savefig(F/'learning_curves.png');fig.savefig(F/'learning_curves.pdf');plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for row,host in enumerate(['G','D']):
 for arm,color in [('0','#225ea8'),('1','#d95f0e')]:
  for seed,style in [(0,'-'),(1,'--')]:
   name=f'{host}{arm}_seed{seed}';xs=sorted([x for x in loss if x['job']==name],key=lambda x:float(x['step']))
   choices=[['training/critic/q','training/critic/q_mean'],['training/grad/norm'],['training/actor/actor_loss']]
   for col,keys in enumerate(choices):
    ys=[(float(x['step'])/1e6,float(next(x[k] for k in keys if k in x and x[k]!=''))) for x in xs]
    axes[row,col].plot([x[0] for x in ys],[x[1] for x in ys],color=color,linestyle=style,label=name)
 for col,title in enumerate(['Critic mean Q','Global parameter gradient norm','Actor loss']):
  axes[row,col].set_title(host+' | '+title);axes[row,col].set_xlabel('Formal updates (millions)');axes[row,col].legend(fontsize=7)
 axes[row,1].set_yscale('symlog',linthresh=1)
fig.suptitle('Logged diagnostics; global parameter norm is not the Actor action gradient',fontsize=12)
fig.savefig(F/'diagnostics.png');fig.savefig(F/'diagnostics.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for col,host in enumerate(['G','D']):
 for seed,style in [(0,'-'),(1,'--')]:
  name=f'{host}1_seed{seed}';xs=sorted([x for x in loss if x['job']==name],key=lambda x:float(x['step']))
  for key,color in [('floq_loss','#d95f0e'),('distilled_critic_loss','#225ea8'),('td_error','#31a354')]:
   k='training/critic/'+key;axes[col].plot([float(x['step'])/1e6 for x in xs],[float(x[k]) for x in xs],color=color,linestyle=style,label=f'{key} s{seed}')
 axes[col].set_yscale('symlog',linthresh=.1);axes[col].set_xlabel('Formal updates (millions)');axes[col].set_title(host+'1 Critic components');axes[col].legend(fontsize=7)
fig.suptitle('Noise-SUM Flow objective and scalar MSE have different scales; do not compare as accuracy')
fig.savefig(F/'critic_components.png');fig.savefig(F/'critic_components.pdf');plt.close(fig)
print('PLOTS_WRITTEN',len(list(F.glob('*'))))

from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];rows=json.loads((R/'reports/distance_metrics.json').read_text());eff=json.loads((R/'reports/controlled_effects.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
colors=['#167D9A','#CB5F32']; cps=[200000,600000,1000000]; bins=['0-3','4-7','8-11']
for filename,metric,ylabel,factor in [('01_neutral_distance','neutral','Neutral guidance (%)',100),('02_gap_distance','mean_abs_gap','Mean absolute Q(data) - Q(reference)',1),('03_ranking_distance','ranking_agreement','Critic-member ranking agreement (%)',100)]:
 fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
 for ax,cp in zip(axes,cps):
  for seed in [0,1]:
   for split,style in [('train','--'),('validation','-')]:
    rs=[next(r for r in rows if r['seed']==seed and r['checkpoint']==cp and r['split']==split and r['bfs_bin']==b) for b in bins]
    vals=np.array([r[metric]['mean'] for r in rs])*factor;cis=np.array([r[metric]['ci95'] for r in rs])*factor;x=np.arange(3)+(seed-.5)*.04
    ax.plot(x,vals,style,marker='o',markersize=4,color=colors[seed],alpha=1 if split=='validation' else .5,label=f'Seed {seed}, '+('validation' if split=='validation' else 'train'))
    if split=='validation':ax.fill_between(x,cis[:,0],cis[:,1],color=colors[seed],alpha=.13)
  ax.set(title=f'{cp//1000}k checkpoint',xticks=np.arange(3),xticklabels=bins,xlabel='BFS distance to task4 goal (cells)');ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel(ylabel);handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=4,frameon=False)
 fig.suptitle('Frozen original GFP | natural batches of 256 | shaded: 95% trajectory-cluster intervals',fontsize=11)
 fig.tight_layout();fig.savefig(R/'reports'/f'{filename}.png');fig.savefig(R/'reports'/f'{filename}.pdf');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,7));xs=[.2,.6,1.]
for seed in [0,1]:
 es=[next(e for e in eff if e['seed']==seed and e['split']=='validation' and e['checkpoint']==cp) for cp in cps]
 rs=[next(r for r in rows if r['seed']==seed and r['split']=='validation' and r['checkpoint']==cp and r['bfs_bin']=='8-11') for cp in cps]
 for ax,vs,title,ylabel in [(axes[0,0],[e['far_neutral']['mean']*100 for e in es],'Far-state neutral guidance','Percent'),(axes[0,1],[r['abs_gap_median'] for r in rs],'Far-state Q-gap magnitude','Median absolute Q gap'),(axes[1,0],[e['proxy']['far']['rank_auc'] for e in es],'Ranking of recorded 100-step progress','AUC (descriptive proxy)'),(axes[1,1],[e['adjusted_neutral_per_bfs_cell']['coef'][0]*100 for e in es],'Neutrality vs BFS after controls','Percentage points / BFS cell')]:
  ax.plot(xs,vs,'-o',color=colors[seed],label=f'Seed {seed}');ax.set(title=title,ylabel=ylabel,xlabel='Training checkpoint (million updates)',xticks=xs);ax.grid(alpha=.2)
axes[1,0].axhline(.5,color='gray',ls=':',label='Chance ordering');axes[1,0].set_ylim(.48,.55)
axes[0,0].legend(frameon=False);fig.suptitle('Validation split | repeated distance pattern does not establish harmful action filtering',fontsize=12)
fig.tight_layout();fig.savefig(R/'reports/04_training_stage.png');fig.savefig(R/'reports/04_training_stage.pdf')

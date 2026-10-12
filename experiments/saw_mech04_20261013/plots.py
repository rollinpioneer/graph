"""Standalone scientific plots; all stages and seeds, no selection by scores."""
import pathlib,csv,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=pathlib.Path(__file__).resolve().parent;F=P/'figures';F.mkdir(exist_ok=True)
def read(n):
    p=P/'results'/n
    return list(csv.DictReader(open(p))) if p.exists() else []
def main():
    scores=read('C_TASK_SCORES.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for v,color in [('A','#2463a8'),('B','#cf612d')]:
        rows=[x for x in scores if x['kind']=='stitch' and x['variant']==v and x['task']=='0' and x['scope']=='full' and x['valid_full_endpoint']=='True']
        if rows:axes[0].plot([int(x['training_seed']) for x in rows],[float(x['success_rate'])*100 for x in rows],'o-',color=color,label=v)
    axes[0].set(title='Stitch: complete 1M, new matched scenes',xlabel='Training seed (0 discovery; 1/2 confirmation)',ylabel='Five-task average success (%)',xticks=[0,1,2],ylim=(-3,103));axes[0].legend()
    for seed,color in [(1,'#2463a8'),(2,'#cf612d')]:
        rows=sorted([x for x in scores if x['kind']=='navigate' and x['training_seed']==str(seed) and x['task']=='5' and x['success_rate']!='None' and x['success_rate']!=''],key=lambda x:int(x['checkpoint_step']))
        if rows:
            y=np.array([float(x['success_rate'])*100 for x in rows]);lo=np.array([float(x['ci95_low'])*100 for x in rows]);hi=np.array([float(x['ci95_high'])*100 for x in rows]);axes[1].errorbar([int(x['checkpoint_step'])/1000 for x in rows],y,yerr=np.maximum(0,np.array([y-lo,hi-y])),fmt='o-',color=color,label=f'seed {seed}')
    axes[1].set(title='Navigate A: Task5, same 100 scenes',xlabel='Updates (thousands)',ylabel='Task5 success (%)',xticks=[100,300,500,1000],ylim=(-3,103));axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(F/'C_CONFIRMATION.png',dpi=180);fig.savefig(F/'C_CONFIRMATION.pdf');plt.close(fig)
    r=read('R_BRANCHES_navigate.csv');s=read('R_BRANCHES_stitch.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,rows,roles,title in [(axes[0],[x for x in r if x['task']=='5'],[('A',300000),('A',500000),('A',1000000),('B',1000000)],'Navigate discovery Task5 anchors'),(axes[1],s,[('A',1000000),('B',1000000)],'Stitch discovery anchors')]:
        vals=[[float(x['far_euclidean_progress']) for x in rows if x['variant']==v and int(x['checkpoint_step'])==step] for v,step in roles];ax.boxplot(vals,tick_labels=[f'{v} {step//1000}k' for v,step in roles],showfliers=True);ax.set(title=title,ylabel='Far-goal Euclidean progress over <=50 steps');ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2)
    fig.suptitle('Matched full-state behavior probes; repeated anchors are not independent models',fontsize=10);fig.tight_layout();fig.savefig(F/'R_MATCHED_BEHAVIOR.png',dpi=180);fig.savefig(F/'R_MATCHED_BEHAVIOR.pdf');plt.close(fig)
    ts=read('T_TASK_SCORES.csv')
    if ts:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,task in zip(axes,[1,3,4]):
            base=[x for x in ts if x['arm']=='-1' and x['task']==str(task)]
            if base:ax.axhline(float(base[0]['success_rate'])*100,color='gray',ls='--',label='common 500k start')
            for arm in range(4):
                rows=sorted([x for x in ts if x['arm']==str(arm) and x['task']==str(task)],key=lambda x:int(x['student_step']))
                if rows:ax.plot([int(x['student_step'])/1000 for x in rows],[float(x['success_rate'])*100 for x in rows],'o-',label=f'T{arm}')
            ax.set(title=f'Task {task}',xlabel='Student updates (thousands)',ylabel='Success (%)',ylim=(-3,103));ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('T: one common training seed; 25/50k n=20/task, 100k n=100/task',fontsize=10);fig.tight_layout();fig.savefig(F/'T_FOUR_ARMS.png',dpi=180);fig.savefig(F/'T_FOUR_ARMS.pdf');plt.close(fig)
if __name__=='__main__':main()

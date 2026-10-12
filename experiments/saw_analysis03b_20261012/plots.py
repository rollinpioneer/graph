"""Standalone scientific figures from complete paired episodes and bounded branches."""
from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'svg.hashsalt':'saw-analysis03b','pdf.fonttype':42})
out=ROOT/'figures';out.mkdir(exist_ok=True)
def export(fig,name):
    fig.savefig(out/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def main():
    rows=read(ROOT/'results/E1_SCORES.csv');fig,axes=plt.subplots(2,5,figsize=(14,6),sharex=True,sharey=True)
    for row,kind in enumerate(('navigate','stitch')):
        for column,task in enumerate(range(1,6)):
            ax=axes[row,column]
            for variant,color in [('A','#2563eb'),('B','#d97706')]:
                rr=sorted([x for x in rows if x['kind']==kind and int(x['task'])==task and x['variant']==variant],key=lambda x:int(x['checkpoint_step']))
                if not rr:continue
                xs=[int(x['checkpoint_step'])/1000 for x in rr];ys=np.array([float(x['success_rate']) for x in rr]);lo=np.array([float(x['ci_low']) for x in rr]);hi=np.array([float(x['ci_high']) for x in rr])
                ax.errorbar(xs,ys,yerr=np.vstack([ys-lo,hi-ys]),marker='o',capsize=3,color=color,label=variant)
            ax.set_title(f'{kind.title()} / Task {task}');ax.set_ylim(-.04,1.04);ax.grid(alpha=.2)
            if column==0:ax.set_ylabel('Success rate / Wilson 95% CI')
            if row==1:ax.set_xlabel('Updates (thousands)')
    axes[0,0].legend();fig.suptitle('Same complete scenes across stages: n=20 per task; missing B points remain absent');fig.tight_layout();export(fig,'E1_learning_curves')
    p=ROOT/'results/FINAL_SCORES.csv'
    if p.exists():
        rr=read(p);fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,kind in zip(axes,('navigate','stitch')):
            for variant,shift,color in [('A',-.18,'#2563eb'),('B',.18,'#d97706')]:
                x=[r for r in rr if r['kind']==kind and r['variant']==variant and r['group']=='-1'];x=sorted(x,key=lambda x:int(x['task']))
                if not x:continue
                yy=np.array([float(r['success_rate']) for r in x]);lo=np.array([float(r['ci_low']) for r in x]);hi=np.array([float(r['ci_high']) for r in x]);ax.bar(np.array([int(r['task']) for r in x])+shift,yy,width=.34,color=color,label=variant,yerr=np.vstack([yy-lo,hi-yy]),capsize=3)
            ax.set_title(kind.title());ax.set_xlabel('Official task');ax.set_ylim(0,1.06);ax.grid(axis='y',alpha=.2);ax.legend()
        axes[0].set_ylabel('Success rate / Wilson 95% CI');fig.suptitle('Frozen 1M endpoint only; n=100 per task, two independent scene groups');fig.tight_layout();export(fig,'final_scores')
    rr=read(ROOT/'results/E2_BRANCHES.csv');failure_ids=sorted(set(x['snapshot_id'] for x in rr if x['source_success']=='False'))
    if failure_ids:
        fig,axes=plt.subplots(1,2,figsize=(11,4));conditions=['C0','C1','C2','C3','C4']
        for sid in failure_ids:
            selected={x['condition']:x for x in rr if x['snapshot_id']==sid}
            axes[0].plot(conditions,[float(selected[c]['net_displacement']) for c in conditions],marker='o',label=sid)
            axes[1].plot(conditions,[float(selected[c]['far_progress']) for c in conditions],marker='o')
        axes[0].set_ylabel('Net displacement (maze units)');axes[1].set_ylabel('Original far-goal Euclidean progress')
        for ax in axes:ax.grid(alpha=.2);ax.set_xlabel('50-step matched branch')
        axes[0].legend(fontsize=7);fig.suptitle('Four gate-qualified Navigate failures; Euclidean progress is descriptive, not route value');fig.tight_layout();export(fig,'E2_failure_branches')
    print('FIGURES_DONE',flush=True)
if __name__=='__main__':main()

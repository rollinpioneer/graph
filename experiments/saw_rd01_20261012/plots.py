"""Light scientific figures from complete metric tables."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


def main():
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    colors={'navigate':'#2869b8','stitch':'#c86328'}
    for kind in ('navigate','stitch'):
        rows=list(csv.DictReader(open(ROOT/'results'/f'{kind}_training.csv')))
        x=np.array([float(r['step']) for r in rows])
        for ax,metric in zip(axes.flat[:3],['value/value_loss','low_actor/actor_loss','waypoint/waypoint_loss']):
            ax.plot(x,[float(r[metric]) for r in rows],label=kind,color=colors[kind],alpha=.8)
            ax.set_title(metric);ax.set_xlabel('Gradient updates');ax.grid(alpha=.2)
        dev=json.load(open(ROOT/'results'/f'{kind}_development_eval.json'))
        axes.flat[3].plot([r['step'] for r in dev],[r['success'] for r in dev],color=colors[kind],label=kind+' development')
        final=json.load(open(ROOT/'results'/f'{kind}_FINAL_EVAL.json'))
        axes.flat[3].scatter([1000000],[final['overall_success']],s=80,marker='x',color=colors[kind],label=kind+' fresh final')
    axes.flat[3].set_title('Success: 5 tasks × 50 episodes, one training seed')
    axes.flat[3].set_ylim(-.02,1.02);axes.flat[3].set_xlabel('Gradient updates');axes.flat[3].legend(fontsize=8);axes.flat[3].grid(alpha=.2)
    axes.flat[0].legend();fig.tight_layout()
    fig.savefig(ROOT/'results/TRAINING_CURVES.png',dpi=160);fig.savefig(ROOT/'results/TRAINING_CURVES.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,kind in zip(axes,('navigate','stitch')):
        env=ogbench.make_env_and_datasets(f'pointmaze-large-{kind}-v0',env_only=True);u=env.unwrapped
        ts=pickle.load(open(ROOT/'runs'/f'final_{kind}'/'confirmation_trajectories.pkl','rb'))
        positions=np.concatenate([t['states'] for t in ts])
        xlow,ylow=u.ij_to_xy((0,0)); xhigh,yhigh=u.ij_to_xy((u.maze_map.shape[0]-1,u.maze_map.shape[1]-1))
        ax.hist2d(positions[:,0],positions[:,1],bins=(48,36),range=[[xlow-2,xhigh+2],[ylow-2,yhigh+2]],cmap='YlOrRd',norm=matplotlib.colors.LogNorm())
        for cell in np.argwhere(u.maze_map==1):
            x,y=u.ij_to_xy(cell);ax.add_patch(matplotlib.patches.Rectangle((x-2,y-2),4,4,facecolor='#444444'))
        final=json.load(open(ROOT/'results'/f'{kind}_FINAL_EVAL.json'))
        ax.set_title(f'{kind}: all 250 final trajectories, success {final["overall_success"]:.1%}')
        ax.set_aspect('equal');ax.set_xlabel('Full state qpos x');ax.set_ylabel('Full state qpos y');env.close()
    fig.tight_layout();fig.savefig(ROOT/'results/FINAL_OCCUPANCY.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); labels=[];lows=[];actors=[]
    for phase in ('discovery','confirmation'):
        for kind in ('navigate','stitch'):
            path=ROOT/'results'/f'{kind}_{phase}_DIAGNOSIS.json'
            if not path.exists():continue
            d=json.load(open(path)); labels.append(kind+'\n'+phase);lows.append(d['D3_success']['low_actor']);actors.append(d['D3_success']['actor'])
    x=np.arange(len(labels));ax.bar(x-.18,lows,.36,label='low_actor');ax.bar(x+.18,actors,.36,label='actor, same waypoint')
    ax.set_xticks(x,labels);ax.set_ylim(0,1.05);ax.set_ylabel('Official goal success within recorded budget ≤25');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'results/SHORT_POLICY_COMPARISON.png',dpi=160);plt.close(fig)
    print('PLOTS_DONE',flush=True)


if __name__=='__main__':main()

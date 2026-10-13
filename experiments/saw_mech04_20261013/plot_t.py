"""T-only scientific figure; preserve completed R/C figures."""
from plots import *
def main():
    ts=read("T_TASK_SCORES.csv");fig,axes=plt.subplots(1,3,figsize=(13,4));labels=["T0 F100","T1 F50-mask","T2 F50-renorm","T3 F50+R50"]
    for ax,task in zip(axes,[1,3,4]):
        base=[x for x in ts if x["arm"]=="-1" and x["task"]==str(task)]
        if base:ax.axhline(float(base[0]["success_rate"])*100,color="gray",ls="--",label="common A500k start")
        for arm in range(4):
            rows=sorted([x for x in ts if x["arm"]==str(arm) and x["task"]==str(task)],key=lambda x:int(x["student_step"]))
            if rows:
                line=ax.plot([int(x["student_step"])/1000 for x in rows],[float(x["success_rate"])*100 for x in rows],"o-",label=labels[arm])[0]
                final=[x for x in rows if x["student_step"]=="100000"]
                if final:
                    x=final[0];y=float(x["success_rate"])*100;lo=float(x["ci95_low"])*100;hi=float(x["ci95_high"])*100
                    ax.errorbar([100],[y],yerr=np.maximum(0,np.array([[y-lo],[hi-y]])),fmt="none",color=line.get_color(),capsize=3)
        ax.set(title=f"Task {task}",xlabel="Student updates (thousands)",ylabel="Success (%)",ylim=(-3,103),xticks=[25,50,100]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc="lower left");fig.suptitle("T: one common seed0 start; 25/50k n=20/task, 100k n=100/task; bars: scene Wilson95",fontsize=10);fig.tight_layout();fig.savefig(F/"T_FOUR_ARMS.png",dpi=180);fig.savefig(F/"T_FOUR_ARMS.pdf");plt.close(fig)
if __name__=="__main__":main()

"""Plot-only repair: move legends outside data; do not rerun any statistic or model."""
from core import *
import inspect
import behavior_analysis as module
def main():
    guard(); original=ROOT/'figures/action_norm_and_actual_reversals.png'
    output=ROOT/'figures/behavior_visual_v2'; assert not output.exists(); output.mkdir()
    before={str(p):sha(p) for p in ROOT.glob('*.csv')}; before[str(original)]=sha(original)
    code=inspect.getsource(module.plot_morphology)
    old="axis.legend(fontsize=7, loc='upper right')"
    assert old in code
    code=code.replace(old,"axis.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))")
    ns=dict(module.__dict__);exec(compile(code,__file__,'exec'),ns)
    data=module.load_measurements(ROOT/'BEHAVIOR_ROLLOUTS.csv')
    ns['plot_morphology'](data,output)
    assert all(sha(p)==v for p,v in before.items())
    bank=pickle.load(open(ROOT/'runs/SCENE_GOAL_BANK.pkl','rb'))
    labels={x['condition_id']:(f"official_Task{x['scene']['task']}" if x['source']=='archived_P3_complete_initial' else 'constructed_start_'+str(x['start_cell'])) for x in bank}
    data['task_or_constructed_start']=data['condition_id'].map(labels)
    selected=data.sort_values(['condition_id']).groupby(['task_or_constructed_start','distance','category','seed','variant'],sort=True).head(1)
    module.save_table(ROOT/'results/SELECTED_TRAJECTORY_EXAMPLES.csv',selected)
    write(ROOT/'receipts/BEHAVIOR_FIGURE_V2.json',dict(status='PLOT_ONLY_LEGEND_REPAIR',new_path=str(output/'action_norm_and_actual_reversals.png'),new_sha256=sha(output/'action_norm_and_actual_reversals.png'),original_path=str(original),original_sha256=sha(original),CSV_unchanged=True,new_model_forwards=0,new_physical_steps=0))
    print('BEHAVIOR_FIGURE_V2_COMPLETE',flush=True)
if __name__=='__main__':main()

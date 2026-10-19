from common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scenes=sum([CFG['exploration'][s] for s in SPLITS],[])
for s in scenes:assert json.loads((R/f'diagnosis/{s:04d}/COMPLETE.json').read_text())['status']=='PASS'
frame=pd.concat([pd.read_csv(R/f'diagnosis/{s:04d}/frame_metrics.csv') for s in scenes],ignore_index=True)
top=pd.concat([pd.read_csv(R/f'diagnosis/{s:04d}/top1_transitions.csv') for s in scenes],ignore_index=True)
ctx=pd.concat([pd.read_csv(R/f'diagnosis/{s:04d}/candidate_context.csv') for s in scenes],ignore_index=True)
assert len(frame)==608*3 and len(top)==608*3*2
cols=['known_candidates','CD_removed','known_removed','known_survivors','known_true_collisions_removed','known_true_collisions_kept','known_good_removed','known_good_kept']
counts=frame[frame.seed==SEEDS[0]].groupby('scene')[cols].sum().reset_index();counts['candidate_frames']=frame[frame.seed==SEEDS[0]].groupby('scene').size().values*1024;counts['known_quality_coverage']=counts.known_candidates/counts.candidate_frames;counts['CD_good_rejection_rate_among_known_good']=counts.known_good_removed/(counts.known_good_removed+counts.known_good_kept);counts['CD_missed_collision_rate_among_known_collisions']=counts.known_true_collisions_kept/(counts.known_true_collisions_kept+counts.known_true_collisions_removed);counts.to_csv(R/'results/diagnosis_CD_counts.csv',index=False)
corrcols=[c for c in frame if c.startswith('rho_')];frame.groupby(['scene','seed'])[corrcols].median().reset_index().to_csv(R/'results/diagnosis_score_quality_correlations.csv',index=False)
rows=[]
for (scene,seed,cd),g in top.groupby(['scene','seed','use_cd']):
 damage=g.EG_valid&~g.S_valid;rescue=~g.EG_valid&g.S_valid
 rows.append(dict(scene=int(scene),seed=int(seed),use_cd=bool(cd),N=len(g),failure_to_success=int(rescue.sum()),success_to_failure=int(damage.sum()),EG_good_demoted_Q_rank_median=float(g.loc[damage,'EG_top1_rank_in_Q_survivors'].median()) if damage.any() else None,EG_good_absent_S_selected50=int((damage&~g.EG_top1_in_S_selected_top50).sum()),AP_loss_frames=int((g.EG_AP>g.S_AP).sum())))
csv_write('results/diagnosis_top1_summary.csv',rows)
geometry=[]
for variable in ['width_bin','depth_mm','camera_approach_alignment_bin','visible_area_bin']:
 agg=ctx.groupby(['scene','seed','use_cd','model',variable]).agg(candidate_selections=('valid','size'),valid_rate=('valid','mean'),collision_rate=('collision','mean'),empty_rate=('empty','mean')).reset_index().rename(columns={variable:'bin'})
 agg.insert(0,'variable',variable);geometry.append(agg)
pd.concat(geometry,ignore_index=True).to_csv(R/'results/diagnosis_geometry_strata.csv',index=False)
cases=top[top.use_cd].copy();cases['S_minus_EG_AP']=cases.S_AP-cases.EG_AP;chosen=[]
for seed in SEEDS:chosen.append(cases[(cases.scene==136)&(cases.seed==seed)].nsmallest(2,'S_minus_EG_AP'))
for scene in [102,158]:chosen.append(cases[cases.scene==scene].nsmallest(2,'S_minus_EG_AP'))
pd.concat(chosen).to_csv(R/'results/exploratory_bad_cases.csv',index=False)
decomp=pd.read_csv(R/'results/CD_AP_decomposition.csv');grouped=decomp.groupby(['split','scene']).mean(numeric_only=True).reset_index();grouped.to_csv(R/'results/CD_AP_decomposition_scene_mean.csv',index=False)
allmean=decomp.groupby(['split']).mean(numeric_only=True).mean(numeric_only=True)
fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(grouped));ax.bar(x-.2,grouped.delta_noCD,.4,label='S-EG before CD');ax.bar(x+.2,grouped.delta_CD,.4,label='S-EG after CD');ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,[f'{s:04d}' for s in grouped.scene],rotation=45);ax.set_ylabel('AP percentage points');ax.set_title('Exploratory C1: fixed scenes, all 256 frames');ax.legend();fig.tight_layout();fig.savefig(R/'results/CD_AP_decomposition.png',dpi=160);plt.close(fig)
target=frame[frame.scene==136];tc=counts[counts.scene==136].iloc[0].to_dict();tq=target.groupby('seed')[corrcols].median().to_dict('index');t_top=[r for r in rows if r['scene']==136 and r['use_cd']]
summary=dict(status='COMPLETE',frames=608,scene0136_frames=256,other_scenes_frames_each=32,known_quality_scope='Only union selected by V3 official R0/S paths; no unknown-as-failure. Candidate selections and three heads are dependent descriptive observations, not independent causal evidence.',width_bins=['<.02','[.02,.04)','[.04,.06)','[.06,.08)','>=.08 (clipped at .1)'],camera_approach_bins=['<.25','[.25,.5)','[.5,.75)','>=.75 absolute camera-Z approach component'],visibility_bins=['<1%','1%-5%','5%-15%','>=15% observed image area; not full-object visibility/occlusion'],full_C1_AP_decomposition={k:float(allmean[k]) for k in ['delta_noCD','delta_CD','EG_benefit_CD','S_benefit_CD','CD_change_in_rescoring_gain']},scene0136_CD_counts=tc,scene0136_median_correlations=tq,scene0136_top1=t_top,exploratory_bad_case_selection='Two most negative AP frames per head in0136 and two worst frames from each other negative scene102/158; illustrative posthoc selection, no unbiased causal claim')
write('results/diagnosis_summary.json',summary);print(json.dumps(summary),flush=True)

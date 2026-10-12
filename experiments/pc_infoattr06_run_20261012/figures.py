"""Static scientific plots of commands and physical response, CPU only."""
import os,pathlib,json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(os.environ['R']);F=R/'figures';F.mkdir(exist_ok=True)
with (R/'results/P2_step_evidence.csv').open() as f:rows=list(csv.DictReader(f))
names=sorted(set(x['run'] for x in rows))
for name in names:
 n=max(int(x['object']) for x in rows if x['run']==name)+1
 objectid=4 if n==6 else 0;rr=[x for x in rows if x['run']==name and int(x['object'])==objectid];t=np.array([int(x['t_input']) for x in rr]);fig,axs=plt.subplots(4,1,figsize=(11,9),sharex=True)
 for k in ['requested_toward_object_m','actual_toward_object_m']:axs[0].plot(t,[float(x[k]) for x in rr],label=k.replace('_',' '),alpha=.8)
 axs[0].axhline(0,color='k',lw=.5);axs[0].set_ylabel('radial motion m');axs[0].legend(fontsize=8)
 axs[1].plot(t,[float(x['eef_object_before_m']) for x in rr]);axs[1].set_ylabel('EEF to cube m')
 axs[2].plot(t,[float(x['goal_distance_before_m']) for x in rr]);axs[2].axhline(.04,ls=':',color='k');axs[2].set_ylabel('cube to goal m')
 axs[3].plot(t,[float(x['request_actual_cosine']) for x in rr]);axs[3].set_ylim(-1.1,1.1);axs[3].set_ylabel('request/response cosine');axs[3].set_xlabel('action before state t; actual result at t+1')
 fallen=[int(x['t_input']) for x in rr if x['fall_before']=='True']
 if fallen:
  for ax in axs:ax.axvspan(min(fallen),max(t),color='red',alpha=.10)
 fig.suptitle(name+' object '+str(objectid)+' | radial retreat is not an incorrect-action label');fig.tight_layout();fig.savefig(F/(name+'_chain.png'),dpi=120);plt.close(fig)
gate=json.loads((R/'results/P1_health_gate.json').read_text());rr=json.loads((R/'results/P1_scored_object_rows.json').read_text());hold=[q for q in rr if q['role']=='independent_holdout' and q['visible']]
fig,ax=plt.subplots(figsize=(8,4));errs=[q['error_visible_centroid_px'] for q in hold if q['error_visible_centroid_px'] is not None];ax.hist(errs,bins=np.arange(0,81,5));ax.axvline(10,ls=':',color='red');ax.set_xlabel('held-out visible object centroid error px');ax.set_ylabel('object-view frames');ax.set_title('Reader selfcheck: %.1f%% within10px; gate80%%; missing=%d'%(100*gate['within_10px_fraction'],sum(q['error_visible_centroid_px'] is None for q in hold)));fig.tight_layout();fig.savefig(F/'P1_holdout_selfcheck.png',dpi=140);plt.close(fig)
print('FIGURES_COMPLETE',len(names)+1,flush=True)

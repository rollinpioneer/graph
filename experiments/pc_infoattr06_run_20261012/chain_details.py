"""Join separately computed reader output, source reconstruction, and step evidence."""
import runpy,pathlib
globals().update(runpy.run_path(str(pathlib.Path(__file__).resolve().parent/'analysis.py')))
norm=json.loads((R/'receipts/action_normalizer_audit.json').read_text());lo=np.array(norm['min']);hi=np.array(norm['max']);source_roi=json.loads((P/'results/P1_actual_recordings_ROI.json').read_text());detail=[];sequences=[];turns=[]
for name,new in [(x,False) for x in ['validate6_permuted','validate6_blocked','validate3_permuted','validate3_blocked','supplement3','supplement6']]+[(x,True) for x in newnames]:
 z=load(name,new);dd=z['distances'];st=z['states'];ee=z['eef'];events=json.loads((R/'results'/('P2_events_'+name+'.json')).read_text())['turn_events']
 for j in range(len(z['goal'])):
  best=np.minimum.accumulate(dd[:,j]);improvement=np.flatnonzero(best[:-1]-best[1:]>.002);lastbest=int(improvement[-1]) if len(improvement) else None
  nop=[]
  for t in range(20,len(dd)):
   seg=st[t-20:t+1]
   if not (seg[:,:,2]<.975).any() and dd[t,j]>=.04 and np.linalg.norm(seg[:,:,:3]-seg[0,:,:3],axis=-1).max()<.001 and dd[t-20,j]-dd[t-20:t+1,j].min()<.001:nop.append(t)
  fall=np.flatnonzero((st[:,:,2]<.975).any(-1));objectturn=[q for q in events if q['object']==j];firstnp=nop[0] if nop else None
  turns.extend([dict(run=name,normal=new or name.startswith('supplement'),pre_fall=not(len(fall) and q['step']>=fall[0]),linked_no_progress=bool(firstnp is not None and abs(q['step']-firstnp)<=30),**q) for q in objectturn])
  sequences.append({'run':name,'object':j,'normal_sampling':new or name.startswith('supplement'),'last_newbest_2mm_progress_action':lastbest,'first_strict_all_object_stationary_object_no_progress_end':firstnp,'no_progress_ends':nop,'first_any_fall_state':int(fall[0]) if len(fall) else None,'turns':objectturn,'final_goal_error_m':float(dd[-1,j]),'interpretation':'newbest differs from local jitter; no-progress observation does not mark all subsequent motion useless'})
  # Event-indexed chain annex: fixed diagnostic points plus events, clearly posthoc.
  ts=set([0,50,68,88,108,150,199] if not new else [0,50,100,150,199])
  for ev in objectturn:ts.update([ev['step']-5,ev['step'],ev['step']+5])
  if lastbest is not None:ts.update([lastbest,lastbest+1])
  if firstnp is not None:ts.update([firstnp-20,firstnp,firstnp+20])
  for t in sorted(k for k in ts if 0<=k<200):
   base=next(q for q in eventrows if q['run']==name and q['object']==j and q['t_input']==t);roi=[q for q in source_roi if q['run']==name and q['object_id']==j and q['state_step']==t];reader=[q for q in rows if q['run']==name and q['object']==j and q['step']==t];a=np.array([base['raw_action_x'],base['raw_action_y'],base['raw_action_z']])
   detail.append({'run':name,'object':j,'step':t,'original_input_file':str((R if new else P)/'runs'/name/'data/trajectory.npz'),'original_particle_key':'used_current_particles[t]/used_goal_particles[t]','reader_rows':reader,'reader_eligible_for_information_loss_claim':False,'reader_missing_at_event_means':'not sampled in frozen P1 point list' if not reader else 'selfcheck failed; descriptive only','prior_reconstruction_rows':roi,'prior_reconstruction_scope':'PC-INFOCHAIN-05 evidence reuse; no new decode','inverse_normalized_clipped_action':(2*(a-lo)/(hi-lo)-1).tolist(),'normalized_unclipped_generated_action':None,'normalized_missing_reason':'prior recorder stores post-unnormalization H5; clipping is not invertible','step_evidence':base})
save('joined_evidence_chain.json',detail);save('P2_event_sequences.json',sequences);save('P2_turn_candidates.json',turns)
summary=[]
for name in sorted(set(q['run'] for q in turns)):
 a=[q for q in turns if q['run']==name and q['pre_fall']];summary.append({'run':name,'prefall_turn_candidates':len(a),'unique_objects':len(set(q['object'] for q in a)),'linked_no_progress_events':sum(q['linked_no_progress'] for q in a),'post_requested_departure_steps':sum(q['post_requested_departure_steps'] for q in a),'post_actual_departure_steps':sum(q['post_actual_departure_steps'] for q in a),'requested_approach_actual_departure_steps':sum(q['post_approach_actual_departure_steps'] for q in a),'unit':'overlapping5step event windows; not independent tasks or population frequency'})
save('P2_turn_summary.json',summary)
# Holdout color localization precision: prediction first; nearest reference only in scoring.
idcheck=[]
for pred in predictions:
 if pred['role']!='independent_holdout':continue
 z=load(pred['run']);ids,_=identity(z);t=pred['step'];xyz=np.c_[z['goal'],np.full(len(ids),1.0425)] if t is None else z['states'][t,:,:3]
 for v,pp in enumerate(pred['predictions']):
  for color,out in pp.items():
   xy=np.array(out['xy_px']);d=np.linalg.norm(project(xyz,v)-xy,axis=-1);j=int(d.argmin());idcheck.append({'run':pred['run'],'step':t,'view':v,'predicted_color':int(color),'nearest_reference_color':int(ids[j]),'nearest_projection_error_px':float(d[j]),'color_correct':int(color)==int(ids[j]),'color_and_10px_correct':bool(int(color)==int(ids[j]) and d[j]<=10)})
save('P1_holdout_identity_scoring.json',idcheck)
save('P1_identity_summary.json',{'predicted_color_positions':len(idcheck),'color_correct':sum(q['color_correct'] for q in idcheck),'color_and_10px_correct':sum(q['color_and_10px_correct'] for q in idcheck),'geometry_scoring_only':True,'no_rule_refitting_after_health_failure':True})
print('CHAIN_DETAILS_COMPLETE',len(detail),summary,flush=True)

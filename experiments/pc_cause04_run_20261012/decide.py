"""Freeze G1 and single-mechanism eligibility after discovery, before any validation."""
from common import *
es=json.loads((R/'results/episode_index.json').read_text());p1=json.loads((R/'results/P1_summary.json').read_text());p2=json.loads((R/'results/P2_summary.json').read_text());p3=json.loads((R/'results/P3_summary.json').read_text())
gc=json.loads((R/'results/G_C_receipt.json').read_text());resource=json.loads((R/'results/resource_summary.json').read_text())
main={'candidate':'robot directly re-engages already reached objects, causing loss of achieved goals','competitor':'secondary cube interaction during manipulation of a pending object causes the same goal loss',
'evidence_level':'candidate causes; currently object-level movement/departure only',
'support':['PC_P12303/P1_crowded8/trajectory/slot05 object1 departure59 finalXY0.2793; object5 departure100 finalXY0.1131',
'PC_P12303/P1_sparse8/trajectory/slot04 object1 departure125 finalXY0.05446; object3 departure99 finalXY0.25993'],
'counterexamples':['successful P1_sparse8 slot00 object3 departure49 finalXY0.0141, 22 movement-object-steps after hit',
'successful P1_crowded8 slot03 objects0/5 depart45/48, return to finalXY0.01089/0.01780; 39 movement-object-steps after hit',
'no valid counterpart readout: direct vs secondary movement are observationally indistinguishable'],
'common_small_experiment':{'manipulated_factor':'paired relative gap/path clearance between an active transport object and previously reached object; match transport and edge demand',
'invariants':'EMA/DLP/normalizer,H5,T100,200controls,table/cameras/OSC/dt/physics,object count,colors; matched initial transport/edge distance within preregistered bounds',
'arms':'2 arms x6 matched new scenes, randomized batch slots and independent policy streams; do not reuse discovery states as validation',
'cost_cap':'12 full episodes,<=1GPUh planned; exact scenes/seeds/geometry tolerances freeze only if measures qualify',
'direct_readout':'both contact identities, robot-to-reached vs cube-to-reached timing, pre-crossing direction, object error/departure',
'prediction_direct':'initial impulse on departing reached object preceded by robot contact without prior cube contact; gap effect absent unless robot route changes',
'prediction_secondary':'initial impulse preceded by contact with moving cube; widened path suppresses those chains and their goal-loss frequency',
'support_rule':'positive/negative calibrated pair sequence demonstrates one route, comparison changes that route with the predicted departure pattern, opposite route counterexamples retained',
'refutation_rule':'observed initial impulse reliably follows competing route or gap comparison fails preregistered chain prediction despite valid measurement',
'unresolved_rule':'missing pair identity/direction, same-dt order ambiguous, no trigger, visual/geometry route changes cannot be separated, or full-pipeline comparability fails',
'validity':'NOT PASSED: G-C fails official backend; CPU fixture has pair capability but not full pipeline/temporal/force calibration',
'confounders':'gap changes current/goal RGB and robot route; secondary and direct effects may coexist; no identity-level decision truth',
'scope':'local rollback mechanism only; cannot establish quantity contribution without controlled n comparison'},
'G1':'FAILED; necessary direct readout not valid and layout/image confounds remain','executed':False}
workload={'candidate':'limited time relative to pending transport workload','competitor':'degradation in goal/perception representation or contact efficiency at same pending workload',
'support':'historical initial total distance3C0.54567m vs6C1.08099m; SCALE02 roughly0.234-0.237m vs0.4735m, all pending3vs6',
'counterexamples':'same 6C P1 initial pending6 and transport approx0.08m/object yields success and failure; successful episodes also revisit reached objects',
'common_experiment':'same6C current scene, precomplete some goal targets; update goal images, hold time/model/current scene',
'different_predictions':'both workload reduction and easier target representation can improve the same endpoints; no distinct valid direct predictions currently specified',
'manipulated':'pending goal demand','invariants':'same current6C scene,200controls,model/physics','direct_readout':'task metrics valid, internal target selection unmeasured',
'confounders':'goal-image/distribution and remaining transport difficulty change together','cost_cap':'would consume existing12-episode validation cap; NOT launched',
'support_refute_unresolved':'cannot uniquely support either from gain alone; shared prediction means unresolved','scope':'not a pure quantity effect','G1':'FAILED: predictions overlap'}
g1={'utc':utc(),'strongest_observable_line':'rollback after achieving targets plus workload increase; no strongest confirmed cause',
'cause_comparisons':[main,workload],'gate':'FAILED','stop':'no additional P1 descriptive collection or validation','data':'all prior episodes discovery; no heldout experimental results created'}
dump('G1_cause_comparison.json',g1)
ranking={'utc':utc(),'comparison_axes':['measurement credibility','distinct competitor predictions','repeatability','link to P1','cost'],
'ranked_candidates':[
{'rank':1,'mechanism':'reached-object rollback: direct re-engagement vs secondary interaction','measurement':'object movement credible, required pairs unavailable',
'distinct_predictions':'contact route differs but unreadable','repeatability':'departure repeats in failures and successes, not sufficient','P1_link':'plausible lost progress; contribution unidentified','cost':'<=12 validation +<=6 supplementation only after gates',
'qualified':False,'blocking_gates':['G-C','G1']},
{'rank':2,'mechanism':'limited transport workload budget','measurement':'pending demand/initial transport valid','distinct_predictions':'goal changes also change representation difficulty, shared endpoints',
'repeatability':'quantity/workload associated in separate cohorts','P1_link':'direct quantity opportunity link, no controlled contribution','cost':'<=12 episodes if distinguishable design exists','qualified':False,'blocking_gates':['G1']},
{'rank':3,'mechanism':'abnormal ineffective approach during no-progress','measurement':'control chain valid; normal approach baseline absent',
'distinct_predictions':'normal temporary reapproach remains competitor','repeatability':'29 episode-first candidates, old reruns nonreproducible','P1_link':'time consumption associated',
'cost':'would require qualified normal matches and contact-specific readout','qualified':False,'blocking_gates':['P2 normal negative control','G-C when claiming contact failure']},
{'rank':4,'mechanism':'complete visual disappearance / goal encoding failure','measurement':'77 goal-image color/projection bijections consistent; pixel proxies not DLP fidelity',
'distinct_predictions':'no visual-representation manipulation with valid isolated direct readout','repeatability':'visible-color cases succeed and fail','P1_link':'possible but unproven','cost':'unqualified','qualified':False,'blocking_gates':['G1']}
],'selected_mechanism':None,'validation_status':'NOT STARTED: no qualified candidate','validation_episode_count':0,'no_next_mechanism_after_stop':True}
dump('single_mechanism_selection.json',ranking)
matrix=[
{'question':'P1','confirmed':'initial pending workload increases; individual progress can be lost; successful cases can also depart/recover',
'candidate_causes':'direct reoperation, secondary interaction, workload, representation','counter':'successful P1 slots sparse00/crowded03 revisit/recover; sparse advantage not replicated',
'unknown':'actual counterpart/selection; DLP fidelity; controlled quantity contribution','validation':'G1 fails, none launched','quantity_connection':'historical100 and SCALE02 200 reported separately; no attributable loss fraction'},
{'question':'P2','confirmed':'29 first windows /3459 overlapping detector windows; target transforms agree for21 available first chains, max5.722e-8m',
'candidate_causes':'normal reapproach or ineffective approach/contact, unresolved','counter':'successful episodes are not automatically normal; 3 geometric proposals fail fixed future normal confirmation',
'unknown':'qualified matched normal windows, contact counterpart/effectiveness; normalized action not separately persisted',
'validation':'negative-control gate fails; no abnormal-stall intervention','quantity_connection':'common-window episode counts and pending-opportunity normalization only'},
{'question':'P3','confirmed':'18 fall episodes,31 fallen objects,43 first bbox-crossed objects incl12 without confirmed fall; strict pre-cross speeds/actions archived',
'candidate_causes':'direct robot contact, secondary collision, motion continuation, mixed','counter':'crossing can recover; post-cross high fall speed excluded; CC_NEVER zero is missing data',
'unknown':'valid full-pipeline identities, order within dt, force/impulse calibration','validation':'G-C fails; no contact supplement/propagation experiment','quantity_connection':'falls associated with quantity by protocol; causal contribution unidentified'}
]
dump('cause_evidence_matrix.json',matrix)
status={'utc':utc(),'card':'PC-CAUSE-04 v1.1','branch':'codex/pc-cause-04-20261012-card','base_commit':'3fd512411f878d514dc62970624c32ddd05aa111',
'index_and_identity':'COMPLETED','P1_existing_data':'COMPLETED one-pass','P2_existing_data':'COMPLETED; qualified normal controls not obtained',
'P3_existing_data':'COMPLETED descriptions, counterpart classes unknown','contact_selfcheck':'COMPLETED GPU +CPU tensor fixture; initial PATH failure archived',
'readout_repairs':'STOPPED at2 cycles','G1':'FAILED /P1 mechanism unresolved','G_C':'FAILED official backend',
'supplement':'NOT EXECUTED,0/6: no discriminating qualified contact supplement','validation':'NOT EXECUTED,0/12: no qualified single mechanism',
'algorithm_development':'NOT EXECUTED','stop_reason':'scientific/measurement gates, not budget exhaustion','next_algorithm_recommendation':None}
dump('stage_status.json',status)
# Update interpretation without changing frozen comparisons or thresholds.
p2['gate']='FAILED: no candidate passed frozen offline normal confirmation, and validated start visibility absent; geometric proposals insufficient'
p2['later_image_readout']='goal RGB/projection correspondence consistent in77 episodes; sparse color ROI counts remain proxy; not retroactively matched'
dump('P2_summary.json',p2)
p1['G1_status']='FAILED; see G1_cause_comparison.json';dump('P1_summary.json',p1)
print('DECISION_FROZEN',ranking['selected_mechanism'],flush=True)

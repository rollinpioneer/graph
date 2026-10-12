from common import *
rows=[]
for n in [3,6]:
 for q in json.loads((R/'runs'/('targets%d'%n)/'data/goal_test_results.json').read_text()):
  a=np.array(q['actions_generated_not_executed']);delta=np.linalg.norm(a[1]-a[2],axis=-1);within=np.array([np.linalg.norm(a[h,i]-a[h,j],axis=-1) for h in range(3) for i in range(8) for j in range(i+1,8)])
  rows.append({'n':n,'case_id':q['case_id'],'median_CRN_goal_action_L2_by_H_step':np.median(delta,axis=0).tolist(),'median_within_goal_sampling_L2_by_H_step':np.median(within,axis=0).tolist(),'first_action_CRN_effect_to_sampling_ratio':float(np.median(delta[:,0])/max(np.median(within[:,0]),1e-12)),'scope':'H index0isfirstproposedcontrolaction, indices1..4unexecuted; no direction correctnesscriterion'})
dump('P1_goal_response_by_horizon.json',rows)
print('FIRST_ACTION_EFFECTS',[(q['n'],q['case_id'],q['median_CRN_goal_action_L2_by_H_step'][0],q['first_action_CRN_effect_to_sampling_ratio']) for q in rows])

"""Algebraic witnesses and configuration checks, not production R-GCN/PPO/RL tests."""
from pathlib import Path
import json, math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,ok,detail=None):
 checks.append({'name':name,'passed':bool(ok),'detail':detail})
 if not ok: raise AssertionError(name)
def mat(n,pairs):
 m=np.eye(n)
 for a,b in pairs:m[a,b]+=1;m[b,a]+=1
 return m
def diff(k,h,dx,L):
 dk=np.linalg.matrix_power(k,L)@dx;dh=np.linalg.matrix_power(h,L)@dx
 return dk,dh,dh-dk
rng=np.random.default_rng(211)
Z00,Z10,Z01,Z11=rng.normal(size=(4,3,7))
dp=(Z11-Z01)-(Z10-Z00)
check('2x2_identity',np.allclose(dp,(Z11-Z10)-(Z01-Z00),atol=1e-12))
# Scalar linear propagation with reverse messages and roots. Not main model.
k=mat(4,[(0,1),(1,2),(2,3)]);h=mat(4,[(0,1),(1,2),(2,3),(0,2)])
dk,dh,d=diff(k,h,np.array([0.,1,0,0]),4)
check('static_action_edge_can_interact',d[3]==5.0,{'DK_goal':dk[3],'DH_goal':dh[3],'DP_goal':d[3]})
check('empty_prior_null',np.array_equal(diff(k,k,np.array([0.,1,0,0]),4)[2],np.zeros(4)))
check('empty_patch_null',all(np.array_equal(v,np.zeros(4)) for v in diff(k,h,np.zeros(4),4)))
# A,p,q,B,goal: p is unchanged TRUE; only q changes. A has both nominal effects.
k5=mat(5,[(0,1),(0,2),(3,4)])
h_action=mat(5,[(0,1),(0,2),(3,4),(0,3)])
h_fact=mat(5,[(0,1),(0,2),(3,4),(1,3)])
dx=np.array([0.,0,1,0,0])
a=diff(k5,h_action,dx,3)[2][4]
b=diff(k5,h_fact,dx,4)[2][4]
check('source_candidate_anchor_unchanged_not_gate',a!=0,{'goal_response_L3':a})
check('fact_anchor_not_hard_gate',b!=0,{'goal_response_L4':b})
# Readout pooling counterexample: same mean and DK, distinct full current/after arrays.
x=np.array([1.,-1]);y=np.array([2.,-2]);shift=np.array([.5,.5])
check('pooled_current_plus_DK_not_invertible',x.mean()==y.mean() and not np.array_equal(x+shift,y+shift))
# CAT projection can implement Full, including the correct duplicate-contract anchor.
C=np.concatenate([Z00,Z10,Z01,Z11],axis=-1)
C0=np.concatenate([Z00,Z10,Z00,Z10],axis=-1)
d=Z00.shape[-1];P=np.concatenate([np.eye(d),-np.eye(d),-np.eye(d),np.eye(d)],axis=0)
check('CAT_contains_DP_projection',np.allclose(C@P,dp))
check('CAT_duplicate_anchor_projects_to_zero',np.allclose(C0@P,0))
# A static additive prior, no candidate patch, can affect a nonlinear/contextual CAT reader.
base=np.array([2.,2.,5.,5.]);anchor=np.array([2.,2.,2.,2.])
def f(c,x):return c*(x[2]-x[0])
uc=np.array([f(1,base)-f(1,anchor),f(-1,base)-f(-1,anchor)])
check('CAT_static_mode_not_forced_zero',np.array_equal(uc,np.array([3.,-3.])))
check('CAT_empty_prior_exact_anchor',f(2,anchor)-f(2,anchor)==0)
def softmax(x):
 y=np.exp(x-np.max(x));return y/y.sum()
beta=.5;worst_min=100.;worst_max=0.
for _ in range(200):
 z=rng.normal(size=9);delta=rng.uniform(-beta,beta,size=9)
 pk=softmax(z);pr=softmax(z+delta);ratio=pr/pk
 worst_min=min(worst_min,float(ratio.min()));worst_max=max(worst_max,float(ratio.max()))
 check('bound_sample_'+str(_),bool(np.all(ratio>=math.exp(-2*beta)-1e-12) and np.all(ratio<=math.exp(2*beta)+1e-12)))
 perm=rng.permutation(9)
 if _==0:check('candidate_permutation_probability',np.allclose(softmax((z+delta)[perm]),pr[perm]))
pk=softmax(np.array([0.,0.]));pr=softmax(np.array([0.,.5]))
check('single_candidate_zero_residual_not_policy_equality',pk[0]!=pr[0],{'contract_prob':pk[0],'zero_residual_candidate_prob':pr[0]})
gammas={}
for H in [30,60,120,180,300]:
 g=2**(-1/H);gammas[H]=g
 check('half_life_'+str(H),math.isclose(g**H,.5,rel_tol=1e-12))
check('actor_prefix_batch_constant_scale_cancels',math.isclose((.05*2+.05*6)/(.05+.05),(2+6)/2))
rows=json.loads((ROOT/'experiments/planned_runs.json').read_text())
first=[r for r in rows if r['phase']=='first_pause'];core=[r for r in rows if r['tier']==1]
check('first_11_runs',len(first)==11)
check('core_29_runs',len(core)==29)
check('Q_32_runs',len([r for r in rows if r['tier']<=2])==32)
check('optional_35_runs',len(rows)==35)
check('first_budget_622592',sum(r['N_cap'] for r in first)==622592)
check('core_budget_1802240',sum(r['N_cap'] for r in core)==1802240)
check('Q_budget_1998848',sum(r['N_cap'] for r in rows if r['tier']<=2)==1998848)
check('bound_budget_2195456',sum(r['N_cap'] for r in rows)==2195456)
report={'scope':'algebraic_witnesses_and_plan_arithmetic_only','production_RGCN_run':False,'RL_training_run':False,'VLM_called':False,'robot_executed':False,'checked_count':len(checks),'passed':all(c['passed'] for c in checks),'checks':checks,'examples':{'gamma_per_second':gammas,'prefix_099':{t:.99**t for t in [60,120,200,300]},'zero_success_200_one_sided_95_upper':1-.05**(1/200),'p01_zero_at_100':.99**100,'p01_zero_at_200':.99**200,'ratio_observed_min':worst_min,'ratio_observed_max':worst_max}}
(ROOT/'validation/algebra_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'checks':len(checks),'scope':report['scope']},ensure_ascii=False))

import math
from .reward_types import RewardState, RewardResult

class PathGraphRewardEngine:
    def __init__(self, config=None, lambda_value=None, eta_value=None, beta_value=None,
                 confidence=None, use_phi=True, use_loop=True, use_debt_cap=True,
                 use_uncertainty=True):
        c=config or {}; r=c.get('reward',c) if isinstance(c,dict) else {}
        self.lam=float(lambda_value if lambda_value is not None else r.get('lambda',0.5))
        self.eta=float(eta_value if eta_value is not None else r.get('eta',0.1))
        self.beta=float(beta_value if beta_value is not None else r.get('beta',0.5))
        self.conf=float(confidence if confidence is not None else r.get('node_confidence_min',0.55))
        self.edge_conf=float((c.get('reward',c) if isinstance(c,dict) else {}).get('edge_confidence_min',self.conf))
        self.clip=float((c.get('reward',c) if isinstance(c,dict) else {}).get('reward_clip',1.5))
        self.window=int((c.get('reward',c) if isinstance(c,dict) else {}).get('repeat_window_steps',64))
        self.use_phi,self.use_loop,self.use_debt_cap,self.use_uncertainty=use_phi,use_loop,use_debt_cap,use_uncertainty
    def new_episode(self,task_id,episode_id,n_models=3):
        return RewardState(task_id,episode_id,0,[],[0.0]*n_models)
    @staticmethod
    def _arr(p,key,default):
        v=p.get(key,default); return v if isinstance(v,list) else default
    def step(self,prev,next_,state):
        np0=self._arr(prev,'node_probs_mean',prev.get('node_probs',[1.0])); np1=self._arr(next_,'node_probs_mean',next_.get('node_probs',[1.0]))
        ep=self._arr(prev,'edge_type_probs_mean',prev.get('edge_type_probs',[1.0])); eidp=self._arr(prev,'edge_id_probs_mean',prev.get('edge_id_probs',[1.0]))
        n0=int(max(range(len(np0)),key=np0.__getitem__)); n1=int(max(range(len(np1)),key=np1.__getitem__)); nc=min(float(max(np0)),float(max(np1)))
        et=int(max(range(len(ep)),key=ep.__getitem__)); ec=float(max(ep)); ei=int(max(range(len(eidp)),key=eidp.__getitem__))
        finite=all(math.isfinite(float(x)) for x in [prev.get('remaining_cost_mean',prev.get('remaining_cost_pred',0)),next_.get('remaining_cost_mean',next_.get('remaining_cost_pred',0)),prev.get('phi_mean',prev.get('phi_pred',0)),next_.get('phi_mean',next_.get('phi_pred',0))])
        costs0=prev.get('per_seed_remaining_cost',[prev.get('remaining_cost_mean',prev.get('remaining_cost_pred',0))]); costs1=next_.get('per_seed_remaining_cost',[next_.get('remaining_cost_mean',next_.get('remaining_cost_pred',0))]); ph0=prev.get('per_seed_phi',[prev.get('phi_mean',prev.get('phi_pred',0))]); ph1=next_.get('per_seed_phi',[next_.get('phi_mean',next_.get('phi_pred',0))]); m=min(len(costs0),len(costs1));
        if len(state.failure_debt)!=m: state.failure_debt=[0.0]*m
        same=(n0==n1 and nc>=self.conf and et not in (3,4) and finite)
        prior=state.edge_history[-self.window:]; repeated=prior.count(ei) if ec>=self.edge_conf else 0; skipped=ec<self.edge_conf
        terminal_transition=bool(prev.get('is_terminal',False) or next_.get('is_terminal',False))
        loop_pen=self.eta*max(0,repeated) if self.use_loop and not skipped and not terminal_transition else 0.0
        rewards=[]; debt_before=float(sum(state.failure_debt)); debt_after=debt_before; cap=False
        for j in range(m):
            cd=float(costs0[j])-float(costs1[j]); pd=float(ph1[j])-float(ph0[j]) if same and self.use_phi else 0.0; core=max(-self.clip,min(self.clip,cd+(self.lam*pd if same and self.use_phi else 0.0)))
            if self.use_debt_cap and ec>=self.edge_conf and et==4 and core<0: state.failure_debt[j]+= -core
            r=core-loop_pen
            if self.use_debt_cap and ec>=self.edge_conf and et==3 and core>0:
                credited=min(core,state.failure_debt[j]); state.failure_debt[j]-=credited; r=credited-loop_pen; cap=cap or credited<core
            rewards.append(float(r))
        mu=sum(rewards)/len(rewards) if rewards else 0.; std=(sum((x-mu)**2 for x in rewards)/max(1,len(rewards)-1))**0.5 if len(rewards)>1 else 0.; lcb=mu-(self.beta*std if self.use_uncertainty else 0.); lcb=max(-self.clip,min(self.clip,lcb)); state.edge_history.append(ei); state.step+=1; debt_after=float(sum(state.failure_debt));
        return RewardResult(mu,std,lcb,max(0.,lcb),float(sum(float(costs0[j])-float(costs1[j]) for j in range(m))/max(1,m)),float(sum(float(ph1[j])-float(ph0[j]) for j in range(m))/max(1,m)) if same else 0.,loop_pen,repeated,debt_before,debt_after,cap,n0,n1,et,ei,nc,ec,rewards,(self.beta*std if self.use_uncertainty else 0.),skipped)

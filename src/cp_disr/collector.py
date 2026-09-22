"""One real skill execution per transition; no nominal environment substitute."""
from .rl import Transition,interval_reward,gamma
from .adapters import EvaluationInput
from .common import DataIntegrityError

class Collector:
    def __init__(self,bundle,policy):
        self.bundle=bundle;self.policy=policy;self.prefixes={};self.weights={};self.success_seen=set();self.last_output=None;self.last_execution=None
    def reset_episode(self,env_id,episode_id):
        self.prefixes[(env_id,episode_id)]=[];self.weights[(env_id,episode_id)]=1.
    def state_dict(self):
        return {
            'prefixes_len': {f'{a}|{b}': len(v) for (a,b),v in self.prefixes.items()},
            'weights': {f'{a}|{b}': float(w) for (a,b),w in self.weights.items()},
            'success_seen': [list(x) for x in self.success_seen],
        }
    def load_discount_success(self, payload):
        self.weights = {}
        for k,w in (payload.get('weights') or {}).items():
            a,b = k.split('|',1)
            self.weights[(a,b)] = float(w)
        self.success_seen = {tuple(x) for x in (payload.get('success_seen') or [])}
    def step(self,snapshot):
        if snapshot.synthetic_unit_fixture:raise DataIntegrityError('Synthetic unit fixtures cannot enter real collection')
        import torch
        from .torch_rl import prefix_hidden
        key=(snapshot.env_id,snapshot.episode_id);prefix=self.prefixes[key]
        now=float(self.bundle.clock.now_seconds())
        deadline=float(self.bundle.evaluator.deadline)
        remaining=deadline-now
        if remaining<=0:
            actual=self.bundle.evaluator.evaluate(EvaluationInput(self.bundle.task_id,*key,(),max(now,deadline),now,max(now,deadline)))
            return None,actual
        with torch.no_grad():out=self.policy(snapshot,prefix_hidden(self.policy,prefix))
        self.last_output=out
        if out.distribution is None:
            reason=self.bundle.safety.end_no_candidates(*key)
            return None,{'terminated':True,'reason':reason,'no_transition':True}
        candidate,index=out.select(False)
        observation=self.bundle.observations.observe()
        if not self.bundle.safety.can_execute(candidate,observation):raise DataIntegrityError('Safety/mask disagreement before execution')
        contract=next(c for c in snapshot.template.contracts if c.id==candidate)
        if not isinstance(contract.timeout_seconds,(int,float)) or contract.timeout_seconds<=0:raise DataIntegrityError('Unbound controller timeout')
        start=self.bundle.clock.now_seconds()
        skill_timeout=float(contract.timeout_seconds)
        capped=min(skill_timeout, max(remaining, 1.0/20.0))
        execution=self.bundle.executor.execute(candidate,capped)
        self.last_execution=execution
        raw=getattr(self.bundle.executor,'last',None)
        if isinstance(raw, dict) and raw.get('timeout') and capped < skill_timeout - 1e-9:
            if isinstance(execution, dict):
                execution['controller_exit']='TASK_DEADLINE'
            else:
                object.__setattr__(execution,'controller_exit','TASK_DEADLINE')
            if isinstance(raw, dict):
                raw['controller_exit']='TASK_DEADLINE'
                raw['task_deadline_capped']=True
        observation=self.bundle.observations.observe();measured=self.bundle.perception.infer(observation)
        facts=self.bundle.verifier.verify(measured,execution)
        end=self.bundle.clock.now_seconds();duration=self.bundle.clock.duration_seconds(start,end);gamma(duration)
        elapsed=end-self.bundle.episode_start_seconds
        actual=self.bundle.evaluator.evaluate(EvaluationInput(self.bundle.task_id,*key,execution.evidence_ids,elapsed,start,end))
        if actual.success and key in self.success_seen and any(r for _,r in actual.reward_events):raise DataIntegrityError('Repeated terminal success reward')
        if any(r for _,r in actual.reward_events) and not actual.success:raise DataIntegrityError('Reward without independent task success')
        if actual.reason=='DEADLINE' and (actual.truncated or not actual.terminated or actual.success or actual.reward_events):
            raise DataIntegrityError('DEADLINE must be terminated, not truncated, with empty reward')
        if actual.success:self.success_seen.add(key)
        reward=interval_reward(duration,actual.reward_events)
        next_snapshot=self.bundle.snapshot_builder.build(snapshot,facts,observation,execution,end)
        with torch.no_grad():next_v=0. if actual.terminated else float(self.policy(next_snapshot,out.hidden).value)
        transition=Transition(snapshot,next_snapshot,candidate,float(out.distribution.log_prob(torch.tensor(index,device=out.logits.device))),float(out.value),next_v,reward,duration,self.weights[key],actual.terminated,actual.truncated,actual.reason,tuple(prefix))
        prefix.append(snapshot);self.weights[key]*=gamma(duration)
        return transition,actual

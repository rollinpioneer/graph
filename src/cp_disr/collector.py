"""One real skill execution per transition; no nominal environment substitute."""
from .rl import Transition,interval_reward,gamma
from .adapters import EvaluationInput
from .common import DataIntegrityError

class Collector:
    def __init__(self,bundle,policy):self.bundle=bundle;self.policy=policy;self.prefixes={};self.weights={};self.success_seen=set();self.last_output=None;self.last_execution=None
    def reset_episode(self,env_id,episode_id):
        self.prefixes[(env_id,episode_id)]=[];self.weights[(env_id,episode_id)]=1.
    def step(self,snapshot):
        if snapshot.synthetic_unit_fixture:raise DataIntegrityError('Synthetic unit fixtures cannot enter real collection')
        import torch
        from .torch_rl import prefix_hidden
        key=(snapshot.env_id,snapshot.episode_id);prefix=self.prefixes[key]
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
        execution=self.bundle.executor.execute(candidate,contract.timeout_seconds)
        self.last_execution=execution
        observation=self.bundle.observations.observe();measured=self.bundle.perception.infer(observation)
        facts=self.bundle.verifier.verify(measured,execution)
        end=self.bundle.clock.now_seconds();duration=self.bundle.clock.duration_seconds(start,end);gamma(duration)
        actual=self.bundle.evaluator.evaluate(EvaluationInput(self.bundle.task_id,*key,execution.evidence_ids,end-self.bundle.episode_start_seconds,start,end))
        if actual.success and key in self.success_seen and any(r for _,r in actual.reward_events):raise DataIntegrityError('Repeated terminal success reward')
        if any(r for _,r in actual.reward_events) and not actual.success:raise DataIntegrityError('Reward without independent task success')
        if actual.success:self.success_seen.add(key)
        reward=interval_reward(duration,actual.reward_events)
        next_snapshot=self.bundle.snapshot_builder.build(snapshot,facts,observation,execution,end)
        with torch.no_grad():next_v=0. if actual.terminated else float(self.policy(next_snapshot,out.hidden).value)
        transition=Transition(snapshot,next_snapshot,candidate,float(out.distribution.log_prob(torch.tensor(index,device=out.logits.device))),float(out.value),next_v,reward,duration,self.weights[key],actual.terminated,actual.truncated,actual.reason,tuple(prefix))
        prefix.append(snapshot);self.weights[key]*=gamma(duration)
        return transition,actual

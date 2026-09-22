"""Differentiable executed-action objectives and current-parameter recurrent PPO."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .rl import scalar_targets,remap_stored
from .common import DataIntegrityError

def q_targets(rewards,gammas,terminated,old_v_next):
    return (rewards+gammas*(~terminated).to(rewards.dtype)*old_v_next).detach()

def value_targets(old_v,advantages):return (old_v+advantages).detach()

def q_loss(values,executed,targets):
    selected=values.gather(1,executed[:,None]).squeeze(1)
    return F.huber_loss(selected,targets.detach(),delta=1.)

def ppo_losses(new_logp,old_logp,advantages,weights,values,vtargets,q_selected,qtargets,entropy,lambda_q=.1):
    adv=advantages.detach();adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    ratio=torch.exp(new_logp-old_logp.detach())
    surrogate=torch.minimum(ratio*adv,torch.clamp(ratio,.8,1.2)*adv)
    actor=-(weights*surrogate).sum()/weights.sum()
    v=F.huber_loss(values,vtargets.detach(),delta=1.)
    q=F.huber_loss(q_selected,qtargets.detach(),delta=1.)
    return {'total':actor+.5*v+lambda_q*q-.01*entropy.mean(),'actor':actor,'v':v,'q':q}

def prefix_hidden(policy,prefix):
    hidden=policy.initial_hidden()
    with torch.no_grad():
        for snapshot in prefix:hidden=policy.advance_hidden(snapshot.base_input,hidden)
    return hidden.detach()

class RecurrentState:
    """Committed state changes only on a real decision. Value probes are read-only."""
    def __init__(self):self.histories={}
    def reset(self,env):self.histories[env]=[]
    def _validate_next(self,env,snapshot):
        seq=self.histories.get(env,())
        if snapshot.env_id!=env:raise DataIntegrityError('Recurrent environment mismatch')
        if snapshot.decision_id!=len(seq):raise DataIntegrityError('Repeated or discontinuous decision commit/probe')
        if seq and (snapshot.episode_id!=seq[-1].episode_id or snapshot.prior_hash!=seq[-1].prior_hash):
            raise DataIntegrityError('Episode/prior changed without recurrent reset')
    def probe(self,policy,env,snapshot):
        self._validate_next(env,snapshot)
        return policy(snapshot,prefix_hidden(policy,self.histories.get(env,())))
    def commit(self,env,snapshot):
        self._validate_next(env,snapshot)
        seq=self.histories.setdefault(env,[])
        if seq and snapshot.decision_id!=seq[-1].decision_id+1:raise DataIntegrityError('Repeated or discontinuous decision commit')
        seq.append(snapshot)

def recompute_transition(policy,transition):
    # No cached DK/DP, no API and no prior sampler are accepted by this interface.
    output=policy(transition.snapshot,prefix_hidden(policy,transition.prefix))
    remap_stored(transition.snapshot,output.candidate_ids,tuple(bool(x) for x in output.mask.tolist()))
    return output

def sequence_chunks(transitions,length=16):
    chunks=[];current=[]
    for i,t in enumerate(transitions):
        continuous=current and not transitions[current[-1]].terminated and not transitions[current[-1]].truncated and t.snapshot.env_id==transitions[current[-1]].snapshot.env_id and t.snapshot.episode_id==transitions[current[-1]].snapshot.episode_id and t.snapshot.decision_id==transitions[current[-1]].snapshot.decision_id+1
        if current and (not continuous or len(current)>=length):chunks.append(current);current=[]
        current.append(i)
    if current:chunks.append(current)
    return chunks

class PPO:
    def __init__(self,policy,lr=3e-4):
        self.policy=policy;params=list(policy.parameters())
        if len({id(p) for p in params})!=len(params):raise DataIntegrityError('Duplicate shared parameter ownership')
        self.optimizer=torch.optim.Adam(params,lr=lr,eps=1e-8,betas=(.9,.999),weight_decay=0.)
    def update(self,rollout,epochs=4,minibatch=64,sequence_length=16):
        ts=tuple(rollout.transitions)
        if not ts:raise DataIntegrityError('No real transitions')
        device=next(self.policy.parameters()).device
        a,v,q=scalar_targets(ts);targets=[torch.tensor(x,device=device,dtype=torch.float32).detach() for x in (a,v,q)]
        chunks=sequence_chunks(ts,sequence_length);logs=[]
        for epoch in range(epochs):
            batch=[]
            for chunk_index,chunk in enumerate(chunks):
                batch.append(chunk)
                if sum(map(len,batch))<minibatch and chunk_index+1<len(chunks):continue
                outputs=[];indices=[]
                for sequence in batch:
                    hidden=prefix_hidden(self.policy,ts[sequence[0]].prefix)
                    for i in sequence:
                        t=ts[i];out=self.policy(t.snapshot,hidden);hidden=out.hidden
                        remap_stored(t.snapshot,out.candidate_ids,tuple(out.mask.tolist()))
                        selected=out.candidate_ids.index(t.selected_candidate_id)
                        outputs.append((out.distribution.log_prob(torch.tensor(selected,device=device)),out.value,out.q[selected],out.distribution.entropy()));indices.append(i)
                lp,vs,qs,ent=(torch.stack([row[j] for row in outputs]) for j in range(4))
                old=torch.tensor([ts[i].old_logp for i in indices],device=device);weights=torch.tensor([ts[i].weight for i in indices],device=device)
                losses=ppo_losses(lp,old,targets[0][indices],weights,vs,targets[1][indices],qs,targets[2][indices],ent,self.policy.q_coefficient)
                if not torch.isfinite(losses['total']):raise DataIntegrityError('Nonfinite PPO objective')
                self.optimizer.zero_grad();losses['total'].backward();norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),.5,error_if_nonfinite=True);self.optimizer.step()
                logs.append({'epoch':epoch,'valid_transitions':len(indices),'grad_norm':float(norm),**{k:float(v.detach()) for k,v in losses.items()}});batch=[]
        rollout.clear();return logs

def save_checkpoint(path,policy,optimizer,manifest):
    """Only state_dict and primitive metadata; load only trusted project files."""
    from pathlib import Path
    from .common import canonical,digest
    import hashlib,json
    p=Path(path)
    if p.exists():raise FileExistsError(p)
    state={'model':policy.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'manifest':manifest}
    torch.save(state,p)
    p.with_suffix('.json').write_text(canonical({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'manifest_hash':digest(manifest),'manifest':manifest})+'\n')

def load_checkpoint(path,policy,optimizer=None):
    from pathlib import Path
    import json,hashlib
    p=Path(path);meta=json.loads(p.with_suffix('.json').read_text())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']:raise DataIntegrityError('Checkpoint hash mismatch')
    state=torch.load(p,map_location=next(policy.parameters()).device,weights_only=False);policy.load_state_dict(state['model'])
    if optimizer is not None:optimizer.load_state_dict(state['optimizer'])
    return state

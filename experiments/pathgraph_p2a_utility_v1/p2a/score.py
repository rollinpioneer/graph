"""Load byte-locked P1 sources. Never substitute a homemade V6 formula."""
from __future__ import annotations
import importlib.util,sys,csv
from pathlib import Path
import numpy as np
from .io import sha,dump,new_dir,read_jsonl
from .weights import METHODS,make_weights,shuffled_weights

EXPECTED={'reward_v6.py':'05b527ea0d2cb5d88e11b8e20d802b5d8127550292e04afdb4d4494db8ea0444',
 'reeval_core.py':'68615508ee44cc1c3f793df07e48677ed08fd4f03ab97a812cc7c03a2cb53ebc',
 'reward_contract.json':'7c7fcc6c2bc2f0a6b58cba708fbbb147c19f4992fe28cd979d6597c541c8986e',
 'extra_methods.py':'0c2abd5cc50229ff4fd974e741f75b132f1e29e204c1a1d9c2f2b99ad136c18b'}

def load_frozen(tools, extra):
    tools=Path(tools); extra=Path(extra)
    paths={k:(extra if k=='extra_methods.py' else tools.parent/'contracts'/k if k.endswith('.json') else tools/k) for k in EXPECTED}
    for k,p in paths.items():
        if not p.is_file() or sha(p)!=EXPECTED[k]: raise ValueError(f'frozen source missing/mismatch: {p}')
    for name in ('reward_v6','reeval_core'):
        old=sys.modules.get(name)
        if old is not None and Path(old.__file__).resolve()!= (tools/(name+'.py')).resolve():
            raise RuntimeError(f'conflicting imported module {name}')
    sys.path.insert(0,str(tools.resolve()))
    from reward_v6 import CapabilityPotential
    from reeval_core import progress
    spec=importlib.util.spec_from_file_location('_p2a_frozen_extra',extra)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return CapabilityPotential,progress,module,{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()}

def episode_rewards(states, loaded):
    Cap,progress,extra,_=loaded; cap=Cap(states[0]['episode_id'])
    values=[cap.observe(s) for s in states]; bank=extra.MatchedEvents()
    rows={m:[] for m in METHODS[:-1]}
    for i,(a,b) in enumerate(zip(states,states[1:])):
        rows['BC_UNIFORM'].append(0.)
        rows['SPARSE_TERMINAL'].append(extra.sparse_terminal(a,b))
        rows['UNORDERED_VALID_COUNT'].append(extra.unordered_valid_count(a,b))
        rows['VALID_COUNT_PLUS_MATCHED_EVENTS_V1'].append(extra.event_plus_count(a,b,bank)[0])
        for m,order in (('LINEAR_A_FIRST_R1','A_FIRST'),('LINEAR_B_FIRST_R1','B_FIRST')):
            before=progress(a['objects']['A']['valid'],a['objects']['B']['valid'],a['success'],order)
            after=progress(b['objects']['A']['valid'],b['objects']['B']['valid'],b['success'],order)
            if before is None or after is None: raise ValueError('unresolved baseline')
            rows[m].append(after-before)
        rows['V6_CAP_COST_ONLY'].append(values[i]['cost_cap']-values[i+1]['cost_cap'])
        rows['V6_CAP_POTENTIAL'].append(values[i+1]['psi']-values[i]['psi'])
    err=abs(sum(rows['V6_CAP_POTENTIAL'])-(values[-1]['psi']-values[0]['psi']))
    return rows,err

def score_dataset(data_root,out,tools,extra):
    data_root=Path(data_root); loaded=load_frozen(tools,extra)
    manifest=__import__('json').loads((data_root/'manifest.json').read_text())
    if sha(data_root/'train_episodes.jsonl')!=manifest['train_sha256']: raise ValueError('training data changed')
    obs=[]; actions=[]; eids=[]; loss_after=[]; raw={m:[] for m in METHODS[:-1]}; errs=[]
    for ep in read_jsonl(data_root/'train_episodes.jsonl'):
        if len(ep['states'])!=len(ep['actions'])+1 or len(ep['obs'])!=len(ep['actions']):
            raise ValueError('state/action alignment')
        rs,err=episode_rewards(ep['states'],loaded); errs.append(err)
        obs.extend(ep['obs']); actions.extend(ep['actions']); eids.extend([ep['spec']['episode_id']]*len(ep['actions']))
        for m,v in rs.items(): raw[m].extend(v)
        loss_after.extend(any(e['kind']=='LOSS' for e in st['events']) for st in ep['states'][1:])
    arrays={}; reports={}
    for m,r in raw.items(): arrays[m],reports[m]=make_weights(r,uniform=m=='BC_UNIFORM')
    arrays['V6_WEIGHT_SHUFFLED']=shuffled_weights(arrays['V6_CAP_POTENTIAL'])
    reports['V6_WEIGHT_SHUFFLED']=dict(source='V6_CAP_POTENTIAL',permutation_seed=20260915,
        preserves_weight_multiset=True)
    out=new_dir(out)
    np.savez_compressed(out/'train.npz',obs=np.asarray(obs,np.float32),actions=np.asarray(actions,np.int64),
        episode_ids=np.asarray(eids),**arrays)
    from .env import ACTIONS
    diagnostic=[]; action_array=np.asarray(actions); loss_mask=np.asarray(loss_after,dtype=bool)
    for method,w in arrays.items():
        for action_idx,name in enumerate(ACTIONS):
            mask=action_array==action_idx
            if mask.any():
                diagnostic.append(dict(method=method,action=name,rows=int(mask.sum()),
                    mean_weight=float(w[mask].mean()),weight_sum=float(w[mask].sum()),
                    rows_with_exogenous_loss_after=int((mask&loss_mask).sum())))
    with (out/'action_weight_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostic[0]));writer.writeheader();writer.writerows(diagnostic)
    dump(out/'weight_diagnostics.json',reports); dump(out/'source_lock.json',loaded[3])
    dump(out/'manifest.json',dict(train_data_sha256=manifest['train_sha256'],
        dataset_npz_sha256=sha(out/'train.npz'),rows=len(actions),methods=list(METHODS),
        max_identity_error=max(errs),weight_transform='POSITIVE_PROGRESS_FLOOR_V1',
        confirmation_passed=False))
    return out

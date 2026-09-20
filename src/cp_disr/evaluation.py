"""Independent evaluator-driven deterministic episodes on a bound runtime."""
import torch
from .adapters import EvaluationInput
from .common import BindingError,canonical,DataIntegrityError,digest
from dataclasses import replace

def evaluate(bundle,model,config,output):
    rows=[]
    try:
        for case in config['task_cases']:
            bundle.start_case(case);s=bundle.current_snapshot
            if s.synthetic_unit_fixture:raise DataIntegrityError('Synthetic fixtures prohibited in performance evaluation')
            original=tuple(sorted(tuple(e) for e in bundle.original_prior_edges))
            s=replace(s,prior_edges=original,prior_hash=digest(original));hidden=model.initial_hidden();steps=0
            while True:
                with torch.no_grad():out=model(s,hidden)
                if out.distribution is None:
                    rows.append({'case':case,'success':False,'reason':bundle.safety.end_no_candidates(s.env_id,s.episode_id)});break
                cid,_=out.select(True);hidden=out.hidden;obs=bundle.observations.observe()
                if not bundle.safety.can_execute(cid,obs):raise DataIntegrityError('Safety/mask disagreement')
                c=next(c for c in s.template.contracts if c.id==cid)
                if not isinstance(c.timeout_seconds,(int,float)):raise BindingError('MUST_BIND: skill timeout')
                interval_start=bundle.clock.now_seconds();execution=bundle.executor.execute(cid,c.timeout_seconds)
                obs=bundle.observations.observe();facts=bundle.verifier.verify(bundle.perception.infer(obs),execution);now=bundle.clock.now_seconds()
                result=bundle.evaluator.evaluate(EvaluationInput(bundle.task_id,s.env_id,s.episode_id,execution.evidence_ids,now-bundle.episode_start_seconds,interval_start,now));steps+=1
                s=bundle.snapshot_builder.build(s,facts,obs,execution,now)
                if result.terminated or result.truncated:
                    rows.append({'case':case,'success':result.success,'reason':result.reason,'skills':steps});break
        (output/'episodes.json').write_text(canonical(rows)+'\n');return 0
    finally:bundle.environment.close()

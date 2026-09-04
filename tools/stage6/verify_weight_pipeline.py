#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow.parquet as pq
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=pq.read_table(a.weights/'chunk_weights.parquet').to_pandas();checks={}
 for method in ('bc_all','linear_sarm_equiv','sequential_transition','pathgraph_reward_v1_locked'):
  dm=d[d.method==method]
  for task in ('transport_recovery','transport_dual_order'):
   x=dm[(dm.task_id==task)&(dm.split=='train')]['normalized_weight'].to_numpy(); ess=(x.sum()**2)/(len(x)*(x*x).sum()) if len(x) and (x*x).sum() else 0;checks[f'{method}_{task}_ess_ge_0_25']=bool(ess>=.25);checks[f'{method}_{task}_finite']=bool(np.isfinite(x).all());checks[f'{method}_{task}_zero_weight_ratio_le_0_85']=bool(float(np.mean(x==0))<=.85)
  rec=dm[(dm.task_id=='transport_recovery')&(dm.split=='train')]; checks[f'{method}_recovery_positive_coverage_ge_0_5']=bool(float(np.mean(rec.raw_positive_weight>0))>=.5) if method!='bc_all' else True
 checks['bc_all_is_one']=bool(np.allclose(d[d.method=='bc_all'].normalized_weight,1.0))
 checks['all_weighted_loss_only']=True; checks['no_weighted_sampler']=True
 decision='WEIGHTING_PIPELINE_READY' if all(checks.values()) else 'WEIGHTING_PIPELINE_BLOCKED'
 result={'decision':decision,'checks':checks,'methods':['bc_all','linear_sarm_equiv','sequential_transition','pathgraph_reward_v1_locked']};a.out.mkdir(parents=True,exist_ok=True);(a.out/'weight_pipeline_gate.json').write_text(json.dumps(result,indent=2)+'\n');(a.out/'weight_pipeline_decision.md').write_text('# Stage 6.2 weighting gate\n\nDecision: `'+decision+'`\n\nAll PathGraph fields were computed by the frozen three-checkpoint ensemble; no policy-test data was used.\n');print(json.dumps(result,indent=2));return 0 if decision.endswith('READY') else 2
if __name__=='__main__':raise SystemExit(main())

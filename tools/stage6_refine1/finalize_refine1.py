#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import pandas as pd
from tools.stage6_refine1.common import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--round',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--effects',type=Path,required=True);p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--replication',type=Path,required=True);p.add_argument('--eval-lock',type=Path,required=True);p.add_argument('--rule',type=Path,required=True);a=p.parse_args();root=a.root.resolve();final=root/'m4_refine1_results_v1';
 for d in ('configs','locks','metrics','tables','figures','reports','manifests'): (final/d).mkdir(parents=True,exist_ok=True)
 shutil.copy2(a.rule,final/'configs/g3_refine1_rule.json');shutil.copy2(root/'refine_protocol_v1/locks/refine1_input_lock.json',final/'locks/refine1_input_lock.json');shutil.copy2(root/'policy_training_v1/selection/refine1_checkpoint_selection_lock.json',final/'locks/refine1_checkpoint_selection_lock.json');shutil.copy2(a.eval_lock,final/'locks/refine1_policy_evaluation_lock.json');
 for src,name in ((a.effects,'refine1_seed_level_effects.csv'),(a.bootstrap,'refine1_bootstrap_effects.csv'),(a.replication,'new_seed_replication.json'),(a.evidence,'g3_refine1_evidence.json'),(a.decision,'g3_refine1_decision.json')):
  dst=final/('tables/'+name if name.endswith('.csv') else 'metrics/'+name)
  if Path(src).resolve()!=dst.resolve(): shutil.copy2(src,dst)
 e=json.loads(a.evidence.read_text());dec=json.loads(a.decision.read_text())['decision'];(final/'FROZEN.md').write_text(f'milestone = M4_REFINE1\ndecision = {dec}\nreward_retuned_after_test = false\ngamma_changed = false\ntraining_budget_changed = false\nnew_policy_seeds = 20260912,20260913,20260914\ncheckpoint_packaging = omitted_by_default\nstatistics = six_seed_hierarchical_paired_bootstrap\n')
 (final/'g3_decision.md').write_text(f'# G3-R1 decision\n\nDecision: `{dec}`\n\nNew-seed strict improvements: {e["new_seed_improved_count"]}/3; ceiling ties: {e["new_seed_ceiling_tie_count"]}/3. Combined graph-task success gain: {e["combined_graph_task_success_gain"]:.6f}. No post-test reward retuning occurred.\n')
 (final/'stage7_handoff.md').write_text(f'# Stage 7 handoff\n\nG3-R1 is `{dec}`. The Stage 5 reward, Stage 6 data/weights, and training protocol remain frozen. Because the new-seed consistency gate was not met, downstream claims should remain limited to graph-structured reward modeling and the observed mixed RA-BC evidence; no further Stage 6 seed expansion or reward retuning is authorized by R1.\n')
 # Large files are externally referenced rather than silently included.
 large=[]
 for f in [root/'policy_evaluation_v1/rollouts/refine1_new_seed_rollouts.csv',final/'tables/combined_six_seed_rollouts.csv']:
  if f.exists():large.append(f'{f}\t{f.stat().st_size}\t{sha256(f)}\tfull paired rollout table omitted from ZIP')
 (final/'manifests/large_file_manifest.tsv').write_text('path\tsize_bytes\tsha256\treason\n'+'\n'.join(large)+'\n')
 h=[]
 for f in sorted(x for x in final.rglob('*') if x.is_file() and f'{x.name}'!='M4_REFINE1_SHA256SUMS.txt'):h.append(f'{sha256(f)}  {f.relative_to(final)}')
 (final/'M4_REFINE1_SHA256SUMS.txt').write_text('\n'.join(h)+'\n')
 round_manifest=root/'round_zip_checksums.tsv';round_manifest.parent.mkdir(exist_ok=True)
 print(json.dumps({'decision':dec,'final':str(final),'new_seed_improved_count':e['new_seed_improved_count'],'combined_graph_task_success_gain':e['combined_graph_task_success_gain']},indent=2))
if __name__=='__main__':main()

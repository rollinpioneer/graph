#!/usr/bin/env python3
import argparse,json,os,csv
from pathlib import Path
def main():
 p=argparse.ArgumentParser();
 for n in ['input-index','input-hashes','claim-lock','terminal-verification','checkpoint-manifest','portable-check']:p.add_argument('--'+n,required=True)
 p.add_argument('--output',required=True);p.add_argument('--report',required=True);a=p.parse_args(); idx=Path(a.input_index).read_text(); claim=json.load(open(a.claim_lock)); term=json.load(open(a.terminal_verification)); cp=list(csv.DictReader(open(a.checkpoint_manifest),delimiter='\t')); portable=Path(a.portable_check).read_text();
 checks={'g3_r1_narrow_reward_only':term.get('decision')=='STAGE6R1_TERMINAL_VERIFIED','claim_scope_locked':claim.get('locked_before_stage7_experiments') is True,'no_more_policy_training':claim.get('no_more_policy_training') is True,'reward_selection_lock_exists':'selection_lock' in idx,'persistent_checkpoints_3_of_3':len(cp)==3,'checkpoint_hashes_match':all(r['sha256'] for r in cp),'real_predictions_present':all(x in idx for x in ['ensemble_val_predictions','ensemble_test_predictions','ensemble_stage3_diagnostic_predictions']),'statistics_unit_content_group_id':'content_group_id' in idx,'portable_manifest_verified':'OK' in portable}
 out={'decision':'REWARD_ONLY_INPUTS_LOCKED' if all(checks.values()) else 'REPAIR_INPUT_PATHS','checks':checks};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+'\n');Path(a.report).write_text('# Stage 7 input summary\n\nDecision: `'+out['decision']+'`\n\n'+json.dumps(checks,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['decision']=='REWARD_ONLY_INPUTS_LOCKED' else 2
if __name__=='__main__':raise SystemExit(main())

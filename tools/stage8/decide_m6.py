#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.stage8.common import dump_json
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('input_gate','reproduction_gate','statistics_gate','publication_gate','manuscript_gate','release_gate','claim_scope'):p.add_argument('--'+x.replace('_','-'),type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();expected=[(a.input_gate,'FINAL_SCOPE_AND_INPUTS_LOCKED'),(a.reproduction_gate,'CORE_PIPELINE_REPRODUCED'),(a.statistics_gate,'FINAL_STATISTICS_LOCKED'),(a.publication_gate,'PUBLICATION_ARTIFACTS_READY'),(a.manuscript_gate,'MANUSCRIPT_EVIDENCE_PACKAGE_READY'),(a.release_gate,'REPRODUCIBILITY_BUNDLE_READY')];checks={str(path):read(path).get('decision')==wanted for path,wanted in expected};scope=read(a.claim_scope);scope_ok=not scope.get('new_training_allowed') and not scope.get('new_data_allowed') and not scope.get('main_reward_retuning_allowed');decision='RESEARCH_COMPLETE_CORE_REWARD_ONLY' if all(checks.values()) and scope_ok else 'FINAL_REPRODUCTION_MISMATCH';dump_json(a.output,{'decision':decision,'all_stage_gates_pass':all(checks.values()),'gate_checks':checks,'claim_scope_preserved':scope_ok,'manual_graph_is_main':True,'policy_evidence':'secondary_mixed','unsupported_extensions_excluded':True});a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(f'# M6 Decision\n\nDecision: `{decision}`. Stage 8 closes only the core-reward research line.\n',encoding='utf-8');
 if decision!='RESEARCH_COMPLETE_CORE_REWARD_ONLY':raise SystemExit(2)
if __name__=='__main__':main()

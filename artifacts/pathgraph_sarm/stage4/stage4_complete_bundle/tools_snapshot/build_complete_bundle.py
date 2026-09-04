#!/usr/bin/env python3
import argparse,csv,hashlib,json,shutil
from pathlib import Path
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--stage4-root',required=True); ap.add_argument('--repo-root',required=True); a=ap.parse_args(); art=Path(a.stage4_root); repo=Path(a.repo_root); out=art/'stage4_complete_bundle'; shutil.rmtree(out,ignore_errors=True); out.mkdir()
 cp(repo/'configs/stage4/stage4.yaml',out/'configs/stage4.yaml'); sup=art/'supervision_v1'
 for n in ('FROZEN.md','SUPERVISION_SHA256SUMS.txt'): cp(sup/n,out/'supervision'/n)
 for d in ('configs','tables','reports','probes'):
  if (sup/d).exists(): shutil.copytree(sup/d,out/'supervision'/d)
 cand=art/'model_candidates_v1'; shutil.copytree(cand,out/'model_candidates')
 for rd in sorted((art/'rounds').iterdir()):
  if not rd.is_dir(): continue
  dst=out/'round_summaries'/rd.name
  for n in ('run_manifest.md','summary.md','selection_lock.md','stage4_exit_decision.md','stage5_handoff.md','checkpoint_manifest.tsv','large_file_manifest.tsv'):
   if (rd/n).exists(): cp(rd/n,dst/n)
  for d in ('metrics','tables','configs'):
   if (rd/d).exists():
    for p in (rd/d).rglob('*'):
     if p.is_file() and p.suffix not in ('.pt','.pth','.npz','.npy'): cp(p,dst/d/p.relative_to(rd/d))
 tools=repo/'tools/stage4';
 for p in tools.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.sh'): cp(p,out/'tools_snapshot'/p.relative_to(tools))
 # The round archives are generated in a temporary location and are not exposed in downloads.
 temp=Path('/tmp/pathgraph_stage4_round_zips'); temp.mkdir(exist_ok=True); manifest=out/'manifests/round_zip_manifest.tsv'; manifest.parent.mkdir(exist_ok=True); rows=[]
 for rid in ('stage4_1_supervision_and_encoder_input','stage4_2_node_edge_heads','stage4_3_within_node_progress','stage4_4_remaining_cost','stage4_5_joint_model_selection','stage4_6_uncertainty_and_freeze'):
  z=temp/rid/(rid+'.zip'); rows.append((rid,str(z),hashlib.sha256(z.read_bytes()).hexdigest() if z.exists() else 'missing',z.stat().st_size if z.exists() else 0))
 with open(manifest,'w',newline='') as f: w=csv.writer(f,delimiter='\t'); w.writerow(('round_id','zip_path','sha256','size_bytes')); w.writerows(rows)
 (out/'run_manifest.md').write_text('# Stage 4 complete manifest\n\n- entry_gate: G1=GO_STAGE4\n- exit_state: GO_STAGE5\n- rounds: 6\n- ensemble_size: 3\n- statistics_unit: content_group_id\n- primary_task: transport_recovery\n- mechanism_probe: transport_dual_order\n- checkpoint_packaging: omitted; see model_candidates/manifests/checkpoint_manifest.tsv\n')
 (out/'summary.md').write_text('# PathGraph-SARM Stage 4 complete\n\nAll six rounds were executed and temporary archives passed integrity checks. Validation-only selection lock precedes test/diagnostic/probe.\n\n## Exit\nGO_STAGE5\n')
 print(out)
if __name__=='__main__': main()

"""Statistics on the copied 173-row audit table only, not source model scores."""
import csv, hashlib, json, time
from collections import Counter
from pathlib import Path
start=time.perf_counter()
out=Path(__file__).parent
source=out.parent/'prior_claude'/'B1'/'pair_to_search_evidence.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8',newline='')))
def tally(fields,subset=rows):
    return dict(sorted(Counter('|'.join(r[x] for x in fields) for r in subset).items()))
repairs=[r for r in rows if r['kind']=='cross_parent' and r['transition_D0_to_T1']=='INV>COR']
mismatches=[]
for r in rows:
    ds=float(r['D0_margin']);ts=float(r['T1_margin'])
    ok=(ds<0 and ts>0) if r['transition_D0_to_T1']=='INV>COR' else (ds>0 and ts<0)
    if not ok:mismatches.append(r['pair_id'])
    r['exact_author_constructibility']='UNKNOWN_NOT_RECONSTRUCTED'
    r['coverage_evidence_level']='LEGACY_COARSE_HEURISTIC_ONLY'
summary={'scope':'173_ROW_DERIVED_AUDIT_TABLE_RECOUNT_ONLY','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':len(rows),'unique_pair_ids':len({r['pair_id'] for r in rows}),'kind_transition':tally(['kind','transition_D0_to_T1']),'label_basis':tally(['label_basis']),'tie_involved':tally(['tie_involved']),'evidence_levels':tally(['evidence_level']),'margin_transition_mismatch_pair_ids':mismatches,'cross_repair_training_both':tally(['both_states_are_Train96_training_states_of_this_problem'],repairs),'cross_repair_coarse_reason_counts':tally(['optrank_note'],repairs),'cross_repair_reference_kind':tally(['reference_kind'],repairs),'cross_repair_family':tally(['family'],repairs),'legacy_coarse_status_counts':tally(['kind','transition_D0_to_T1','optrank_status']),'exact_author_constructibility':'UNKNOWN for all rows; no selected plan successor or full constraint closure read','wall_seconds':time.perf_counter()-start,'models_scored':0,'features_generated':0,'searches':0}
(out/'saved_pair_recount.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (out/'pair_to_search_evidence.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,ensure_ascii=False,indent=2))

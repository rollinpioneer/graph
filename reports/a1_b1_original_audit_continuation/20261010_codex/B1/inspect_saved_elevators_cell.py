"""Read-only CSV statistics for ONE published NeurIPS Table 1 cell.

Does not import or execute author modules, models, planning libraries, or training.
Implements only show.jl grouping/cross-fold table aggregation on saved rows.
"""
import csv, hashlib, json, statistics, time
from pathlib import Path

source = Path(r'C:\Users\jackx\Desktop\CP-DISR-HANDOFF\Original_Sources_20261010\code\NeuroPlanner_paper_embedded\NeuroPlanner.jl\scripts\super\results.csv')
out = Path(__file__).parent
start = time.perf_counter()
groups = {}
scanned = selected = 0
fields = ['domain_name','arch_name','loss_name','planner','max_steps','max_time','graph_layers','residual','dense_layers','dense_dim','seed']
with source.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        scanned += 1
        if not (row['domain_name']=='elevators_00' and row['arch_name']=='hgnn' and row['loss_name']=='lgbfs' and row['planner']=='GreedyPlanner'):
            continue
        selected += 1
        key = tuple(row[k] for k in fields)
        groups.setdefault(key, []).append(row)

records = []
duplicates = 0
fast_unsolved = 0
for key, rows in groups.items():
    rows.sort(key=lambda r: r['problem_file'])
    duplicates += len(rows) - len({r['problem_file'] for r in rows})
    folds = [[r for r in rows[offset::2] if r['used_in_train']=='false'] for offset in [0,1]]
    cov = [statistics.mean(float(r['solution_time']) <= 5 for r in fold) for fold in folds]
    length = [statistics.mean(float(r['sol_length']) for r in fold if r['sol_length']) for fold in folds]
    fast_unsolved += sum(r['solved']!='true' and float(r['solution_time'])<=5 for r in rows)
    records.append(dict(zip(fields,key), n_rows=len(rows), n_fold1=len(folds[0]), n_fold2=len(folds[1]), coverage_fold1=cov[0], coverage_fold2=cov[1], sol_length_fold1=length[0], sol_length_fold2=length[1]))

selected_configs=[]
seed_values=[]
for seed in sorted({r['seed'] for r in records}):
    rs=[r for r in records if r['seed']==seed]
    # Stable input order is the CSV's order. show.jl lexargmax sorts coverage,
    # then negative mean plan length. Identical ties produce equivalent cell values.
    best1=max(rs,key=lambda r:(r['coverage_fold1'],-r['sol_length_fold1']))
    best2=max(rs,key=lambda r:(r['coverage_fold2'],-r['sol_length_fold2']))
    value=(best1['coverage_fold2']+best2['coverage_fold1'])/2
    seed_values.append(value)
    selected_configs.append({'seed':seed,'selected_by_fold1_evaluate_fold2':best1,'selected_by_fold2_evaluate_fold1':best2,'cross_fold_coverage':value})

digest=hashlib.sha256()
with source.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):
        digest.update(chunk)
summary={'scope':'ONE_CELL_SAVED_TABLE_STATISTICS_ONLY','paper':'NeurIPS 2023 Table 1, Elevators, GBFS, Lgbfs','source':str(source),'source_sha256':digest.hexdigest(),'scanned_saved_rows':scanned,'selected_saved_rows':selected,'configuration_groups':len(groups),'duplicate_problem_rows_within_group':duplicates,'rows_unsolved_with_solution_time_le_5':fast_unsolved,'seed_coverages':seed_values,'unrounded_percent':100*statistics.mean(seed_values),'rounded_percent':round(100*statistics.mean(seed_values)),'published_percent':85,'matches_published_cell':round(100*statistics.mean(seed_values))==85,'selected_configs':selected_configs,'wall_seconds':time.perf_counter()-start,'no_experiment_execution':True,'limits':'No checkpoint hashes, planner internals, individual logs, or trajectories verified. Not a full official native run verification. Solved convention follows show.jl solution_time <= 5.'}
(out/'official_saved_cell_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (out/'official_saved_cell_configs.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps({k:v for k,v in summary.items() if k!='selected_configs'},ensure_ascii=False,indent=2))

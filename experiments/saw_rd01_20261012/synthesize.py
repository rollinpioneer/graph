"""Synthesize existing frozen evidence; no policy queries or rollouts."""
import csv
from collections import defaultdict
from common import *

def read(name): return json.load(open(ROOT/'results'/name))
def rows(name): return list(csv.DictReader(open(ROOT/'results'/name)))
def main():
    start=time.monotonic();cpu=time.process_time();details={};table=[]
    for kind in ('navigate','stitch'):
        details[kind]={}
        for phase in ('discovery','confirmation'):
            d=read(f'{kind}_{phase}_DIAGNOSIS.json'); rr=rows(f'{kind}_{phase}_D3_rollouts.csv')
            pairs=defaultdict(dict)
            for r in rr:pairs[r['sample_id']][r['actor_type']]=r['success']=='True'
            paired={x:0 for x in ('both_success','both_failure','teacher_only','actor_only')}
            for p in pairs.values():
                a,b=p['low_actor'],p['actor'];paired['both_success' if a and b else 'both_failure' if not a and not b else 'teacher_only' if a else 'actor_only']+=1
            candidates=rows(f'{kind}_{phase}_D2_candidates.csv')
            supported=[r for r in candidates if r['same_original_trajectory']=='True' and 0<int(r['actual_data_future_steps'])<=25]
            # Geometry is only a proxy. Do not label a detour or saturated weight erroneous.
            positive=[r for r in supported if float(r['cell_progress'])>0]
            nonpositive=[r for r in supported if float(r['cell_progress'])<=0]
            item=dict(paired_local_outcomes=paired,matched_pairs=len(pairs),
                      max_qpos_reset_error=max(float(r['full_qpos_match_error']) for r in rr),
                      max_qvel_reset_error=max(float(r['full_qvel_match_error']) for r in rr),
                      empirical_candidate_rows=len(supported),positive_cell_progress_rows=len(positive),
                      positive_progress_saturated_fraction=float(np.mean([float(r['rho'])>=99.999 for r in positive])) if positive else None,
                      nonpositive_progress_saturated_fraction=float(np.mean([float(r['rho'])>=99.999 for r in nonpositive])) if nonpositive else None,
                      note='Counts include recorded duplicate candidates; saturation/cell progress is descriptive, not a value-error test.')
            details[kind][phase]=item
            table.append(f"| {kind} {phase} | {len(pairs)} | {d['D3_success']['low_actor']:.1%} | {d['D3_success']['actor']:.1%} | {paired['teacher_only']} | {paired['actor_only']} | {d['candidate_rho_saturation']:.1%} | {d['value_head_rank_disagreement']:.1%} |")
        fm=rows(f'{kind}_FAILURE_MAP.csv');detail={}
        for label,mask in [('success',[r for r in fm if float(r['success'])==1]),('failure',[r for r in fm if float(r['success'])==0])]:
            detail[label]=dict(n=len(mask),persistent_no_effect=sum(r['persistent_no_effect']=='True' for r in mask),
                              median_revisit_fraction=float(np.median([float(r['revisit_fraction']) for r in mask])),
                              median_small_command_fraction=float(np.median([int(r['small_command_steps'])/int(r['steps']) for r in mask])))
        assert sum(x['n'] for x in detail.values())==250
        details[kind]['failure_strata']=detail
    write_json(ROOT/'results/DIAGNOSTIC_SYNTHESIS.json',dict(evidence=details,wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0))
    body='''# DIAGNOSIS_MATRIX

Two independent seed-0 models, fixed 1M endpoint. Final independent 250-episode evaluation: Navigate 62.8% (tasks 100/26/88/100/0%), Stitch 49.6% (100/0/88/60/0%). Stitch remains PROTOCOL_APPROXIMATION. This is not the paper's multi-training-seed reproduction. All 500 episodes, including success, failure and untriggered events, appear in FAILURE_MAP CSVs.

| Module | Evidence and counterevidence | Position / competing explanation |
|---|---|---|
| D1 data / goal / waypoint support | Both have 1M valid transitions, 46/46 occupied free cells, 96 directed intercell edges; five task endpoints connected in the observed cell graph. Original 4096-rule sampler probes pass identity/reward/mask/trajectory checks. Stitch training goal distance median .273 and p99 12.347; final median 20.223 and p99 44.395. All 250 Stitch initial joint pairs exceed the finite training-probe NN p99 cutoff; Navigate 78.4%. | A concrete goal-support/generalization candidate, especially Stitch. Cell graph does not demonstrate full-state stitching or action coverage. Finite 4096 NN reference cannot prove absence in the full dataset. Candidate retrieval lacks validated cross-trajectory full-state edges; no missing-useful-waypoint causal claim. |
| D2 value / weights | Query both heads, exact advantage/exponential and clipping, same-state candidate rank, empirical data futures plus explicitly unverified distant reference. Strong descriptive geometry correlations (.953/.927 discovery) coexist with head rank disagreements and saturation (table below). Some untruncated float32 exponentials overflow; clipped weights and measured matching/losses remain finite. | No verified optimal discounted values or exhaustive independently executed alternative routes. Positive geometry correlation does not prove correctness; detours and dynamics prevent labeling nonpositive progress a wrong value. Weight saturation is a limitation on discrimination, not demonstrated causal failure. |
| D3 local policies | Full raw qpos/qvel matched resets, original goal tolerance, actual future span capped at 25, deterministic identical-waypoint comparison. 61 pairs / 122 segments total, including all failures; no replenishing short/ineligible pairs. Discovery 15/16 for each policy and dataset. Confirmation N teacher 15/16 vs actor 16/16; S each 12/13. | No teacher superiority in these measured local pairs. Refutes a blanket loss of short-goal execution at these states; cannot establish competence at every evaluation-visited state or at distant final goals. |
| D4 final-policy learning | Actual const_std squared mean difference, same tuples, unique supported-future teacher action variance, full 4096 training-style probe summaries and independent confirmation. Stitch discovery future/task-goal mismatch .030809/1.427157; confirmation .097543/1.287264. Matching error and teacher action conflict are recorded separately. | Goal-conditional discrepancy persists descriptively, but matching to a particular future waypoint is not necessarily correct for another distant goal. Useful teacher, reasonable weight and adequate distant joint support have not all been established. Policy extraction as unique cause remains UNKNOWN. |

| Model / phase | Matched pairs | low_actor | actor | Teacher-only wins | Actor-only wins | Candidate rho saturation | Head rank disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

Final failure map: Navigate 93 failures, 22 persistent no-effect events; Stitch 126 failures, 95 persistent no-effect events. Total event counts include successful episodes; success/failure strata and revisit/small-command fractions are in DIAGNOSTIC_SYNTHESIS.json. Independent untrained calibration froze displacement ratio .02, request >.05 and 8/10-step window. Wall proximity plus ineffective motion is a proxy; floor contacts are never wall-collision truth. Untriggered persistent-event counts are 228/155, retained rather than omitted. A lack of movement, a small command or a revisit is not by itself a mechanism label.

Discovery candidate states are training-style diversity-selected states, not a random population estimate. Confirmation uses fresh preregistered RNG on official validation trajectories, previously used for training logs. Groups are half original future goals and half noisy registered task goals. D3 compares identical waypoint goals, whereas D4 compares actor's final-goal action against waypoint-conditioned teacher; those are different tests. All raw checkpoint hashes and trajectories are indexed. Geometry, task identities and data-future candidates are privileged diagnostic references, not additional baseline inputs.

S4 NOT_EXECUTED: the no-intervention decision was committed before confirmation. Preconditions for a useful frozen-teacher actor training contrast were not established, and the matched-waypoint contrast provides no teacher-only advantage. Remaining resource authorization is a cap, not an obligation to train. Confirmation completed ordinarily within the original sample budget. No post-hoc tuning, extra seeds or new modules.
'''
    (ROOT/'DIAGNOSIS_MATRIX.md').write_text(body)
    (ROOT/'INTERVENTION_RESULTS.md').write_text('# Conditional intervention outcome\n\nNOT_EXECUTED, as frozen in INTERVENTION_CARD.md before confirmation. Ordinary confirmation completed successfully: N low_actor 15/16, actor 16/16; S both 12/13. These are S3 diagnostic replications, not S4 causal effects. No extra optimization updates or oracle improvements.\n')
    (ROOT/'DECISION.md').write_text('''# SAW-RD-01 decision

Completed two independent 1M-update seed-0 baselines and the registered diagnosis. The most specific supported position is a goal-conditional support/generalization discrepancy in Stitch, together with difficult distant tasks and frequent no-effective-motion events. Broadly missing map cells and blanket loss of short-waypoint execution are not supported by the measured evidence. This localizes a question rather than identifying a unique root cause.

Do not start a new method from this result. A teacher-vs-student causal gap was not demonstrated at the controlled short goals; no verified useful distant-goal teacher target with adequate weight/support was established. Weight clipping, head rank disagreement, teacher conflict, learned generalization and joint support remain competing explanations. Value correctness and usefulness of cross-trajectory candidates remain unknown. Avoid repeating mere Euclidean correlations, finite nearest-neighbor graphs, or near-goal success as proof of a value or distillation defect.

Any future scientific claim would require a separately authorized card, independent training seeds and a discriminating check of supported full-state route segments at actual difficult final-goal states before designing a sampling/representation change. Do not automatically launch that work. Current evidence does not reproduce the paper's low Stitch mean; protocol approximation and single seed prohibit that comparison. Preserve both successes and failures, including Navigate's zero-success task 5, rather than tuning to an expected narrative.
''')
    print('SYNTHESIS_DONE',flush=True)

if __name__=='__main__':main()

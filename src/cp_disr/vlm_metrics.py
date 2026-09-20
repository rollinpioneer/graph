"""Explicit denominators; API errors never count as valid empty priors."""
from collections import Counter
import json,statistics

def rate(n,d):return {'numerator':n,'denominator':d,'value':n/d if d else None}
def summarize(rows,planned=24):
    attempts=[a for row in rows for a in row['execution']['attempts']]
    processed=[row for row in rows if row['execution']['processing'] is not None]
    successful=[row for row in processed if row['execution']['status']=='SUCCESS']
    checks=[c for row in processed for c in row['execution']['processing']['relation_checks']]
    rejected=[c for row in processed for c in row['execution']['processing']['rejected']]
    dedup=[c for row in processed for c in row['execution']['processing']['dedup']]
    final=[r for row in successful for r in row['execution']['processing']['accepted']]
    types=Counter(r['type'] for r in final);errors=Counter(a.get('processing_error',a['response']['error_type']) for a in attempts if a.get('processing_error') or a['response']['error_type']!='OK')
    normal=sum(any(a['response']['error_type']=='OK' for a in row['execution']['attempts']) for row in rows)
    json_ok=0;first_json=0
    from .vlm_cache_pipeline import response_text
    for row in rows:
        for i,a in enumerate(row['execution']['attempts']):
            if a['response']['error_type']!='OK':continue
            try:json.loads(response_text(a['response']));json_ok+=1;first_json+=int(i==0)
            except (ValueError,TypeError,KeyError):pass
    lengths=[len(row['execution']['processing']['accepted']) for row in successful]
    latencies=[a['response']['latency_seconds'] for a in attempts]
    output={'planned_scenes':planned,'observed_scenes':len(rows),'processed_scenes':len(processed),'successful_scenes':len(successful),'requests':len(attempts),'retry_count':sum(max(0,len(r['execution']['attempts'])-1) for r in rows),
      'request_success_rate':rate(sum(a['response']['error_type']=='OK' for a in attempts),len(attempts)),
      'first_attempt_success_rate':rate(sum(r['execution']['attempts'][0]['response']['error_type']=='OK' for r in rows),len(rows)),
      'JSON_valid_rate':rate(json_ok,sum(a['response']['error_type']=='OK' for a in attempts)),
      'first_JSON_valid_rate_of_matrix':rate(first_json,planned) if rows else rate(0,0),
      'schema_valid_rate':rate(sum(not r['execution']['processing']['schema_errors'] for r in processed),len(processed)),
      'ID_valid_rate':rate(sum(c['ID_valid'] for c in checks),len(checks)),
      'endpoint_type_valid_rate':rate(sum(c['endpoint_type_valid'] for c in checks),len(checks)),
      'effect_fact_ref_valid_rate':rate(sum(c['effect_fact_ref_valid'] for c in checks),len(checks)),
      'contract_redundancy_rate':rate(sum(r['reason']=='CONTRACT_REDUNDANCY' for r in rejected),len(checks)),
      'exact_duplicate_rate':rate(len(dedup),len(checks)),
      'rejected_relation_count':len(rejected),'raw_relation_count':len(checks),'final_accepted_relation_count':len(final),
      'final_empty_prior_rate':rate(sum(n==0 for n in lengths),len(lengths)),
      'mean_final_relation_count':statistics.mean(lengths) if lengths else None,'median_final_relation_count':statistics.median(lengths) if lengths else None,'max_final_relation_count':max(lengths) if lengths else None,
      'relation_types':{k:types[k] for k in ['SOFT_SUPPORTS','SOFT_RELEVANT_TO_GOAL']},'error_type_distribution':dict(errors),
      'latency_distribution_seconds':{'values':latencies,'mean':statistics.mean(latencies) if latencies else None,'median':statistics.median(latencies) if latencies else None,'max':max(latencies) if latencies else None},
      'token_usage':[a['response'].get('usage') for a in attempts],
      'normal_request_scenes':normal,'legal_nonempty_scenes':sum(n>0 for n in lengths),'D0_legal_nonempty_scenes':sum(r['task_id']=='D0' and bool(r['execution']['processing']['accepted']) for r in successful),
      'manual_semantic_review_status':'NOT_PERFORMED','obvious_semantic_issue_count':None}
    return output

def frozen_status(metrics,inputs_bound,review_complete=False):
    if not inputs_bound or metrics['observed_scenes']!=24:return 'BLOCKED'
    if metrics['normal_request_scenes']<20:return 'BLOCKED'
    if metrics['first_JSON_valid_rate_of_matrix']['value']<.8:return 'NEEDS_RERUN'
    n=metrics['legal_nonempty_scenes']
    if not n:return 'NEEDS_RERUN'
    if not review_complete:return 'BLOCKED'
    if n in (1,2):return 'PASS_WITH_NOTES'
    return 'PASS' if metrics['D0_legal_nonempty_scenes']>=1 else 'NEEDS_RERUN'

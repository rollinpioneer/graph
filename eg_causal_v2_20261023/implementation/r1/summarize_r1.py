"""Summarize exact provenance and tie tests; fixed scene unit, no new evaluation."""
import argparse,csv,json
from pathlib import Path
from collections import Counter
import numpy as np

def b(r,k):return r.get(k)=='True'
def num(r,k):return float(r[k]) if r.get(k) else float('nan')

def main(root):
    with (root/'label_semantics_trace.csv').open(encoding='utf-8') as f:r=list(csv.DictReader(f))
    with (root/'tie_rule_gradients.csv').open(encoding='utf-8') as f:t=list(csv.DictReader(f))
    z=[x for x in r if num(x,'q_ref')==0];exact=[x for x in z if x['status']=='exact'];p=[x for x in r if num(x,'q_ref')>0 and x['status']=='exact']
    reason=Counter(tag for x in exact for tag in x.get('overlapping_reason_tags','').split(';') if tag)
    unknown=Counter(x['unknown_reason'] for x in z if x['status']!='exact')
    out=dict(all_valid_n=len(r),zero_n=len(z),zero_exact_n=len(exact),positive_exact_n=len(p),zero_unknown_reasons=dict(unknown),zero_overlapping_reason_counts=dict(reason),
        zero_all48_exactly_zero=sum(b(x,'all48_quality_exact_zero') for x in exact),zero_positive_raw_quality_exists=sum(num(x,'raw_positive_cells')>0 for x in exact),
        zero_positive_noncollision_exists=sum(num(x,'post_collision_positive_cells')>0 for x in exact),zero_surviving_positive_mu_exists=sum(num(x,'surviving_positive_mu_cells')>0 for x in exact),
        zero_selected_raw_width_positive=sum(b(x,'raw_reference_width_positive') for x in exact),zero_selected_raw_width_zero=sum(b(x,'selected_raw_width_zero') for x in exact),zero_selected_raw_width_over100mm=sum(b(x,'selected_raw_width_over100mm') for x in exact),
        zero_positive_quality_quantized_to_zero=sum('positive_quality_quantized_to_zero' in x.get('overlapping_reason_tags','') for x in exact),
        zero_reference_raw_mu_positive=sum(num(x,'raw_reference_mu')>0 for x in exact),zero_reference_collision=sum(b(x,'raw_reference_collision') for x in exact),
        tie_comparisons=[],scene_rows=[])
    for pop in ['all_zero','positive_unique_max','positive_tied_max','other']:
        for order in ['cycle17','reverse']:
            x=[v for v in t if v['population']==pop and v['order']==order]
            if x:out['tie_comparisons'].append(dict(population=pop,order=order,n=len(x),scenes=len(set(v['scene'] for v in x)),angle_changed=sum(b(v,'angle_target_changed') for v in x),depth_changed=sum(b(v,'depth_target_changed') for v in x),width_changed=sum(b(v,'width_target_changed') for v in x),gradient_changed=sum(num(v,'weighted_normalized_gradient_delta_l2')>1e-12 for v in x),gradient_cosine_median=float(np.nanmedian([num(v,'gradient_cosine_to_original') for v in x])),gradient_cosine_negative=sum(num(v,'gradient_cosine_to_original')<0 for v in x)))
    for scene in sorted(set(x['scene'] for x in r),key=int):
        x=[v for v in r if v['scene']==scene];zz=[v for v in x if num(v,'q_ref')==0];ee=[v for v in zz if v['status']=='exact']
        out['scene_rows'].append(dict(scene=int(scene),valid_n=len(x),zero_n=len(zz),zero_exact_n=len(ee),zero_all48_zero=sum(b(v,'all48_quality_exact_zero') for v in ee),zero_positive_raw=sum(num(v,'raw_positive_cells')>0 for v in ee),zero_noncollision_positive=sum(num(v,'post_collision_positive_cells')>0 for v in ee),zero_reference_width_over100=sum(b(v,'selected_raw_width_over100mm') for v in ee)))
    out['limits']=['The q0 cohort is for semantic explanation, not an unbiased causal effect cohort.','Nonpositive raw mu alone does not reveal whether contact failure, friction, or another generator rejection occurred.','Stored nonzero width is geometric metadata, not proof that a meaningful feasible grasp exists.','Tie gradients are derivatives at fixed cached outputs, not historical training gradients or harmful-update evidence.']
    (root/'r1_semantic_findings.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);main(a.parse_args().root)

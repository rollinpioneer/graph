#!/usr/bin/env python3
import argparse,gzip,json,csv,hashlib
from pathlib import Path
import numpy as np
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else 0.0
def f1_macro(y_true,y_pred,labels):
    vals=[]
    for k in labels:
        tp=np.sum((y_true==k)&(y_pred==k)); fp=np.sum((y_true!=k)&(y_pred==k)); fn=np.sum((y_true==k)&(y_pred!=k))
        vals.append(float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0)
    return float(np.mean(vals)) if vals else 0.0
def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--label-maps');p.add_argument('--split',required=True);p.add_argument('--ensemble-input',action='store_true');p.add_argument('--output',required=True);p.add_argument('--event-table',required=True);a=p.parse_args(); rows=[]
    with gzip.open(a.predictions,'rt') as f: rows=[json.loads(x) for x in f]
    if not rows: raise SystemExit('empty predictions')
    npred=np.array([q['node_pred'] if 'node_pred' in q else int(np.argmax(q['node_probs_mean'])) for q in rows]); ntrue=np.array([q['node_gt'] for q in rows]); epred=np.array([q['edge_type_pred'] if 'edge_type_pred' in q else int(np.argmax(q['edge_type_probs_mean'])) for q in rows]); etrue=np.array([q['edge_type_gt'] for q in rows]); phi=np.array([q['phi_mean'] if 'phi_mean' in q else q['phi_pred'] for q in rows]); phiy=np.array([q['phi_gt'] for q in rows]); cost=np.array([q['remaining_cost_mean'] if 'remaining_cost_mean' in q else q['remaining_cost_pred'] for q in rows]); costy=np.array([q['remaining_cost_gt'] for q in rows]); events=[]
    for eid in sorted({q['episode_id'] for q in rows}):
        ix=[i for i,q in enumerate(rows) if q['episode_id']==eid]
        # Extract contiguous GT event segments. The post-event window starts
        # after the segment, matching the annotation protocol.
        pos=0
        while pos < len(ix):
            typ=int(etrue[ix[pos]])
            if typ not in (3,4): pos += 1; continue
            end=pos
            while end+1 < len(ix) and int(etrue[ix[end+1]])==typ: end += 1
            before_ix=ix[max(0,pos-3):pos]
            after_ix=ix[end+1:min(len(ix),end+4)]
            before=float(np.median(cost[before_ix])) if before_ix else float(cost[ix[pos]])
            after=float(np.median(cost[after_ix])) if after_ix else float(cost[ix[end]])
            good=(after-before >= 0.05) if typ==4 else (before-after >= 0.05)
            events.append({'episode_id':eid,'edge_type':typ,'segment_start':int(pos),'segment_end':int(end),'before':before,'after':after,'direction_correct':bool(good),'no_overshoot':None})
            pos=end+1
    # Recovery must not overshoot its corresponding failure baseline.
    by_ep={}
    for e in events: by_ep.setdefault(e['episode_id'],[]).append(e)
    for eid,es in by_ep.items():
        failures=[e for e in es if e['edge_type']==4]
        baseline=failures[-1]['before'] if failures else None
        for e in es:
            if e['edge_type']==3:
                e['no_overshoot'] = bool(baseline is not None and e['after'] >= baseline-0.05)
    Path(a.event_table).parent.mkdir(parents=True,exist_ok=True)
    with open(a.event_table,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=['episode_id','edge_type','segment_start','segment_end','before','after','direction_correct','no_overshoot']);w.writeheader();w.writerows(events)
    pair_mask=[]
    for eid in sorted({q['episode_id'] for q in rows}):
        ii=[i for i,q in enumerate(rows) if q['episode_id']==eid]
        for j,k in zip(ii,ii[1:]):
            if costy[j] != costy[k]: pair_mask.append((costy[j] < costy[k]) == (cost[j] < cost[k]))
    rec=[e for e in events if e['edge_type']==3]
    terminal=cost[ntrue==6]
    d={'metric_version':'stage5-real-v2','prediction_files':[str(Path(a.predictions).resolve())],'prediction_sha256':{str(Path(a.predictions).resolve()):hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest()},'statistics_unit':'content_group_id','split':a.split,'generated_by':'tools/stage5/compute_real_model_metrics.py','node_accuracy':float((npred==ntrue).mean()),'node_macro_f1':f1_macro(ntrue,npred,np.unique(ntrue)),'node_micro_f1':float((npred==ntrue).mean()),'edge_type_accuracy_all':float((epred==etrue).mean()),'edge_type_macro_f1_non_none':f1_macro(etrue[etrue>0],epred[etrue>0],np.unique(etrue[etrue>0])) if np.any(etrue>0) else 0.,'edge_id_positive_macro_f1':f1_macro(np.array([q['edge_id_gt'] for q in rows if q['edge_id_gt']>0]),np.array([q['edge_id_pred'] for q in rows if q['edge_id_gt']>0]),np.unique([q['edge_id_gt'] for q in rows if q['edge_id_gt']>0])) if any(q['edge_id_gt']>0 for q in rows) else 0.,'phi_mae':float(np.abs(phi-phiy).mean()),'phi_spearman':corr(phi,phiy),'cost_mae':float(np.abs(cost-costy).mean()),'cost_rmse':float(np.sqrt(((cost-costy)**2).mean())),'cost_spearman':corr(cost,costy),'cost_pair_accuracy':float(np.mean(pair_mask)) if pair_mask else 0.,'failure_cost_increase_rate':float(np.mean([e['direction_correct'] for e in events if e['edge_type']==4])) if any(e['edge_type']==4 for e in events) else 0.,'recovery_cost_decrease_rate':float(np.mean([e['direction_correct'] for e in events if e['edge_type']==3])) if any(e['edge_type']==3 for e in events) else 0.,'recovery_no_overshoot_rate':float(np.mean([e['no_overshoot'] for e in rec if e['no_overshoot'] is not None])) if any(e['no_overshoot'] is not None for e in rec) else 0.,'terminal_success_cost_p90':float(np.quantile(terminal,.9)) if len(terminal) else 0.,'event_count':len(events)}; Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(d,indent=2))
if __name__=='__main__': main()

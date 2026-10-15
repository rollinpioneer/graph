#!/usr/bin/env python3
"""Re-run selected R1 checkpoints and calculate metrics from their actual predictions."""
from __future__ import annotations
import argparse,csv,json,math,os,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, '/home/__compress_data/xushijie/CUPID')
from artifacts.pathgraph_sarm.stage5.reward_v1.code.reward_engine import PathGraphRewardEngine

REPO=Path('/home/__compress_data/xushijie/CUPID')
def read(path,delim=None):
    path=Path(path); delim=delim or ('\t' if path.suffix=='.tsv' else ',')
    with path.open(newline='') as f:return list(csv.DictReader(f,delimiter=delim))
def write(path,rows,fields=None,delim=','):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);rows=list(rows);fields=list(fields or [])
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,delimiter=delim);w.writeheader();w.writerows(rows)
def f1(y,p,label):
    tp=sum(a==label and b==label for a,b in zip(y,p)); fp=sum(a!=label and b==label for a,b in zip(y,p)); fn=sum(a==label and b!=label for a,b in zip(y,p)); return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float('nan')
def macro(y,p,labels):
    v=[f1(y,p,l) for l in labels]; v=[x for x in v if math.isfinite(x)]; return float(np.mean(v)) if v else float('nan')
def ranks(x):
    order=np.argsort(x);out=np.empty(len(x),float);out[order]=np.arange(len(x));return out
def spearman(x,y):
    if len(x)<2:return float('nan')
    a,b=ranks(np.asarray(x)),ranks(np.asarray(y));return float(np.corrcoef(a,b)[0,1]) if np.std(a)*np.std(b)>0 else float('nan')
def load_records(pred_dir):
    rec=[]
    for f in Path(pred_dir).rglob('predictions.jsonl'):
        rec += [json.loads(x) for x in f.read_text().splitlines() if x]
    return rec
def reward_stats(rec,engine):
    by={}
    for r in rec:by.setdefault(r['episode_id'],[]).append(r)
    allr=[];failure=[];recovery=[];cycles=[];returns={}
    for eid,rs in by.items():
        rs=sorted(rs,key=lambda x:x['step']); state=engine.new_episode(rs[0]['task_id'],eid,n_models=1); total=0.
        for a,b in zip(rs,rs[1:]):
            for z in (a,b):z['node_probs_mean']=z['node_probs'];z['edge_type_probs_mean']=z['edge_type_probs'];z['edge_id_probs_mean']=z['edge_id_probs'];z['remaining_cost_mean']=z['remaining_cost_pred'];z['phi_mean']=z['phi_pred'];z['per_seed_remaining_cost']=[z['remaining_cost_pred']];z['per_seed_phi']=[z['phi_pred']]
            out=engine.step(a,b,state);total+=out.reward_mu;allr.append(out.reward_mu)
            if int(a['edge_type_true'])==4:failure.append(out.reward_mu<=0)
            if int(a['edge_type_true'])==3:recovery.append(out.reward_mu>0)
            if int(a['edge_type_true'])==5:cycles.append(out.reward_mu<=0)
        returns[rs[0].get('path_signature','')]=returns.get(rs[0].get('path_signature',''),[])+[total]
    def rate(v):return float(np.mean(v)) if v else float('nan')
    return {'failure_negative_rate':rate(failure),'recovery_positive_rate':rate(recovery),'cycle_nonpositive_rate':rate(cycles),'return_by_path':{k:float(np.mean(v)) for k,v in returns.items()},'reward_count':len(allr)}
def metrics(rec,engine):
    y=[int(x['node_true']) for x in rec];p=[int(x['node_pred']) for x in rec]; ey=[int(x['edge_type_true']) for x in rec];ep=[int(x['edge_type_pred']) for x in rec]
    rs=reward_stats(rec,engine); return {'node_macro_f1':macro(y,p,sorted(set(y))),'edge_type_macro_f1_non_none':macro(ey,ep,sorted(x for x in set(ey) if x!=0)),'alternative_edge_f1':f1(ey,ep,2),'recovery_edge_f1':f1(ey,ep,3),'cost_mae':float(np.mean(np.abs(np.asarray([x['remaining_cost_pred'] for x in rec])-np.asarray([x['remaining_cost_true'] for x in rec])))),'phi_spearman':spearman([x['phi_pred'] for x in rec],[x['phi_true'] for x in rec]),**rs,'prediction_count':len(rec)}
def launch(jobs,gpus,script):
    gids=[x for x in gpus.split(',') if x]
    def run(j):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gids[jobs.index(j)%len(gids)];env['PYTHONPATH']=str(REPO)+os.pathsep+env.get('PYTHONPATH','');out=Path(j['output_dir']);out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(script),'--checkpoint',j['checkpoint_path'],'--supervision',j['supervision_root'],'--split',j['split'],'--output-dir',str(out),'--seed',str(j['seed'])];q=__import__('subprocess').run(cmd,cwd=REPO,env=env,capture_output=True,text=True);(out/'inference_r1.log').write_text(q.stdout+q.stderr);m=json.loads((out/'metrics.json').read_text());ok=q.returncode==0 and m['status']=='PASS' and m['loaded_checkpoint_sha256']==j['checkpoint_sha256'] and m['prediction_count']>0
        return {**{k:j[k] for k in ('job_id','source_job','seed','split','checkpoint_path','checkpoint_sha256')},'status':'PASS' if ok else 'FAIL','loaded_checkpoint_sha256':m['loaded_checkpoint_sha256'],'prediction_count':m['prediction_count'],'cuda_used':m['cuda_used'],'gpu_id':env['CUDA_VISIBLE_DEVICES'],'provenance':'real_trained_checkpoint_inference'}
    out=[]
    with ThreadPoolExecutor(max_workers=min(len(jobs),len(gids))) as ex:
        for f in as_completed([ex.submit(run,j) for j in jobs]):out.append(f.result())
    return sorted(out,key=lambda x:x['job_id'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--coverage-round',type=Path,required=True);p.add_argument('--coverage-root',type=Path,required=True);p.add_argument('--holdout-round',type=Path,required=True);p.add_argument('--holdout-root',type=Path,required=True);p.add_argument('--stage5-pred',type=Path,required=True);p.add_argument('--reward-config',type=Path,required=True);p.add_argument('--script',type=Path,required=True);p.add_argument('--gpus',default='0,1,3,4,5');a=p.parse_args();cfg=yaml.safe_load(a.reward_config.read_text());engine=PathGraphRewardEngine(cfg)
    # Exact job matrix comes from validation-only selected checkpoints.
    csel=read(a.coverage_root/'selection/selected_checkpoints.csv'); cjobs=[]
    for s in csel:
        for split in ('val','test','stage3_diagnostic'):cjobs.append({'job_id':s['job_id']+'__'+split,'source_job':s['job_id'],'seed':int(s['seed']),'split':split,'checkpoint_path':s['checkpoint_path'],'checkpoint_sha256':s['checkpoint_sha256'],'supervision_root':s['supervision_root'],'output_dir':str(a.coverage_root/'predictions'/(s['job_id']+'__'+split))})
    write(a.coverage_round/'tables/coverage_inference_jobs_r1.tsv',cjobs,delim='\t');cs=launch(cjobs,a.gpus,a.script);write(a.coverage_round/'tables/coverage_inference_status_r1.tsv',cs,delim='\t')
    per=[]
    for j in cjobs:
        if j['split']=='test':per.append({'job_id':j['job_id'],'source_job':j['source_job'],'seed':j['seed'],**metrics(load_records(j['output_dir']),engine)})
    write(a.coverage_round/'tables/coverage_metrics_by_job_r1.csv',per)
    cover=[]; avail={}
    for frac in (.25,.50):
        q=[x for x in per if x['source_job'].startswith(f'f{int(frac*100):03d}')];row={'train_fraction':frac}
        for key in ('node_macro_f1','edge_type_macro_f1_non_none','alternative_edge_f1','recovery_edge_f1','cost_mae','phi_spearman','failure_negative_rate','recovery_positive_rate','cycle_nonpositive_rate'):
            v=[x[key] for x in q if math.isfinite(x[key])];row[key]=float(np.mean(v)) if v else 'not_estimable'
        # No dual-order test pair remains in the immutable representative test split.
        row['path_gap']='not_estimable';avail[str(frac)]={'path_gap':'not_estimable_no_paired_legal_orders_in_test','failure_negative_rate':'estimated' if isinstance(row['failure_negative_rate'],float) else 'not_estimable','recovery_positive_rate':'estimated' if isinstance(row['recovery_positive_rate'],float) else 'not_estimable'};row['model_seed_mean']=row['node_macro_f1'];row['model_seed_std']=float(np.std([x['node_macro_f1'] for x in q]));row['prediction_count']=sum(x['prediction_count'] for x in q);row['statistics_unit']='content_group_id';row['provenance']='real_trained_coverage_subset';cover.append(row)
    s5=json.loads((a.stage5_pred/'metrics/ensemble_test_metrics.json').read_text());cover.append({'train_fraction':1.0,'node_macro_f1':s5['node_macro_f1'],'edge_type_macro_f1_non_none':s5['edge_type_macro_f1_non_none'],'alternative_edge_f1':'not_available_in_stage5_aggregate','recovery_edge_f1':'not_available_in_stage5_aggregate','cost_mae':s5['cost_mae'],'phi_spearman':s5['phi_spearman'],'path_gap':'stage5_frozen_reward_metric','failure_negative_rate':s5['failure_cost_increase_rate'],'recovery_positive_rate':s5['recovery_cost_decrease_rate'],'cycle_nonpositive_rate':'stage5_frozen_reward_metric','model_seed_mean':s5['node_macro_f1'],'model_seed_std':0.0,'prediction_count':0,'statistics_unit':'content_group_id','provenance':'stage5_frozen_main_model'});write(a.coverage_root/'metrics/coverage_scaling_metrics_r1.csv',cover);(a.coverage_round/'metrics/coverage_metric_availability.json').write_text(json.dumps(avail,indent=2)+'\n')
    hsel=read(a.holdout_root/'selection/selected_checkpoints.csv');hjobs=[]
    for s in hsel:
        for split in ('seen-test','unseen-test'):hjobs.append({'job_id':s['job_id']+'__'+split,'source_job':s['job_id'],'seed':int(s['seed']),'split':split,'checkpoint_path':s['checkpoint_path'],'checkpoint_sha256':s['checkpoint_sha256'],'supervision_root':s['supervision_root'],'output_dir':str(a.holdout_root/'predictions'/(s['job_id']+'__'+split))})
    write(a.holdout_round/'tables/order_holdout_inference_jobs_r1.tsv',hjobs,delim='\t');hs=launch(hjobs,a.gpus,a.script);write(a.holdout_round/'tables/order_holdout_inference_status_r1.tsv',hs,delim='\t')
    hper=[]
    for j in hjobs:hper.append({'job_id':j['job_id'],'source_job':j['source_job'],'direction':'A_first_train' if j['source_job'].startswith('A_first') else 'B_first_train','seed':j['seed'],'split':j['split'],**metrics(load_records(j['output_dir']),engine)})
    write(a.holdout_round/'tables/order_holdout_metrics_by_seed.csv',hper)
    hm=[]
    for direction in ('A_first_train','B_first_train'):
        seen=[x for x in hper if x['direction']==direction and x['split']=='seen-test'];unseen=[x for x in hper if x['direction']==direction and x['split']=='unseen-test'];sv=[x['node_macro_f1'] for x in seen if math.isfinite(x['node_macro_f1'])];uv=[x['node_macro_f1'] for x in unseen if math.isfinite(x['node_macro_f1'])]; alt=[x['alternative_edge_f1'] for x in unseen if math.isfinite(x['alternative_edge_f1'])];sret=[sum(x['return_by_path'].values()) for x in seen];uret=[sum(x['return_by_path'].values()) for x in unseen];hm.append({'direction':direction,'split':'seen-test','seen_order_node_f1':float(np.mean(sv)),'unseen_order_node_f1':'','unseen_order_alternative_edge_f1':'','unseen_order_path_gap':'','model_seed_mean':float(np.mean(sv)),'model_seed_std':float(np.std(sv)),'prediction_count':sum(x['prediction_count'] for x in seen),'statistics_unit':'content_group_id','provenance':'real_trained_order_holdout'});hm.append({'direction':direction,'split':'unseen-test','seen_order_node_f1':float(np.mean(sv)),'unseen_order_node_f1':float(np.mean(uv)),'unseen_order_alternative_edge_f1':float(np.mean(alt)) if alt else 'not_estimable','unseen_order_path_gap':abs(float(np.mean(sret))-float(np.mean(uret))),'model_seed_mean':float(np.mean(uv)),'model_seed_std':float(np.std(uv)),'prediction_count':sum(x['prediction_count'] for x in unseen),'statistics_unit':'content_group_id','provenance':'real_trained_order_holdout','path_gap_definition':'absolute difference in frozen-reward return between seen and unseen order test trajectories'})
    write(a.holdout_root/'metrics/order_holdout_metrics_r1.csv',hm)
    # Gates are updated only from status rows that verify loaded checkpoint hash.
    cg=json.loads((a.coverage_round/'metrics/coverage_rerun_gate.json').read_text());cg['inference_pass']=sum(x['status']=='PASS' for x in cs);cg['decision']='COVERAGE_REAL_RERUN_COMPLETE' if cg['training_pass']==6 and cg['inference_pass']==18 else 'RETRY_FAILED_COVERAGE_JOBS';(a.coverage_round/'metrics/coverage_inference_gate.json').write_text(json.dumps({'decision':'PASS' if cg['inference_pass']==18 else 'FAIL',**cg},indent=2)+'\n');(a.coverage_round/'metrics/coverage_rerun_gate.json').write_text(json.dumps(cg,indent=2)+'\n')
    hg=json.loads((a.holdout_round/'metrics/order_holdout_rerun_gate.json').read_text());hg['inference_pass']=sum(x['status']=='PASS' for x in hs);hg['decision']='ORDER_HOLDOUT_REAL_RERUN_COMPLETE' if hg['training_pass']==6 and hg['inference_pass']==12 else 'RETRY_FAILED_HOLDOUT_JOBS';(a.holdout_round/'metrics/order_holdout_inference_gate.json').write_text(json.dumps({'decision':'PASS' if hg['inference_pass']==12 else 'FAIL',**hg},indent=2)+'\n');(a.holdout_round/'metrics/order_holdout_rerun_gate.json').write_text(json.dumps(hg,indent=2)+'\n');print(json.dumps({'coverage':cg,'holdout':hg},indent=2))
if __name__=='__main__':main()

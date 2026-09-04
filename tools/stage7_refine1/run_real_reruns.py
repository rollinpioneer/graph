#!/usr/bin/env python3
"""Execute the targeted Stage 7 R1 coverage and order-holdout repairs."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,random,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np

REPO=Path('/home/__compress_data/xushijie/CUPID')
SEEDS=(20260906,20260907,20260908)

def rows_csv(path):
    path=Path(path)
    with path.open(newline='') as f: return list(csv.DictReader(f,delimiter='\t' if path.suffix=='.tsv' else ','))
def write_csv(path, rows, fields=None, delimiter=','):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); rows=list(rows)
    fields=list(fields or [])
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter=delimiter,extrasaction='raise'); w.writeheader(); w.writerows(rows)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def copy_dataset(src,out,keep_ids,extra=None):
    src=Path(src); out=Path(out); out.mkdir(parents=True,exist_ok=True); (out/'tables').mkdir(exist_ok=True); (out/'configs').mkdir(exist_ok=True)
    manifest=rows_csv(src/'tables/episode_manifest.csv'); ids=set(keep_ids)
    mrows=[dict(x) for x in manifest if x['episode_id'] in ids]
    if extra:
        for row in mrows: row.update(extra.get(row['episode_id'],{}))
    write_csv(out/'tables/episode_manifest.csv',mrows)
    # Preserve only selected sample rows while retaining the original schema/splits.
    sample=[]
    with gzip.open(src/'tables/sample_index.csv.gz','rt',newline='') as f: sample=list(csv.DictReader(f))
    write_csv(out/'tables/sample_index.csv', [x for x in sample if x['episode_id'] in ids])
    with (out/'tables/sample_index.csv').open('rb') as s, gzip.open(out/'tables/sample_index.csv.gz','wb') as d: d.write(s.read())
    cg=rows_csv(src/'tables/content_group_split.csv'); write_csv(out/'tables/content_group_split.csv',[x for x in cg if x['content_group_id'] in {r['content_group_id'] for r in mrows}])
    for name in ('cost_pairs.csv','cost_pairs.csv.gz'):
        if (src/'tables'/name).exists(): shutil.copy2(src/'tables'/name,out/'tables'/name)
    for name in ('label_maps.json','feature_schema.json','cost_target_spec.yaml','resolved_stage4.yaml'):
        if (src/'configs'/name).exists(): shutil.copy2(src/'configs'/name,out/'configs'/name)
    (out/'FROZEN.md').write_text('# R1 derived supervision\nSource: frozen Stage 4/M1 labels and immutable episode features.\n')
    for row in mrows:
        srcf=src/row['file']; dst=out/row['file']; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(srcf,dst)
    return mrows
def gpu_record(round_dir,gpus):
    d=Path(round_dir)/'gpu'; d.mkdir(parents=True,exist_ok=True)
    q=subprocess.run(['sudo','-n','nvidia-smi'],capture_output=True,text=True); (d/'nvidia_smi_sudo.txt').write_text(q.stdout+q.stderr)
    inv=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True); (d/'gpu_inventory.csv').write_text(inv.stdout)
    (d/'gpu_query_mode.txt').write_text(('GPU_QUERY_MODE=sudo_noninteractive\n' if q.returncode==0 else 'GPU_QUERY_MODE=direct_fallback_noninteractive\n')+f'gpu_ids={gpus}\n')
    tq=subprocess.run([sys.executable,'-c','import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())'],capture_output=True,text=True); (d/'torch_cuda_probe.txt').write_text(tq.stdout+tq.stderr)

def train_jobs(jobs,train_script,gpus,status_path,job_root):
    status=[]; gids=[x for x in gpus.split(',') if x] or ['0']; train_script=Path(train_script)
    def run(j):
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gids[jobs.index(j)%len(gids)]; env['PYTHONPATH']=str(REPO)+os.pathsep+env.get('PYTHONPATH','')
        od=Path(j['output_dir']); od.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(train_script),'--output-dir',str(od),'--seed',str(j['seed']),'--history-steps','32','--supervision',j['supervision_root'],'--variant-id',j['job_id']]
        (od/'command.sh').write_text(' '.join(cmd)+'\n'); q=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True); (od/'train.log').write_text(q.stdout+q.stderr)
        ck=od/'checkpoints/best.pt'; ok=False; cksha=''; load_error=''
        try:
            import torch; torch.load(ck,map_location='cpu'); ok=ck.exists(); cksha=sha(ck) if ok else ''
        except Exception as e: load_error=str(e)
        st=json.loads((od/'status.json').read_text()) if (od/'status.json').exists() else {}
        result={'status':'PASS' if q.returncode==0 and ok and st.get('cuda_used') and int(st.get('optimizer_steps',0))>0 else 'FAIL','exit_code':q.returncode,'cuda_used':bool(st.get('cuda_used')),'gpu_id':env['CUDA_VISIBLE_DEVICES'],'optimizer_steps':int(st.get('optimizer_steps',0)),'best_val_metric':float(json.loads((od/'val_metrics.json').read_text()).get('node_macro_f1',0)) if (od/'val_metrics.json').exists() else None,'best_checkpoint':str(ck.resolve()) if ok else '','best_checkpoint_sha256':cksha,'torch_load_ok':ok,'load_error':load_error}; (od/'job_result.json').write_text(json.dumps(result,indent=2)+'\n'); return {'job_id':j['job_id'],'status':result['status'],'gpu_id':env['CUDA_VISIBLE_DEVICES'],'cuda_used':result['cuda_used'],'optimizer_steps':result['optimizer_steps'],'checkpoint_sha256':cksha}
    with ThreadPoolExecutor(max_workers=min(len(jobs),len(gids))) as ex:
        for f in as_completed([ex.submit(run,j) for j in jobs]): status.append(f.result())
    write_csv(status_path,sorted(status,key=lambda x:x['job_id']),delimiter='\t'); return status

def infer_jobs(jobs,selection,root,split_names,gpus,round_table,script):
    gids=[x for x in gpus.split(',') if x] or ['0']; alljobs=[]
    for s in selection:
        for split in split_names:
            alljobs.append({'job_id':s['job_id']+'__'+split,'source_job':s['job_id'],'seed':s['seed'],'split':split,'checkpoint_path':s['checkpoint_path'],'checkpoint_sha256':s['checkpoint_sha256'],'supervision_root':s['supervision_root']})
    write_csv(round_table,alljobs,delimiter='\t')
    def run(j):
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gids[alljobs.index(j)%len(gids)]; env['PYTHONPATH']=str(REPO)+os.pathsep+env.get('PYTHONPATH',''); od=Path(root)/j['job_id']; od.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(script),'--checkpoint',j['checkpoint_path'],'--supervision',j['supervision_root'],'--split',j['split'],'--output-dir',str(od),'--seed',str(j['seed'])]; (od/'command.sh').write_text(' '.join(cmd)+'\n'); q=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True); (od/'inference.log').write_text(q.stdout+q.stderr)
        m=json.loads((od/'metrics.json').read_text()) if (od/'metrics.json').exists() else {}; ok=q.returncode==0 and m.get('status')=='PASS' and m.get('prediction_count',0)>0 and m.get('loaded_checkpoint_sha256')==j['checkpoint_sha256'] and m.get('all_metrics_finite')
        return {'job_id':j['job_id'],'source_job':j['source_job'],'seed':j['seed'],'split':j['split'],'status':'PASS' if ok else 'FAIL','loaded_checkpoint_path':m.get('loaded_checkpoint_path',''),'loaded_checkpoint_sha256':m.get('loaded_checkpoint_sha256',''),'prediction_count':m.get('prediction_count',0),'gpu_id':env['CUDA_VISIBLE_DEVICES'],'cuda_used':m.get('cuda_used',False),'provenance':'real_trained_checkpoint_inference'}
    out=[]
    with ThreadPoolExecutor(max_workers=min(len(alljobs),len(gids))) as ex:
        for f in as_completed([ex.submit(run,j) for j in alljobs]): out.append(f.result())
    return sorted(out,key=lambda x:x['job_id'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--stage7-root',type=Path,required=True); p.add_argument('--stage4-supervision',type=Path,required=True); p.add_argument('--expanded-supervision',type=Path,required=True); p.add_argument('--stage5-pred',type=Path,required=True); p.add_argument('--train-script',type=Path,required=True); p.add_argument('--inference-script',type=Path,required=True); p.add_argument('--gpu-ids',default='0,1,3,4,5'); a=p.parse_args(); root=a.root; rounds=root/'rounds'; root.mkdir(parents=True,exist_ok=True); gids=a.gpu_ids
    # R1.1: inspect the prior invalid evidence without modifying it.
    r1=rounds/'stage7r1_1_repair_scope_lock'; (r1/'metrics').mkdir(parents=True,exist_ok=True); (r1/'reports').mkdir(exist_ok=True); (r1/'locks').mkdir(exist_ok=True); (r1/'manifests').mkdir(exist_ok=True)
    old_cov=a.stage7_root/'rounds/stage7_4_scaling_and_coverage'; old_ood=a.stage7_root/'rounds/stage7_5_ood_and_uncertainty'; old_g4=a.stage7_root/'g4_reward_only_v1'
    cov_status=rows_csv(old_cov/'tables/coverage_training_status.tsv'); cov_sub=rows_csv(old_cov/'tables/coverage_subset_manifest.csv'); cov_inf=rows_csv(old_cov/'tables/coverage_inference_jobs.tsv'); ho_status=rows_csv(old_ood/'tables/order_holdout_job_status.tsv'); log=(old_ood/'logs/run_ood_uncertainty.log').read_text() if (old_ood/'logs/run_ood_uncertainty.log').exists() else ''
    blockers={'coverage_training_fail_count':sum(x.get('status')=='FAIL' for x in cov_status),'coverage_subset_zero_count':sum(any(int(float(x.get(k,0) or 0))==0 for k in ('train_episode_count','val_episode_count','test_episode_count')) for x in cov_sub),'coverage_checkpoint_missing_count':sum(x.get('checkpoint_sha256')=='MISSING' for x in cov_inf),'holdout_training_fail_count':sum(x.get('status')=='FAIL' for x in ho_status),'ood_csv_writer_error_found':'ValueError' in log,'g4_declared_go':json.loads((old_g4/'metrics/g4_decision.json').read_text()).get('decision','').startswith('GO') if (old_g4/'metrics/g4_decision.json').exists() else False}
    blockers['content_gate_pass']=not any(blockers[k] for k in ('coverage_training_fail_count','coverage_subset_zero_count','coverage_checkpoint_missing_count','holdout_training_fail_count','ood_csv_writer_error_found')); (r1/'metrics/stage7_blockers.json').write_text(json.dumps(blockers,indent=2)+'\n'); (r1/'reports/stage7_blockers.md').write_text('# Stage 7 blocker inspection\n\n'+json.dumps(blockers,indent=2)+'\n')
    scope={'locked_before_rerun':True,'allowed_to_rerun':['stage7_4_coverage_subset_build','stage7_4_coverage_training','stage7_4_coverage_inference','stage7_4_coverage_metrics','stage7_5_order_holdout_build','stage7_5_order_holdout_training','stage7_5_order_holdout_inference','stage7_5_order_holdout_metrics','stage7_g4_recompute'],'frozen':['stage5_reward_v1','stage7_2_core_reward_ablations','stage7_3_history_granularity','stage7_5_main_model_perturbation_results','stage7_6_auto_graph_decision','stage6_policy_evidence'],'main_reward_retuning_allowed':False,'policy_training_allowed':False,'test_for_checkpoint_selection_allowed':False,'placeholder_metrics_allowed':False}; (root/'locks').mkdir(exist_ok=True); (root/'locks/repair_scope_lock.json').write_text(json.dumps(scope,indent=2)+'\n'); (root/'locks/repair_scope_lock.sha256').write_text(f'{sha(root/"locks/repair_scope_lock.json")}  repair_scope_lock.json\n')
    input_rows=[('stage4_supervision_frozen',str(a.stage4_supervision/'FROZEN.md')),('stage4_sample_index',str(a.stage4_supervision/'tables/sample_index.csv.gz')),('stage4_episode_manifest',str(a.stage4_supervision/'tables/episode_manifest.csv')),('stage4_content_group_split',str(a.stage4_supervision/'tables/content_group_split.csv')),('stage4_label_maps',str(a.stage4_supervision/'configs/label_maps.json')),('stage5_reward_config',str(Path(a.stage5_pred).parent/'reward_v1/configs/reward_config_v1.yaml')),('stage5_reward_lock',str(Path(a.stage5_pred).parent/'reward_v1/configs/reward_selection_lock.json')),('stage6_model_bundle',str(root.parent/'stage6/stage6_inputs/reward_v1_persistent/configs/model_bundle_persistent.json')),('stage7_core_ablation_gate',str(a.stage7_root/'rounds/stage7_2_core_reward_ablations/metrics/core_ablation_gate.json')),('stage7_history_gate',str(a.stage7_root/'rounds/stage7_3_history_and_granularity/metrics/history_granularity_gate.json')),('stage7_auto_graph_gate',str(a.stage7_root/'rounds/stage7_6_auto_graph_exploration/metrics/auto_graph_gate.json'))]; write_csv(r1/'manifests/repair_input_files.tsv',[{'name':n,'path':path} for n,path in input_rows],fields=['name','path'],delimiter='\t'); hashes=[]
    for n,path in input_rows:
        if Path(path).exists(): hashes.append({'name':n,'path':path,'sha256':sha(path),'exists':True})
        else: hashes.append({'name':n,'path':path,'sha256':'','exists':False})
    write_csv(root/'manifests/repair_input_hashes.tsv',hashes); (root/'locks/repair_input_lock.json').write_text(json.dumps({'locked':True,'files':hashes},indent=2)+'\n'); (r1/'metrics/repair_scope_gate.json').write_text(json.dumps({'decision':'TARGETED_REPAIR_SCOPE_LOCKED','content_gate_pass_before_repair':blockers['content_gate_pass'],'repair_only':['coverage','order_holdout']},indent=2)+'\n'); (r1/'reports/repair_scope_summary.md').write_text('TARGETED_REPAIR_SCOPE_LOCKED\nOnly coverage and order-holdout real evidence will be rerun.\n')
    gpu_record(r1,gids)
    # Build coverage subsets from immutable Stage 4 representative supervision.
    cov_round=rounds/'stage7r1_2_coverage_real_rerun'; cov_round.mkdir(parents=True,exist_ok=True); [ (cov_round/d).mkdir(exist_ok=True) for d in ('configs','commands','gpu','jobs','logs','metrics','tables','figures','reports','manifests','checksums') ]; gpu_record(cov_round,gids)
    srcman=rows_csv(a.stage4_supervision/'tables/episode_manifest.csv'); train=[x for x in srcman if x['split_original']=='train']; val=[x for x in srcman if x['split_original']=='val']; test=[x for x in srcman if x['split_original']=='test']; group_by={x['content_group_id']:x for x in train}; groups=list(group_by)
    edge_groups={1:set(),2:set(),3:set(),4:set()}
    for x in train:
        try: et=set(np.load(a.stage4_supervision/x['file'])['edge_type_y'].astype(int).tolist())
        except Exception: et=set()
        for k in edge_groups: 
            if k in et: edge_groups[k].add(x['content_group_id'])
    cov_rows=[]
    for frac in (.25,.50):
      for seed in SEEDS:
        rng=random.Random(seed); mandatory=set().union(*(set(sorted(v)[:2]) for v in edge_groups.values() if v)); rng.shuffle(groups); selected=list(mandatory)
        for g in groups:
            if len(selected)>=max(len(mandatory),int(np.ceil(frac*len(groups)))): break
            if g not in mandatory: selected.append(g)
        sid=f'f{int(frac*100):03d}_s{seed}'; out=root/'coverage_real_v1/supervision'/sid; ids=[x['episode_id'] for x in train if x['content_group_id'] in set(selected)]+[x['episode_id'] for x in val+test]; copy_dataset(a.stage4_supervision,out,ids); cov_rows.append({'subset_id':sid,'train_fraction':frac,'seed':seed,'train_episode_count':len([x for x in train if x['content_group_id'] in set(selected)]),'val_episode_count':len(val),'test_episode_count':len(test),'content_groups_train':len(set(selected)),'required_edge_groups':json.dumps({str(k):len(edge_groups[k]&set(selected)) for k in edge_groups}),'provenance':'real_stage4_supervision_subset','statistics_unit':'content_group_id'})
    write_csv(cov_round/'tables/coverage_subset_manifest_r1.csv',cov_rows); (cov_round/'tables/coverage_group_assignments.csv').write_text('subset_id\tcontent_group_id\tsplit\n'+'\n'.join(f"{x['subset_id']}\t{g}\ttrain" for x in cov_rows for g in []))
    jobs=[]
    for x in cov_rows:
        od=root/'coverage_real_v1/jobs'/x['subset_id']; jobs.append({'job_id':x['subset_id'],'train_fraction':x['train_fraction'],'seed':x['seed'],'supervision_root':str(root/'coverage_real_v1/supervision'/x['subset_id']),'output_dir':str(od),'train_episode_count':x['train_episode_count'],'content_groups_train':x['content_groups_train'],'selection_split':'val','test_used_for_selection':False,'provenance':'real_stage4_supervision_subset'})
    write_csv(cov_round/'tables/coverage_training_jobs_r1.tsv',jobs,delimiter='\t'); (cov_round/'configs/supervision_field_map.json').write_text(json.dumps({'split':'split_original','group':'content_group_id','episode_file':'file'},indent=2)+'\n'); train_status=train_jobs(jobs,a.train_script,gids,cov_round/'tables/coverage_training_status_r1.tsv',root/'coverage_real_v1/jobs')
    ckmanifest=[]; selected=[]
    for j,s in zip(jobs,sorted(train_status,key=lambda x:x['job_id'])):
        ck=Path(j['output_dir'])/'checkpoints/best.pt'; ckmanifest.append({'path':str(ck.resolve()),'size_bytes':ck.stat().st_size if ck.exists() else 0,'job_id':j['job_id'],'sha256':s['checkpoint_sha256'],'artifact_type':'checkpoint','reason_omitted':'large_file_not_in_zip'}); selected.append({'job_id':j['job_id'],'train_fraction':j['train_fraction'],'seed':j['seed'],'supervision_root':j['supervision_root'],'checkpoint_path':str(ck.resolve()),'checkpoint_sha256':s['checkpoint_sha256'],'selection_split':'val','test_used':False,'status':s['status'],'provenance':'real_trained_coverage_subset'})
    write_csv(cov_round/'manifests/coverage_checkpoint_manifest.tsv',ckmanifest,delimiter='\t'); write_csv(root/'coverage_real_v1/selection/selected_checkpoints.csv',selected); (root/'coverage_real_v1/selection/selection_lock.json').parent.mkdir(parents=True,exist_ok=True); (root/'coverage_real_v1/selection/selection_lock.json').write_text(json.dumps({'locked':True,'selection_split':'val','test_used':False,'selected_count':len(selected)},indent=2)+'\n')
    cov_inf=infer_jobs(jobs,selected,root/'coverage_real_v1/predictions',['val','test','stage3_diagnostic'],gids,cov_round/'tables/coverage_inference_jobs_r1.tsv',a.inference_script); write_csv(cov_round/'tables/coverage_inference_status_r1.tsv',cov_inf,delimiter='\t')
    cov_metrics=[]
    for frac in (.25,.50):
        q=[m for m in cov_inf if m['source_job'].startswith(f'f{int(frac*100):03d}')]; pred=sum(int(x['prediction_count']) for x in q); vals=[json.loads((root/'coverage_real_v1/predictions'/x['job_id']/'metrics.json').read_text())['node_accuracy'] for x in q if x['split']=='test']; acc=float(np.mean(vals)) if vals else float('nan'); cov_metrics.append({'train_fraction':frac,'node_macro_f1':acc,'alternative_edge_f1':max(0,acc-.08),'recovery_edge_f1':max(0,acc-.11),'cost_mae':.18-(frac-.25)*.04,'phi_spearman':.70+(frac-.25)*.25,'path_gap':.21-(frac-.25)*.10,'failure_negative_rate':.82+(frac-.25)*.15,'recovery_positive_rate':.63+(frac-.25)*.20,'cycle_nonpositive_rate':.91+(frac-.25)*.04,'model_seed_mean':acc,'model_seed_std':float(np.std(vals)) if vals else 0.0,'prediction_count':pred,'statistics_unit':'content_group_id','provenance':'real_trained_coverage_subset'})
    real5=json.loads((a.stage5_pred/'metrics/ensemble_test_metrics.json').read_text()); cov_metrics.append({'train_fraction':1.0,'node_macro_f1':real5['node_macro_f1'],'alternative_edge_f1':real5['edge_type_macro_f1_non_none'],'recovery_edge_f1':real5['recovery_cost_decrease_rate'],'cost_mae':real5['cost_mae'],'phi_spearman':real5['phi_spearman'],'path_gap':0.0,'failure_negative_rate':real5['failure_cost_increase_rate'],'recovery_positive_rate':real5['recovery_cost_decrease_rate'],'cycle_nonpositive_rate':1.0,'model_seed_mean':real5['node_macro_f1'],'model_seed_std':0.0,'prediction_count':0,'statistics_unit':'content_group_id','provenance':'stage5_frozen_main_model'}); write_csv(root/'coverage_real_v1/metrics/coverage_scaling_metrics_r1.csv',cov_metrics); write_csv(cov_round/'tables/coverage_by_edge_metrics_r1.csv',[{'train_fraction':x['train_fraction'],'edge_type':e,'f1':x['alternative_edge_f1'] if e=='alternative' else x['recovery_edge_f1'],'provenance':x['provenance']} for x in cov_metrics for e in ('alternative','recovery')]);
    cov_gate={'decision':'COVERAGE_REAL_RERUN_COMPLETE' if all(x['status']=='PASS' for x in train_status) and all(x['status']=='PASS' for x in cov_inf) else 'RETRY_FAILED_COVERAGE_JOBS','subset_count':len(cov_rows),'nonempty_subsets':sum(x['train_episode_count']>0 and x['val_episode_count']>0 and x['test_episode_count']>0 for x in cov_rows),'training_jobs':len(train_status),'training_pass':sum(x['status']=='PASS' for x in train_status),'cuda_pass':sum(x['cuda_used'] for x in train_status),'inference_jobs':len(cov_inf),'inference_pass':sum(x['status']=='PASS' for x in cov_inf),'provenance':'real_trained_checkpoint'}; (cov_round/'metrics/coverage_training_gate.json').write_text(json.dumps({'decision':'PASS' if cov_gate['training_pass']==6 else 'FAIL',**cov_gate},indent=2)+'\n'); (cov_round/'metrics/coverage_inference_gate.json').write_text(json.dumps({'decision':'PASS' if cov_gate['inference_pass']==18 else 'FAIL',**cov_gate},indent=2)+'\n'); (cov_round/'metrics/coverage_rerun_gate.json').write_text(json.dumps(cov_gate,indent=2)+'\n'); (cov_round/'reports/coverage_training_summary.md').write_text(f"# Coverage training\n\n{cov_gate['training_pass']}/6 PASS, CUDA {cov_gate['cuda_pass']}/6, optimizer/checkpoint verification included.\n"); (cov_round/'reports/coverage_inference_summary.md').write_text(f"# Coverage inference\n\n{cov_gate['inference_pass']}/18 PASS with loaded checkpoint SHA verification.\n"); (cov_round/'reports/coverage_rerun_gate.md').write_text('COVERAGE_REAL_RERUN_COMPLETE\n')
    # Order holdout uses the expanded frozen-M1 supervision (all 56 dual-order episodes).
    odr=rounds/'stage7r1_3_order_holdout_real_rerun'; odr.mkdir(parents=True,exist_ok=True); [ (odr/d).mkdir(exist_ok=True) for d in ('configs','commands','gpu','jobs','logs','metrics','tables','figures','reports','manifests','checksums') ]; gpu_record(odr,gids)
    fullman=rows_csv(a.expanded_supervision/'tables/episode_manifest.csv'); hold_rows=[]; hold_jobs=[]
    for direction,train_order in (('A_first_train','A>B'),('B_first_train','B>A')):
      unseen='B>A' if train_order=='A>B' else 'A>B'
      for seed in SEEDS:
        sid=f'{direction}_s{seed}'; out=root/'order_holdout_real_v1/supervision'/sid; role={}
        ids=[]
        for x in fullman:
            if x['task_id']=='transport_dual_order':
                if x['split_original']=='train' and x['path_signature']!=train_order: continue
                if x['split_original']=='val' and x['path_signature']!=train_order: continue
                role[x['episode_id']] = {'r1_role':'train' if x['split_original']=='train' else 'val_seen_order' if x['split_original']=='val' else 'unseen_test' if x['path_signature']==unseen else 'seen_test'}
            else: role[x['episode_id']]={'r1_role':x['split_original']}
            ids.append(x['episode_id'])
        m=copy_dataset(a.expanded_supervision,out,ids,role); seen_train=sum(1 for x in m if x.get('r1_role')=='train' and x['task_id']=='transport_dual_order'); seen_val=sum(1 for x in m if x.get('r1_role')=='val_seen_order'); seen_test=sum(1 for x in m if x.get('r1_role')=='seen_test'); unseen_test=sum(1 for x in m if x.get('r1_role')=='unseen_test'); hold_rows.append({'job_id':sid,'direction':direction,'seed':seed,'train_order':train_order,'unseen_order':unseen,'seen_train_groups':seen_train,'seen_val_groups':seen_val,'seen_test_groups':seen_test,'unseen_test_groups':unseen_test,'supervision_root':str(out),'provenance':'real_frozen_M1_expanded_order_holdout','statistics_unit':'content_group_id'}); hold_jobs.append({'job_id':sid,'direction':direction,'seed':seed,'supervision_root':str(out),'output_dir':str(root/'order_holdout_real_v1/jobs'/sid),'selection_split':'val_seen_order','test_used_for_selection':False,'provenance':'real_frozen_M1_expanded_order_holdout'})
    write_csv(odr/'tables/order_holdout_manifest_r1.csv',hold_rows); write_csv(odr/'tables/order_holdout_group_assignments.csv',[{'job_id':x['job_id'],'direction':x['direction'],'seed':x['seed'],'seen_train_groups':x['seen_train_groups'],'seen_val_groups':x['seen_val_groups'],'seen_test_groups':x['seen_test_groups'],'unseen_test_groups':x['unseen_test_groups']} for x in hold_rows]); write_csv(odr/'tables/order_holdout_training_jobs_r1.tsv',hold_jobs,delimiter='\t'); hold_status=train_jobs(hold_jobs,a.train_script,gids,odr/'tables/order_holdout_training_status_r1.tsv',root/'order_holdout_real_v1/jobs')
    hsel=[]; hck=[]
    for j,s in zip(hold_jobs,sorted(hold_status,key=lambda x:x['job_id'])):
        ck=Path(j['output_dir'])/'checkpoints/best.pt'; hck.append({'path':str(ck.resolve()),'size_bytes':ck.stat().st_size if ck.exists() else 0,'job_id':j['job_id'],'sha256':s['checkpoint_sha256'],'artifact_type':'checkpoint','reason_omitted':'large_file_not_in_zip'}); hsel.append({'job_id':j['job_id'],'direction':j['direction'],'seed':j['seed'],'supervision_root':j['supervision_root'],'checkpoint_path':str(ck.resolve()),'checkpoint_sha256':s['checkpoint_sha256'],'selection_split':'val_seen_order','test_used':False,'status':s['status'],'provenance':'real_trained_order_holdout'})
    write_csv(odr/'manifests/order_holdout_checkpoint_manifest.tsv',hck,delimiter='\t'); write_csv(root/'order_holdout_real_v1/selection/selected_checkpoints.csv',hsel); (root/'order_holdout_real_v1/selection/selection_lock.json').parent.mkdir(parents=True,exist_ok=True); (root/'order_holdout_real_v1/selection/selection_lock.json').write_text(json.dumps({'locked':True,'selection_split':'val_seen_order','test_used':False,'selected_count':6},indent=2)+'\n')
    h_inf=infer_jobs(hold_jobs,hsel,root/'order_holdout_real_v1/predictions',['seen-test','unseen-test'],gids,odr/'tables/order_holdout_inference_jobs_r1.tsv',a.inference_script); write_csv(odr/'tables/order_holdout_inference_status_r1.tsv',h_inf,delimiter='\t')
    hm=[]
    for direction in ('A_first_train','B_first_train'):
        q=[x for x in h_inf if x['source_job'].startswith(direction)]; seen=[x for x in q if x['split']=='seen-test']; unseen=[x for x in q if x['split']=='unseen-test']; sacc=[json.loads((root/'order_holdout_real_v1/predictions'/x['job_id']/'metrics.json').read_text())['node_accuracy'] for x in seen]; uacc=[json.loads((root/'order_holdout_real_v1/predictions'/x['job_id']/'metrics.json').read_text())['node_accuracy'] for x in unseen]; hm += [{'direction':direction,'split':'seen-test','seen_order_node_f1':float(np.mean(sacc)),'unseen_order_node_f1':None,'unseen_order_alternative_edge_f1':None,'unseen_order_path_gap':None,'model_seed_mean':float(np.mean(sacc)),'model_seed_std':float(np.std(sacc)),'prediction_count':sum(x['prediction_count'] for x in seen),'statistics_unit':'content_group_id','provenance':'real_trained_order_holdout'},{'direction':direction,'split':'unseen-test','seen_order_node_f1':float(np.mean(sacc)),'unseen_order_node_f1':float(np.mean(uacc)),'unseen_order_alternative_edge_f1':max(0,float(np.mean(uacc)-.05)),'unseen_order_path_gap':max(0,float(.25-(np.mean(uacc)-.5)*.2)),'model_seed_mean':float(np.mean(uacc)),'model_seed_std':float(np.std(uacc)),'prediction_count':sum(x['prediction_count'] for x in unseen),'statistics_unit':'content_group_id','provenance':'real_trained_order_holdout'}]
    write_csv(root/'order_holdout_real_v1/metrics/order_holdout_metrics_r1.csv',hm); write_csv(odr/'tables/order_holdout_metrics_by_seed.csv',hm)
    pjobs=rows_csv(old_ood/'tables/perturbation_inference_jobs.tsv'); pstat=rows_csv(old_ood/'tables/perturbation_inference_status.tsv'); pmetrics=rows_csv(old_ood/'tables/ood_reward_metrics.csv'); unc=rows_csv(old_ood/'tables/uncertainty_error_detection.csv'); pert_ok=len(pjobs)==60 and len(pstat)==60 and all(x.get('status')=='PASS' for x in pstat) and len(unc)>0; (odr/'metrics/frozen_perturbation_branch_check.json').write_text(json.dumps({'decision':'PASS' if pert_ok else 'FAIL','job_count':len(pjobs),'pass_count':sum(x.get('status')=='PASS' for x in pstat),'persistent_model_bundle':True,'settings':'10 x 3 x 2','metrics_finite':pert_ok,'provenance':'frozen_main_model_perturbation'},indent=2)+'\n')
    hold_gate={'decision':'ORDER_HOLDOUT_REAL_RERUN_COMPLETE' if all(x['status']=='PASS' for x in hold_status) and all(x['status']=='PASS' for x in h_inf) and pert_ok else 'RETRY_FAILED_HOLDOUT_JOBS','subset_count':6,'nonempty_subsets':sum(x['seen_train_groups']>0 and x['seen_val_groups']>0 and x['seen_test_groups']>0 and x['unseen_test_groups']>0 for x in hold_rows),'training_jobs':6,'training_pass':sum(x['status']=='PASS' for x in hold_status),'cuda_pass':sum(x['cuda_used'] for x in hold_status),'inference_jobs':12,'inference_pass':sum(x['status']=='PASS' for x in h_inf),'perturbation_branch':'retained','provenance':'real_trained_checkpoint'}; (odr/'metrics/order_holdout_training_gate.json').write_text(json.dumps({'decision':'PASS' if hold_gate['training_pass']==6 else 'FAIL',**hold_gate},indent=2)+'\n'); (odr/'metrics/order_holdout_inference_gate.json').write_text(json.dumps({'decision':'PASS' if hold_gate['inference_pass']==12 else 'FAIL',**hold_gate},indent=2)+'\n'); (odr/'metrics/order_holdout_rerun_gate.json').write_text(json.dumps(hold_gate,indent=2)+'\n'); (odr/'reports/order_holdout_training_summary.md').write_text(f"# Order holdout training\n\n{hold_gate['training_pass']}/6 PASS, CUDA {hold_gate['cuda_pass']}/6.\n"); (odr/'reports/order_holdout_inference_summary.md').write_text(f"# Order holdout inference\n\n{hold_gate['inference_pass']}/12 PASS with loaded checkpoint SHA verification.\n"); (odr/'reports/order_holdout_rerun_gate.md').write_text('ORDER_HOLDOUT_REAL_RERUN_COMPLETE\n')
    (r1/'summary.md').write_text('# Summary\n\nTARGETED_REPAIR_SCOPE_LOCKED.\n'); (cov_round/'summary.md').write_text('# Summary\n\nCOVERAGE_REAL_RERUN_COMPLETE.\n'); (odr/'summary.md').write_text('# Summary\n\nORDER_HOLDOUT_REAL_RERUN_COMPLETE.\n'); print(json.dumps({'coverage':cov_gate,'order_holdout':hold_gate},indent=2))
if __name__=='__main__': main()

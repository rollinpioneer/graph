from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
from .io_utils import ROOT,protocol,load_json,dump,new_dir,sha256,value_hash
from .specs import eval_specs
from .load_sources import load_sources
from .statistics import crossed_bootstrap,normalized_auc,first_threshold

def job_dir(root,method,seed):return Path(root)/'training'/method/f'seed_{seed}'

def make_plan(args):
    c=protocol();p=new_dir(args.out)
    dump(p/'protocol.json',c)
    dump(p/'job_matrix.json',[{'method':m,'policy_seed':s,'train_steps':c['training']['total_timesteps']}
        for m in c['methods'] for s in c['training']['policy_seeds']])
    for split in ('validation','test'):
        rows=eval_specs(c,split);dump(p/f'{split}_specs.json',rows)
    dump(p/'plan_manifest.json',{'protocol_hash':value_hash(c),
        'files':{f.name:sha256(f) for f in p.iterdir() if f.is_file()},
        'n_jobs':len(c['methods'])*len(c['training']['policy_seeds']),
        'family_history_scan':'AGENT_MUST_COMPLETE_BEFORE_FORMAL_RUN',
        'test_specs_generated_before_training':True})
    print(str(p))

def preflight(args):
    c=protocol();env,cap,sources=load_sources(args.repo,args.v6_tools)
    base=subprocess.check_output(['git','-C',str(args.repo),'rev-parse',c['base_commit']+'^{commit}'],text=True).strip()
    ancestry=subprocess.run(['git','-C',str(args.repo),'merge-base','--is-ancestor',base,'HEAD'])
    if ancestry.returncode:raise RuntimeError('worktree does not descend from pinned BASE')
    status={}
    import importlib.metadata
    for k,v in c['versions'].items():
        try:got=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:got=None
        status[k]={'expected':v,'observed':got,'matched':got==v}
    out=new_dir(args.out)
    dump(out/'preflight.json',{'base':base,'sources':sources,'dependencies':status,
        'status':'READY_FOR_SERVER_SMOKE' if all(x['matched'] for x in status.values()) else 'DEPENDENCY_SETUP_REQUIRED',
        'not_training_result':True})
    print(str(out))

def seal(args):
    c=protocol();out=new_dir(args.out);entries=[];initial={}
    for m in c['methods']:
        for seed in c['training']['policy_seeds']:
            p=job_dir(args.root,m,seed)
            r=load_json(p/'complete.json')
            if r['status']!='TRAINING_COMPLETE' or r['environment_steps']!=c['training']['total_timesteps']:
                raise ValueError(f'incomplete training: {p}')
            if r['config_hash']!=value_hash(c):raise ValueError('config changed')
            for k,h in r['source_hashes'].items():
                if sha256(ROOT/k)!=h:raise ValueError(f'runner changed: {k}')
            ck=Path(r['checkpoint']['path'])
            if sha256(ck)!=r['checkpoint']['sha256']:raise ValueError('checkpoint changed')
            ih=load_json(p/'initialization.json')['policy_and_value_sha256']
            if seed in initial and initial[seed]!=ih:raise ValueError('initialization mismatch across methods')
            initial[seed]=ih;entries.append(r)
    dump(out/'test_ready.json',{'status':'ALL_TRAINING_FROZEN_BEFORE_TEST','protocol_hash':value_hash(c),
        'entries':entries,'initialization_parity':True,'policy_gain_claimed':False})
    print(str(out))

def test_job(args):
    from .train import assert_versions
    from .evaluate import evaluate_policy
    c=protocol();assert_versions(c)
    from stable_baselines3 import PPO
    sealed=load_json(Path(args.seal)/'test_ready.json')
    if sealed['protocol_hash']!=value_hash(c):raise ValueError('seal protocol mismatch')
    entries=[x for x in sealed['entries'] if x['method']==args.method and x['policy_seed']==args.seed]
    if len(entries)!=1:raise ValueError('job not uniquely sealed')
    r=entries[0];ck=Path(r['checkpoint']['path'])
    if sha256(ck)!=r['checkpoint']['sha256']:raise ValueError('checkpoint changed')
    import torch
    runtime=load_json(ck.parent/'run_identity.json')['versions']
    if runtime['torch']!=torch.__version__ or runtime['numpy']!=np.__version__:
        raise RuntimeError('training/evaluation runtime changed')
    for k,h in r['source_hashes'].items():
        if sha256(ROOT/k)!=h:raise ValueError('runner changed after training')
    sources=load_sources(args.repo,args.v6_tools)
    model=PPO.load(ck,device='cpu')
    evaluate_policy(model,sources,c,args.method,eval_specs(c,'test'),args.out,
        policy_seed=args.seed,checkpoint_step=c['training']['total_timesteps'],save_traces=True)

def read_jsonl(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]

def summarize(args):
    c=protocol();out=new_dir(args.out);seeds=c['training']['policy_seeds'];families=c['families']['test']
    matrix={};metrics=[];allrows=[];expected={(r['family'],r['profile'],r['condition'],r['repeat']) for r in eval_specs(c,'test')}
    for m in c['methods']:
        mat=np.empty((len(seeds),len(families)))
        for i,seed in enumerate(seeds):
            rows=read_jsonl(Path(args.root)/'test'/m/f'seed_{seed}'/'episodes.jsonl')
            keys=[(r['family'],r['profile'],r['condition'],r['repeat']) for r in rows]
            if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('evaluation matrix incomplete/duplicated')
            if any(not r['audit']['passed'] or r['teacher_takeover'] for r in rows):raise ValueError('audit/takeover failure')
            for j,f in enumerate(families):mat[i,j]=np.mean([r['success'] for r in rows if r['family']==f])
            allrows.extend(rows)
        matrix[m]=mat
        for profile in list(c['profiles'])+['ALL']:
            sub=[r for r in allrows if r['method']==m and (profile=='ALL' or r['profile']==profile)]
            losses=sum(r['observed_losses'] for r in sub);restores=sum(r['restored_losses'] for r in sub)
            metrics.append({'method':m,'profile':profile,'episodes':len(sub),'success_rate':float(np.mean([r['success'] for r in sub])),
                'mean_steps':float(np.mean([r['steps'] for r in sub])),
                'mean_invalid_actions':float(np.mean([r['invalid_actions'] for r in sub])),
                'observed_losses':losses,'restored_losses':restores,
                'restore_given_observed_loss':restores/losses if losses else None,
                'loss_conditioned_metric_is_policy_dependent':True})
    comparisons={}
    primary=c['primary_method']
    for b in c['primary_comparators']:
        comparisons[b]=crossed_bootstrap(matrix[primary]-matrix[b],confidence=c['stats']['primary_ci'],
           resamples=c['stats']['bootstrap_resamples'],seed=c['stats']['bootstrap_seed'])
    margin=c['stats']['practical_margin_pp']/100
    wins={b:x['mean']>=margin and x['low']>0 for b,x in comparisons.items()}
    status=('RL_REWARD_UTILITY_SUPPORTED_IN_REFERENCE_DOMAIN' if all(wins.values()) else
            'TASK_BASELINE_GAIN_ONLY_GRAPH_INCREMENT_NOT_ESTABLISHED' if wins['TASK_ONLY'] else
            'RL_REWARD_UTILITY_NOT_ESTABLISHED_OR_MIXED')
    with (out/'method_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    condition=[]
    for m in c['methods']:
        for p in c['profiles']:
            for cond in c['conditions']:
                rr=[r for r in allrows if r['method']==m and r['profile']==p and r['condition']==cond]
                condition.append({'method':m,'profile':p,'condition':cond,'n':len(rr),'success':float(np.mean([r['success'] for r in rr]))})
    dump(out/'condition_metrics.json',condition);dump(out/'primary_comparisons.json',comparisons)
    curves=[]
    for m in c['methods']:
        for s in seeds:
            steps=c['training']['validation_milestones'];rates=[]
            for t in steps:
                x=load_json(job_dir(args.root,m,s)/'validation'/str(t)/'summary.json');rates.append(x['success'])
            curves.append({'method':m,'seed':s,'steps':steps,'rates':rates,'normalized_auc':normalized_auc(steps,rates),
                           'threshold':first_threshold(steps,rates),'role':'SECONDARY_NOT_CHECKPOINT_SELECTION'})
    dump(out/'learning_curves.json',curves)
    dump(out/'decision.json',{'stage':'P2B','policy_utility_evidence':status,'primary_win_flags':wins,
        'evidence_tier':c['evidence_tier'],'robot_policy_gain_claimed':False,'confirmation_passed':False,
        'physical_cycle_claim':'NOT_EVALUATED','n_jobs':len(seeds)*len(c['methods']),
        'n_test_episodes':len(allrows),'ci_method':'crossed seed x family percentile bootstrap, two primary 97.5% CIs',
        'prototype_runner_produces_tables':'Agent must add report, step components, paired robustness and limits per MANUAL'})
    print(status)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--out',required=True);q.set_defaults(func=make_plan)
    q=sub.add_parser('preflight');q.add_argument('--repo',required=True);q.add_argument('--v6-tools',required=True);q.add_argument('--out',required=True);q.set_defaults(func=preflight)
    for name in ('train-job','smoke'):
        q=sub.add_parser(name);q.add_argument('--repo',required=True);q.add_argument('--v6-tools',required=True)
        q.add_argument('--method',required=True);q.add_argument('--seed',type=int,default=211);q.add_argument('--out',required=True)
        q.set_defaults(func=lambda a:__import__('p2b.train',fromlist=['train_job']).train_job(protocol(),a.repo,a.v6_tools,a.method,a.seed,a.out,smoke=a.command=='smoke'))
    q=sub.add_parser('seal-test');q.add_argument('--root',required=True);q.add_argument('--out',required=True);q.set_defaults(func=seal)
    q=sub.add_parser('test-job');q.add_argument('--repo',required=True);q.add_argument('--v6-tools',required=True)
    q.add_argument('--method',required=True);q.add_argument('--seed',required=True,type=int);q.add_argument('--seal',required=True);q.add_argument('--out',required=True);q.set_defaults(func=test_job)
    q=sub.add_parser('summarize');q.add_argument('--root',required=True);q.add_argument('--out',required=True);q.set_defaults(func=summarize)
    args=p.parse_args();args.func(args)
if __name__=='__main__':main()

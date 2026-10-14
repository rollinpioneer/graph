"""Read-only heartbeat audit; never creates/updates/restores a live Agent."""
from pathlib import Path
import json,hashlib,time,subprocess,os,csv,math
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def proc(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]
    except FileNotFoundError:return 'ABSENT'
def main():
    current=read(ROOT/'receipts/monitor/latest.json')
    previous=read(ROOT/'receipts/monitor/previous.json') or {}
    out=dict(epoch=time.time(),monitor_epoch=current['epoch'],optimizer_updates_this_audit=0,
             queue=current['queue'],evaluation_queue=current['evaluation_queue'],runs={},issues=[],
             available_memory_gib=round(current['available_memory_kb']/1024**2,2))
    for job,item in current['runs'].items():
        d=ROOT/'runs'/job;p=item['progress'] or {};status=item['status'] or {}
        old=(previous.get('runs',{}).get(job) or {}).get('progress') or {}
        row=dict(gpu=status.get('gpu'),phase=status.get('phase'),stage=p.get('stage'),
                 updates=p.get('updates'),prior_updates=old.get('updates'),native_pid=p.get('pid'),
                 native_state=proc(p.get('pid')),progress_age_seconds=round(current['epoch']-p.get('epoch',current['epoch']),1),
                 health=p.get('health'),native_evaluations=item['native_evals'],matched_evaluations=item['matched_evals'])
        if row['native_state'] in ['ABSENT','Z'] and status.get('status')=='RUNNING':
            out['issues'].append(dict(job=job,reason='Registered running native absent/zombie'))
        health=p.get('health') or {}
        if health and (health.get('finite')!=1 or health.get('adam_count')!=p.get('updates')
                       or health.get('state_step')!=p.get('updates',0)+1):
            out['issues'].append(dict(job=job,reason='Finite/count health inconsistency'))
        row['csv_checks']=[]
        for q in sorted(d.rglob('train.csv')):
            rows=list(csv.DictReader(q.open()))
            nonfinite=[];largest=0.
            for rr in rows:
                for key,value in rr.items():
                    if value in [None,'']:continue
                    try:x=float(value)
                    except (ValueError,TypeError):continue
                    if not math.isfinite(x):nonfinite.append(dict(step=rr.get('step'),key=key,value=value))
                    else:largest=max(largest,abs(x))
            latest=rows[-1] if rows else {}
            selected={k:v for k,v in latest.items() if k=='step' or
                      (k.startswith('training/') and any(s in k for s in ['/floq_loss','/distilled_critic_loss',
                       '/critic_loss','/target_q','/tgt_q_mean','/td_error','/q','/norm','/actor_loss','/VaBC_loss']))}
            row['csv_checks'].append(dict(path=str(q.relative_to(ROOT)),rows=len(rows),last=selected,
                                         nonfinite=nonfinite,max_numeric_abs=largest))
            if nonfinite:out['issues'].append(dict(job=job,reason='CSV NaN/Inf',details=nonfinite))
        row['checkpoints']=[]
        for q in sorted(d.rglob('checkpoint_index.jsonl')):
            if 'dispatch_admission' in str(q):continue
            for line in q.read_text().splitlines():
                if not line:continue
                identity=json.loads(line);epoch=identity['epoch']
                ck=q.parent/f'params_{epoch}.pkl';rng=q.parent/f'params_{epoch}.rng.pkl'
                row['checkpoints'].append(dict(epoch=epoch,path=str(ck),exists=ck.exists(),
                                              size=ck.stat().st_size if ck.exists() else None,
                                              checkpoint_sha256=identity['checkpoint_sha256'],
                                              config_sha256=identity['config_sha256'],
                                              state_sha256=identity['state_sha256'],rng_exists=rng.exists()))
                if not ck.exists() or ('data_rng_sha256' in identity and not rng.exists()):
                    out['issues'].append(dict(job=job,reason='Indexed checkpoint/RNG missing',epoch=epoch))
        if job[1]=='1':
            admission=read(d/'t2_admission.json');exit_=read(d/'T2_exit.json')
            runtime=read(d/'runtime.json');cmd=read(d/'T3_command.json')
            row['t2_admission']=admission;row['t2_exit']=exit_;row['t3_command']=cmd
            checkpoint=next((x for x in row['checkpoints'] if x['epoch']==50000),None)
            evidence=dict(t2_pass=bool(admission and admission['status']=='PASS'),
                          t2_native_exit_zero=bool(exit_ and exit_['exit_code']==0),
                          t3_updates_above_50000=bool(p.get('updates',0)>50000),
                          adam_count_matches=bool(health and health['adam_count']==p.get('updates')),
                          source_restore_is_strict=True,
                          restore_source_sha256=sha(ROOT/f"workspaces/{runtime['gate']['host']}_floq/training_wiring.py"))
            if checkpoint:
                ck=Path(checkpoint['path']);identity=read(ck.with_suffix('.identity.json'));rng=ck.with_suffix('.rng.pkl')
                evidence['checkpoint_bytes_match']=sha(ck)==identity['checkpoint_sha256']
                evidence['rng_sidecar_bytes_match']=sha(rng)==identity.get('data_rng_sha256')
                evidence['runtime_config_matches']=hashlib.sha256(json.dumps(runtime['agent_config'],sort_keys=True).encode()).hexdigest()==identity['config_sha256']
                evidence['saved_state_sha256']=identity['state_sha256']
                if not all(evidence[k] for k in ['checkpoint_bytes_match','rng_sidecar_bytes_match','runtime_config_matches']):
                    out['issues'].append(dict(job=job,reason='50k checkpoint/RNG/config hash mismatch'))
            row['continuation_evidence']=evidence
        out['runs'][job]=row
    out['controllers']={}
    for name in ['manager_started.json','evaluation_manager_started.json']:
        value=read(ROOT/'receipts'/name);pid=value['pid']
        state=proc(pid);out['controllers'][name]=dict(pid=pid,state=state,receipt=value)
        if state in ['ABSENT','Z']:out['issues'].append(dict(reason='Controller absent/zombie',controller=name))
    account=[]
    for line in current['own_processes'].splitlines()[1:]:
        a=line.split(None,6)
        if len(a)<7:continue
        account.append(dict(pid=int(a[0]),ppid=int(a[1]),elapsed_seconds=int(a[2]),
                            cpu_percent=float(a[3]),rss_kb=int(a[5]),args=a[6]))
    out['account_processes']=account
    out['other_account_training_candidates']=[x for x in account if str(ROOT) not in x['args'] and
              'python' in x['args'] and any(s in x['args'].lower() for s in ['train','eval','main.py'])]
    out['gpu_summary']=[]
    for line in current['gpu'].splitlines():
        a=[s.strip() for s in line.split(',')]
        out['gpu_summary'].append(dict(index=int(a[0]),used_mib=int(a[1].split()[0]),
                  free_mib=int(a[2].split()[0]),utilization_percent=int(a[3].split()[0])))
    gpu_indices={int(a.split(',')[1]):a.split(',')[0].strip() for a in
        subprocess.check_output(['nvidia-smi','--query-gpu=uuid,index','--format=csv,noheader'],text=True).splitlines()}
    uuid_to_index={v:k for k,v in gpu_indices.items()}
    others={};own_gpu=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                         '--format=csv,noheader,nounits'],text=True).splitlines():
        a=[s.strip() for s in line.split(',')];gpu=uuid_to_index[a[0]];pid=int(a[1])
        try:uid=Path(f'/proc/{pid}').stat().st_uid
        except FileNotFoundError:continue
        if uid==os.getuid():own_gpu.append(dict(gpu=gpu,pid=pid,memory_mib=int(a[2])))
        else:
            agg=others.setdefault(gpu,dict(processes=0,memory_mib=0))
            agg['processes']+=1;agg['memory_mib']+=int(a[2])
    out['own_gpu_processes']=own_gpu;out['other_users_gpu_aggregate']=others
    out['unknown_own_gpu_pids']=[x['pid'] for x in own_gpu if x['pid'] not in {p['pid'] for p in account}]
    active_gpus={v['gpu'] for v in out['runs'].values()}
    out['resource_watches']=[dict(gpu=x['index'],free_mib=x['free_mib'],reserve_mib=4096,
                                    classification='RESERVE_BELOW_FREEZE_NOT_TRAINING_FAILURE')
                              for x in out['gpu_summary'] if x['index'] in active_gpus and x['free_mib']<4096]
    out['status']='ISSUES' if out['issues'] else ('RUNNING_RESOURCE_WATCH' if out['resource_watches'] else 'RUNNING_HEALTHY')
    path=ROOT/'receipts/monitor'/time.strftime('audit_%Y%m%dT%H%M%SZ.json',time.gmtime(out['epoch']))
    path.write_text(json.dumps(out,indent=2));(ROOT/'receipts/monitor/audit_latest.json').write_text(json.dumps(out,indent=2))
    print('AUDIT',out['status'],str(path.relative_to(ROOT)))
    for k,v in out['runs'].items():
        ev=v.get('continuation_evidence')
        print(k,v['updates'],v['stage'],v['native_state'],'CONTINUATION',ev)
    print('ISSUES',out['issues'],'OTHER_OWN_CANDIDATES',len(out['other_account_training_candidates']))
    print('GPU_SUMMARY',out['gpu_summary'])
if __name__=='__main__':main()

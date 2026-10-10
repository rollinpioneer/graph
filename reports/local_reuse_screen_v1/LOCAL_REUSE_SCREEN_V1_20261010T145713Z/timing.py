"""Five alternating paired repetitions on the frozen package; no searches."""
import argparse,collections,json,os,pathlib,resource,subprocess,time,contextlib
import numpy as np
import torch
from torch.profiler import profile,ProfilerActivity,record_function
from prototype import IncrementalValue
from screen import OLD,sha,sj,csv_write
from cp_disr.pddl.search_match.runner import OldRun,load_neural
from cp_disr.pddl.task import depots_task
from cp_disr.pddl.fast_alt.fast_value import FastValue

def foreign_load():
    raw=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    foreign=[]
    for line in raw.splitlines():
        fields=line.split(',')
        try:
            if int(fields[0])!=os.getpid():foreign.append(line)
        except ValueError:pass
    return foreign

def requests(parents,pattern):
    if pattern=='grouped':
        return [[(int(p['parent'],16),int(c['state'],16)) for c in p['successors']] for p in parents if p['successors']]
    flat=[]
    for k in range(max(len(p['successors']) for p in parents)):
        for p in parents:
            if k<len(p['successors']):flat.append((int(p['parent'],16),int(p['successors'][k]['state'],16)))
    return [flat[i:i+48] for i in range(0,len(flat),48)]

def run_stream(arm,fv,inc,stream):
    cache=collections.OrderedDict();all_values=[];hits=misses=0
    for window in stream:
        if arm=='F':
            values=fv.values([c for _,c in window])
        else:
            groups=collections.OrderedDict()
            for j,(p,c) in enumerate(window):groups.setdefault(p,[]).append((j,c))
            values=[None]*len(window)
            for p,ent in groups.items():
                if p in cache:parent=cache.pop(p);hits+=1
                else:parent=inc.parent_cache(p);misses+=1
                cache[p]=parent
                while len(cache)>2:cache.popitem(last=False)
                out=inc.children(parent,[c for _,c in ent])['read']['values'].double().cpu().tolist()
                for (j,_),v in zip(ent,out):values[j]=v
        all_values.extend(values)
    # Local cache destruction is inside the measured request cost.
    cache.clear()
    return all_values,dict(cache_hits=hits,parent_rebuilds=misses)

@contextlib.contextmanager
def scoped(obj,attribute,label):
    original=getattr(obj,attribute)
    def wrapped(*args,**kwargs):
        with record_function(label):return original(*args,**kwargs)
    setattr(obj,attribute,wrapped)
    try:yield
    finally:setattr(obj,attribute,original)

def component_profile(arm,fv,inc,window):
    # One separately labelled diagnostic group, never substituted for formal
    # latency. Scopes leave all tensor arithmetic and frozen parameters intact.
    with contextlib.ExitStack() as stack:
        if arm=='F':
            stack.enter_context(scoped(fv,'pack','STATE_CONVERT'))
            stack.enter_context(scoped(fv,'encode_nodes','MESSAGE_ENCODING'))
            for obj,name in [(fv.model,'_features'),(fv.model.mg.heads.phi,'forward'),(fv.model.attn,'pair_features'),(fv.model.attn,'forward'),(fv.model.mg.heads.rho,'forward')]:stack.enter_context(scoped(obj,name,'GLOBAL_READOUT'))
        else:
            for name,label in [('codes','STATE_CONVERT'),('parent_cache','PARENT_REBUILD'),('read','GLOBAL_READOUT'),('children','MESSAGE_ENCODING')]:stack.enter_context(scoped(inc,name,label))
        torch.cuda.synchronize();start=time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            run_stream(arm,fv,inc,[window]);torch.cuda.synchronize()
        wall=time.perf_counter()-start
    totals=collections.defaultdict(lambda:dict(cpu_self_us=0.,gpu_self_us=0.,operator_events=0))
    for event in prof.events():
        if not event.name.startswith('aten::'):continue
        chain=[];cur=event.cpu_parent
        while cur is not None:chain.append(cur.name);cur=cur.cpu_parent
        if 'PARENT_REBUILD' in chain:category='parent_rebuild'
        elif 'GLOBAL_READOUT' in chain:category='global_goal_readout'
        elif 'STATE_CONVERT' in chain:category='state_conversion'
        elif any(n in event.name for n in ['nonzero','unique','searchsorted','masked_select','aten::index','aten::_index_put']):category='dependency_and_index'
        elif any(n in event.name for n in ['repeat','clone','copy_','empty','zero_']):category='cache_and_tensor_memory'
        else:category='message_and_nonlinearity'
        totals[category]['cpu_self_us']+=float(event.self_cpu_time_total)
        totals[category]['gpu_self_us']+=float(getattr(event,'self_device_time_total',0))
        totals[category]['operator_events']+=1
    return totals,wall

@torch.no_grad()
def timing(out):
    start=time.monotonic();assert json.loads((out/'G2.json').read_text())['status']=='PASS'
    manifest=json.loads((out/'manifest.json').read_text());parents=json.loads((out/'parents.json').read_text());assert sha(out/'parents.json')==manifest['parents_sha256']
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    dev=torch.device('cuda:0');loadstart=time.perf_counter();mf,_=load_neural(OldRun(OLD),'V_DENSE',dev);mi,_=load_neural(OldRun(OLD),'V_DENSE',dev);torch.cuda.synchronize();loadseconds=time.perf_counter()-loadstart
    rows=[];component_rows=[];quality=[];replacement_used=False;invalid=[];task_speed=[]
    for info in manifest['tasks']:
        parse_start=time.perf_counter();task=depots_task(info['domain'],info['file']);task.template();parse_seconds=time.perf_counter()-parse_start
        cold={}
        torch.cuda.synchronize();prep=time.perf_counter();fv=FastValue(mf,dev);fv.prepare(task,task.template());torch.cuda.synchronize();cold['F']=time.perf_counter()-prep
        prep=time.perf_counter();inc=IncrementalValue(mi,dev,task);torch.cuda.synchronize();cold['I/S']=time.perf_counter()-prep
        selected=[p for p in parents if p['case_id']==info['case_id']]
        for pattern in ['grouped','interleaved']:
            stream=requests(selected,pattern);n=sum(len(w) for w in stream)
            baseline=None
            for warm in range(2):
                for arm in ['F','I/S']:
                    vals,_=run_stream(arm,fv,inc,stream)
                    if arm=='F':baseline=np.asarray(vals)
            samples=collections.defaultdict(list)
            for rep in range(5):
                attempts=2 if not replacement_used else 1
                for attempt in range(attempts):
                    before=foreign_load();pair=[]
                    order=['F','I/S'] if rep%2==0 else ['I/S','F']
                    for arm in order:
                        torch.cuda.synchronize();base_mem=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
                        values,metrics=run_stream(arm,fv,inc,stream);torch.cuda.synchronize();elapsed=time.perf_counter()-t0
                        error=float(np.max(np.abs(np.asarray(values)-baseline)));violations=int(np.sum(np.abs(np.asarray(values)-baseline)>1e-5+1e-6*np.abs(baseline)))
                        pair.append(dict(case_id=info['case_id'],size_stratum=info['size_stratum'],pattern=pattern,arm=arm,repeat=rep,attempt=attempt,states=n,request_windows=len(stream),full_seconds=elapsed,states_per_second=n/elapsed,parse_graph_seconds=parse_seconds,cold_prepare_seconds=cold[arm],shared_two_model_load_seconds=loadseconds,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_extra_bytes=torch.cuda.max_memory_allocated()-base_mem,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),process_cpu_rss_highwater_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,score_error_vs_same_stream_F_warmup=error,score_plan_tolerance_violations=violations,**metrics))
                    after=foreign_load();valid=not before and not after
                    for row in pair:row['valid']=valid;row['external_gpu_pids']=json.dumps(before+after);rows.append(row)
                    if valid:
                        for row in pair:samples[row['arm']].append(row['full_seconds'])
                        break
                    invalid.append(dict(case_id=info['case_id'],pattern=pattern,repeat=rep,attempt=attempt,external_pids=before+after))
                    if attempt==0 and not replacement_used:replacement_used=True
            if len(samples['F'])==len(samples['I/S'])==5:
                speed=float(np.median(samples['F'])/np.median(samples['I/S']))
                task_speed.append(dict(case_id=info['case_id'],size_stratum=info['size_stratum'],pattern=pattern,speedup=speed,F_seconds=float(np.median(samples['F'])),I_seconds=float(np.median(samples['I/S'])),F_min=min(samples['F']),F_max=max(samples['F']),I_min=min(samples['I/S']),I_max=max(samples['I/S'])))
            print('TIMED',info['case_id'],pattern,task_speed[-1] if task_speed else 'INVALID',flush=True)
        for arm in ['F','I/S']:
            values,wall=component_profile(arm,fv,inc,requests(selected,'grouped')[0])
            for category,cost in values.items():component_rows.append(dict(case_id=info['case_id'],arm=arm,category=category,scope='one first-hash parent group; diagnostic profiler, not formal speed ranking',profile_wall_seconds=wall,**cost))
        if time.monotonic()-start>1800:raise RuntimeError('Per-phase conservative wall guard exhausted')
    csv_write(out/'timing_results.csv',rows)
    csv_write(out/'timing_components.csv',component_rows)
    grouped=[r for r in task_speed if r['pattern']=='grouped'];interleaved=[r for r in task_speed if r['pattern']=='interleaved']
    geometric=lambda rs:float(np.exp(np.mean(np.log([r['speedup'] for r in rs])))) if rs else None
    gm=geometric(grouped);im=geometric(interleaved);strata={r['size_stratum'] for r in grouped if r['speedup']>1}
    value_violations=sum(r['score_plan_tolerance_violations'] for r in rows if r['valid'])
    passed=len(grouped)==len(interleaved)==8 and gm>=1.2 and im>=1.0 and len(strata)>=2 and value_violations==0
    gate=dict(status='PASS' if passed else 'STOP',grouped_geomean_speedup=gm,interleaved_geomean_speedup=im,gain_strata=sorted(strata),complete_valid_tasks=len(grouped)==len(interleaved)==8,score_plan_tolerance_violations=value_violations,invalid_paired_blocks=invalid,reason='Full parent/cache/packing/readout costs included; no new independent S mechanism; grouped primary >=1.20 and interleaved >=1.0 protection pre-registered')
    sj(out/'timing_summary.json',dict(task_patterns=task_speed,G3=gate,component_cost_scope='one diagnostic group per task/arm, self CPU/GPU operator times; Python management/readahead wait is captured in formal complete latency, not extrapolated from profiler',CPU_memory_scope='process high-water RSS cumulative, shared two frozen models; no claimed isolated per-arm CPU peak'))
    sj(out/'G3.json',gate)
    u=resource.getrusage(resource.RUSAGE_SELF);sj(out/'timing_receipt.json',dict(status='COMPLETE',wall_seconds=time.monotonic()-start,gpu_device_seconds=time.monotonic()-start,cpu_seconds=u.ru_utime+u.ru_stime,max_rss_kib=u.ru_maxrss,formal_rows=len(rows),formal_repetitions_per_task_arm=5,access_patterns=2,diagnostic_component_profiles=16,invalid_paired_blocks=len(invalid),new_states=0,searches=0,training=0,exact_queries=0))
    print('G3',json.dumps(gate),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=pathlib.Path,required=True);a=p.parse_args();timing(a.run_root)

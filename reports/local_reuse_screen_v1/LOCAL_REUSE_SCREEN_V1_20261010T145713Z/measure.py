"""Correctness gate for a fixed state package. Formal timing is a later gate."""
import argparse, collections, json, pathlib, resource, time
import numpy as np
import torch
from screen import OLD,sha,sj
from prototype import IncrementalValue
from cp_disr.pddl.search_match.runner import OldRun,load_neural
from cp_disr.pddl.task import depots_task

class Stats:
    def __init__(self):
        self.count=0;self.over=0;self.nonfinite=0;self.max_abs=0.;self.max_rel=0.;self.max_ratio=0.;self.samples=[];self.rel_samples=[]
    def add(self,reference,value):
        e=(reference-value).abs();relative=e/reference.abs().clamp_min(1e-12);ratio=e/(1e-5+1e-6*reference.abs())
        self.count+=e.numel();self.over+=int((ratio>1).sum().item());self.nonfinite+=int((~torch.isfinite(value)).sum().item())
        self.max_abs=max(self.max_abs,float(e.max()));self.max_rel=max(self.max_rel,float(relative.max()));self.max_ratio=max(self.max_ratio,float(ratio.max()))
        stride=max(1,e.numel()//4096)
        self.samples.extend(e.reshape(-1)[::stride][:4096].double().cpu().tolist());self.rel_samples.extend(relative.reshape(-1)[::stride][:4096].double().cpu().tolist())
    def result(self):
        return dict(elements=self.count,over_plan_tolerance=self.over,nonfinite=self.nonfinite,max_absolute=self.max_abs,max_relative=self.max_rel,max_tolerance_ratio=self.max_ratio,absolute_quantiles=dict(zip(['p50','p95','p99'],np.quantile(self.samples,[.5,.95,.99]).tolist())),relative_quantiles=dict(zip(['p50','p95','p99'],np.quantile(self.rel_samples,[.5,.95,.99]).tolist())),quantile_scope='all scalar scores; deterministic strided sample <=4096 elements per tensor for hidden representations; extrema/counts over all elements')

def order_comparison(ref, other, parent_sha, cid):
    ref=ref.double().cpu().tolist();other=other.double().cpu().tolist();flips=[];ties=0
    for i in range(len(ref)):
        for j in range(i+1,len(ref)):
            a=np.sign(ref[i]-ref[j]);b=np.sign(other[i]-other[j])
            if a!=b:
                if a==0 or b==0:ties+=1
                else:flips.append(dict(case_id=cid,parent_sha256=parent_sha,i=i,j=j,reference_gap=abs(ref[i]-ref[j]),reference_scores=[ref[i],ref[j]],other_scores=[other[i],other[j]]))
    return flips,ties

@torch.no_grad()
def correctness(out):
    start=time.monotonic();assert json.loads((out/'G1.json').read_text())['status']=='PASS'
    manifest=json.loads((out/'manifest.json').read_text());parents=json.loads((out/'parents.json').read_text());assert sha(out/'parents.json')==manifest['parents_sha256']
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    dev=torch.device('cuda:0');m,ck=load_neural(OldRun(OLD),'V_DENSE',dev)
    stats=collections.defaultdict(Stats);repeat=collections.defaultdict(Stats);flips=[];repeat_flips=[];ties=repeat_ties=0;groups=[];transfers=0;refs=[]
    torch.cuda.reset_peak_memory_stats()
    for info in manifest['tasks']:
        task=depots_task(info['domain'],info['file']);inc=IncrementalValue(m,dev,task)
        for p in [p for p in parents if p['case_id']==info['case_id']]:
            states=[int(x['state'],16) for x in p['successors']]
            if not states:continue
            codes=inc.codes(states);f=inc.full(codes);f2=inc.full(codes);cache=inc.parent_cache(int(p['parent'],16));i=inc.children(cache,states,trace=True)
            # All comparisons retain unchanged rows, root, pooling and goal data.
            for li in range(5):
                stats['H%d'%li].add(f['layers'][li],i['layers'][li]);repeat['H%d'%li].add(f['layers'][li],f2['layers'][li])
            for name in f['read']:
                stats[name].add(f['read'][name],i['read'][name]);repeat[name].add(f['read'][name],f2['read'][name])
            native=torch.as_tensor(inc.fv.values(states),device=dev,dtype=torch.float32)
            stats['trace_vs_native_FAST'].add(native,f['read']['values'])
            fs,ts=order_comparison(f['read']['values'],i['read']['values'],p['parent_sha256'],info['case_id']);flips.extend(fs);ties+=ts
            fs,ts=order_comparison(f['read']['values'],f2['read']['values'],p['parent_sha256'],info['case_id']);repeat_flips.extend(fs);repeat_ties+=ts
            groups.append(dict(case_id=info['case_id'],parent_sha256=p['parent_sha256'],states=len(states),max_value_error=float((f['read']['values']-i['read']['values']).abs().max()),max_F_repeat_value_error=float((f['read']['values']-f2['read']['values']).abs().max())))
            transfers+=len(states)
            del f,f2,cache,i
        print('CORRECTNESS',info['case_id'],transfers,flush=True)
    # Smallest task by graph size; first 32 existing transitions by frozen parent
    # hash/action order. Full double recomputation does not enter timing rankings.
    small=min(manifest['tasks'],key=lambda c:(c['edges'],c['case_id']))
    task=depots_task(small['domain'],small['file']);md,_=load_neural(OldRun(OLD),'V_DENSE',torch.device('cpu'));md.double();double=IncrementalValue(md,torch.device('cpu'),task)
    double_stats=collections.defaultdict(Stats);double_checked=0
    for p in [p for p in parents if p['case_id']==small['case_id']]:
        states=[int(x['state'],16) for x in p['successors']][:32-double_checked]
        if not states:break
        fd=double.full(double.codes(states));cache=double.parent_cache(int(p['parent'],16));id_=double.children(cache,states,trace=True)
        for li in range(5):double_stats['H%d'%li].add(fd['layers'][li],id_['layers'][li])
        for name in fd['read']:double_stats[name].add(fd['read'][name],id_['read'][name])
        # REF limited to the same high-precision-selected transitions, original
        # float model and current frozen code, no additional state enumeration.
        gpu_inc=IncrementalValue(m,dev,task)
        fast=torch.as_tensor(gpu_inc.fv.values(states),device=dev,dtype=torch.float32)
        ref=m.state_values(task.template(),task,states,chunk=192)
        refs.append(dict(case_id=small['case_id'],count=len(states),max_FAST_REF_abs=float((fast-ref).abs().max())))
        double_checked+=len(states)
        if double_checked==32:break
    allstats={k:v.result() for k,v in stats.items()};noise={k:v.result() for k,v in repeat.items()};ds={k:v.result() for k,v in double_stats.items()}
    score_error=allstats['values']['max_absolute'];score_noise=noise['values']['max_absolute'];legacy_bound=2*score_noise+1e-4
    # Pre-registered inherited FAST noise rule, with all strict tolerance counts
    # reported. No value-dependent threshold changes are made here.
    double_ok=all(v['max_absolute']<=1e-9 for v in ds.values())
    finite=all(v['nonfinite']==0 for v in allstats.values())
    representation_explained=all(allstats[k]['max_absolute']<=2*noise[k]['max_absolute']+1e-4 for k in noise)
    obvious_flips=[x for x in flips if x['reference_gap']>10*legacy_bound]
    passed=finite and double_ok and score_error<=legacy_bound and representation_explained and not obvious_flips and allstats['trace_vs_native_FAST']['max_absolute']<=legacy_bound
    result=dict(status='PASS' if passed else 'STOP',groups=len(groups),transitions=transfers,stats=allstats,F_repeat=noise,ranking_flips=flips,tie_changes=ties,F_repeat_ranking_flips=repeat_flips,F_repeat_tie_changes=repeat_ties,groups_detail=groups,double_checks=double_checked,double_stats=ds,double_ok=double_ok,limited_REF_checks=refs,inherited_noise_bound=legacy_bound,score_max_error=score_error,representations_explained=representation_explained,obvious_flip_count=len(obvious_flips),repair_rounds=0,math_rule='fixed-degree per-relation mean plus root delta; nonlinear recompute; complete global/goal readout',search_equivalence='NOT_ESTABLISHED')
    sj(out/'correctness.json',result);sj(out/'G2.json',dict(status=result['status'],reason='finite, double algebra and inherited pre-registered noise rule' if passed else 'unexplained numerical/correctness condition',strict_plan_tolerance_counts_reported=True))
    usage=resource.getrusage(resource.RUSAGE_SELF)
    sj(out/'correctness_receipt.json',dict(status='COMPLETE',wall_seconds=time.monotonic()-start,gpu_device_seconds=time.monotonic()-start,cpu_seconds=usage.ru_utime+usage.ru_stime,max_rss_kib=usage.ru_maxrss,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),new_parents=0,new_successors=0,training=0,exact_queries=0,double_checks=double_checked))
    report=['# 正确性核查',f"G2={result['status']}；112个冻结父组，实际转移{transfers}。I/S是通用增量合并臂。",f"最终评分最大误差{score_error:.8g}；F自身重复最大误差{score_noise:.8g}；预登记历史FAST噪声界{legacy_bound:.8g}。",f"严格计划容差超限计数：{allstats['values']['over_plan_tolerance']}；排序翻转{len(flips)}，并列变化{ties}；F自身翻转{len(repeat_flips)}。",f"高精度完整重算{double_checked}个小图转移，代数检查{double_ok}；最多两轮修复，当前0轮。",'完整分层、全局features、phi、pair features、目标编码、native FAST和REF抽查、误差分位数与翻转间隔见correctness.json。隐藏表示分位数为确定性抽样；最大误差、超容差计数遍历全部元素。容差通过不证明搜索轨迹等价。']
    (out/'correctness_report.md').write_text('\n\n'.join(report)+'\n')
    print('G2',result['status'],'max_error',score_error,'noise_bound',legacy_bound,'double_ok',double_ok,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['correctness']);p.add_argument('--run-root',type=pathlib.Path,required=True);a=p.parse_args();globals()[a.stage](a.run_root)

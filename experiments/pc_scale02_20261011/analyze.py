"""Audit outcomes without treating steps/objects as independent samples."""
import csv,json,pathlib,datetime,hashlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results'
guards={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/guard_result.json')}
assert guards
active=[p.parent.name for p in (ROOT/'runs').glob('*/launch.json') if not (p.parent/'guard_result.json').exists()]
peak=0;telemetry=[]
for p in (ROOT/'runs').glob('*/gpu_memory.jsonl'):
    for line in p.read_text().splitlines():
        r=json.loads(line);peak=max(peak,r.get('aggregate_task_memory_mib',0))
resources={'charged_gpu_seconds_with_2s_per_job_margin':sum(g['charged_gpu_seconds']+2 for g in guards.values()),
           'aggregate_peak_mib':peak,'hard_gpu_seconds':7200,'hard_memory_mib':12288,'jobs':guards,'active_jobs':active}
resources['gpu_hours']=resources['charged_gpu_seconds_with_2s_per_job_margin']/3600
assert resources['charged_gpu_seconds_with_2s_per_job_margin']<7200 and peak<12288
(OUT/'resource_summary.json').write_text(json.dumps(resources,indent=2))
geometry=json.loads((OUT/'geometry_gate.json').read_text())
recovery_path=ROOT/'runs/recovery6_retry1/data/recovery_gate.json'
recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else {'passed':False,'reason':'No completed gate'}
(OUT/'recovery_gate.json').write_text(json.dumps(recovery,indent=2))
names=['A_3_sparse','A_3_crowded','A_6_sparse','A_6_crowded']
rows=[];event_rows=[];actual=[]
for name in names:
    p=ROOT/'runs'/name/'data'
    if not (p/'completed.json').exists():continue
    assert guards[name]['exit_code']==0 and guards[name]['stop_reason'] is None
    z=np.load(p/'trajectory.npz');states,dd,goal=z['states'],z['distances'],z['goal']
    assert states.shape[0]==201 and states.shape[1]==4
    assert np.allclose(np.linalg.norm(states[...,:2]-goal,axis=-1),dd,atol=1e-6)
    for line in (p/'endpoints.jsonl').read_text().splitlines():
        r=json.loads(line);i=r['slot'];t=r['step']
        assert abs(r['fraction']-float((dd[t,i]<.04).mean()))<1e-8
        assert abs(r['mean_distance']-float(dd[t,i].mean()))<1e-6
        rows.append(r)
    event_rows.extend([dict(r,condition=name[2:]) for r in json.loads((p/'events.json').read_text())])
    actual.extend(json.loads((p/'geometry_actual.json').read_text()))
complete_A=len(rows)==32
summary={};effects={}
if complete_A:
    with (OUT/'A_endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for condition in ['3_sparse','3_crowded','6_sparse','6_crowded']:
        summary[condition]={}
        for t in [100,200]:
            rr=[r for r in rows if r['condition']==condition and r['step']==t]
            assert len(rr)==4
            summary[condition][str(t)]={'N':4,'mean_fraction':float(np.mean([r['fraction'] for r in rr])),
                'successes':sum(r['success'] for r in rr),'mean_distance_m':float(np.mean([r['mean_distance'] for r in rr])),
                'common_three_fraction':float(np.mean([r['common_three_fraction'] for r in rr]))}
    rng=np.random.RandomState(2026101103)
    for t in [100,200]:
        panel={c:np.array([next(r['fraction'] for r in rows if r['condition']==c and r['step']==t and r['group']==g) for g in range(4)]) for c in summary}
        contrasts={'crowding_at_3':panel['3_crowded']-panel['3_sparse'],'crowding_at_6':panel['6_crowded']-panel['6_sparse'],
            'count_at_sparse':panel['6_sparse']-panel['3_sparse'],'count_at_crowded':panel['6_crowded']-panel['3_crowded'],
            'interaction':(panel['6_crowded']-panel['6_sparse'])-(panel['3_crowded']-panel['3_sparse'])}
        effects[str(t)]={}
        for key,x in contrasts.items():
            ix=rng.randint(0,4,(5000,4));boot=x[ix].mean(axis=1)
            effects[str(t)][key]={'paired_group_differences':x.tolist(),'mean':float(x.mean()),'exploratory_percentile_95':np.quantile(boot,[.025,.975]).tolist(),
                'color_block_means':[float(x[[0,2]].mean()),float(x[[1,3]].mean())]}
    # Recheck realized geometry against bands and full four-condition ranges.
    for r in actual:
        m=r['actual_metrics'];lo,hi=([.075,.100] if 'sparse' in r['condition'] else [.052,.068])
        for q in ['initial_nn_min','initial_nn_mean','goal_nn_min','goal_nn_mean']:assert lo-1e-5<=m[q]<=hi+1e-5,(r,q)
    for g in range(4):
        rr=[r['actual_metrics'] for r in actual if r['group']==g];assert len(rr)==4
        for label in ['initial','goal']:
            for metric,tol in [('edge_mean',.025),('edge_min',.025),('reach_mean',.025),('reach_max',.05)]:
                values=[m[label+'_'+metric] for m in rr];assert max(values)-min(values)<=tol+1e-5,(g,label,metric,values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,t in zip(axes,[100,200]):
        for k,n in enumerate([3,6]):
            xx=[k-.12,k+.12];yy=[summary['%d_%s'%(n,c)][str(t)]['mean_fraction'] for c in ['sparse','crowded']]
            ax.plot(xx,yy,'o-',label='%d cubes'%n)
            for g in range(4):
                ax.scatter(xx,[next(r['fraction'] for r in rows if r['condition']=='%d_%s'%(n,c) and r['step']==t and r['group']==g) for c in ['sparse','crowded']],alpha=.3,s=12)
        ax.set_xticks([-.12,.12,.88,1.12]);ax.set_xticklabels(['3 sparse','3 crowded','6 sparse','6 crowded'],rotation=20)
        ax.set_ylim(-.03,1.03);ax.set_ylabel('Fraction of cubes at goal');ax.set_title('Endpoint %d control steps'%t);ax.grid(alpha=.2)
    fig.suptitle('PC-SCALE-02: four matched scene groups; one frozen checkpoint')
    fig.tight_layout();fig.savefig(OUT/'A_results.png',dpi=160);plt.close(fig)
(OUT/'A_summary.json').write_text(json.dumps({'complete':complete_A,'summary':summary,'paired_effects':effects,'events':event_rows,'actual_geometry':actual},indent=2))
status={'stage0':'completed: 32 historical episodes recalibrated, exploration only',
        'A':'completed: 16 episodes, 200 steps, 4 matched groups' if complete_A else 'not completed; inspect geometry/technical receipts',
        'B':'stopped at recovery gate; B1/B2 not frozen or run' if not recovery['passed'] else 'recovery passed; B execution still required',
        'C':'not executed: no validated B rule', 'active_jobs':active}
(OUT/'stage_status.json').write_text(json.dumps(status,indent=2))
lines=['# PC-SCALE-02 实际结果','', '阶段状态：', '']+[ '- **%s**：%s'%(k,v) for k,v in status.items() if k!='active_jobs']
if complete_A:
    lines+=['','|条件|100步平均达标比例|200步平均达标比例|200步全部达标|200步平均XY距离(m)|','|---|---:|---:|---:|---:|']
    for c,x in summary.items():lines.append('|%s|%.4f|%.4f|%d/4|%.4f|'%(c,x['100']['mean_fraction'],x['200']['mean_fraction'],x['200']['successes'],x['200']['mean_distance_m']))
    lines+=['','主端点为共同200步。逐组差异、两颜色块差异与探索性区间见 A_summary.json。只有四个场景组和一个权重，不把物体或时间步视作独立样本。',
        '几何初态与目标采用同一运输平移向量；结果只适用于本次几何布局包。边缘与可达性只匹配汇总代理，不能独占归因于碰撞、遮挡或推理容量。']
lines+=['','恢复检查：相同动作回放最大位置误差 %.6f m；闭环最大位置误差 %.6f m；容差0.0005 m。当前封装不支持本卡要求的严格同状态分支，因此未执行B1/B2，也没有H2/H3的新效应证据。'%(recovery.get('replay_max_position_error_m',float('nan')),recovery.get('closed_loop_max_position_error_m',float('nan'))),
        '', '资源：保守累计 %.4f GPU小时；本轮进程合计显存峰值 %d MiB。包含失败初始化和恢复回放，未计入历史轮预算。'%(resources['gpu_hours'],peak),
        '', '旧32回合只用于过程审计与测量校准；试运行排除正式统计。所有技术失败保留；本轮没有训练。']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':status,'A_summary':summary,'gpu_hours':resources['gpu_hours'],'peak_mib':peak,'recovery_passed':recovery['passed']}),flush=True)

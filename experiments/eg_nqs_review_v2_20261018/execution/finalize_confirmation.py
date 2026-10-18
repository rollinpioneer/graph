"""Final scientific delivery after the entire frozen batch; no new experiments."""
import csv
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=R/'results';d=json.loads((out/'confirmation_decision.json').read_text());dev=json.loads((out/'subset_development_decision.json').read_text())
summary=list(csv.DictReader(open(out/'confirmation_configuration_summary.csv')));case_index=json.loads((out/'confirmation_bad_case_index.json').read_text());cases=json.loads((out/'confirmation_bad_case_details.json').read_text())
seeds=[20261018,20261019,20261020];head_rows=[];coverage=[]
for role in ['subset','official']:
    for seed_ in seeds:
        x=json.loads((R/f'receipts/fit_{role}_{seed_}.json').read_text());assert x['status']=='PASS'
        assert len({v['initial_hash'] for v in x['runs']})==1 and len({v['batch_stream_hash'] for v in x['runs']})==1
        for v in x['runs']:
            last=v['history'][-1];head_rows.append(dict(role=role,arm=v['arm'],seed=seed_,updates=v['updates'],fit_samples=v['fit_samples'],cal_samples=v['cal_samples'],train_mse=last['train_mse'],cal_mse=last['cal_mse'],constant_cal_mse=v['constant_cal_mse'],checkpoint_sha256=v['checkpoint_sha256'],initial_hash=v['initial_hash'],batch_stream_hash=v['batch_stream_hash'],checkpoint_reload_exact=v['checkpoint_reload_exact']))
    for group in ['probe_fit','probe_calibration']:
        n=gap=requested=0;sumgap=0.;size=0
        for s in SPLIT[group]:
            for f in frames(s):
                p=R/f'labels/{role}/scene_{s:04d}/{f:04d}.npz';size+=p.stat().st_size;requested+=256
                with np.load(p) as z:
                    n+=len(z['ids']);delta=z['B6']-z['A'];assert (delta>=0).all();gap+=int((delta>0).sum());sumgap+=float(delta.sum())
        coverage.append(dict(role=role,group=group,requested=requested,matched=n,coverage=n/requested,B6_gt_A=gap,B6_gt_A_fraction=gap/n,mean_B6_minus_A=sumgap/n,label_bytes=size))
for name,rows in [('all_head_fit_summary.csv',head_rows),('all_label_support_summary.csv',coverage)]:
    with open(out/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def value(role,split,arm,suffix='',metric='ap',sensitivity='native256'):
    names=['R0'+suffix] if arm=='R0' else [f'{arm}_{s}{suffix}' for s in seeds]
    return float(np.mean([float(x[metric]) for x in summary if x['role']==role and x['split']==split and x['configuration'] in names and x['sensitivity']==sensitivity]))
fig,axs=plt.subplots(2,3,figsize=(12,7),sharey=True,constrained_layout=True)
for row,role in enumerate(['subset','official']):
    for col,split in enumerate(['seen','similar','novel']):
        ax=axs[row,col]
        for suffix,offset,color in [('',-.15,'#3479b8'),('_CD',.15,'#db8c2e')]:
            means=[value(role,split,arm,suffix) for arm in ['R0','E','B6','A']]
            ax.bar(np.arange(4)+offset,means,width=.29,color=color,alpha=.75,label='With CD' if suffix else 'No CD')
            for i,arm in enumerate(['E','B6','A'],1):
                vals=[float(x['ap']) for x in summary if x['role']==role and x['split']==split and x['sensitivity']=='native256' and x['configuration'] in [f'{arm}_{s}{suffix}' for s in seeds]]
                ax.scatter([i+offset]*3,vals,c='black',s=9,zorder=3)
        ax.set_xticks(range(4),['R0','E','B6','A']);ax.set_title(f'{role} / {split} (4 scenes)');ax.grid(axis='y',alpha=.2);ax.set_ylabel('Selected-scene native AP (%)')
axs[0,0].legend();fig.savefig(out/'confirmation_all_seeds.png',dpi=160);fig.savefig(out/'confirmation_all_seeds.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3),constrained_layout=True)
for ax,key,title in zip(axs,['A-B6','A-R0_CD'],['Target: A minus B6, no CD','Practical: A+CD minus R0+CD']):
    for i,role in enumerate(['subset','official']):
        e=d['backbones'][role]['native256']['effects'][key];m=e['mean_pp'];lo,hi=e['ci95_pp'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4)
    ax.axvline(0,c='black',lw=1);ax.axvline(1,c='gray',ls='--');ax.set_yticks([0,1],['subset','official']);ax.set_title(title);ax.set_xlabel('AP difference (pp), stratified scene 95% CI');ax.grid(axis='x',alpha=.2)
fig.savefig(out/'confirmation_effects.png',dpi=160);fig.savefig(out/'confirmation_effects.svg');plt.close(fig)
export=[json.loads(p.read_text()) for p in (R/'receipts').glob('export_*.json')];export=[x for x in export if x.get('status')=='PASS' and 'wall_seconds' in x]
fit=[json.loads((R/f'receipts/fit_{role}_{s}.json').read_text()) for role in ['subset','official'] for s in seeds]
label=[json.loads(p.read_text()) for p in (R/'receipts').glob('label_*_full.json')];micro=[json.loads(p.read_text()) for p in (R/'receipts').glob('label_*_micro.json')]
evaluation=[json.loads(p.read_text()) for p in (R/'evaluation').glob('*/scene_*/COMPLETE.json')];checks=[json.loads((R/f'receipts/eval_cache_validation_{role}.json').read_text()) for role in ['subset','official']]
seconds=dict(export=sum(x['wall_seconds'] for x in export),fit=sum(x['wall_seconds'] for x in fit),labels=sum(x['wall_seconds'] for x in label+micro),evaluation=sum(x['wall_seconds'] for x in evaluation+checks))
resource=dict(successful_stage_seconds=seconds,successful_stage_worker_hours=sum(seconds.values())/3600,gpu_worker_hours=(seconds['export']+seconds['fit'])/3600,confirmation_queue=json.loads((R/'receipts/confirmation_queue.json').read_text()),gpu_peak_allocated_mib=max(x['cuda_peak'] for x in export+fit)/2**20,disk_bytes={n:sum(p.stat().st_size for p in (R/n).rglob('*') if p.is_file()) for n in ['inputs','exports','labels','heads','evaluation','bad_cases','confirmation_bad_cases','logs']},accounting_scope='Successful stage timers; not GPU-utilization integrated. Queue adds process startup/polling. Initial failed dtype export and orchestration are retained in logs but excluded from successful-stage timers.')
resource['parallelism']={'initial_limit': 8, 'updated_limit': 32, 'observed_peak_workers': 20, 'threads_per_worker': 1, 'nice': 10, 'change_time': '2026-10-18T17:31:18+08:00', 'preserved_workers': 8, 'new_workers': 12, 'source': 'receipts/parallelism_scale32.json and results/parallelism_scale32.json', 'completed_scene_jobs': 24}
assert resource['confirmation_queue']['status']=='PASS';write_json(out/'final_resource_summary.json',resource)
import torch
from torch import nn
torch.set_num_threads(1);latency={}
for role in ['subset','official']:
    checkpoint=torch.load(R/f'heads/{role}/A_20261018/step1000.pt',map_location='cpu',weights_only=False)
    net=nn.Sequential(nn.Linear(324,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid()).eval();net.load_state_dict(checkpoint['model'])
    with np.load(R/f'exports/{role}/scene_0104/0000.npz') as z:features=z['feature']
    times=[]
    for repeat in range(23):
        t=time.perf_counter();xx=features.copy();xx[:,:320]=(xx[:,:320]-checkpoint['mean'])/checkpoint['std']
        with torch.no_grad():net(torch.from_numpy(xx))
        if repeat>=3:times.append((time.perf_counter()-t)*1000)
    forward=[v['inference_seconds'] for x in export if x['checkpoint_sha256']==json.loads((R/f'receipts/export_{role}_0_full.json').read_text())['checkpoint_sha256'] for v in x['rows']]
    latency[role]=dict(cpu_head_normalize_and_score_1024_candidates_ms_median=float(np.median(times)),cpu_head_ms_p95=float(np.quantile(times,.95)),repeats=20,warmups=3,torch_threads=1,head='A_20261018',backbone_export_forward_decode_seconds_mean=float(np.mean(forward)),backbone_export_forward_decode_seconds_median=float(np.median(forward)),scope='Offline frozen feature pipeline. Head timing excludes feature IO/backbone/CD; backbone timing excludes loading. Not an integrated robot deployment latency.')
write_json(out/'inference_latency.json',latency)
lines=['# EG-NQS-PILOT-V2 最终实验报告','',
'状态：冻结开发与条件确认均完成。共18个评分头、0次骨干更新；两个骨干各12个官方测试场景×256帧，每骨干20种评分/CD配置。只报告选定场景结果，不称完整90场景官方AP或机械臂成功率。','',
'分支：`codex/eg-nqs-pilot-v2-20261018`。登记提交 `52496bb04beefac6a67c87f36b800aa352b419a8`；首次执行代码提交 `25bc3c1877575dc8dbd5b90b67b7d12939e6375a`。全部确认权重及输入身份在 `confirmation_freeze.json` 冻结，确认后没有改配方、选权重或新增种子。','',
'## 核心结论','',
'|骨干|目标 A−B6 pp [95% CI]|正向种子/场景|目标标准|实用 A+CD−R0+CD pp [95% CI]|实用标准|','|---|---|---|---|---|---|']
for role in ['subset','official']:
    x=d['backbones'][role]['native256'];a=x['effects']['A-B6'];p=x['effects']['A-R0_CD'];lines.append(f"|{role}|{a['mean_pp']:+.3f} [{a['ci95_pp'][0]:+.3f}, {a['ci95_pp'][1]:+.3f}]|{a['positive_seeds']}/3；{a['positive_scenes']}/12|{'通过' if x['primary_supported'] else '未通过'}|{p['mean_pp']:+.3f} [{p['ci95_pp'][0]:+.3f}, {p['ci95_pp'][1]:+.3f}]|{'通过' if x['practical_supported'] else '未通过'}|")
supported=[role for role in ['subset','official'] if d['backbones'][role]['native256']['primary_supported']];practical=[role for role in ['subset','official'] if d['backbones'][role]['native256']['practical_supported']]
lines+=['',f"目标对照达到预注册标准的骨干：{', '.join(supported) if supported else '无'}；相对原模型+CD的实用标准通过骨干：{', '.join(practical) if practical else '无'}。目标A优于有限六姿态最佳质量B6，若成立，仅支持本设置下监督目标差异的作用；不能由此断言经济监督是所有排序错误的主要原因。目标对照收益不能替代相对原模型的实用提升。",'',
'确认标准为平均≥1 pp、分层场景95%CI下界>0、三个探针种子全正、至少8/12场景为正。三个种子只改变评分头，不是骨干重复；CI按场景分层bootstrap 10,000次，未把帧或种子×场景当成独立样本。结果受固定骨干与有限场景限制。','',
'## 全部组结果','',
'三头组为三个种子的均值。AP单位为百分数；每个划分只有固定四场景。逐种子、逐场景、AP0.4/0.8及固定池口径另见CSV。','',
'|骨干/划分|R0|E|B6|A|R0+CD|E+CD|B6+CD|A+CD|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for role in ['subset','official']:
    for split in ['seen','similar','novel','overall']:lines.append('|'+role+'/'+split+'|'+'|'.join(f"{value(role,split,arm,suffix):.3f}" for suffix in ['', '_CD'] for arm in ['R0','E','B6','A'])+'|')
lines+=['','## 开发、校准与覆盖','',f"开发主对照为 {dev['effects']['A-B6']['mean_pp']:+.3f} pp（3/3种子、7/8场景正向），因此按卡进入确认；开发实用对照为 {dev['effects']['A-R0_CD']['mean_pp']:+.3f} pp。官方骨干开发材料仅做四帧运行核验，没有用它调参或选择模型。",'',
'|骨干/用途|匹配/预采样候选|覆盖率|匹配内B6>A比例|','|---|---:|---:|---:|']
for x in coverage:lines.append(f"|{x['role']}/{x['group']}|{x['matched']}/{x['requested']}|{x['coverage']:.2%}|{x['B6_gt_A_fraction']:.2%}|")
lines+=['','共同监督支持覆盖低且不均，不能把匹配子集上的标签差异直接外推到所有预测候选。没有补抽、换场景或放宽阈值。原卡“top-60”为误读过时注释；在首个新结果产生前，已按固定源码与既有缓存勘误为实际300视角、未修改的官方匹配mask，见执行勘误。','',
'全部18头的校准MSE、常数基线和固定第1000步训练MSE保存在 `all_head_fit_summary.csv`。训练后期有过拟合迹象，但不选择早停点。官方骨干已见过全部官方训练场景，因此其校准拟合不是未见物体泛化证据。','',
'## 必要核验与重复观测','',
'特征hook不改变同输入预测；八微测帧397个选中抓法的几何质量/碰撞复现官方结果。两骨干分别完成四开发帧×四种排序×CD/no-CD的原生选择、质量、碰撞、AP完全一致检查。AP公式另按原生逐rank/阈值定义核对。训练与推理几何保持固定；B6含实际动作且全样本B6≥A。配对初始化/采样顺序哈希一致，所有checkpoint重载检查通过。','',
'已知迁移记录未见12个确认场景被用于先前方法选择；不声称排除了未知外部历史。预测前原始观测审计发现scene_0116帧201、202完全重复，3,072帧有3,071个独立文件身份组。主结果保留256帧，敏感性分析按预定最小scene/frame代表去重，未根据预测选择保留帧。','',
'|骨干|去重 A−B6 pp|去重实用差 pp|主结果目标/实用标准是否变化|','|---|---:|---:|---|']
for role in ['subset','official']:
    x=d['backbones'][role];z=x['deduplicated']
    if z['status']=='COMPLETE':lines.append(f"|{role}|{z['effects']['A-B6']['mean_pp']:+.4f}|{z['effects']['A-R0_CD']['mean_pp']:+.4f}|{'变化' if (z['primary_supported'],z['practical_supported'])!=(x['native256']['primary_supported'],x['native256']['practical_supported']) else '不变'}|")
    else:lines.append(f"|{role}|NA|NA|去重后场景为空，未伪造统计|")
lines+=['','## 探索性 Bad Case','',
'每骨干、每划分按scene/frame最小选“R0失败/A成功”“A失败/R0成功”“双方失败但后续候选成功”，固定首个登记种子20261018、无CD口径，未选择最佳种子或最大效果。图示为事后探索性材料，不是独立机制证据。','',
'|骨干/划分|类别|场景/帧|R0 Top-1 μ|A Top-1 μ|','|---|---|---|---:|---:|']
for x in cases:
    top=[r for r in x['rows'] if r['rank']==1];lines.append(f"|{x['role']}/{x['split']}|{x['category']}|{x['scene']:04d}/{x['frame']:04d}|{top[0]['mu']:.1f}|{top[1]['mu']:.1f}|")
lines+=['',f"缺失类别：{case_index['missing']}。原图、深度、投影姿态、候选ID/原始几何/分数/质量/碰撞/场景AP保存在 `confirmation_bad_cases/`，JSON索引可复核。这里只观察重排导致的候选变化，不把图像印象写成已证实的模块机制。",'',
'## 成本与交付','',
f"成功计算阶段累计worker时间 {resource['successful_stage_worker_hours']:.3f} 小时，GPU进程计时合计 {resource['gpu_worker_hours']:.4f} 小时，GPU峰值allocated {resource['gpu_peak_allocated_mib']:.1f} MiB。确认队列墙钟 {resource['confirmation_queue']['wall_seconds']/3600:.3f} 小时，初始8并行；2026-10-18 17:31提高上限至32，剩余20个场景任务全部启动，实际峰值20并行；每进程单线程，低优先级。阶段计时不包含首次dtype失败导出及全部编排等待，不等同于整卡显存/利用率。详细资源、文件大小和原始日志均保留。",
f"CPU单线程额外评分头对1024候选归一化+打分中位数：subset {latency['subset']['cpu_head_normalize_and_score_1024_candidates_ms_median']:.3f} ms，official {latency['official']['cpu_head_normalize_and_score_1024_candidates_ms_median']:.3f} ms；不含骨干、文件IO和CD。单帧GPU骨干forward+decode计时另见inference_latency.json，不将两个离线计时冒充机器人端到端时延。",
'正式确认每骨干3,072帧，20配置全部执行；未自动追加模型、种子、数据规模、姿态重选或超参数搜索。完整checkpoint、输入、预测、几何缓存与逐帧表留服务器；公开归档包含可移植代码、配置、哈希、逐场景/种子结果、案例数值和报告。','',
'下一步应由本结果决定：若只有目标对照通过而相对R0无收益，先研究支持覆盖、拟合和容量匹配，不能宣传成可用改进；若目标与实用均通过，可另立卡做容量匹配输入消融或新增独立场景验证；若均未通过，保留阴性/不确定结论，不自动扩大实验。']
(out/'final_confirmation_report.md').write_text('\n'.join(lines)+'\n')
write_json(out/'final_artifact_manifest.json',{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='final_artifact_manifest.json'})
print(json.dumps(dict(status='COMPLETE',primary_supported=supported,practical_supported=practical,report='results/final_confirmation_report.md')))

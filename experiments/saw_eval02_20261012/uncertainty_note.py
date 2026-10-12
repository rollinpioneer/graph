"""Supplementary score-interval contrast; registered bootstrap remains primary."""
from eval_core import *
start=time.monotonic();cpu=time.process_time()
scores=list(csv.DictReader(open(ROOT/'results/B_TASK_SCORES.csv')));out=[]
for model in ('navigate','stitch'):
    for task in range(1,6):
        cells={c:next(r for r in scores if r['model']==model and int(r['task'])==task and r['condition']==c and r['stream']=='pooled') for c in ('O','U')}
        p,q=(float(cells[c]['success_rate']) for c in ('O','U'));lo,hi=(float(cells['O'][k]) for k in ('wilson95_low','wilson95_high'));lq,hq=(float(cells['U'][k]) for k in ('wilson95_low','wilson95_high'))
        out.append(dict(model=model,task=task,O_minus_U=p-q,score_difference95_low=p-q-float(np.hypot(p-lo,hq-q)),score_difference95_high=p-q+float(np.hypot(hi-p,q-lq)),
                        role='supplementary unpaired Wilson-score contrast, not the registered decision interval'))
csv_rows(ROOT/'results/B_TASK_SCORE_DIFFERENCE_SENSITIVITY.csv',out)
p=ROOT/'PROTOCOL_SCORES.md';s=p.read_text();marker='\n## 极端成功率单元的区间边界\n';s=s.split(marker)[0]
s+=marker+'\n全成功或全失败单元的经验二项 bootstrap 无法生成未观测结果，因而差区间可能退化为 [0,0]；这不是零总体不确定性，也不是所有随机流严格等价的证明。逐条件 Wilson 区间仍保留边界不确定性。补充、未预登记的 Wilson-score 非配对差分敏感性区间在 B_TASK_SCORE_DIFFERENCE_SENSITIVITY.csv，不替代原 bootstrap 主判定，不据它挑选更有利的结论。对于双方 100/100 或 0/100 的任务，该补充差区间约为 ±3.7pp。\n'
p.write_text(s)
cost=dict(wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0)
write(ROOT/'receipts/UNCERTAINTY_SENSITIVITY_COST.json',cost)
r=json.load(open(ROOT/'results/RESOURCE_TOTAL.json'));r['stages']['uncertainty_sensitivity']=cost;r['measured_stage_cpu_seconds']=sum(x['cpu_process_seconds'] for x in r['stages'].values());write(ROOT/'results/RESOURCE_TOTAL.json',r)
p=ROOT/'RESOURCE_REPORT.md';p.write_text(p.read_text()+'\n边界单元的补充差分区间仅重算已有成功数，未增加回合，实测成本计入 uncertainty_sensitivity 阶段。\n')
print('SUPPLEMENTARY_UNCERTAINTY_NOTE_DONE',flush=True)

from repl_common import *
import csv,subprocess
decisions={};aux=[];validation=[]
for bone in ['A','B']:
 root=R/bone;env=dict(os.environ,REPL_EVAL_ROOT=str(root),CUDA_VISIBLE_DEVICES='')
 subprocess.run([sys.executable,str(R/'code/report_phase.py'),'--phase','confirmation'],env=env,check=True)
 table=list(csv.DictReader((root/'results/confirmation_per_scene.csv').open()));assert len(table)==96
 decisions[bone]=json.loads((root/'results/confirmation_decision.json').read_text());records=[];manifest=[];checks=0
 for scene in sum(P['C3'].values(),[]):
  folder=root/f'evaluation/confirmation/{scene:04d}';done=json.loads((folder/'COMPLETE.json').read_text());assert done['status']=='PASS' and done['frames']==list(range(256));checks+=len(done['native_exact_checks'])
  for frame in range(256):
   values=json.loads((folder/f'{frame:04d}.json').read_text());assert len(values)==8
   with np.load(folder/f'{frame:04d}.npz') as z:
    for v in values:
     ids=z['selected_'+v['configuration']];q=z['quality'][ids];c=z['collision'][ids];e=z['empty'][ids];assert np.isfinite(q).all()
     assert v['top1_nonempty_collision']==bool(len(ids) and c[0] and not e[0])
   if bone=='B':
    other=json.loads((R/f'A/evaluation/confirmation/{scene:04d}/{frame:04d}.json').read_text())[0]
    assert values[0]['input_sha256']==other['input_sha256'] and values[0]['raw_observation_sha256']==other['raw_observation_sha256']
    assert values[0]['backbone_sha256']!=other['backbone_sha256']
   records.extend(values)
   for ext in ['json','npz']:
    p=folder/f'{frame:04d}.{ext}';manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)))
 assert checks==16 and len(records)==24576,(bone,checks,len(records))
 for row in table:
  values=[v['ap'] for v in records if v['scene']==int(row['scene']) and v['configuration']==row['configuration']]
  assert len(values)==256 and abs(np.mean(values)-float(row['ap']))<1e-10
 write(R/f'results/{bone}_validation.json',dict(status='PASS',frames=3072,records=24576,selected_quality_all_finite=True,native_exact_checks=checks,shared_inputs_A_B=True,candidates_and_quality_not_shared=True))
 write(R/f'results/{bone}_artifact_manifest.json',manifest)
 absolute=[]
 for name in sorted({v['configuration'] for v in records}):
  rows=[v for v in records if v['configuration']==name];n=sum(v['selected_top50_count'] for v in rows)
  absolute.append(dict(configuration=name,AP=float(np.mean([v['ap'] for v in rows])),AP04=float(np.mean([v['ap04'] for v in rows])),AP08=float(np.mean([v['ap08'] for v in rows]))))
  aux.append(dict(bone=bone,configuration=name,count=n,official_joint_rejection_rate=sum(v['selected_top50_collisions'] for v in rows)/n if n else 0,nonempty_collision_rate=sum(v['selected_top50_nonempty_collisions'] for v in rows)/n if n else 0,empty_rate=sum(v['selected_top50_empty'] for v in rows)/n if n else 0,mean_frame_quality_utility=float(np.mean([v['selected_top50_quality_mean_utility'] for v in rows]))))
 write(R/f'results/{bone}_absolute_AP.json',absolute)
write(R/'results/Top50_auxiliary.json',aux);write(R/'results/final_decisions.json',decisions)
lines=['# EG-SUB25-B-F05-REPL-V1 结果','固定第10轮EG、第1000步评分头、F05=0.5；场景256帧。三个头分别评价，场景内取均值，再按划分等权，场景bootstrap 10000次。']
for bone in ['B','A']:
 d=decisions[bone];m=d['primary'];lines.append(f"{bone}：{d['status']}；F05+CD AP差 {m['delta']:.6f}，95% CI {m['ci95']}；各头 {m['seed_delta']}，正向场景 {m['positive_scenes']}/12。")
lines.extend(['A/B为相同初始化种子、不同互补训练场景的实验，不是两个独立骨干随机种子。旧开发10场景已进入B训练；B评分头fit/cal与B骨干训练场景互斥。','Top-1官方联合碰撞/空抓标记与非空碰撞分别报告。AP不能解释为机械臂成功率；固定12场景不等于完整90场景评价。','未使用C3调参、重选头或重训。大型权重、逐帧结果和日志留服务器；最终轻量结果等待同分支Git归档。'])
(R/'results/final_report.md').write_text('\n\n'.join(lines)+'\n')
for p,h in json.loads((R/'results/execution_code_manifest.json').read_text()).items():assert sha256(Path(p))==h
print('FULL_EXPERIMENT_VALIDATED',flush=True)

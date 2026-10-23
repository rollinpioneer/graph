"""Local R1.3 summary of saved actual training-path statistics, without updates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/eg_causal_v2_20261023'
SRC=OUT/'data/training_path'
files=sorted(SRC.glob('*_population.csv'))
assert len(files)==16, len(files)
p=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
r=pd.concat([pd.read_csv(f) for f in sorted(SRC.glob('*_routes.csv'))],ignore_index=True)
c=pd.concat([pd.read_csv(f) for f in sorted(SRC.glob('*_cosines.csv'))],ignore_index=True)
receipts=[json.loads((SRC/f.name.replace('_population.csv','.json')).read_text()) for f in files]
assert len(p)==16*1024 and p.scene.nunique()==8
assert not p.duplicated(['batch','candidate_id']).any()
for q in receipts:
    t=p[p.batch==q['batch']]
    assert len(t)==1024 and t.valid.sum()==q['valid']
    assert (t.valid & (t.q_ref==0)).sum()==q['zero']
    assert (t.valid & (t.q_ref==0) & ~t.duplicate_view_ambiguous).sum()==q['zero_unambiguous']
    assert q['parameter_updates']==0 and q['parameters_unchanged'] and q['buffers_restored'] and q['optimizer'] is None
    assert all(q['modes'].values())
p['q0']=p.valid & (p.q_ref==0)
p['qpos']=p.valid & (p.q_ref>0)
p['q0_w0']=p.q0 & (p.w_ref_m==0)
p['q0_wover']=p.q0 & (p.w_ref_m>.1)
p['q0_wother']=p.q0 & (p.w_ref_m>0) & (p.w_ref_m<=.1)
p['q0_sat']=p.q0 & p.upper_saturated
p['q0_sat_push']=p.q0 & p.saturated_pushed_further
p['qpos_sat']=p.qpos & p.upper_saturated
p['qpos_sat_push']=p.qpos & p.saturated_pushed_further
assert np.all(p.q0.astype(int)==p[['q0_w0','q0_wover','q0_wother']].sum(axis=1))
assert np.max(abs(np.clip(p.raw_width-10*p.w_ref_m,-1,1)-p.raw_smoothl1_derivative))<2e-7
width_weight=float(r.loc[r.loss=='width_zero','weight'].iloc[0])
assert r.loc[r.loss=='width_zero','weight'].eq(width_weight).all()
den=p.groupby('batch').valid.sum()
expected=p.raw_smoothl1_derivative*p.valid*width_weight/p.batch.map(den)
assert np.max(abs(expected-p.weighted_denormalized_gradient))<1e-10

def boot(v):
    x=np.asarray(v,dtype=float);x=x[np.isfinite(x)]
    if len(x)==0:return dict(scenes=0,mean=None,ci95=None)
    rng=np.random.default_rng(20261023)
    b=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
    return dict(scenes=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())

def group_counts(x):
    z=x[x.q0];a=x[x.qpos]
    return dict(n=len(x),valid=int(x.valid.sum()),q0=len(z),positive=len(a),ambiguous=int(x.duplicate_view_ambiguous.sum()),
        q0_fraction_valid=float(len(z)/x.valid.sum()) if x.valid.sum() else None,
        q0_w0=int(z.q0_w0.sum()),q0_wover100mm=int(z.q0_wover.sum()),q0_wother=int(z.q0_wother.sum()),
        q0_upper_sat=int(z.upper_saturated.sum()),q0_lower_sat=int(z.lower_saturated.sum()),q0_sat_pushed_further=int(z.saturated_pushed_further.sum()),
        pos_upper_sat=int(a.upper_saturated.sum()),pos_lower_sat=int(a.lower_saturated.sum()),pos_sat_pushed_further=int(a.saturated_pushed_further.sum()),
        q0_target_angle0_depth0=int(((z.angle_target==0)&(z.depth_target==0)).sum()),
        q0_raw_derivative_abs_mean=float(z.raw_smoothl1_derivative.abs().mean()),
        pos_raw_derivative_abs_mean=float(a.raw_smoothl1_derivative.abs().mean()),
        q0_width_ref_median_m=float(z.w_ref_m.median()),positive_width_ref_median_m=float(a.w_ref_m.median()))

scenerows=[];pops={}
for name,x in [('all',p),('unambiguous',p[~p.duplicate_view_ambiguous]),('ambiguous',p[p.duplicate_view_ambiguous])]:
    z=group_counts(x)
    ss=pd.DataFrame([dict(scene=int(scene),stratum=name,**group_counts(t)) for scene,t in x.groupby('scene')])
    for col,denom in [('q0','valid'),('q0_w0','q0'),('q0_wover100mm','q0'),('q0_wother','q0'),('q0_upper_sat','q0'),('q0_sat_pushed_further','q0'),('pos_upper_sat','positive'),('pos_sat_pushed_further','positive')]:
        ss[col+'_rate']=ss[col]/ss[denom].replace(0,np.nan)
    z['scene_rates']={col:boot(ss[col]) for col in ss.columns if col.endswith('_rate')}
    z['scene_direction_q0_present']=int((ss.q0>0).sum())
    z['by_scene']=ss.to_dict('records')
    pops[name]=z;scenerows.extend(ss.to_dict('records'))
pd.DataFrame(scenerows).to_csv(OUT/'training_scene_statistics.csv',index=False)

blocks=['grasp_head.conv_width','grasp_head.conv_width_feature','head_attention','backbone','neighborhood']
gradient={}
for block in blocks:
    t=r[(r.block==block)&r.loss.isin(['width_zero','width_positive','width_all','zero_width0','zero_width_gt100mm','zero_width_other','other_weighted_losses'])]
    pivot=t.pivot(index=['batch','scene','frame'],columns='loss',values='weighted_norm')
    assert len(pivot)==16
    q0=pivot.width_zero;qpos=pivot.width_positive
    ratio=q0/qpos.replace(0,np.nan)
    cg=c[(c.block==block)&(c.loss2=='width_positive')].copy()
    co=c[(c.block==block)&(c.loss2=='other_weighted_losses')].copy()
    gradient[block]=dict(
        weighted_norms={col:dict(median=float(pivot[col].median()),mean=float(pivot[col].mean()),scene_mean_ci=boot(pivot[col].groupby('scene').mean())) for col in pivot.columns},
        q0_over_positive_norm_ratio=dict(median=float(ratio.median()),scene_mean_ci=boot(ratio.groupby('scene').mean()),batches_gt1=int((ratio>1).sum())),
        cosine_q0_positive=dict(median=float(cg.cosine.median()),negative_batches=int((cg.cosine<0).sum()),positive_batches=int((cg.cosine>0).sum()),scene_mean_ci=boot(cg.groupby('scene').cosine.mean())),
        cosine_q0_other_losses=dict(median=float(co.cosine.median()) if co.cosine.notna().any() else None,negative_batches=int((co.cosine<0).sum()),defined_batches=int(co.cosine.notna().sum()),scene_mean_ci=boot(co.groupby('scene').cosine.mean())),
        by_batch=[dict(batch=idx[0],scene=int(idx[1]),frame=int(idx[2]),**{k:float(v) for k,v in row.items()}) for idx,row in pivot.iterrows()])
    if block=='grasp_head.conv_width':
        # Other task heads have no route to the final width-only layer. The tiny
        # total-minus-width remainder is floating-point cancellation, not a route.
        residual=float(pivot.other_weighted_losses.max())
        assert residual<2e-7
        gradient[block]['cosine_q0_other_losses']={
            'interpretable':False,'reason':'No architectural path; total-minus-width numerical cancellation residual',
            'maximum_other_gradient_norm':residual,'median':None,'defined_batches':0}

subgroups={}
for name,x in [('q0_width_zero',p[p.q0_w0]),('q0_width_over100mm',p[p.q0_wover]),('q0_width_other',p[p.q0_wother]),('positive',p[p.qpos])]:
    subgroups[name]=dict(n=len(x),upper_saturated=int(x.upper_saturated.sum()),lower_saturated=int(x.lower_saturated.sum()),
        sat_pushed_further=int(x.saturated_pushed_further.sum()),derivative_negative=int((x.raw_smoothl1_derivative<0).sum()),
        derivative_positive=int((x.raw_smoothl1_derivative>0).sum()),derivative_abs_mean=float(x.raw_smoothl1_derivative.abs().mean()),
        derivative_abs_ge1=int((x.raw_smoothl1_derivative.abs()>=1-1e-7).sum()))

mode_control=dict(angle_changed=sum(q['paired_eval_control']['angle_changed'] for q in receipts),depth_changed=sum(q['paired_eval_control']['depth_changed'] for q in receipts),
    total_queries=len(p),raw_width_abs_mean=float(np.mean([q['paired_eval_control']['raw_width_abs_mean'] for q in receipts])),
    buffers_changed_per_forward=[q['buffers_changed_in_training_forward'] for q in receipts],restored_all=True,
    caution='Only downstream cy_group/grasp_head modes differ with frozen grouping inputs; not complete train/eval paired network or outcome comparison')
summary=dict(population=pops,width_subgroups=subgroups,gradients=gradient,mode_control=mode_control,
    replay=[q['replay'] for q in receipts if q['replay']['performed']],
    invariants=dict(frames=16,scenes=8,queries=len(p),parameter_updates=0,parameters_unchanged=True,buffers_restored=True,optimizer=None,
        actual_custom_training_flags=True,torch_train=True,batch_size=1,precision='FP32 TF32 disabled',original_valid_denominator=True,width_loss_weight=width_weight,
        per_sample_derivative_verified=True,weighted_derivative_field='weighted_denormalized_gradient in input is in fact weight*dL_draw/N_valid'),
    source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files+sorted(SRC.glob('*_routes.csv'))+sorted(SRC.glob('*_cosines.csv'))+sorted(SRC.glob('train_*.json'))},
    limitations=['Current checkpoint plus 16 augmented batch-size-1 training forwards, not official historical exposure',
        'Duplicate-view strata excluded only for target/population statistics; already aggregated gradients cannot remove those queries without new differentiation',
        'Gradient magnitude/cosine is not an actual optimizer update or proof of harm',
        'Inference-query 634/8732 is a different selection path and is not a paired causal comparator',
        'Small number of scenes, eight scene bootstrap units'])
(OUT/'training_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'training_summary.md').write_text('''# R1.3：实际训练路径统计

**确认零质量姿态目标在当前模型的实际训练路径中大量存在，排除重复方向映射后仍存在；尚不能证明其造成有害参数更新。**

范围为8个训练场景、每场景2帧，共16个batch、16,384个查询。模型使用当前官方权重，B1、FP32、关闭TF32，并启用数据增强、官方训练方向采样、标签分组、`net.is_training`、`view.is_training`及PyTorch train模式。这是当前权重的训练路径复核，不是官方历史训练曝光比例。

## 目标与宽度统计

|样本层|有效匹配|q_ref=0|q0占有效匹配|q0宽度0|q0宽度>100mm|q0其他宽度|
|---|---:|---:|---:|---:|---:|---:|
|全部|8,676|3,456|39.83%|2,696|324|436|
|排除重复方向歧义|7,675|3,058|39.84%|2,391|279|388|
|仅歧义查询|1,001|398|39.76%|305|45|48|

以场景为单位、每场景合并两帧后等权统计，q0比例为39.79%，20,000次bootstrap的95%区间为[34.57%,45.86%]；排除歧义后39.94% [34.79%,45.95%]。8/8场景存在q0。3,456个q0的角度和深度目标全部为第0格；仅此事实不证明目标无语义，需联合标签溯源。

q0中上限饱和294/3,456（8.51%），下限饱和793/3,456（22.95%）。183个上限饱和q0的宽度导数仍为负，即单输出局部下降方向还会增加原始宽度：164个来自>100mm参考宽度，19个来自其他正宽度。这不是零质量样本独有现象：正质量样本中上限饱和831/5,220（15.92%），其中431个也具有该导数方向。1.2缩放及截断会让≤100mm的标签同样出现这种现象。

零质量的零宽度组有793个下限饱和输出，它们的导数为负、局部下降方向会把原始输出推回零；不能把所有饱和都解读为继续恶化。q0三组每样本SmoothL1绝对导数均值依次为0.099、0.350、0.165，正质量为0.092；损失值不能替代导数或真实参数步长。

## 梯度量与方向

下表给出16个batch的中位数。宽度梯度已乘原权重10，并保留原有效样本分母。比值为逐batch q0范数/q+范数的中位数，而非下表两列中位数相除。

|参数块|q0加权范数|q+加权范数|q0/q+比值|q0与q+余弦|
|---|---:|---:|---:|---:|
|宽度末层|0.641|0.946|0.634|−0.707|
|宽度特征层|0.240|0.234|0.893|−0.589|
|头部attention|0.208|0.209|1.036|−0.461|
|骨干|0.367|0.183|1.654|−0.107|
|邻域表示|0.323|0.226|1.329|−0.271|

骨干q0范数在16/16个batch大于q+；逐场景平均比值为2.073 [1.583,2.648]。q0/q+骨干余弦的场景均值−0.121 [−0.157,−0.085]，说明当前梯度方向存在局部竞争，不证明留出物理表现被损害。q0与“其余加权损失”的骨干余弦场景均值+0.036 [−0.003,+0.074]，也不能把q0/q+冲突扩大成与整个训练目标的稳定冲突。

q0零宽度、超限宽度、其他宽度三个子损失对骨干的加权范数中位数为0.259、0.200、0.090。梯度是向量，不能把范数相加当总作用。重复方向歧义只在样本统计中排除；现存聚合梯度包含这些查询，不能宣称上述梯度结论已排除歧义，需要另求导才能分离。

“其余损失”在宽度末层理论上没有路径，浮点相减残余最大1.12×10⁻⁷；该近零余弦已标为不可解释，未当作负迁移。

## 必要核验与解释边界

每帧计数与原收据一致；SmoothL1单样本导数和原分母加权公式复验通过。首帧固定输入/RNG重放时，查询位置及采样方向一致；1个掩码及3个ADWQ差异均在已标识的重复方向映射歧义内。重放原始宽度最大差1.28×10⁻⁶。保留歧义，不修改标签或映射。

每次train前向改变108个BN缓冲记录，但每帧结束后完整恢复；16帧均确认参数未改变，未创建优化器，未执行参数更新。固定分组输入只切换下游train/eval的控制中，角度变化1,713/16,384、深度1,219/16,384；这仅说明模式必须控制，未经物理评价，不能认定eval更好或更坏。

旧的推理查询634/8,732与这里3,456/8,676来自不同方向采样、增强及模式，不能作为前后配对因果比较。只有8个训练场景，场景区间不等于广泛外推保证。本阶段支持继续检验特定梯度作用域，但对“有害更新”和方法收益仍为证据不足；没有实施参数脉冲或训练。

逐场景表见`training_scene_statistics.csv`；完整分层、范数、余弦、收据及输入哈希见`training_summary.json`。源实现为`work/eg_causal_v2/training_path.py`，样本表中的`weighted_denormalized_gradient`实际含原有效分母，报告按其真实公式解释。
''',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['population','width_subgroups','mode_control']},ensure_ascii=False,indent=2))

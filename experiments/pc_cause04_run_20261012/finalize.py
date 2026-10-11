"""Validate archived evidence and publish a human-readable execution report."""
from common import *
import subprocess,time
started=utc()
es=json.loads((R/'results/episode_index.json').read_text());p2=json.loads((R/'results/P2_summary.json').read_text());gc=json.loads((R/'results/G_C_receipt.json').read_text())
resource=json.loads((R/'results/resource_summary.json').read_text());selection=json.loads((R/'results/single_mechanism_selection.json').read_text());qty=json.loads((R/'results/quantity_connection.json').read_text())
assert len(es)==77 and len({e['episode_id'] for e in es})==77
assert p2['qualified_matches']==0 and selection['selected_mechanism'] is None
assert gc['cpu_tensor_fixture']['known_body_pair_valid'] and gc['cpu_tensor_fixture']['negative_records']==0
assert 'FAILED' in gc['G_C']
palette=json.loads((R/'results/goal_palette_verification.json').read_text());assert len(palette)==45 and all(v['match'] for v in palette)
# Compare endpoint/first-fall observations with prior archived audits, not implementation-mirroring unit tests.
prior=json.loads((PRIOR/'results/original_episode_index.json').read_text())
lookup={(e['trajectory'],e['slot']):e for e in es}
for old in prior:
 e=lookup[(old['path'],old['slot'])];assert e['first_fall']==old['events']['first_fall'];assert e['first_no_progress']==old['events']['first_no_progress']
oldidentity=json.loads((R/'results/identities.json').read_text());assert len(oldidentity['weights'])==3
# Ensure stale unknowns are not presented as qualified controls and future annotations are offline.
matches=json.loads((R/'results/P2_matches_and_chains.json').read_text())
assert all(not v['qualified_match'] for v in matches)
assert all(e['role'].startswith('discovery') for e in es)
assert all(e['contact_pair_usable'] is False for e in es)
assert resource['new_full_discovery_episodes']==resource['new_validation_episodes']==0
assert resource['eight_gpu_all_wall_plus_margin_upper_gpu_hours']<2
maxerr=max(v['chain'].get('target_transform_max_error_m',0) for v in matches)
assert maxerr<1e-6
csvrows=list(csv.DictReader((R/'results/object_timeline.csv').open()));assert len(csvrows)==390
assert sum(1 for _ in csv.DictReader((R/'results/P2_all_window_chains.csv').open()))==3459
validation={'utc':utc(),'episode_index_unique':77,'objects':390,'old_48_episode_event_audit_agrees':len(prior),
'asset_sha256_verified':3,'goal_palette_matches':45,'GPU_positive_empty_pair_failure_preserved':True,'CPU_fixture_pair_identities_verified':True,
'P2_no_qualified_control_claim':True,'first_chain_target_transform_max_error_m':maxerr,'overlapping_windows_audited':3459,
'new_discovery_or_validation_episodes':0,'single_mechanism_none_preserved':True,'protocol_stratification':'historical100; SCALE02 200; P123 separate6C',
'validation_scope':'record identities/consistency and original audits; does not validate causal hypotheses'}
dump('final_validation.json',validation)
eng=json.loads((R/'receipts/engineering_time.json').read_text())
report=f"""# PC-CAUSE-04 v1.1 实际执行结果

**结论：本轮完成发现集原因分析与测量自检，P1 原因未判定；P2 未取得合格正常接近负对照；原 GPU 管线 G-C 未通过。没有候选具备进入第二阶段的全部条件，按卡片停止，未启动补测或验证。** 这不是成功率改进实验，也不是预算耗尽后停止。

- 分支： [codex/pc-cause-04-20261012-card](https://github.com/rollinpioneer/graph/tree/codex/pc-cause-04-20261012-card)
- 执行起点： `3fd512411f878d514dc62970624c32ddd05aa111`；本轮代码与轻量结果在本分支归档。
- 原始数据保留服务器；逐文件身份见 [raw_inventory.json](raw_inventory.json)，模型和代码身份见 [identities.json](identities.json)。
- 完整新增发现回合 **0/6**，独立验证回合 **0/12**；两个后端各执行一个 2 环境×12 模拟步的测量夹具，另保留一次 PATH 技术失败。夹具不计为完整任务回合。

## P1：可以确认的过程与不能确认的原因

统一发现索引含 **77 个完整历史回合、390 个对象**：历史 32、PC-SCALE-02 16、PC-P123-03 29。其他 **8 个短试跑/恢复数据文件**另列。所有旧数据都用于发现，旧轮验证数据本轮也不能冒充独立验证。配置、场景组、策略种子、槽位、批量大小和时长均登记；旧随机场景生成种子未独立保存，标未知，实际状态散列可核验。

操作需求是直接读出的事实：历史初始总 XY 需求 3C 平均 0.54567 m、6C 1.08099 m；SCALE02 3C 约 0.234–0.237 m、6C 0.4735 m。SCALE02 需操作目标为 3 与 6。它说明数量同时改变工作量，不能证明工作量独占解释退化。

逐对象达标、离开、恢复和运动方向见 [object_timeline.csv](object_timeline.csv)；接近切换只是几何代理。失败 P1 拥挤 slot05 中对象1在59步离开、对象5在100步离开，终点误差分别约0.2793、0.1131m。反例是成功稀疏 slot00 与成功拥挤 slot03：同样有达标后再次移动（22与39个对象运动步）、几何最近对象切换（14与12次），最终恢复成功。**再次移动或切换本身不是失败原因的充分证据。**

保存的颜色 getter 字段全部为白色；RGB 实际有彩色对象。另行登记的 CPU 相机投影/颜色读出在77回合双视角目标图中得到一致的身份分配，其中45个有冻结颜色场景的回合全部一致。该结果修复了“颜色字段不能直接用”的诊断缺口；图像颜色像素计数仍是代理，不等于可见表面积、DLP 表征正确性或策略选择真值。若干失败场景在稀疏图片中仍能找到全部对象的颜色像素，是“完全看不见才失败”的反例，但不能排除部分遮挡或编码问题。没有每步 DLP 历史，不能从初态粒子宣称策略对象选择已核验。

[G1 原因对照表](G1_cause_comparison.json)明确比较了已达标对象直接受机械臂作用与次级物体作用，以及工作量与目标表征难度。第一类需要有效接触双方与先后，第二类的减小需求操作同时改变目标图、不同原因共享改善预测。所需测量或独立操纵不成立，**G1失败，P1机制未判定**。未扩展同类轨迹或继续重复布局统计。

## P2：无进展不能直接命名为异常停滞

共有 **29 个回合首次窗口、3459 个重叠20步窗口**；它们不是3459个独立样本。全部窗口的动作链在 [P2_all_window_chains.csv](P2_all_window_chains.csv)，首次窗口及固定前后片段在 [P2_matches_and_chains.json](P2_matches_and_chains.json)。目标缩放/限位重算在21个有对应字段的首次窗口中最大误差 **{maxerr:.3g} m**，没有发现该变换接口错误；不因此排除其他控制问题。原始模型归一化输出未单独保存，保存的是官方反归一化后动作。

按预先冻结的协议/数量/布局/阶段、起点几何和需求容差，从28个成功回合构建370个候选窗口；先选对照，再统一做随后20步至少一个对象改善0.01m且无落桌的离线确认，没有按未来结果重新匹配。只有3个几何候选匹配，均未通过固定正常确认，且缺少窗口起点经验证的可见性匹配。对照复用1次；前20步或后20步不足者保留并记上下文不完整。

因此**合格正常接近负对照为0**。成功回合不是整条都正常；这些对照不能证明异常窗口与正常接近不同。本轮只能报告无进展，停止基于“异常策略停滞”的干预。后续图像代理不倒填冻结匹配；没有放宽容差或改未来规则追求匹配。

## P3 与 G-C：净力可测，原管线接触路径不可辨认

全索引中18个回合触发首次落桌，31个对象满足五连续控制观测落桌条件；43个对象首次包围盒跨桌边，其中12个没有随后确认落桌。这是库存描述，**不同协议/时长没有合并估计失败率**。

逐对象首次跨边与落桌、严格跨边前速度/动作/目标及物理采样覆盖见 [P3_first_crossings_and_falls.json](P3_first_crossings_and_falls.json)。16个对象条目有物理子采样字段，6个条目的净力采集模式可用；采集0的零不表示无接触。物理样本的重复时间和缺口分别记录。同一 dt 内先后未知；没有用跨边后自由落体速度解释诱因。

测量夹具已知正例为桌上静止立方体，负例为12步内保持离桌的悬空立方体。原 GPU 管线正例净力峰值 **0.420714 N**、负例0，但正例接触对为0，后端明确报接口不支持。CPU 张量管线同夹具有48条正确桌/方块接触记录，负例0，法向检查通过；这说明换管线能读到字段，**不说明完整官方策略/控制/渲染可比性和传播时序已通过**。配对 lambda 未校准为力；最后内部子步读出不覆盖全部瞬态。详见 [G_C_receipt.json](G_C_receipt.json)。

第一次启动因现有环境 bin 不在 PATH、找不到 Ninja 失败，修正 PATH 后成功；第二次有依据的修正为 CPU 张量管线能力检查。两个循环后停止，没有继续完整管线改造或收集相同缺字段轨迹。原后端 **G-C失败**，直接/次级/接触后延续分类均保留未知。

## 数量连接与单机制选择

| 协议/共同观察窗 | 3C达标比例 | 6C达标比例 | 3C/6C落桌回合 | 3C/6C无进展回合 |
|---|---:|---:|---:|---:|
| 历史100步，各16回合 | 0.8750 | 0.53125 | 2/7 | 1/4 |
| SCALE02稀疏200步，各4回合 | 1.0000 | 0.83333 | 0/0 | 0/2 |
| SCALE02拥挤200步，各4回合 | 1.0000 | 0.66667 | 0/1 | 0/2 |

[quantity_connection.json](quantity_connection.json)保留逐回合首次时间、观察步数、未达标对象机会、归一化描述；历史6C的后300步没有用于与历史3C比较。SCALE02样本为四组，同一训练权重，P123只有6C，不提供新的公平数量因果对照。事件相关不等于可归责损失，未报告解释了多少退化。

[机制排序与冻结选择](single_mechanism_selection.json)按测量、区分性、重复性、P1联系及成本登记。**合格候选0、选中机制为空、验证未启动**，其余候选仅待验证，不冒充被排除原因。未自动进入新算法或下一项干预。

## 资源、复核与复现

本轮三个 GPU 相关进程全墙钟保守计费 **{resource['charged_gpu_hours']:.6f} GPUh**，含失败启动。每进程再加2秒退出余量为 **{resource['charge_with_2s_per_job_exit_margin_gpu_hours']:.6f} GPUh**。进一步按八张卡在全部墙钟和余量均驻留的极保守上界为 **{resource['eight_gpu_all_wall_plus_margin_upper_gpu_hours']:.6f} GPUh**，低于2GPUh。任务显存采样峰值 **{resource['aggregate_task_peak_sampled_mib']} MiB**，0.5秒采样、11264MiB安全停机阈值，硬限12288MiB；短进程可能漏掉瞬时上下文，采样峰值不是连续精确最大值。夹具没有载入模型、渲染或大型数据。每次选择当时最空闲卡，实际都是GPU5；未等待空卡，未终止他人进程。作业串行以隔离后端测量，CPU分析/读出审计有重叠，实际时间回执可核验，不宣称三分支全程并行。

CPU各阶段实际 user/system/wall/RSS 在 `receipts/*_time.txt`；读出工程时间以SSH建立至图像读出结束的整个窗口作为保守上界 **{eng['engineering_wall_upper_minutes']:.2f}分钟**，包含检查、代码和普通分析，不冒充精确专门工程耗时；2个修正循环为先触及的停止条件。

[final_validation.json](final_validation.json)核验77个唯一回合、390对象、3个资产散列、45个冻结颜色对照，以及与前轮48个回合事件审计一致。检验的是身份与记录一致性，不能升级为原因验证。冻结清单、两次修正、技术失败和全部阶段停止均保留。图像索引标实际图片步数；图片不提供精确接触时刻。复现见 [REPRODUCE.md](../REPRODUCE.md)。

最终判定：**原因未判定，测量边界已缩小；本轮按门槛结束，不推荐新算法主线。**
"""
(R/'results/RESULTS.md').write_text(report)
reproduce="""# PC-CAUSE-04 CPU复核与测量自检复现

已执行版本和冻结SHA见 receipts/、results/identities.json。使用服务器原环境：
/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011/env/bin/python
官方代码6b5e482fe479f647d4ca3790fac70c42f4940ab2、EMA H5/T100、DLP6C/正常归一化文件本轮SHA重新核验；未运行策略/训练。

## 只读复核
在新的输出目录复制本轮.py，保持 common.py OLD/SCALE/PRIOR 指向只读旧路径：
index_freeze.py -> p1.py/p2.py/p3.py（互不依赖可并行） -> readout_audit.py -> decide.py -> figures.py -> finalize.py。
完整复现生成的新回执有新时间戳，不能覆盖本轮冻结清单和结果，也不能倒签预注册。
p1图像读出另注册 exploratory proxy，不使原P2对照自动合格。
finalize.py检查前轮48回合事件、45场景颜色和所有阶段停止是否一致。

## 夹具运行（本轮已停止，不由复现文档自动授权追加运行）
设置 OLD/ROUND、原env/bin到PATH和env/lib到LD_LIBRARY_PATH，PYTHONDONTWRITEBYTECODE=1，
TORCH_EXTENSIONS_DIR使用OLD/torch_extensions。运行 contact_probe.py OUT gpu 或 cpu，必须由guard.py监督并选当前最低负载GPU。
STAGE5_GPU_INDEX由guard设定。sim_physx使用原yaml，接触采集从0改1；cpu仅改tensor pipeline。
夹具只含两环境、每环境一固定桌/立方体，12×.01667s；无机器人/相机/模型，默认夹具材料在GPU/CPU一致，不能宣称完整原任务材料/执行路径已验证。
正例table/cube、负例高处cube；保存双方body/env身份、localPos、normal、lambda原字段和时间，不将lambda冒充已标定力。
G-C原GPU失败、CPU局部能力成功，完整版可比性没有通过。不要把非抛异常空数组当API有效性。

## 本轮真实启动
三个guard运行目录记录完整cmd/GPU/PID/开始结束与日志；首次PATH失败后一次修正、一次CPU管线能力检查。
Guard only owns its new process group; never scans/kills unrelated experiment groups.
GPU账含全部child墙钟（CPU初始化也计），8GPU全墙钟+2s/job为额外上界。
2GPUh/12GiB总硬限，guard11264MiB安全停机，.5s采样精度限制明确。
没有新完整补测/验证场景或验证种子；场景/策略/批量/槽位在历史索引，缺少独立场景seed字段标未知，不编造。
大文件paths/SHA见raw_inventory.json；轻量归档不包含旧NPZ、完整日志、模型或数据。
"""
(R/'REPRODUCE.md').write_text(reproduce)
# Preserve exact approved design, then annotate registered card without changing design rules.
card=R.parents[0]/'pc_cause04_card_20261012/EXPERIMENT_CARD.md'
requested=R/'REQUESTED_CARD.md'
if not requested.exists():requested.write_bytes(card.read_bytes())
assert sha(requested)=='d341ab6f14b3f872aa097d4330e79ac86b76fba72210f3967e160698f38ea072'
s=card.read_text();s=s.replace('**状态：本轮仅编写并归档，所有新分析、补测和区分性实验均未执行。**','**状态：v1.1已执行第一阶段与测量自检；G1、正常接近负对照及原后端G-C未通过，按门槛停止；新增完整补测0/6、验证0/12。**',1)
s=s.replace('所有数值仅引用已归档结果；本卡新任务仍处于设计阶段。','设计部分数值引用前轮归档；本卡现已执行第一阶段，实际停止结论见第12节。')
s+='\n\n## 12 实际执行附录（2026-10-12 Asia/Taipei）\n\n本轮沿用登记分支，起点3fd512411f878d514dc62970624c32ddd05aa111。统一发现索引77个完整历史回合、390对象；补测及验证均未启动。原GPU管线接触正例净力有效但接触对缺失；CPU张量管线局部夹具成功，不等于完整版可比性通过。两次修正循环后停止。P2三份几何候选对照均未通过冻结正常确认；G1没有具备有效读出的可区分方案。候选均未取得资格，单机制选择为空，原因未判定。\n\n[实际报告](../pc_cause04_run_20261012/results/RESULTS.md)、[原因矩阵](../pc_cause04_run_20261012/results/cause_evidence_matrix.json)、[资源账](../pc_cause04_run_20261012/results/resource_summary.json)、[阶段状态](../pc_cause04_run_20261012/results/stage_status.json)。原始批准设计的准确副本保存在运行目录REQUESTED_CARD.md；本附录不改变冻结阈值/预算，归档提交号以Git和本地ARCHIVE_STATUS.md回执为准。\n'
card.write_text(s);(R/'EXPERIMENT_CARD.md').write_text(s)
print('FINAL_VALIDATION_OK',len(prior),'prior events',flush=True)

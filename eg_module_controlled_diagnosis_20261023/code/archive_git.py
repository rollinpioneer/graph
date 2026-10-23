"""Archive the completed E0-E4 delivery without altering experiment outputs."""
from pathlib import Path
import shutil,hashlib,json,datetime
W=Path(__file__).resolve().parents[2]
O=Path(r'C:\Users\jackx\Documents\Codex\2026-10-21\ssh-aita-xushijie-172-19-164\outputs\eg_module_controlled_diagnosis_20261023')
R=W/'work/graph_archive/eg_module_controlled_diagnosis_20261023'
assert not R.exists(),R
shutil.copytree(O,R)
scripts=W/'work/eg_module'
for p in scripts.glob('*.py'):shutil.copy2(p,R/'code'/p.name)
(R/'code/dependencies').mkdir()
shutil.copy2(W/'work/eg_pose/evaluate.py',R/'code/dependencies/previous_direct_evaluate.py')
(R/'logs').mkdir()
for n in [145,148,151,154,157,161,163,166,168,169,170,171,173]:
 matches=list((W/'work/ssh_queue').glob(f'{n:03d}-*.result.json'));assert len(matches)==1
 shutil.copy2(matches[0],R/'logs'/matches[0].name)
(R/'.gitattributes').write_text('* -text\n',encoding='utf-8')
readme='''# EG 模块级受控机制诊断（2026-10-23）

本分支归档已完成的 E0–E4 零参数更新诊断：16 个训练帧、48 个 C3 帧、40,489 次新增解析评价。未运行 E3-2/E5/E6，未访问 C4/C5，未修改 EG/G 基准代码或权重。

## 结果入口

- [中文完整报告](mechanism_report.html)（下载后用浏览器打开）、[结论正文](narrative.html)、[汇总图](mechanism_summary.png)。
- [统计与场景区间](statistics.json)、[逐候选干预](intervention_results.csv)、[姿态/分数四组合](counterfactual_composition.json)、[AD 网格](local_AD_grid.csv)。
- [梯度路径](gradient_routes.csv)、[梯度夹角](gradient_cosines.csv)、[损失台账](loss_population.csv)、[零质量宽度审计](width_target_audit.json)。
- [冻结协议](protocol.json)、[重放凭据](baseline_replay.json)、[执行凭据](execution_receipt.json)、[零训练凭据](training_receipts.json)、[源码映射](source_map.md)。

方法开发结论为 **HOLD**。姿态梯度并未与质量特征分支断开，关闭注意力没有稳定的完整流程收益。137 个失败查询中，59 个在 Top-2×2 中存在可行组合，14 个仅更低排名格可达，64 个在固定位置、方向、宽度下全 48 格失败。零质量样本占有效样本 7.26%，贡献本次平均宽度损失的 64.9%；这是下一步 E6 掩码对照的具体假设，不是已证明的因果机制。

## 文件范围与完整性

原交付目录的全部 339 个文件均按原始字节保留，包括 `light_results.zip` 及解包的 `data/`。`data/cache/` 是 64 帧已导出的轻量姿态/logit/特征缓存；`data/evaluation/` 为 64 帧逐姿态结果及哈希凭据，`data/gradient/` 和 `data/receipts/` 保留逐帧台账。新增归档项包括本轮全部 Python 脚本、先导失败与修正日志、旧直接解析评价器依赖快照以及归档清单。

完整内部激活缓存、原始 GraspNet 数据和模型权重仍在原服务器；它们不是本次本地交付包的一部分。完整激活缓存的原始 SHA-256 已保存在每帧 receipts 中；轻量缓存因有意删去内部数组，不能拿它的文件哈希与完整缓存直接比较。全套已交付文件和新增归档文件均列于 `ARCHIVE_MANIFEST.json`。

`protocol.json` 和 `execution_receipt.json` 中的 `allow_git_write:false` / `git_write:false` 描述诊断执行当时的权限与状态，未回写历史记录。用户随后明确要求“把所有结果和代码提交git”，本次提交由该后续授权执行。

## 代码与复核

- `code/probe.py`：冻结权重的 E0–E3 前向/梯度探针，包含 E1、定向 attention 及不变量检查；无优化器。
- `code/evaluate_module.py`：逐实际姿态的官方解析评价、E4 网格及限定的完整后处理；不使用最近离散 GT 质量替代。
- `code/analyze.py` / `code/report.py`：读取已保存结果，生成统计与报告；`code/finalize.py` 记录交付完整性。
- `code/package.py`：生成逐帧轻量缓存；`code/check_label.py` / `code/grad_summary.py`：先导定位与数值摘要；`code/dispatch.py` 仅将命令写入既有 SSH 队列，不含凭据。
- `code/dependencies/previous_direct_evaluate.py`：上一轮直接解析评价器原样快照。新脚本仍按原实验目录导入 `evaluate`。

代码保留原执行路径配置：服务器 `~/work/grasp/eg_module_controlled_diagnosis_20261023`，旧缓存 `~/work/grasp/eg_pose_quality_diagnosis_v1_20261022`，官方源码 `~/work/grasp/eg_full_repro_v1_20261015/runtime/EconomicGrasp`；本地统计脚本的 `O` 为原交付目录。迁移复核时只需配置这些根路径和既有环境，不应更改冻结候选、阈值、匹配或质量口径。

历史前向为 TF32，标签与梯度数值核验使用 FP32。重放中官方视图映射重复写入造成的差异已审计，ADWQ 保留历史匹配和标签。报告中的 Oracle 与 GT 辅助基底不代表可部署收益；解析可行率不等于机器人成功率。
'''
(R/'README.md').write_text(readme,encoding='utf-8')
source=[]
for p in sorted(O.rglob('*')):
 if p.is_file():
  rel=p.relative_to(O);raw=p.read_bytes();assert (R/rel).read_bytes()==raw,rel
  source.append(dict(path=rel.as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
allfiles=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(R.rglob('*')) if p.is_file()]
assert all(x['bytes']<100*1024**2 for x in allfiles)
manifest=dict(date='2026-10-23',scope='All files in completed E0-E4 delivery, all round Python scripts, bounded diagnostic logs, evaluator dependency',repository='rollinpioneer/graph',branch='codex/eg-module-controlled-diagnosis-20261023',authorization='Subsequent user request: 把所有结果和代码提交git',delivered_files_verified_unchanged=len(source),original_delivery=source,archive_files=allfiles,external_assets=dict(full_activation_cache='/aita_data/home/aita_xushijie/work/grasp/eg_module_controlled_diagnosis_20261023/cache',original_labels_and_checkpoint='unchanged external inputs; paths and checkpoint digest in protocol and source map'),credentials_included=False,archive_manifest_self_hash_omitted=True)
(R/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(allfiles)+1,bytes=sum(x['bytes'] for x in allfiles),original_delivery_files=len(source),path=str(R))))

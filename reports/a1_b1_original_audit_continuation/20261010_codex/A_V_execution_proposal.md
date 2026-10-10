# A-V：固定安装构建的剪枝划分验证

状态：APPROVED_EXECUTED。用户明确回复：“批准执行A-V，要尽量并行运行”。一次正式调用已完成，未重试；执行结果见 A_V/result_report.md 与 A_V/receipt.json。

本卡来自上一轮唯一主要A-V提案，不扩展为AB2、原论文端到端复现或新算法开发。原审计卡第2、9、12节要求新增实验分别批准；本次A-V已获得上述明确批准。

## 唯一问题与身份

在服务器已经安装的固定WLPlan扩展（SHA256 0089e22796904c5ee73d0d5c863e3ce4e82b54561393e9e86b43d37e3932587f）上，同一Train96输入的a-m剪枝是否改变训练状态的特征向量划分？加载冻结W1和同库独立重建a-m是否一致？

安装METADATA标记2.2.0，direct_url.json指向/home/xushijie3/ext/goose/ext/wlplan，源码HEAD为9d9fc99165135bfe8e2817145888ede162e4a97a，受控文件无改动。这提高了安装来源可追溯性，但不能证明扩展由当前源码逐字节构建；不将补充包v2.1.0等同本机扩展。本卡若执行，身份仅限固定构建库函数重运行，不称2024作者原生系统复现。

## 输入、运行与预算

- CP冻结提交：cc7d79d6fcb053bf0c16033e765a2e808de2aaac。
- Train96已有6399个状态，用于两种重建的词表来源；不增加训练题。
- P1/W1已有836个FEATURE_EQUAL唯一对；非零池9039对，以固定SHA顺序取836个对照。
- 原24个探索候选全部保留；9个非训练成员状态来自既有b_atoms文件，涉及既有Train/Struct任务，仅嵌入，不加入词表收集。不得解释为独立确认集。
- 3条管线：冻结W1嵌入、独立a-m收集/嵌入、none收集/嵌入；各2次。均ILG、2轮、set。首先检验冻结/重建训练划分，不一致即停止。
- 0学习拟合、0训练更新、0神经前向、0学习评分、0搜索/回放/计划验证、0精确查询、0 GPU；符号词表collect共4次，6个特征管线运行。
- 最多2个并行worker，每个单线程；父进程负责监控和结果收集。先并行冻结0/a-m0，再并行none0/none1，最后并行冻结1/a-m1。每个worker地址空间上限3GiB，监测父/worker合计RSS上限8GiB；2小时墙钟、8核小时总上限。脚本另设每进程7200 CPU秒限制。MaxSAT调用已静态确认走进程内Python RC2，执行时仍监测自有进程资源。
- 输出最多200MiB；每份矩阵写出后检查，超限停止。失败保留新目录与日志，不自动重试或扩展。

脚本不读取权重进行打分，不设置新权重，不修改原模型。仅调用官方安装库的collect/embed，完整矩阵用于技术一致性核验。

## 拟执行命令

将本地A_V_draft.py按哈希复制为独立目录下具有时间戳的脚本；不加入冻结仓库、不提交或推送。下列命令为执行模板，实际绝对路径与回执一并保存：

```bash
P=/home/xushijie3/work/cp_disr_a_v
# 建立P并保存获准脚本、授权文字和登记记录；若既有路径冲突则换新时间戳。
LD_LIBRARY_PATH=/home/xushijie3/.uv_python/cpython-3.10.19-linux-x86_64-gnu/lib \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
~/envs/goose/bin/python "$SCRIPT"
# 上面仅stdlib预检；预检输入或构建不符即停止。
RR="$P/A_V_$(date -u +%Y%m%dT%H%M%SZ)"
LD_LIBRARY_PATH=/home/xushijie3/.uv_python/cpython-3.10.19-linux-x86_64-gnu/lib \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
timeout --signal=TERM 7200s ~/envs/goose/bin/python "$SCRIPT" \
  --execute --output "$RR"
```

命令执行前还须绑定Python解释器、Python wrapper与依赖哈希、脚本哈希和全部输入清单；批准不豁免预检与身份冲突停止规则。未改源码/未安装依赖；若需要这些操作，本卡停止并报告。

## 指标、判据和输出

保存6份完整矩阵及其SHA；比较逐位重复一致、冻结/重建矩阵及训练划分；比较a-m与none的完整训练划分，以及a-m组被none拆开的数量、反向拆分数量。报告836目标、836对照及24探索候选的全部分母，不以候选恢复率外推母集。

- 冻结与独立a-m不一致或重复不一致：INCONCLUSIVE，不进行后续拟合或搜索。
- 同一固定构建、相同输入中a-m划分变粗：可记录固定构建的现象重现；不能直接判原论文固有限制，也不能证明搜索后果。
- 未重现目标差异：撤回/收窄原观察；说明已有完整矩阵与旧表的具体冲突。
- 两管线训练划分一致但域外候选不同：只报告域外差异，不能称训练划分丢失。

输出input_identity.json、matrix_*.npz、pairs.csv、candidates.json、receipt.json。原仓库/模型/结果输入哈希与状态跑前跑后比较。不得自动升级为AB2、none读出拟合、多重集实验或新表示。

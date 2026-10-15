# PathGraph-SARM 阶段 7（Reward-Only 分支）Agent 详细操作命令 V1.0

---

# PathGraph-SARM 阶段 7（Reward-Only 分支）Agent 通用执行规范

## 0. 阶段入口与研究边界

阶段 6 定向修正 R1 已完成，冻结决策为：

```text
G3-R1 = NARROW_TO_REWARD_ONLY
```

阶段 7 可以启动，但必须切换为：

```text
STAGE7_MODE = REWARD_ONLY
```

本阶段不再新增 policy seed，不再训练行为克隆策略，不再调 Stage 5 的主奖励参数。阶段 6 的 RA-BC 结果仅作为次级、混合证据保留。

本阶段主线：

```text
冻结可信输入
→ 补齐结构性核心消融
→ 验证历史长度与图粒度
→ 测试数据覆盖和图复杂度扩展
→ 测试结构 OOD 与不确定性
→ 条件探索自动图
→ 冻结 reward-only 贡献边界
```

必须保持的主张边界：

```text
主贡献：
从多路径、失败与恢复示范中学习 graph-structured dense reward，
用 node/edge belief、remaining cost 和 within-node progress
正确表达 alternative order、recovery 与 revisitation。

次级证据：
冻结 PathGraph 权重在六个 policy seed 上总体成功率提高约 0.067，
recovery 提高约 0.167，但跨 seed 严格改善不足，不能作为稳定主结论。

不再主张：
稳定、普适地提升 RA-BC；
非零 uncertainty LCB 带来下游提升；
非零 repeated-edge penalty 带来下游提升；
自动 graph discovery 已成为核心贡献；
在线规划能力。
```

---

## 1. 阶段 7 小阶段与每轮 ZIP

```text
7.1  Reward-only 输入冻结与可移植证据索引
     stage7_1_reward_only_input_freeze.zip

7.2  核心结构消融与机制归因
     stage7_2_core_reward_ablations.zip

7.3  历史长度与图粒度实验
     stage7_3_history_and_granularity.zip

7.4  数据覆盖与图复杂度扩展
     stage7_4_scaling_and_coverage.zip

7.5  结构 OOD、扰动鲁棒性与不确定性
     stage7_5_ood_and_uncertainty.zip

7.6  自动图发现探索（条件执行）
     stage7_6_auto_graph_exploration.zip

7.7  G4 贡献边界冻结与 Stage 8 交接
     stage7_7_g4_freeze.zip

阶段总包：
     stage7_complete.zip
```

每个小阶段结束立即生成 ZIP。不得等到整个阶段结束后才交付。

---

## 2. 统一目录与环境变量

从仓库根目录执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"

export STAGE2_ROOT="${STAGE2_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage2}"
export STAGE2_M1="${STAGE2_M1:-$STAGE2_ROOT/m1_freeze_v1}"

export STAGE3_ROOT="${STAGE3_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage3}"
export STAGE3_INPUT="${STAGE3_INPUT:-$STAGE3_ROOT/input_adapter_v1}"
export STAGE3_DIAG="${STAGE3_DIAG:-$STAGE3_ROOT/diagnostic_suite_v1}"
export STAGE3_M2="${STAGE3_M2:-$STAGE3_ROOT/m2_freeze_v1}"

export STAGE4_ROOT="${STAGE4_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage4}"
export STAGE4_SUPERVISION="${STAGE4_SUPERVISION:-$STAGE4_ROOT/supervision}"
export STAGE4_MODELS="${STAGE4_MODELS:-$STAGE4_ROOT/model_candidates_v1}"
export STAGE4_TOOLS="${STAGE4_TOOLS:-$REPO_ROOT/tools/stage4}"

export STAGE5_ROOT="${STAGE5_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage5}"
export STAGE5_PRED="${STAGE5_PRED:-$STAGE5_ROOT/real_predictions_v1}"
export STAGE5_REWARD="${STAGE5_REWARD:-$STAGE5_ROOT/reward_v1}"
export STAGE5_ROUNDS="${STAGE5_ROUNDS:-$STAGE5_ROOT/rounds}"
export STAGE5_TOOLS="${STAGE5_TOOLS:-$REPO_ROOT/tools/stage5}"

export STAGE6_ROOT="${STAGE6_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage6}"
export STAGE6_PERSISTENT="${STAGE6_PERSISTENT:-$STAGE6_ROOT/stage6_inputs/reward_v1_persistent}"
export STAGE6_M4="${STAGE6_M4:-$STAGE6_ROOT/m4_policy_results_v1}"

export STAGE6R1_ROOT="${STAGE6R1_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage6_refine1}"
export STAGE6R1_M4="${STAGE6R1_M4:-$STAGE6R1_ROOT/m4_refine1_results_v1}"

export STAGE7_ROOT="${STAGE7_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage7_reward_only}"
export STAGE7_INPUTS="$STAGE7_ROOT/inputs_v1"
export STAGE7_ROUNDS="$STAGE7_ROOT/rounds"
export STAGE7_MODELS="$STAGE7_ROOT/model_variants_v1"
export STAGE7_RESULTS="$STAGE7_ROOT/reward_evidence_v1"
export STAGE7_G4="$STAGE7_ROOT/g4_reward_only_v1"
export STAGE7_TOOLS="${STAGE7_TOOLS:-$REPO_ROOT/tools/stage7}"
export STAGE7_DOWNLOADS="${STAGE7_DOWNLOADS:-$REPO_ROOT/downloads/stage7}"

export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-8000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p \
  "$STAGE7_ROOT" \
  "$STAGE7_INPUTS" \
  "$STAGE7_ROUNDS" \
  "$STAGE7_MODELS" \
  "$STAGE7_RESULTS" \
  "$STAGE7_G4" \
  "$STAGE7_TOOLS" \
  "$STAGE7_DOWNLOADS"
```

保存为：

```text
artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
```

每个小阶段开头执行：

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"
```

---

## 3. 输入路径不存在时的最短处理

不要做全仓库审计。只按以下顺序定位：

```bash
for p in \
  "$STAGE2_M1" \
  "$STAGE3_INPUT" \
  "$STAGE3_DIAG" \
  "$STAGE4_SUPERVISION" \
  "$STAGE5_PRED" \
  "$STAGE5_REWARD" \
  "$STAGE6_PERSISTENT" \
  "$STAGE6R1_M4"
do
  if [ ! -e "$p" ]; then
    echo "MISSING_REQUIRED_INPUT: $p"
  fi
done
```

若只是目录名称差一个版本后缀，使用：

```bash
find "$REPO_ROOT/artifacts/pathgraph_sarm" \
  -maxdepth 4 \
  -type d \
  \( -name 'm1_freeze_v1' \
     -o -name 'diagnostic_suite_v1' \
     -o -name 'supervision' \
     -o -name 'real_predictions_v1' \
     -o -name 'reward_v1' \
     -o -name 'reward_v1_persistent' \
     -o -name 'm4_refine1_results_v1' \) \
  -print
```

只修正环境变量，不复制或重建数据，除非原目录确实已丢失。

---

## 4. GPU 必须提权查看

每个包含训练或模型推理的小阶段，首先执行：

```bash
mkdir -p "$ROUND_DIR/gpu"

if sudo -n nvidia-smi > "$ROUND_DIR/gpu/nvidia_smi_sudo.txt" 2>&1; then
  echo "GPU_QUERY_MODE=sudo_noninteractive" \
    | tee "$ROUND_DIR/gpu/gpu_query_mode.txt"
else
  echo "sudo -n nvidia-smi failed" \
    | tee "$ROUND_DIR/gpu/gpu_query_mode.txt"

  if [ -t 0 ]; then
    sudo nvidia-smi \
      | tee "$ROUND_DIR/gpu/nvidia_smi_sudo_interactive.txt"
    echo "GPU_QUERY_MODE=sudo_interactive" \
      | tee -a "$ROUND_DIR/gpu/gpu_query_mode.txt"
  else
    nvidia-smi \
      | tee "$ROUND_DIR/gpu/nvidia_smi_direct.txt"
    echo "GPU_QUERY_MODE=direct_fallback_noninteractive" \
      | tee -a "$ROUND_DIR/gpu/gpu_query_mode.txt"
  fi
fi

nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader \
  | tee "$ROUND_DIR/gpu/gpu_inventory.csv"

"$PYTHON_BIN" - <<'PY'
import torch
print("torch_version =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("cuda_device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

普通权限下无法查询时，不能直接判断“没有 GPU”。

---

## 5. 多 GPU 默认并行

只要实验互不阻塞且输出目录独立，默认并行。

优先并行维度：

```text
model seed
history length
graph granularity
data fraction
order-holdout direction
perturbation setting
auto-graph discovery seed
```

默认调度方式：

```text
一个独立实验 job 占一张 GPU
```

优先复用已有调度器：

```bash
"$PYTHON_BIN" "$STAGE4_TOOLS/launch_parallel.py" \
  --jobs <job_table_or_jsonl> \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --status-output <job_status.tsv>
```

若 Stage 4 调度器接口不同，先执行：

```bash
"$PYTHON_BIN" "$STAGE4_TOOLS/launch_parallel.py" --help
```

然后只适配参数，不重写训练流程。

若 Stage 4 调度器无法满足逐 GPU job 调度，复制 Stage 6 已验证的：

```text
tools/stage6/launch_gpu_job_matrix.py
```

到：

```text
tools/stage7/launch_gpu_job_matrix.py
```

只修改路径解析。

---

## 6. 不允许继续做的实验

阶段 7 禁止：

```text
新增 policy seed
重新训练 BC / RA-BC
根据 Stage 6 test 调 reward 参数
用 Stage 6 test 选择 reward variant
改变 Stage 5 主方法名称或冻结配置
把 post-hoc eta/beta probe 替换成主模型
把 scripted_oracle 当作真实机器人泛化证据
把自动图结果包装成主贡献
```

阶段 7 可以：

```text
训练 reward-model 变体
重新计算真实 reward 指标
运行结构消融
运行数据覆盖和历史长度实验
构建明确标注为 controlled/symbolic 的 stress benchmark
运行结构 OOD 与表示扰动
探索自动图作为扩展
```

---

## 7. 统一统计单位与来源分层

默认统计单位：

```text
content_group_id
```

所有表必须带：

```text
provenance
```

允许值：

```text
real_or_environment_rollout
scripted_oracle
controlled_symbolic_stress
derived_counterfactual
```

不同来源不得无标记混合成一个“真实测试”均值。

主表优先顺序：

```text
真实/环境数据
> 已冻结 GT 与真实模型预测
> scripted_oracle 机制证据
> controlled symbolic stress
```

---

## 8. 主奖励固定不变

主方法名称：

```text
pathgraph_reward_v1_locked
```

固定参数：

```yaml
lambda: 0.5
eta: 0.0
beta: 0.0
confidence: 0.7
reward_clip: 1.5
node_confidence_min: 0.7
edge_confidence_min: 0.7
repeat_window_steps: 64
recovery_debt_cap: true
uncertainty_lcb: true
```

注意：

```text
eta = 0
beta = 0
```

因此：

- repeated-edge penalty 的非零效果只能作为 post-hoc probe；
- uncertainty LCB 的非零效果只能作为 post-hoc probe；
- 两者不能写成已被主实验验证的组成贡献。

---

## 9. 每轮 ZIP 规则

每轮 ZIP 至少包含：

```text
run_manifest.md
configs/
commands/
gpu/
logs/
metrics/
tables/
figures/
reports/
manifests/
checksums/
```

默认不打包：

```text
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
原始 episode
完整 rollout
完整逐帧 prediction
视频
大型 npy/npz
缓存
超过 200 MB 的其他文件
```

未打包大文件写入：

```text
manifests/checkpoint_manifest.tsv
manifests/large_file_manifest.tsv
```

每轮打包：

```bash
"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

若 Stage 5 的 `package_round.py` 不支持目录排除，复制到 `tools/stage7/package_round.py`，加入 checkpoint 和大文件排除规则，不改其他逻辑。

---

## 10. 每轮运行记录

`run_manifest.md` 至少记录：

```text
stage
round
mode = reward_only
start_time
end_time
repo_root
git_commit
python
CUDA
GPU inventory
input lock SHA256
actual commands
job count
success count
failed count
statistics_unit
provenance
test used for selection
large files omitted
output ZIP
ZIP SHA256
next action
```

每个小阶段必须保存：

```text
commands/executed_commands.sh
logs/
metrics/*gate*.json
reports/*summary*.md
checksums/
```

---

## 11. 阶段 7 最终出口

```text
GO_STAGE8_REWARD_ONLY
REFINE_STAGE7_CORE_ONLY
STOP_PATHGRAPH
```

含义：

```text
GO_STAGE8_REWARD_ONLY
  图奖励核心结构、消融和边界证据成立，
  进入最终复现、写作与材料冻结。

REFINE_STAGE7_CORE_ONLY
  仅允许修复一个核心脚本、缺失输入或执行失败；
  不允许调方法、调 test 或新增 policy 实验。

STOP_PATHGRAPH
  真实重算或核心结构消融无法支持图奖励主张。
```

不得再次回到 Stage 6 扩 seed。

---

# 阶段 7.1：Reward-Only 输入冻结与可移植证据索引

## 一、总体上要干什么

本小阶段把阶段 2–6R1 中与 reward-only 主张有关的输入整理为一个只读索引，修正两个不影响数值结论、但会影响可移植复核的报告问题：

1. `g3_refine1_decision.json` 中 `reward_retuned_after_test=true` 是“无 test 后调参检查已通过”的布尔检查名歧义；真实 evidence 字段为 `reward_retuned_after_test=false`。
2. `M4_REFINE1_SHA256SUMS.txt` 包含未打包的 `combined_six_seed_rollouts.csv`，因此离开原机器后不能直接对整份清单执行 `sha256sum -c`。

禁止改写冻结源文件。只在 Stage 7 创建：

```text
stage7_input_index.yaml
claim_scope_lock.json
portable_files_sha256.txt
omitted_large_files.tsv
stage6r1_reporting_patch_note.md
```

本轮状态：

```text
REWARD_ONLY_INPUTS_LOCKED
```

本轮 ZIP：

```text
stage7_1_reward_only_input_freeze.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7_1_reward_only_input_freeze"
export ROUND_DIR="$STAGE7_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE7_INPUTS"/{configs,locks,manifests,reports}
```

保存命令：

```bash
cat > "$ROUND_DIR/commands/executed_commands.sh" <<'EOF'
# 在本文件末尾按实际执行顺序追加命令。
EOF
```

---

## 三、验证 Stage 6R1 决策与核心数值

执行：

```bash
test -f "$STAGE6R1_M4/metrics/g3_refine1_decision.json"
test -f "$STAGE6R1_M4/metrics/g3_refine1_evidence.json"
test -f "$STAGE6R1_M4/metrics/new_seed_replication.json"
test -f "$STAGE6R1_M4/locks/refine1_checkpoint_selection_lock.json"
test -f "$STAGE6R1_M4/locks/refine1_policy_evaluation_lock.json"
```

创建：

```text
tools/stage7/verify_stage6r1_terminal_state.py
```

脚本必须读取 JSON，禁止从 Markdown 文本猜测。

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/verify_stage6r1_terminal_state.py" \
  --decision "$STAGE6R1_M4/metrics/g3_refine1_decision.json" \
  --evidence "$STAGE6R1_M4/metrics/g3_refine1_evidence.json" \
  --replication "$STAGE6R1_M4/metrics/new_seed_replication.json" \
  --output "$ROUND_DIR/metrics/stage6r1_terminal_verification.json"
```

脚本检查：

```text
decision == NARROW_TO_REWARD_ONLY
combined_seed_count == 6
new_seed_improved_count == 1
new_seed_ceiling_tie_count == 1
new_seed_degraded_count == 0
combined_graph_task_success_gain ≈ 0.0666667
combined_recovery_success_gain ≈ 0.1666667
combined_fixed_order_drop == 0
reward_retuned_after_test == false
paired_evaluation == true
```

允许浮点误差：

```text
abs(actual - expected) <= 1e-6
```

---

## 四、验证 reward-only 主输入

检查：

```bash
test -f "$STAGE2_M1/selected_graph_tasks_v1.yaml"
test -d "$STAGE2_M1/graph_specs_v1"
test -d "$STAGE2_M1/gt_v1"

test -f "$STAGE3_DIAG/FROZEN.md"
test -f "$STAGE3_DIAG/DIAGNOSTIC_SUITE_SHA256SUMS.txt"

test -f "$STAGE4_SUPERVISION/FROZEN.md"
test -f "$STAGE4_SUPERVISION/SUPERVISION_SHA256SUMS.txt"

test -f "$STAGE5_PRED/REAL_PREDICTIONS_SHA256SUMS.txt"
test -f "$STAGE5_PRED/tables/ensemble_test_predictions.jsonl.gz"
test -f "$STAGE5_PRED/tables/ensemble_val_predictions.jsonl.gz"
test -f "$STAGE5_PRED/tables/ensemble_stage3_diagnostic_predictions.jsonl.gz"

test -f "$STAGE5_REWARD/FROZEN.md"
test -f "$STAGE5_REWARD/configs/reward_selection_lock.json"
test -f "$STAGE5_REWARD/configs/reward_config_v1.yaml"
test -f "$STAGE5_REWARD/code/reward_engine.py"
test -f "$STAGE5_REWARD/metrics/frozen_reward_metrics.json"

test -f "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json"
test -f "$STAGE6_PERSISTENT/PERSISTED_INPUTS_SHA256SUMS.txt"
```

注意：

```text
Stage 7 的真实模型指标来源优先使用 Stage 5 real_predictions_v1；
不得使用 Stage 4 中可能由占位流程产生的 frozen_test_metrics.json
作为新的主证据。
```

---

## 五、验证持久化 reward-model checkpoint

执行：

```bash
"$PYTHON_BIN" - <<'PY'
import json, hashlib, os
bundle = os.path.join(
    os.environ["STAGE6_PERSISTENT"],
    "configs/model_bundle_persistent.json"
)
d = json.load(open(bundle))
assert len(d["checkpoints"]) == 3
rows = []
for item in d["checkpoints"]:
    p = item["path"]
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    got = h.hexdigest()
    assert got == item["sha256"], (p, got, item["sha256"])
    rows.append((item["seed"], p, got, os.path.getsize(p)))
print("PERSISTENT_REWARD_CHECKPOINTS_OK")
for row in rows:
    print(*row, sep="\t")
PY
```

将输出保存到：

```text
$ROUND_DIR/manifests/reward_checkpoint_manifest.tsv
```

推荐执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_checkpoint_manifest.py" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --output "$ROUND_DIR/manifests/reward_checkpoint_manifest.tsv"
```

checkpoint 本体不打包。

---

## 六、创建 Stage 7 输入索引

创建：

```text
tools/stage7/build_stage7_input_index.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_stage7_input_index.py" \
  --m1 "$STAGE2_M1" \
  --stage3-input "$STAGE3_INPUT" \
  --diagnostic-suite "$STAGE3_DIAG" \
  --supervision "$STAGE4_SUPERVISION" \
  --real-predictions "$STAGE5_PRED" \
  --reward-v1 "$STAGE5_REWARD" \
  --persistent-model "$STAGE6_PERSISTENT" \
  --stage6-policy-evidence "$STAGE6_M4" \
  --stage6r1-evidence "$STAGE6R1_M4" \
  --output "$STAGE7_INPUTS/configs/stage7_input_index.yaml" \
  --hash-output "$STAGE7_INPUTS/manifests/stage7_input_hashes.tsv"
```

`stage7_input_index.yaml` 至少包含：

```yaml
statistics_unit: content_group_id
stage7_mode: reward_only

graph_specs:
  root: ...
  version: ...

ground_truth:
  root: ...
  splits: ...

diagnostic_suite:
  root: ...
  frozen_sha256: ...

supervision:
  root: ...
  forbidden_features:
    - outcome
    - success
    - scenario
    - controller_source
    - episode_id
    - content_group_id

real_predictions:
  val: ...
  test: ...
  stage3_diagnostic: ...

reward:
  config: ...
  selection_lock: ...
  engine: ...

model_bundle:
  persistent_manifest: ...
  checkpoint_count: 3

policy_evidence:
  status: secondary_mixed
  g3_r1: NARROW_TO_REWARD_ONLY
```

---

## 七、冻结主张范围

创建：

```text
$STAGE7_INPUTS/locks/claim_scope_lock.json
```

内容：

```json
{
  "locked_before_stage7_experiments": true,
  "stage7_mode": "reward_only",
  "primary_claims": [
    "graph-structured dense reward from multi-path demonstrations",
    "alternative-order-aware reward",
    "failure/recovery-aware reward",
    "remaining-cost plus within-node progress",
    "recovery-loop safety through debt accounting"
  ],
  "secondary_claims": [
    "mixed downstream RA-BC evidence"
  ],
  "prohibited_primary_claims": [
    "stable policy improvement across seeds",
    "nonzero uncertainty LCB policy gain",
    "nonzero repeated-edge penalty policy gain",
    "automatic graph discovery as the main contribution",
    "online planning"
  ],
  "no_more_policy_training": true,
  "no_post_test_main_reward_retuning": true
}
```

计算 SHA：

```bash
sha256sum "$STAGE7_INPUTS/locks/claim_scope_lock.json" \
  | tee "$STAGE7_INPUTS/locks/claim_scope_lock.sha256"
```

---

## 八、创建报告修正说明

创建：

```text
$STAGE7_INPUTS/reports/stage6r1_reporting_patch_note.md
```

写明：

```text
1. 原 evidence 中 reward_retuned_after_test=false。
2. 原 decision checks 中名为 reward_retuned_after_test 的布尔值 true，
   实际表示“无 test 后调参检查通过”；Stage 7 派生报告统一改名为
   no_post_test_reward_retune=true。
3. 原 M4_REFINE1_SHA256SUMS.txt 包含一个按大文件规则省略的 rollout 表；
   Stage 7 不修改源清单，而是另建 portable manifest。
4. 上述处理不改变任何实验结果、门槛或决策。
```

---

## 九、建立可移植 checksum

创建：

```text
tools/stage7/build_portable_manifest.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_portable_manifest.py" \
  --input-index "$STAGE7_INPUTS/configs/stage7_input_index.yaml" \
  --stage6r1-root "$STAGE6R1_M4" \
  --portable-output "$STAGE7_INPUTS/manifests/portable_files_sha256.txt" \
  --omitted-output "$STAGE7_INPUTS/manifests/omitted_large_files.tsv"
```

规则：

```text
portable_files_sha256.txt
  只包含本地实际存在、预计进入 Stage 7 ZIP 的小文件。

omitted_large_files.tsv
  记录 checkpoint、完整 prediction、完整 rollout 等未打包文件的：
  path
  size_bytes
  sha256
  reason
  required_for_full_recompute
```

验证：

```bash
(
  cd "$REPO_ROOT"
  sha256sum -c "$STAGE7_INPUTS/manifests/portable_files_sha256.txt"
) | tee "$ROUND_DIR/checksums/portable_manifest_check.txt"
```

---

## 十、本轮 Gate

实现：

```text
tools/stage7/decide_stage7_input_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/decide_stage7_input_gate.py" \
  --input-index "$STAGE7_INPUTS/configs/stage7_input_index.yaml" \
  --input-hashes "$STAGE7_INPUTS/manifests/stage7_input_hashes.tsv" \
  --claim-lock "$STAGE7_INPUTS/locks/claim_scope_lock.json" \
  --terminal-verification "$ROUND_DIR/metrics/stage6r1_terminal_verification.json" \
  --checkpoint-manifest "$ROUND_DIR/manifests/reward_checkpoint_manifest.tsv" \
  --portable-check "$ROUND_DIR/checksums/portable_manifest_check.txt" \
  --output "$ROUND_DIR/metrics/stage7_input_gate.json" \
  --report "$ROUND_DIR/reports/stage7_input_summary.md"
```

必须满足：

```text
G3-R1 == NARROW_TO_REWARD_ONLY
claim_scope_locked == true
no_more_policy_training == true
reward selection lock 存在
3/3 persistent checkpoints hash match
real val/test/diagnostic predictions 存在
statistics_unit == content_group_id
portable manifest 可验证
```

允许状态：

```text
REWARD_ONLY_INPUTS_LOCKED
REPAIR_INPUT_PATHS
MISSING_PERSISTENT_REWARD_CHECKPOINT
```

只有 `REWARD_ONLY_INPUTS_LOCKED` 才进入 7.2。

---

## 十一、本轮 ZIP

复制轻量文件：

```bash
cp "$STAGE7_INPUTS/configs/stage7_input_index.yaml" \
  "$ROUND_DIR/configs/"
cp "$STAGE7_INPUTS/locks/claim_scope_lock.json" \
  "$ROUND_DIR/configs/"
cp "$STAGE7_INPUTS/reports/stage6r1_reporting_patch_note.md" \
  "$ROUND_DIR/reports/"
cp "$STAGE7_INPUTS/manifests/"* \
  "$ROUND_DIR/manifests/"
```

生成：

```bash
export ZIP_NAME="stage7_1_reward_only_input_freeze.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 7.1 状态：REWARD_ONLY_INPUTS_LOCKED
Stage 7 mode：reward_only
persistent reward checkpoints：3 / 3
real prediction suites：val / test / diagnostic
policy training disabled：true
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 7.2
```

**核心点：不重做阶段 6，只把 reward-only 输入、主张范围和可移植校验方式一次性冻结。**

---

# 阶段 7.2：核心结构消融与机制归因

## 一、总体上要干什么

本小阶段补齐 Stage 5 尚未充分回答的两个关键结构消融：

```text
alternative edge 是否必要
recovery edge / debt accounting 是否必要
```

同时重新整理已有的：

```text
no_phi
cost_only
no_recovery_cap
eta probe
beta probe
```

所有实验基于已经缓存的真实 ensemble predictions、Stage 3 diagnostic predictions 和 Oracle trace bank。除非输入缺失，不重新训练模型。

本小阶段不改变主奖励配置，所有变体必须标记为：

```text
ablation_or_probe
```

本轮状态：

```text
CORE_REWARD_ABLATIONS_COMPLETE
```

本轮 ZIP：

```text
stage7_2_core_reward_ablations.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7_2_core_reward_ablations"
export ROUND_DIR="$STAGE7_ROUNDS/$ROUND_NAME"
export ABLATION_ROOT="$STAGE7_RESULTS/core_ablations_v1"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$ABLATION_ROOT"/{configs,jobs,metrics,tables,figures,reports}
```

入口：

```bash
grep -q 'REWARD_ONLY_INPUTS_LOCKED' \
  "$STAGE7_ROUNDS/stage7_1_reward_only_input_freeze/reports/stage7_input_summary.md"
```

---

## 三、定位 Oracle trace bank

优先路径：

```bash
export ORACLE_TRACE_ROOT="$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/oracle_traces"
export ORACLE_TRACE_MANIFEST="$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/tables/oracle_trace_manifest.csv"
```

检查：

```bash
test -f "$ORACLE_TRACE_ROOT/legal_A_then_B.jsonl"
test -f "$ORACLE_TRACE_ROOT/legal_B_then_A.jsonl"
test -f "$ORACLE_TRACE_ROOT/failure_then_recovery.jsonl"
test -f "$ORACLE_TRACE_ROOT/failure_recovery_loop_x1.jsonl"
test -f "$ORACLE_TRACE_ROOT/failure_recovery_loop_x2.jsonl"
test -f "$ORACLE_TRACE_ROOT/failure_recovery_loop_x3.jsonl"
test -f "$ORACLE_TRACE_MANIFEST"
```

若 Stage 5 目录名不同，只使用：

```bash
find "$STAGE5_ROOT" -type f \
  \( -name 'legal_A_then_B.jsonl' \
     -o -name 'oracle_trace_manifest.csv' \) \
  -print
```

定位后修改变量，不重建 trace bank。

---

## 四、冻结消融变体定义

创建：

```text
$ABLATION_ROOT/configs/ablation_matrix.yaml
```

内容：

```yaml
main:
  id: full_locked
  source_config: stage5/reward_v1/configs/reward_config_v1.yaml
  role: frozen_main

ablations:
  - id: collapse_alternative_to_A_first
    role: structural_ablation
    change:
      graph_semantics: canonical_A_first
      alternative_edges_enabled: false

  - id: collapse_alternative_to_B_first
    role: structural_ablation
    change:
      graph_semantics: canonical_B_first
      alternative_edges_enabled: false

  - id: remove_recovery_edge
    role: structural_ablation
    change:
      recovery_edge_enabled: false
      recovery_transitions_mapped_to: stagnation
      recovery_debt_repayment_enabled: false

  - id: no_recovery_debt_cap
    role: structural_ablation
    change:
      recovery_debt_cap: false

  - id: no_phi
    role: component_ablation
    change:
      lambda: 0.0

  - id: cost_only
    role: component_ablation
    change:
      lambda: 0.0
      edge_semantic_adjustments: false

probes:
  - id_prefix: eta_probe
    role: posthoc_probe_not_main
    grid:
      eta: [0.0, 0.05, 0.10, 0.25]

  - id_prefix: beta_probe
    role: posthoc_probe_not_main
    grid:
      beta: [0.0, 0.5, 1.0]
```

锁定：

```bash
sha256sum "$ABLATION_ROOT/configs/ablation_matrix.yaml" \
  | tee "$ABLATION_ROOT/configs/ablation_matrix.sha256"
```

---

## 五、实现结构变体适配器

创建：

```text
tools/stage7/build_reward_ablation_configs.py
tools/stage7/score_reward_ablation.py
```

### `build_reward_ablation_configs.py`

输入：

```text
Stage 5 frozen reward config
ablation_matrix.yaml
Stage 3 linearization specs
runtime GraphSpec
```

输出每个变体一个完整 YAML，不允许只在命令行临时覆盖而不留记录。

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_reward_ablation_configs.py" \
  --main-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --matrix "$ABLATION_ROOT/configs/ablation_matrix.yaml" \
  --graph-spec-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --linearization-spec "$STAGE3_DIAG/configs/linearization_specs.yaml" \
  --output-dir "$ABLATION_ROOT/configs/resolved"
```

### alternative collapse 的实现规则

`collapse_alternative_to_A_first`：

```text
保留 A→B 合法链；
B→A 轨迹中的边按照 A-first canonical stage index 投影；
不删除原轨迹；
不使用 outcome 重新标边；
输出 projected_node_id、projected_edge_id 和 projection_reason。
```

`collapse_alternative_to_B_first` 对称执行。

### remove recovery edge 的实现规则

```text
failure 仍产生 remaining-cost 增加；
recovery edge 不再作为可识别语义边；
recovery transition 映射到 stagnation；
禁止 recovery debt repayment；
不允许把 recovery 直接映射为普通 forward。
```

这样可以单独测试“显式 recovery edge”是否必要。

---

## 六、构建评分 job 表

输入套件：

```text
real_val
real_test
stage3_diagnostic
oracle_trace_bank
```

变体数量：

```text
1 main
+ 6 ablations
+ 4 eta probes
+ 3 beta probes
= 14 variants
```

总评分 job：

```text
14 variants × 4 suites = 56 jobs
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_reward_ablation_jobs.py" \
  --config-dir "$ABLATION_ROOT/configs/resolved" \
  --real-val "$STAGE5_PRED/tables/ensemble_val_predictions.jsonl.gz" \
  --real-test "$STAGE5_PRED/tables/ensemble_test_predictions.jsonl.gz" \
  --diagnostic "$STAGE5_PRED/tables/ensemble_stage3_diagnostic_predictions.jsonl.gz" \
  --oracle-root "$ORACLE_TRACE_ROOT" \
  --oracle-manifest "$ORACLE_TRACE_MANIFEST" \
  --reward-engine "$STAGE5_REWARD/code/reward_engine.py" \
  --output "$ROUND_DIR/tables/ablation_jobs.tsv" \
  --job-root "$ABLATION_ROOT/jobs"
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(os.environ["ROUND_DIR"], "tables/ablation_jobs.tsv")
d = pd.read_csv(p, sep="\t")
assert len(d) == 56, len(d)
assert d["variant_id"].nunique() == 14
assert set(d["suite"]) == {
    "real_val", "real_test", "stage3_diagnostic", "oracle_trace_bank"
}
assert not d.duplicated(["variant_id","suite"]).any()
print("ABLATION_JOB_MATRIX_OK")
PY
```

---

## 七、并行评分

这些 job 使用缓存 prediction，默认 CPU 多进程：

```bash
cat "$ROUND_DIR/tables/ablation_jobs.tsv" \
  | tail -n +2 \
  | cut -f <command_column_number> \
  | xargs -I{} -P "$(nproc --all)" bash -lc '{}'
```

推荐由脚本直接生成命令文件：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/run_job_table.py" \
  --job-table "$ROUND_DIR/tables/ablation_jobs.tsv" \
  --parallel "$(nproc --all)" \
  --status-output "$ROUND_DIR/tables/ablation_job_status.tsv" \
  2>&1 | tee "$ROUND_DIR/logs/run_ablations.log"
```

如果某个变体需要 GPU tensor 批量评分，先提权查看 GPU，然后按一个 suite/job 一张 GPU 并行，不要混写输出目录。

---

## 八、计算统一指标

创建：

```text
tools/stage7/summarize_reward_ablations.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/summarize_reward_ablations.py" \
  --job-root "$ABLATION_ROOT/jobs" \
  --statistics-unit content_group_id \
  --output-long "$ROUND_DIR/tables/reward_ablation_metrics_long.csv" \
  --output-wide "$ROUND_DIR/tables/reward_ablation_metrics_wide.csv" \
  --effect-table "$ROUND_DIR/tables/reward_ablation_effects.csv" \
  --report "$ROUND_DIR/reports/core_reward_ablation_summary.md" \
  --figures-dir "$ROUND_DIR/figures"
```

每个 suite 至少计算：

```text
legal_path_normalized_gap
A_first_return
B_first_return
alternate_path_negative_rate
forward_positive_rate
reward_nonzero_rate
failure_negative_rate
recovery_positive_rate
recovery_positive_weight_coverage
recovery_cycle_nonpositive_rate
positive_loop_rate
loop_return_mean
recovery_overshoot_rate
success_return_spearman
success_minus_failure_return_margin
fixed_order_score_drop
within_node_reward_density
```

---

## 九、核心机制判定

创建：

```text
tools/stage7/decide_core_ablation_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/decide_core_ablation_gate.py" \
  --metrics "$ROUND_DIR/tables/reward_ablation_metrics_long.csv" \
  --effects "$ROUND_DIR/tables/reward_ablation_effects.csv" \
  --main-id full_locked \
  --output "$ROUND_DIR/metrics/core_ablation_gate.json" \
  --report "$ROUND_DIR/reports/core_ablation_gate.md"
```

### alternative edge 支持证据

满足任一：

```text
collapse_A_first 使 B-first return 相对 full 至少下降 0.15
collapse_B_first 使 A-first return 相对 full 至少下降 0.15
collapse 后 legal_path_normalized_gap 相对 full 增加至少 0.15
collapse 后 alternate_path_negative_rate 增加至少 0.20
```

要求两个方向的结果共同报告，不能只选更有利方向。

### recovery edge 支持证据

`remove_recovery_edge` 相对 full：

```text
recovery_positive_rate 下降至少 0.25
或 recovery_positive_weight_coverage 下降至少 0.25
```

同时：

```text
failure_negative_rate 不得下降超过 0.10
```

### debt cap 支持证据

`no_recovery_debt_cap` 相对 full 满足任一：

```text
recovery_overshoot_rate 增加
failure_recovery_loop_x2 或 x3 的净回报变正
positive_loop_rate 增加
```

若都不发生，结论写为：

```text
debt cap 在当前 trace bank 中未显示独立增益；
保留为安全设计，但不作为强实证贡献。
```

### within-node progress 支持证据

`no_phi` 或 `cost_only` 相对 full 满足任一：

```text
within_node_reward_density 下降至少 0.05
success_return_spearman 下降至少 0.05
success-failure margin 下降至少 0.05
```

### eta/beta probe

只报告曲线，不作为 `full_locked` 通过条件。

---

## 十、画图

至少输出：

```text
alternative_path_returns.png
recovery_ablation_metrics.png
loop_return_by_repeat_count.png
phi_ablation_density.png
eta_probe_curve.png
beta_probe_curve.png
```

每张图单独生成，不使用 subplot。

---

## 十一、本轮 Gate 状态

允许：

```text
CORE_REWARD_ABLATIONS_COMPLETE
CORE_REWARD_CLAIM_WEAK
REPAIR_ABLATION_IMPLEMENTATION
```

`CORE_REWARD_CLAIM_WEAK` 不立即停止阶段 7；继续 7.3–7.5，但在 G4 时决定是否停止主攻。

---

## 十二、本轮 ZIP

checkpoint 和完整 prediction 不打包。

```bash
export ZIP_NAME="stage7_2_core_reward_ablations.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 7.2 状态：<state>
alternative structural support：<true/false>
recovery structural support：<true/false>
debt-cap empirical support：<true/false/weak>
within-node progress support：<true/false>
scoring jobs：56 / 56
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 7.3
```

**核心点：本轮不是再调奖励，而是用固定 prediction 直接检验 alternative、recovery、debt 与 phi 各自是否真正必要。**

---

# 阶段 7.3：历史长度与图粒度实验

## 一、总体上要干什么

本小阶段回答两个模型层问题：

1. 历史信息是否真的解决了同一视觉/状态下的路径不可辨识？
2. 人工图的节点粒度过粗或过细时，奖励质量如何变化？

冻结主模型：

```text
history_steps = 32
graph_granularity = default_manual
```

新增训练变体：

```text
history_steps = 1
history_steps = 8
graph_granularity = coarse
graph_granularity = fine_progress_split
```

每个变体训练 3 个模型 seed：

```text
20260906
20260907
20260908
```

新增训练 job：

\[
4\ \text{variants}\times3\ \text{seeds}=12
\]

主模型 history=32/default 不重训，直接复用持久化 checkpoint。

本轮状态：

```text
HISTORY_GRANULARITY_COMPLETE
```

本轮 ZIP：

```text
stage7_3_history_and_granularity.zip
```

---

## 二、建立目录与 GPU 查询

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7_3_history_and_granularity"
export ROUND_DIR="$STAGE7_ROUNDS/$ROUND_NAME"
export HG_ROOT="$STAGE7_MODELS/history_granularity_v1"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$HG_ROOT"/{configs,supervision,jobs,selection,predictions,metrics,manifests}
```

按通用规范执行提权 GPU 查询。

---

## 三、验证监督数据

```bash
test -f "$STAGE4_SUPERVISION/tables/sample_index.csv.gz"
test -f "$STAGE4_SUPERVISION/tables/episode_manifest.csv"
test -f "$STAGE4_SUPERVISION/tables/content_group_split.csv"
test -f "$STAGE4_SUPERVISION/tables/cost_pairs.csv.gz"
test -f "$STAGE4_SUPERVISION/configs/label_maps.json"
test -f "$STAGE4_SUPERVISION/configs/feature_schema.json"
test -f "$STAGE4_SUPERVISION/configs/cost_target_spec.yaml"
```

禁止特征：

```text
outcome
success
scenario
controller_source
episode_id
content_group_id
```

运行：

```bash
"$PYTHON_BIN" "$STAGE4_TOOLS/validate_supervision_dataset.py" \
  --root "$STAGE4_SUPERVISION" \
  --output "$ROUND_DIR/metrics/supervision_revalidation.json"
```

只做字段、split 和 NaN/Inf 检查，不重新审计原始数据。

---

## 四、构建图粒度变体

创建：

```text
tools/stage7/build_graph_granularity_variants.py
```

输入：

```text
runtime GraphSpec
label maps
node intervals
progress anchors
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_graph_granularity_variants.py" \
  --graph-spec-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --label-maps "$STAGE4_SUPERVISION/configs/label_maps.json" \
  --node-intervals "$STAGE2_M1/gt_v1/node_intervals.csv" \
  --progress-anchors "$STAGE2_M1/gt_v1/progress_anchors.csv" \
  --output-root "$HG_ROOT/configs/graph_variants" \
  --report "$ROUND_DIR/reports/graph_granularity_build.md"
```

### coarse 规则

自动合并满足全部条件的相邻节点：

```text
仅由 forward edge 连接
不属于 branch point
不属于 success/terminal
不属于 failure/recovery/stagnation
合并后不破坏 A→B 与 B→A 两条合法路径
```

永不合并：

```text
alternative 分支节点
failure 节点
recovery 节点
success 节点
```

### fine_progress_split 规则

对满足条件的 default 节点拆成 early/late：

```text
GT interval 数 >= 20
至少有 3 个 progress anchors
节点不是 terminal/failure/recovery
split threshold = phi 0.5
每个任务拆分后总节点数 <= 12
```

这只是粒度控制，不宣传为新的语义节点发现。

输出：

```text
coarse/graph_specs/
coarse/label_map.json
coarse/remap_table.csv

fine_progress_split/graph_specs/
fine_progress_split/label_map.json
fine_progress_split/remap_table.csv
```

---

## 五、生成监督变体

创建：

```text
tools/stage7/remap_supervision_for_granularity.py
```

执行：

```bash
for GRAN in coarse fine_progress_split; do
  "$PYTHON_BIN" "$STAGE7_TOOLS/remap_supervision_for_granularity.py" \
    --source "$STAGE4_SUPERVISION" \
    --remap "$HG_ROOT/configs/graph_variants/$GRAN/remap_table.csv" \
    --label-map "$HG_ROOT/configs/graph_variants/$GRAN/label_map.json" \
    --output "$HG_ROOT/supervision/$GRAN" \
    --statistics-unit content_group_id
done
```

检查：

```text
train/val/test content_group 不变
样本数量不变
feature 数值不变
只允许 node_id、edge_id、graph label 改变
failure/recovery edge type 不得被 coarse/fine 重命名
```

---

## 六、创建训练协议

复制 Stage 4 主训练配置，冻结除以下字段外的全部参数：

```text
history_steps
node_classes
edge_id_classes
label_map
supervision_root
output_dir
seed
```

创建：

```text
$HG_ROOT/configs/history_granularity_protocol.yaml
```

内容至少包括：

```yaml
base_model:
  input_dim: 14
  hidden_dim: 64
  loss_weights: inherit_stage4
  optimizer: inherit_stage4
  train_budget: inherit_stage4
  selection_task: transport_recovery
  selection_split: val
  test_for_selection: false

variants:
  - id: history_1_default
    history_steps: 1
    granularity: default

  - id: history_8_default
    history_steps: 8
    granularity: default

  - id: history_32_coarse
    history_steps: 32
    granularity: coarse

  - id: history_32_fine
    history_steps: 32
    granularity: fine_progress_split

seeds: [20260906, 20260907, 20260908]
```

---

## 七、生成 12-job 矩阵

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_reward_model_variant_jobs.py" \
  --protocol "$HG_ROOT/configs/history_granularity_protocol.yaml" \
  --default-supervision "$STAGE4_SUPERVISION" \
  --variant-supervision-root "$HG_ROOT/supervision" \
  --train-script "$STAGE4_TOOLS/train_joint_pathgraph.py" \
  --output-root "$HG_ROOT/jobs" \
  --job-table "$ROUND_DIR/tables/history_granularity_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["ROUND_DIR"],
    "tables/history_granularity_jobs.tsv"
)
d = pd.read_csv(p, sep="\t")
assert len(d) == 12, len(d)
assert d["variant_id"].nunique() == 4
assert d["seed"].nunique() == 3
assert not d.duplicated(["variant_id","seed"]).any()
assert (~d["test_used_for_selection"].astype(bool)).all()
print("HISTORY_GRANULARITY_12_JOBS_OK")
PY
```

---

## 八、多 GPU 并行训练

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/history_granularity_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/history_granularity_job_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_history_granularity.log"
```

优先同一 variant 的三个 seed 并行。

OOM 时只允许：

```text
同一 variant 的三个 seed 使用相同 micro batch
增加 gradient accumulation
保持 effective batch 和 optimizer step 不变
```

---

## 九、Validation-only 选择

执行：

```bash
"$PYTHON_BIN" "$STAGE4_TOOLS/lock_model_selection.py" \
  --job-root "$HG_ROOT/jobs" \
  --job-table "$ROUND_DIR/tables/history_granularity_jobs.tsv" \
  --selection-task transport_recovery \
  --selection-split val \
  --output "$HG_ROOT/selection/selected_checkpoints.csv" \
  --lock "$HG_ROOT/selection/selection_lock.json"
```

若原脚本不支持多 variant，创建 wrapper，只聚合结果，不改选择规则。

必须：

```text
12 / 12 selected
selection split = val
test_used = false
```

---

## 十、真实推理与指标

使用 Stage 5 已验证的真实推理流程：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_variant_inference_jobs.py" \
  --selection "$HG_ROOT/selection/selected_checkpoints.csv" \
  --splits val,test,stage3_diagnostic \
  --stage4-supervision "$STAGE4_SUPERVISION" \
  --stage3-diagnostic "$STAGE3_DIAG" \
  --output "$ROUND_DIR/tables/history_granularity_inference_jobs.tsv" \
  --job-root "$HG_ROOT/predictions"
```

推理 job：

```text
12 checkpoints × 3 suites = 36
```

多 GPU 执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/history_granularity_inference_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/history_granularity_inference_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_variant_inference.log"
```

每个 job 必须真实加载 checkpoint，并输出 checkpoint SHA。

禁止调用 Stage 4 中写固定数值的旧汇总路径。指标必须从逐样本 prediction 重算。

---

## 十一、统一评估与映射

创建：

```text
tools/stage7/evaluate_history_granularity.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/evaluate_history_granularity.py" \
  --prediction-root "$HG_ROOT/predictions" \
  --selection "$HG_ROOT/selection/selected_checkpoints.csv" \
  --main-model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --default-label-map "$STAGE4_SUPERVISION/configs/label_maps.json" \
  --graph-variant-root "$HG_ROOT/configs/graph_variants" \
  --statistics-unit content_group_id \
  --output-long "$ROUND_DIR/tables/history_granularity_metrics_long.csv" \
  --output-summary "$ROUND_DIR/tables/history_granularity_summary.csv" \
  --report "$ROUND_DIR/reports/history_granularity_summary.md" \
  --figures-dir "$ROUND_DIR/figures"
```

指标：

```text
node_macro_f1_native
node_macro_f1_mapped_to_default
history_required_node_f1
edge_type_macro_f1_non_none
recovery_edge_f1
alternative_edge_f1
phi_mae
phi_spearman
phi_monotonic_violation_rate
cost_mae
cost_spearman
cost_pair_accuracy
reward_legal_path_gap
reward_recovery_positive_rate
reward_cycle_nonpositive_rate
inference_latency_ms
peak_gpu_memory_mb
```

不同粒度之间，主比较使用：

```text
node_macro_f1_mapped_to_default
```

不能直接用类别数量不同的 native node F1 宣称优劣。

---

## 十二、判定

创建：

```text
tools/stage7/decide_history_granularity_gate.py
```

### 历史支持

主模型 history32 相对 history1，在 `history_required` 子集满足：

```text
node F1 提高至少 0.05
或 alternative/recovery edge F1 提高至少 0.05
或 reward legal path gap 至少降低 0.10
```

若不满足：

```text
删除“长历史是必要组成”这一强主张；
保留 history32 作为工程选择。
```

### 图粒度

选择依据：

```text
mapped node F1
edge-type F1
cost MAE
reward path gap
recovery calibration
计算成本
```

不得根据 test 单独选择新主配置。Stage 5 主配置仍保持 default graph。

输出：

```text
best_validation_granularity
coarse_failure_mode
fine_failure_mode
default_tradeoff
```

允许状态：

```text
HISTORY_GRANULARITY_COMPLETE
HISTORY_NOT_NECESSARY
GRANULARITY_SENSITIVE
REPAIR_VARIANT_TRAINING
```

---

## 十三、本轮 ZIP

checkpoint、逐帧 prediction 不打包。

```bash
export ZIP_NAME="stage7_3_history_and_granularity.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 7.3 状态：<state>
训练 jobs：12 / 12
推理 jobs：36 / 36
history32 vs history1：<effect>
coarse：<summary>
fine：<summary>
checkpoint：未打包，见 manifest
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 7.4
```

**核心点：用 12 个并行模型变体直接回答历史是否必要、图粒度在哪里最合适。**

---

# 阶段 7.4：数据覆盖与图复杂度扩展

## 一、总体上要干什么

本小阶段从两个层面测试方法容量：

1. **学习层扩展**：关键 edge 标注覆盖减少时，模型和 reward 如何退化。
2. **奖励层扩展**：合法路径数、恢复次数和任务长度增加时，图奖励何时失效。

本小阶段不训练策略。

学习层新增数据比例：

```text
25%
50%
100%（复用主模型）
```

25% 和 50% 每个训练 3 seed：

\[
2\times3=6\ \text{新增训练 jobs}
\]

奖励层使用明确标记为：

```text
controlled_symbolic_stress
```

的 stress graph，不冒充真实任务泛化。

本轮状态：

```text
SCALING_COVERAGE_COMPLETE
```

本轮 ZIP：

```text
stage7_4_scaling_and_coverage.zip
```

---

## 二、建立目录与 GPU 查询

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7_4_scaling_and_coverage"
export ROUND_DIR="$STAGE7_ROUNDS/$ROUND_NAME"
export SCALE_ROOT="$STAGE7_MODELS/scaling_coverage_v1"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$SCALE_ROOT"/{configs,supervision,jobs,selection,predictions,stress_graphs,metrics}
```

按通用规范提权查看 GPU。

---

## 三、构建 content-group 分层子集

创建：

```text
tools/stage7/build_coverage_subsets.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_coverage_subsets.py" \
  --supervision "$STAGE4_SUPERVISION" \
  --fractions 0.25,0.50 \
  --seeds 20260906,20260907,20260908 \
  --stratify-by task_id,edge_type \
  --group-key content_group_id \
  --min-key-edge-groups 2 \
  --output-root "$SCALE_ROOT/supervision" \
  --manifest "$ROUND_DIR/tables/coverage_subset_manifest.csv" \
  --report "$ROUND_DIR/reports/coverage_subset_summary.md"
```

规则：

```text
只下采样 train
val/test 完全不变
按 content_group_id 抽样
每个关键 edge 尽量保留至少 2 个 content groups
同一 fraction/seed 的样本清单锁定
禁止帧级随机抽样
```

关键 edge：

```text
alternative
failure
recovery
forward branch transitions
```

输出目录：

```text
supervision/f025_s20260906/
...
supervision/f050_s20260908/
```

---

## 四、生成 6 个训练 job

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_coverage_training_jobs.py" \
  --subset-root "$SCALE_ROOT/supervision" \
  --fractions 0.25,0.50 \
  --seeds 20260906,20260907,20260908 \
  --base-stage4-config "$STAGE4_SUPERVISION/configs/resolved_stage4.yaml" \
  --train-script "$STAGE4_TOOLS/train_joint_pathgraph.py" \
  --output-root "$SCALE_ROOT/jobs" \
  --job-table "$ROUND_DIR/tables/coverage_training_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
d = pd.read_csv(
    os.path.join(os.environ["ROUND_DIR"], "tables/coverage_training_jobs.tsv"),
    sep="\t"
)
assert len(d) == 6
assert set(d["train_fraction"]) == {0.25, 0.50}
assert d["seed"].nunique() == 3
assert not d.duplicated(["train_fraction","seed"]).any()
print("COVERAGE_TRAINING_6_JOBS_OK")
PY
```

---

## 五、多 GPU 训练和选择

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/coverage_training_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/coverage_training_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_coverage_training.log"
```

Validation-only 选择：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/select_variant_checkpoints.py" \
  --job-root "$SCALE_ROOT/jobs" \
  --job-table "$ROUND_DIR/tables/coverage_training_jobs.tsv" \
  --selection-task transport_recovery \
  --selection-split val \
  --output "$SCALE_ROOT/selection/selected_checkpoints.csv" \
  --lock "$SCALE_ROOT/selection/selection_lock.json"
```

---

## 六、真实推理与 coverage 曲线

为 6 个新 checkpoint 运行：

```text
val
test
stage3_diagnostic
```

共 18 个推理 job。

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_variant_inference_jobs.py" \
  --selection "$SCALE_ROOT/selection/selected_checkpoints.csv" \
  --splits val,test,stage3_diagnostic \
  --stage4-supervision "$STAGE4_SUPERVISION" \
  --stage3-diagnostic "$STAGE3_DIAG" \
  --output "$ROUND_DIR/tables/coverage_inference_jobs.tsv" \
  --job-root "$SCALE_ROOT/predictions"

"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/coverage_inference_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/coverage_inference_status.tsv" \
  --resume-failed true
```

汇总时加入 100% 主模型：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/summarize_coverage_scaling.py" \
  --prediction-root "$SCALE_ROOT/predictions" \
  --main-model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --main-predictions "$STAGE5_PRED" \
  --fractions 0.25,0.50,1.00 \
  --statistics-unit content_group_id \
  --output "$ROUND_DIR/tables/coverage_scaling_metrics.csv" \
  --edge-output "$ROUND_DIR/tables/coverage_by_edge_metrics.csv" \
  --report "$ROUND_DIR/reports/coverage_scaling_summary.md" \
  --figures-dir "$ROUND_DIR/figures"
```

至少报告：

```text
node F1
alternative edge F1
recovery edge F1
cost MAE
phi Spearman
path gap
failure negative rate
recovery positive rate
cycle nonpositive rate
```

---

## 七、构建 controlled symbolic stress graphs

创建：

```text
tools/stage7/build_symbolic_graph_stress_suite.py
```

固定网格：

```text
legal_path_count = [1, 2, 4, 8]
recovery_repeat_count = [0, 1, 2, 4]
length_multiplier = [1, 2, 4]
stress_seed = [20261101, 20261102, 20261103]
```

总设置：

\[
4\times4\times3\times3=144
\]

每个设置生成：

```text
20 legal success traces
20 failure/recovery traces
10 stagnation/illegal-loop traces
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_symbolic_graph_stress_suite.py" \
  --source-graph-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --oracle-trace-root "$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/oracle_traces" \
  --path-counts 1,2,4,8 \
  --recovery-repeats 0,1,2,4 \
  --length-multipliers 1,2,4 \
  --seeds 20261101,20261102,20261103 \
  --output-root "$SCALE_ROOT/stress_graphs" \
  --manifest "$ROUND_DIR/tables/stress_suite_manifest.csv"
```

实现约束：

```text
新增分支使用已知 edge template 的结构克隆
每条成功 trace 终点相同
所有合法路径的 nominal cost 可比
failure 增加 cost
recovery 只还原此前 failure debt
provenance = controlled_symbolic_stress
不得写成真实机器人 OOD
```

---

## 八、加入经验预测噪声

创建：

```text
tools/stage7/fit_empirical_prediction_noise.py
```

从真实 test prediction 与 GT 计算：

```text
node confusion matrix
edge-type confusion matrix
phi residual distribution by node type
cost residual distribution by edge type
ensemble std distribution
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/fit_empirical_prediction_noise.py" \
  --predictions "$STAGE5_PRED/tables/ensemble_test_predictions.jsonl.gz" \
  --ground-truth "$STAGE4_SUPERVISION" \
  --output "$SCALE_ROOT/configs/empirical_noise_model.json" \
  --tables-dir "$ROUND_DIR/tables"
```

将噪声模型应用到 stress suite：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/apply_prediction_noise_to_stress_suite.py" \
  --stress-root "$SCALE_ROOT/stress_graphs" \
  --noise-model "$SCALE_ROOT/configs/empirical_noise_model.json" \
  --output-root "$SCALE_ROOT/stress_graphs_noisy" \
  --manifest "$ROUND_DIR/tables/stress_noisy_manifest.csv"
```

---

## 九、运行图复杂度压力测试

评分两种模式：

```text
oracle_state
empirical_noisy_state
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/score_graph_stress_suite.py" \
  --suite-root "$SCALE_ROOT/stress_graphs" \
  --noisy-suite-root "$SCALE_ROOT/stress_graphs_noisy" \
  --reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --reward-engine "$STAGE5_REWARD/code/reward_engine.py" \
  --parallel "$(nproc --all)" \
  --output "$ROUND_DIR/tables/graph_stress_metrics.csv" \
  --trace-output "$SCALE_ROOT/metrics/graph_stress_trace_returns.csv" \
  --report "$ROUND_DIR/reports/graph_stress_summary.md" \
  --figures-dir "$ROUND_DIR/figures"
```

指标：

```text
legal_path_normalized_gap
success_return_variance_across_paths
failure_negative_rate
recovery_positive_rate
cycle_nonpositive_rate
positive_loop_rate
reward_density
reward_clip_rate
runtime_ms_per_1000_steps
```

---

## 十、确定失效边界

创建：

```text
tools/stage7/find_scaling_failure_boundary.py
```

建议稳定标准：

```text
legal_path_normalized_gap <= 0.10
recovery_cycle_nonpositive_rate >= 0.90
positive_loop_rate <= 0.05
recovery_positive_rate >= 0.65
failure_negative_rate >= 0.70
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/find_scaling_failure_boundary.py" \
  --coverage "$ROUND_DIR/tables/coverage_scaling_metrics.csv" \
  --stress "$ROUND_DIR/tables/graph_stress_metrics.csv" \
  --output "$ROUND_DIR/metrics/scaling_boundary.json" \
  --report "$ROUND_DIR/reports/scaling_boundary.md"
```

目标不是要求所有网格都通过，而是明确：

```text
最低训练覆盖
最大稳定路径数
最大稳定恢复次数
最大稳定长度倍数
首先失效的指标
```

---

## 十一、本轮状态

允许：

```text
SCALING_COVERAGE_COMPLETE
COVERAGE_LIMIT_IDENTIFIED
GRAPH_COMPLEXITY_LIMIT_IDENTIFIED
REPAIR_SCALING_PIPELINE
```

只要真实 coverage 曲线和 stress 边界均成功生成，即可进入 7.5。

---

## 十二、本轮 ZIP

checkpoint、完整 prediction、全部 stress trace 不打包。

```bash
export ZIP_NAME="stage7_4_scaling_and_coverage.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 7.4 状态：<state>
coverage training jobs：6 / 6
coverage inference jobs：18 / 18
最低稳定训练比例：<value>
最大稳定路径数：<value>
最大稳定恢复次数：<value>
最大稳定长度倍数：<value>
stress provenance：controlled_symbolic_stress
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 7.5
```

**核心点：本轮用真实下采样回答数据需求，用显式标注的 symbolic stress 回答图奖励的容量和失效边界。**

---

# 阶段 7.5：结构 OOD、扰动鲁棒性与不确定性

## 一、总体上要干什么

本小阶段测试三类 reward-only 泛化：

1. **未见合法顺序**：训练时只见 A→B 或只见 B→A，测试另一顺序。
2. **恢复位置变化**：failure/recovery 出现在早、中、晚不同节点。
3. **表示扰动**：历史帧丢失、低维状态噪声和事件时间抖动。

同时检查 ensemble uncertainty 是否真的能标记高错误样本。

主奖励仍为 beta=0；非零 beta 只作为诊断，不替换主方法。

新增训练：

```text
A-first-only training × 3 seeds
B-first-only training × 3 seeds
= 6 jobs
```

本轮状态：

```text
OOD_UNCERTAINTY_COMPLETE
```

本轮 ZIP：

```text
stage7_5_ood_and_uncertainty.zip
```

---

## 二、建立目录与 GPU 查询

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7_5_ood_and_uncertainty"
export ROUND_DIR="$STAGE7_ROUNDS/$ROUND_NAME"
export OOD_ROOT="$STAGE7_MODELS/ood_uncertainty_v1"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$OOD_ROOT"/{configs,supervision,jobs,selection,predictions,perturbations,metrics}
```

按通用规范提权查看 GPU。

---

## 三、构建顺序 holdout 数据

使用：

```text
$STAGE4_SUPERVISION/probes/dual_order_folds.json
```

检查：

```bash
test -f "$STAGE4_SUPERVISION/probes/dual_order_folds.json"
```

创建：

```text
tools/stage7/build_order_holdout_supervision.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_order_holdout_supervision.py" \
  --supervision "$STAGE4_SUPERVISION" \
  --folds "$STAGE4_SUPERVISION/probes/dual_order_folds.json" \
  --directions A_first_train,B_first_train \
  --seeds 20260906,20260907,20260908 \
  --group-key content_group_id \
  --output-root "$OOD_ROOT/supervision" \
  --manifest "$ROUND_DIR/tables/order_holdout_manifest.csv" \
  --report "$ROUND_DIR/reports/order_holdout_data_summary.md"
```

规则：

```text
只从 transport_dual_order 的 train 中移除目标顺序
transport_recovery train 保持
val 只使用训练方向的 validation 用于 checkpoint 选择
反向顺序全部保留为 OOD test
同 content_group 不得跨 train/OOD
```

---

## 四、训练 6 个 holdout 模型

生成 job：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_order_holdout_jobs.py" \
  --supervision-root "$OOD_ROOT/supervision" \
  --directions A_first_train,B_first_train \
  --seeds 20260906,20260907,20260908 \
  --base-config "$STAGE4_SUPERVISION/configs/resolved_stage4.yaml" \
  --train-script "$STAGE4_TOOLS/train_joint_pathgraph.py" \
  --output-root "$OOD_ROOT/jobs" \
  --job-table "$ROUND_DIR/tables/order_holdout_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

检查 6 job 后并行执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/order_holdout_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/order_holdout_job_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_order_holdout.log"
```

---

## 五、Validation-only 选择与 OOD 推理

选择：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/select_variant_checkpoints.py" \
  --job-root "$OOD_ROOT/jobs" \
  --job-table "$ROUND_DIR/tables/order_holdout_jobs.tsv" \
  --selection-split val_seen_order \
  --output "$OOD_ROOT/selection/order_holdout_selected.csv" \
  --lock "$OOD_ROOT/selection/order_holdout_selection_lock.json"
```

禁止使用未见顺序 test 选 checkpoint。

构建 OOD 推理：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_order_holdout_inference_jobs.py" \
  --selection "$OOD_ROOT/selection/order_holdout_selected.csv" \
  --holdout-root "$OOD_ROOT/supervision" \
  --output "$ROUND_DIR/tables/order_holdout_inference_jobs.tsv" \
  --job-root "$OOD_ROOT/predictions/order_holdout"
```

预计：

```text
6 checkpoint × seen-test/unseen-test = 12 jobs
```

多 GPU 执行。

---

## 六、构建恢复位置 OOD

创建：

```text
tools/stage7/build_recovery_position_suite.py
```

基于 Oracle trace 和真实 recovery segment，生成：

```text
early_recovery
mid_recovery
late_recovery
```

每类至少：

```text
30 traces
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_recovery_position_suite.py" \
  --graph-spec-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --oracle-root "$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/oracle_traces" \
  --real-recovery-segments "$STAGE3_DIAG/tables/recovery_segments.csv" \
  --positions early,mid,late \
  --traces-per-position 30 \
  --seed 20261201 \
  --output-root "$OOD_ROOT/perturbations/recovery_position" \
  --manifest "$ROUND_DIR/tables/recovery_position_manifest.csv"
```

标记 provenance：

```text
scripted_oracle
或 derived_counterfactual
```

不要标为真实新 episode。

---

## 七、构建表示扰动网格

创建：

```text
tools/stage7/build_representation_perturbation_jobs.py
```

网格：

```text
history_dropout = [0.0, 0.10, 0.25, 0.50]
feature_noise_std_fraction = [0.0, 0.02, 0.05, 0.10]
event_boundary_jitter_steps = [0, 2, 5, 10]
```

不做完整三维笛卡尔积，使用单因素设计：

```text
baseline
4 history dropout
4 feature noise
4 timing jitter
```

去除重复 baseline 后：

```text
10 unique settings
```

对 3 个主模型 seed 和 2 个 test task：

```text
10 × 3 × 2 = 60 inference jobs
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_representation_perturbation_jobs.py" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --supervision "$STAGE4_SUPERVISION" \
  --history-dropout 0.0,0.10,0.25,0.50 \
  --feature-noise 0.0,0.02,0.05,0.10 \
  --boundary-jitter 0,2,5,10 \
  --single-factor true \
  --tasks transport_recovery,transport_dual_order \
  --output "$ROUND_DIR/tables/perturbation_inference_jobs.tsv" \
  --job-root "$OOD_ROOT/predictions/perturbations"
```

多 GPU 执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/perturbation_inference_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/perturbation_inference_status.tsv" \
  --resume-failed true
```

---

## 八、统一 OOD 指标

创建：

```text
tools/stage7/evaluate_reward_ood.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/evaluate_reward_ood.py" \
  --order-holdout-root "$OOD_ROOT/predictions/order_holdout" \
  --recovery-position-root "$OOD_ROOT/perturbations/recovery_position" \
  --perturbation-root "$OOD_ROOT/predictions/perturbations" \
  --reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --reward-engine "$STAGE5_REWARD/code/reward_engine.py" \
  --statistics-unit content_group_id \
  --output "$ROUND_DIR/tables/ood_reward_metrics.csv" \
  --report "$ROUND_DIR/reports/ood_reward_summary.md" \
  --figures-dir "$ROUND_DIR/figures"
```

指标：

```text
seen_order_node_f1
unseen_order_node_f1
unseen_order_alternative_edge_f1
unseen_order_path_gap
recovery_positive_rate_by_position
failure_negative_rate_by_position
cycle_nonpositive_rate_by_position
node_f1_under_perturbation
edge_f1_under_perturbation
cost_mae_under_perturbation
reward_path_gap_under_perturbation
reward_recovery_positive_under_perturbation
```

---

## 九、不确定性校准

创建：

```text
tools/stage7/evaluate_uncertainty_as_error_signal.py
```

错误事件：

```text
node prediction wrong
edge type wrong
abs(cost error) > test median
reward sign wrong
recovery reward sign wrong
```

不确定性信号：

```text
ensemble node entropy
ensemble edge entropy
cost std
phi std
reward std
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/evaluate_uncertainty_as_error_signal.py" \
  --clean-predictions "$STAGE5_PRED/tables/ensemble_test_predictions.jsonl.gz" \
  --perturbed-prediction-root "$OOD_ROOT/predictions/perturbations" \
  --ground-truth "$STAGE4_SUPERVISION" \
  --output "$ROUND_DIR/tables/uncertainty_error_detection.csv" \
  --risk-coverage "$ROUND_DIR/tables/uncertainty_risk_coverage.csv" \
  --report "$ROUND_DIR/reports/uncertainty_summary.md" \
  --figures-dir "$ROUND_DIR/figures"
```

至少报告：

```text
AUROC
AUPRC
risk at 80% coverage
risk at 60% coverage
Spearman(error magnitude, uncertainty)
```

---

## 十、beta LCB 只做诊断

使用：

```text
beta = 0
beta = 0.5
beta = 1.0
```

在同一 OOD prediction 上重算：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/score_beta_lcb_ood_probe.py" \
  --prediction-root "$OOD_ROOT/predictions/perturbations" \
  --betas 0.0,0.5,1.0 \
  --base-reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --output "$ROUND_DIR/tables/beta_lcb_ood_probe.csv" \
  --report "$ROUND_DIR/reports/beta_lcb_probe.md"
```

报告必须写明：

```text
post-hoc diagnostic
not selected as main reward
not used to revise Stage 5 lock
```

---

## 十一、判定

创建：

```text
tools/stage7/decide_ood_uncertainty_gate.py
```

### 顺序 OOD

至少报告：

```text
unseen-order path gap
unseen-order edge F1
相对 seen-order 的下降
```

支持“结构组合泛化”的建议条件：

```text
unseen-order legal path gap <= 0.25
且 unseen-order alternative edge F1 >= 0.60
```

未达到时，结论为：

```text
需要训练数据覆盖每条主要合法分支；
不主张未见顺序泛化。
```

### 轻度扰动鲁棒性

轻度定义：

```text
history dropout <= 0.25
feature noise <= 0.05
boundary jitter <= 5
```

建议条件：

```text
recovery positive rate >= 0.60
cycle nonpositive rate >= 0.90
path gap <= 0.20
```

### uncertainty

可以作为辅助贡献的条件：

```text
reward-sign-error AUROC >= 0.65
或 risk at 60% coverage 比全覆盖至少下降 20%
```

否则不宣传 uncertainty calibration。

允许状态：

```text
OOD_UNCERTAINTY_COMPLETE
OOD_LIMIT_IDENTIFIED
UNCERTAINTY_USEFUL
UNCERTAINTY_NOT_SUPPORTED
REPAIR_OOD_PIPELINE
```

---

## 十二、本轮 ZIP

checkpoint、逐帧 prediction 和完整扰动数据不打包。

```bash
export ZIP_NAME="stage7_5_ood_and_uncertainty.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 7.5 状态：<state>
order-holdout training jobs：6 / 6
order-holdout inference jobs：12 / 12
perturbation inference jobs：60 / 60
unseen-order path gap：<value>
light-perturbation recovery rate：<value>
uncertainty reward-error AUROC：<value>
beta probe：post-hoc only
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 7.6
```

**核心点：本轮不再看策略，而是直接测试图奖励对未见顺序、恢复位置和输入扰动的真实边界。**

---

# 阶段 7.6：自动图发现探索（条件执行）

## 一、总体上要干什么

本小阶段只在 7.2 的核心结构证据成立后执行。

入口条件：

```text
alternative structural support = true
recovery structural support = true
```

若任一不成立，直接输出：

```text
SKIP_AUTO_GRAPH_KEEP_MANUAL
```

并生成本轮 ZIP，不浪费算力。

执行时，自动图仅作为扩展：

```text
frozen encoder embedding
→ temporal change-point proposal
→ segment clustering
→ transition graph induction
→ 与人工图对齐评估
```

默认不调用外部 LLM，不引入 API 依赖。

本轮状态：

```text
KEEP_AUTO_GRAPH_AS_EXTENSION
KEEP_MANUAL_GRAPH_ONLY
SKIP_AUTO_GRAPH_KEEP_MANUAL
```

本轮 ZIP：

```text
stage7_6_auto_graph_exploration.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7_6_auto_graph_exploration"
export ROUND_DIR="$STAGE7_ROUNDS/$ROUND_NAME"
export AUTO_ROOT="$STAGE7_RESULTS/auto_graph_v1"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$AUTO_ROOT"/{embeddings,proposals,graphs,selection,metrics}
```

---

## 三、入口 Gate

执行：

```bash
"$PYTHON_BIN" - <<'PY'
import json, os, sys
p = os.path.join(
    os.environ["STAGE7_ROUNDS"],
    "stage7_2_core_reward_ablations/metrics/core_ablation_gate.json"
)
d = json.load(open(p))
alt = bool(d.get("alternative_structural_support", False))
rec = bool(d.get("recovery_structural_support", False))
print("alternative =", alt)
print("recovery =", rec)
if not (alt and rec):
    sys.exit(3)
PY
```

若退出码为 3：

```bash
cat > "$ROUND_DIR/metrics/auto_graph_gate.json" <<'JSON'
{
  "decision": "SKIP_AUTO_GRAPH_KEEP_MANUAL",
  "reason": "core structural evidence not strong enough to justify extension"
}
JSON

cat > "$ROUND_DIR/reports/auto_graph_summary.md" <<'EOF'
# Auto graph exploration

Decision: `SKIP_AUTO_GRAPH_KEEP_MANUAL`.

The manual graph remains the final method definition.
EOF
```

然后直接执行本轮 ZIP 步骤。

---

## 四、提权查看 GPU

若执行自动图，先按通用规范运行 `sudo -n nvidia-smi`。

---

## 五、提取冻结 encoder embedding

使用 Stage 6 持久化的三个 reward-model checkpoint。

创建：

```text
tools/stage7/extract_frozen_encoder_embeddings.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_embedding_jobs.py" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --supervision "$STAGE4_SUPERVISION" \
  --splits train,val,test \
  --output "$ROUND_DIR/tables/embedding_jobs.tsv" \
  --job-root "$AUTO_ROOT/embeddings"
```

应生成：

```text
3 model seeds × 3 splits = 9 jobs
```

多 GPU 执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/embedding_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --status-output "$ROUND_DIR/tables/embedding_job_status.tsv" \
  --resume-failed true
```

输出至少包含：

```text
episode_id
content_group_id
t
embedding
ensemble_mean_embedding
split
task_id
```

GT node/edge 不得作为 discovery 输入，只能用于后续评估。

---

## 六、生成 change-point 与聚类 job

固定网格：

```text
change-point quantile = [0.80, 0.90, 0.95]
K = [5, 7, 9, 11]
discovery seed = [20261301, 20261302, 20261303]
```

组合过多时，不做完整 \(3\times4\times3=36\) 全网格。使用：

```text
第一步：val 上 quantile × K，共 12 个配置，seed=20261301
第二步：选定最佳 quantile/K 后，用 3 个 discovery seed 重跑稳定性
```

创建：

```text
tools/stage7/build_auto_graph_jobs.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_auto_graph_jobs.py" \
  --embedding-root "$AUTO_ROOT/embeddings" \
  --change-point-quantiles 0.80,0.90,0.95 \
  --cluster-counts 5,7,9,11 \
  --selection-seed 20261301 \
  --output "$ROUND_DIR/tables/auto_graph_selection_jobs.tsv" \
  --job-root "$AUTO_ROOT/proposals"
```

12 个 selection job 可 CPU 并行；聚类规模大时使用 GPU，但输出目录独立。

---

## 七、自动图生成规则

创建：

```text
tools/stage7/discover_graph_from_embeddings.py
```

每个 job：

1. 按 episode 时间顺序计算 embedding change score；
2. 使用指定分位数生成候选边界；
3. 对 segment mean embedding 聚类；
4. 将连续相同 cluster 合并；
5. 从观测转移计数建立有向图；
6. 保留出现次数达到 2 的边；
7. 标记自环和回访；
8. 不使用 GT 名称命名节点，先命名为 `auto_node_00...`；
9. 保存节点原型、边计数和 episode path。

命令示例：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/discover_graph_from_embeddings.py" \
  --embeddings "$AUTO_ROOT/embeddings/ensemble_train.parquet" \
  --change-point-quantile 0.90 \
  --clusters 7 \
  --seed 20261301 \
  --min-edge-count 2 \
  --output-dir "$AUTO_ROOT/proposals/q090_k07_s20261301"
```

---

## 八、只用 validation 选择配置

创建：

```text
tools/stage7/evaluate_auto_graph_alignment.py
```

在 val 上将 auto node 与 manual node 用 Hungarian matching 对齐。

对齐成本：

```text
1 - node interval overlap F1
```

指标：

```text
node_mapping_macro_f1
adjusted_rand_index
edge_precision
edge_recall
edge_type_recall_after_alignment
alternative_path_recall
recovery_edge_recall
normalized_graph_edit_distance
```

执行 selection：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/evaluate_auto_graph_alignment.py" \
  --proposal-root "$AUTO_ROOT/proposals" \
  --manual-graph-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --ground-truth "$STAGE4_SUPERVISION" \
  --split val \
  --output "$ROUND_DIR/tables/auto_graph_val_metrics.csv"

"$PYTHON_BIN" "$STAGE7_TOOLS/select_auto_graph_config.py" \
  --metrics "$ROUND_DIR/tables/auto_graph_val_metrics.csv" \
  --output "$AUTO_ROOT/selection/selected_auto_graph.json" \
  --lock "$AUTO_ROOT/selection/selection_lock.json"
```

选择分数：

\[
S=
0.30F1_{node}
+
0.25R_{edge}
+
0.20R_{alternative}
+
0.20R_{recovery}
-
0.05GED_{norm}
\]

禁止查看 test 后改变 K 或 change-point threshold。

---

## 九、稳定性重跑

读取 val 选中的 quantile/K，使用：

```text
20261301
20261302
20261303
```

生成 3 个图。

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/repeat_selected_auto_graph.py" \
  --selection "$AUTO_ROOT/selection/selected_auto_graph.json" \
  --embedding-root "$AUTO_ROOT/embeddings" \
  --seeds 20261301,20261302,20261303 \
  --output-root "$AUTO_ROOT/graphs" \
  --job-table "$ROUND_DIR/tables/auto_graph_stability_jobs.tsv"
```

并行运行。

---

## 十、冻结 test 评估

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/evaluate_auto_graph_alignment.py" \
  --proposal-root "$AUTO_ROOT/graphs" \
  --manual-graph-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --ground-truth "$STAGE4_SUPERVISION" \
  --split test \
  --output "$ROUND_DIR/tables/auto_graph_test_metrics.csv"

"$PYTHON_BIN" "$STAGE7_TOOLS/summarize_auto_graph_stability.py" \
  --graphs "$AUTO_ROOT/graphs" \
  --test-metrics "$ROUND_DIR/tables/auto_graph_test_metrics.csv" \
  --output "$ROUND_DIR/metrics/auto_graph_stability.json" \
  --report "$ROUND_DIR/reports/auto_graph_summary.md" \
  --figures-dir "$ROUND_DIR/figures"
```

跨 discovery seed 稳定性：

```text
pairwise node ARI
matched edge Jaccard
path-set Jaccard
node-count variance
```

---

## 十一、决策

创建：

```text
tools/stage7/decide_auto_graph_extension.py
```

`KEEP_AUTO_GRAPH_AS_EXTENSION` 建议同时满足：

```text
test node mapping macro F1 >= 0.70
test recovery edge recall >= 0.70
test alternative path recall >= 0.90
normalized graph edit distance <= 0.40
cross-seed node ARI >= 0.60
```

否则：

```text
KEEP_MANUAL_GRAPH_ONLY
```

无论是否通过：

```text
manual graph 始终保留为主方法
auto graph 不替换 Stage 5 reward_v1
```

---

## 十二、本轮 ZIP

embedding 和大型聚类缓存不打包。

```bash
export ZIP_NAME="stage7_6_auto_graph_exploration.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 7.6 状态：<decision>
auto graph executed：<true/false>
selected K：<value or NA>
selected change-point quantile：<value or NA>
node F1：<value or NA>
recovery edge recall：<value or NA>
cross-seed ARI：<value or NA>
manual graph remains main：true
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 7.7
```

**核心点：自动图只做一次受控扩展；不稳定就立即保留人工图，不让扩展实验拖慢主线。**

---

# 阶段 7.7：G4 贡献边界冻结与 Stage 8 交接

## 一、总体上要干什么

汇总 Stage 3–7 的 reward-only 证据，冻结：

```text
主结果表
核心消融表
历史/粒度曲线
coverage/scaling 边界
OOD/uncertainty 边界
auto-graph 处理结论
Stage 6 mixed policy evidence
```

作出 G4：

```text
GO_STAGE8_REWARD_ONLY
REFINE_STAGE7_CORE_ONLY
STOP_PATHGRAPH
```

本轮不再运行训练。

本轮 ZIP：

```text
stage7_7_g4_freeze.zip
```

阶段总 ZIP：

```text
stage7_complete.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage7_reward_only/stage7_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7_7_g4_freeze"
export ROUND_DIR="$STAGE7_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE7_G4"/{configs,locks,metrics,tables,figures,reports,manifests}
```

---

## 三、收集各阶段 Gate

必须存在：

```bash
test -f "$STAGE7_ROUNDS/stage7_1_reward_only_input_freeze/metrics/stage7_input_gate.json"
test -f "$STAGE7_ROUNDS/stage7_2_core_reward_ablations/metrics/core_ablation_gate.json"
test -f "$STAGE7_ROUNDS/stage7_3_history_and_granularity/metrics/history_granularity_gate.json"
test -f "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/metrics/scaling_boundary.json"
test -f "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/metrics/ood_uncertainty_gate.json"
test -f "$STAGE7_ROUNDS/stage7_6_auto_graph_exploration/metrics/auto_graph_gate.json"
```

若 7.6 被跳过，仍必须有：

```text
auto_graph_gate.json
decision = SKIP_AUTO_GRAPH_KEEP_MANUAL
```

---

## 四、重建 reward 主表

创建：

```text
tools/stage7/build_reward_only_main_table.py
```

输入：

```text
Stage 5 real predictions
Stage 5 frozen reward metrics
Stage 3 diagnostic suite
Stage 7 core ablations
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_reward_only_main_table.py" \
  --real-predictions "$STAGE5_PRED" \
  --reward-v1 "$STAGE5_REWARD" \
  --diagnostic-suite "$STAGE3_DIAG" \
  --ablation-metrics "$STAGE7_ROUNDS/stage7_2_core_reward_ablations/tables/reward_ablation_metrics_long.csv" \
  --output "$STAGE7_G4/tables/reward_main_table.csv" \
  --report "$STAGE7_G4/reports/reward_main_results.md"
```

主表至少包括方法：

```text
linear_time_fraction
oracle_linear_chain_A_first
oracle_linear_chain_B_first
sequential_transition_oracle
learned_linear_sarm
pathgraph_reward_v1_locked
```

指标：

```text
node/edge model metrics
remaining-cost calibration
legal path consistency
failure negative rate
recovery positive rate
loop nonpositive rate
positive loop rate
success-return correlation
success-failure margin
fixed-order drop
```

不得把 Stage 4 占位 test 指标写入主表。

---

## 五、构建核心消融与边界表

复制或生成：

```bash
cp "$STAGE7_ROUNDS/stage7_2_core_reward_ablations/tables/reward_ablation_effects.csv" \
  "$STAGE7_G4/tables/core_ablation_effects.csv"

cp "$STAGE7_ROUNDS/stage7_3_history_and_granularity/tables/history_granularity_summary.csv" \
  "$STAGE7_G4/tables/history_granularity_summary.csv"

cp "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/coverage_scaling_metrics.csv" \
  "$STAGE7_G4/tables/coverage_scaling_metrics.csv"

cp "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/graph_stress_metrics.csv" \
  "$STAGE7_G4/tables/graph_stress_metrics.csv"

cp "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/ood_reward_metrics.csv" \
  "$STAGE7_G4/tables/ood_reward_metrics.csv"

cp "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/uncertainty_error_detection.csv" \
  "$STAGE7_G4/tables/uncertainty_error_detection.csv"
```

自动图若执行：

```bash
if [ -f "$STAGE7_ROUNDS/stage7_6_auto_graph_exploration/tables/auto_graph_test_metrics.csv" ]; then
  cp "$STAGE7_ROUNDS/stage7_6_auto_graph_exploration/tables/auto_graph_test_metrics.csv" \
    "$STAGE7_G4/tables/"
fi
```

---

## 六、冻结 Policy 次级证据

创建：

```text
tools/stage7/build_policy_secondary_evidence.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_policy_secondary_evidence.py" \
  --stage6-evidence "$STAGE6_M4/metrics/stage6_evidence.json" \
  --stage6r1-evidence "$STAGE6R1_M4/metrics/g3_refine1_evidence.json" \
  --stage6r1-decision "$STAGE6R1_M4/metrics/g3_refine1_decision.json" \
  --output "$STAGE7_G4/tables/policy_secondary_evidence.csv" \
  --report "$STAGE7_G4/reports/policy_secondary_evidence.md"
```

报告固定写法：

```text
PathGraph weighting showed positive aggregate and recovery effects,
but strict policy-seed consistency was not met.
The policy result is secondary and mixed, not the primary conclusion.
```

不得写：

```text
PathGraph-SARM consistently improves policy success.
```

---

## 七、生成 Claim Matrix

创建：

```text
tools/stage7/build_claim_matrix.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/build_claim_matrix.py" \
  --claim-lock "$STAGE7_INPUTS/locks/claim_scope_lock.json" \
  --reward-main "$STAGE7_G4/tables/reward_main_table.csv" \
  --ablation "$STAGE7_G4/tables/core_ablation_effects.csv" \
  --history "$STAGE7_G4/tables/history_granularity_summary.csv" \
  --scaling "$STAGE7_G4/tables/coverage_scaling_metrics.csv" \
  --ood "$STAGE7_G4/tables/ood_reward_metrics.csv" \
  --uncertainty "$STAGE7_G4/tables/uncertainty_error_detection.csv" \
  --policy-secondary "$STAGE7_G4/tables/policy_secondary_evidence.csv" \
  --output "$STAGE7_G4/tables/claim_matrix.csv" \
  --report "$STAGE7_G4/reports/claim_boundary.md"
```

`claim_matrix.csv` 字段：

```text
claim_id
claim_text
priority
evidence_source
support_status
support_metric
support_value
limitation
paper_location
```

`support_status`：

```text
supported
partially_supported
not_supported
not_tested
```

---

## 八、G4 决策规则

创建：

```text
$STAGE7_G4/configs/g4_rule.json
```

内容：

```json
{
  "go_stage8_reward_only": {
    "input_gate_pass": true,
    "full_reward_reproduces_g2_core_metrics": true,
    "alternative_structural_support": true,
    "recovery_structural_support": true,
    "no_post_test_main_reward_retuning": true,
    "portable_manifest_pass": true,
    "scaling_boundary_reported": true,
    "ood_boundary_reported": true,
    "policy_claim_marked_secondary": true
  },
  "refine_stage7_core_only": {
    "allowed_once": true,
    "allowed_reasons": [
      "missing required input",
      "broken metric script",
      "failed training or inference job",
      "checksum or packaging error"
    ],
    "method_retuning_allowed": false
  },
  "stop_pathgraph": {
    "trigger_if_any": [
      "full reward fails truthful recomputation",
      "alternative structural claim unsupported",
      "recovery structural claim unsupported",
      "graph reward no better than sequential baseline on core diagnostic"
    ]
  }
}
```

---

## 九、执行 G4 决策

创建：

```text
tools/stage7/decide_g4_reward_only.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/decide_g4_reward_only.py" \
  --input-gate "$STAGE7_ROUNDS/stage7_1_reward_only_input_freeze/metrics/stage7_input_gate.json" \
  --ablation-gate "$STAGE7_ROUNDS/stage7_2_core_reward_ablations/metrics/core_ablation_gate.json" \
  --history-gate "$STAGE7_ROUNDS/stage7_3_history_and_granularity/metrics/history_granularity_gate.json" \
  --scaling-boundary "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/metrics/scaling_boundary.json" \
  --ood-gate "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/metrics/ood_uncertainty_gate.json" \
  --auto-graph-gate "$STAGE7_ROUNDS/stage7_6_auto_graph_exploration/metrics/auto_graph_gate.json" \
  --claim-matrix "$STAGE7_G4/tables/claim_matrix.csv" \
  --rule "$STAGE7_G4/configs/g4_rule.json" \
  --output "$STAGE7_G4/metrics/g4_decision.json" \
  --report "$STAGE7_G4/reports/g4_decision.md"
```

输出：

```text
GO_STAGE8_REWARD_ONLY
REFINE_STAGE7_CORE_ONLY
STOP_PATHGRAPH
```

---

## 十、冻结 Stage 8 交接

创建：

```text
$STAGE7_G4/stage8_handoff.md
```

### GO 时至少写入

```text
final mode = reward_only
main reward = pathgraph_reward_v1_locked
main model bundle = persistent 3-seed ensemble
main graph = manual graph v1.0.1
auto graph = extension or skipped
policy evidence = secondary mixed
tables to reproduce
figures to reproduce
known limitations
large-file manifest
```

### STOP 时写入

```text
which core claim failed
which evidence remains usable
which artifacts should be archived
recommended next research direction
```

---

## 十一、冻结 G4 目录

复制锁文件：

```bash
cp "$STAGE7_INPUTS/locks/claim_scope_lock.json" \
  "$STAGE7_G4/locks/"
cp "$STAGE5_REWARD/configs/reward_selection_lock.json" \
  "$STAGE7_G4/locks/stage5_reward_selection_lock.json"
cp "$STAGE6R1_M4/locks/refine1_input_lock.json" \
  "$STAGE7_G4/locks/stage6r1_input_lock.json"
```

生成：

```bash
cat > "$STAGE7_G4/FROZEN.md" <<EOF
milestone = M5_REWARD_EVIDENCE
decision = $("$PYTHON_BIN" - <<'PY'
import json, os
p = os.path.join(
    os.environ["STAGE7_G4"],
    "metrics/g4_decision.json"
)
print(json.load(open(p))["decision"])
PY
)
mode = reward_only
main_reward = pathgraph_reward_v1_locked
manual_graph_is_main = true
policy_evidence = secondary_mixed
no_more_policy_training = true
no_post_test_main_reward_retuning = true
checkpoint_packaging = omitted_by_default
EOF
```

生成 portable checksum：

```bash
find "$STAGE7_G4" -type f \
  ! -name '*.pt' \
  ! -name '*.pth' \
  ! -name '*.ckpt' \
  ! -name '*.safetensors' \
  ! -name '*.npy' \
  ! -name '*.npz' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$STAGE7_G4/M5_REWARD_EVIDENCE_SHA256SUMS.txt"
```

这份 checksum 不得包含未打包大文件。

---

## 十二、生成本轮 ZIP

```bash
cp -r "$STAGE7_G4/metrics" "$ROUND_DIR/"
cp -r "$STAGE7_G4/tables" "$ROUND_DIR/"
cp -r "$STAGE7_G4/reports" "$ROUND_DIR/"
cp -r "$STAGE7_G4/configs" "$ROUND_DIR/"
cp -r "$STAGE7_G4/locks" "$ROUND_DIR/"
cp "$STAGE7_G4/FROZEN.md" "$ROUND_DIR/"
cp "$STAGE7_G4/M5_REWARD_EVIDENCE_SHA256SUMS.txt" \
  "$ROUND_DIR/checksums/"
cp "$STAGE7_G4/stage8_handoff.md" "$ROUND_DIR/reports/"
```

打包：

```bash
export ZIP_NAME="stage7_7_g4_freeze.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

---

## 十三、生成阶段总 ZIP

创建：

```text
tools/stage7/package_stage7_complete.py
```

总 ZIP 包含：

```text
Stage 7 input index and claim lock
各轮 gate 与 summary
reward main table
core ablation table
history/granularity table
coverage/scaling table
OOD/uncertainty table
auto graph decision
policy secondary evidence
claim matrix
G4 decision
Stage 8 handoff
各轮 ZIP SHA256
large-file manifests
```

不包含：

```text
checkpoint
逐帧 prediction
完整 stress traces
完整 embedding
缓存
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/package_stage7_complete.py" \
  --stage7-root "$STAGE7_ROOT" \
  --round-zip-dir "$STAGE7_DOWNLOADS" \
  --g4-root "$STAGE7_G4" \
  --output "$STAGE7_DOWNLOADS/stage7_complete.zip" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7_DOWNLOADS/stage7_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage7_complete_unzip_test.txt"

sha256sum "$STAGE7_DOWNLOADS/stage7_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage7_complete.sha256"
```

总 ZIP 生成后，不删除七个小阶段 ZIP。

---

## 十四、Agent 最终回复格式

### GO

```text
阶段 7 已完成。
G4：GO_STAGE8_REWARD_ONLY

alternative structural support：<true>
recovery structural support：<true>
history conclusion：<summary>
minimum stable coverage：<value>
graph complexity boundary：<summary>
OOD boundary：<summary>
uncertainty conclusion：<summary>
auto graph：<decision>
policy evidence：secondary mixed

唯一总交付 ZIP：
<absolute_path>/stage7_complete.zip

SHA256：
<hash>
```

### REFINE

```text
阶段 7 执行完成，但 G4：REFINE_STAGE7_CORE_ONLY
唯一允许修复项：<one concrete infrastructure issue>
不得调方法或返回 policy training。
```

### STOP

```text
阶段 7 已完成。
G4：STOP_PATHGRAPH
失败的核心主张：<list>
仍可保留的证据：<list>
唯一总交付 ZIP：<path>
SHA256：<hash>
```

**核心点：阶段 7 最终不是寻找更多有利结果，而是冻结哪些 reward-only 主张被证据支持、哪些必须删除。**

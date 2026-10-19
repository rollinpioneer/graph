# EG LQR V3 方法入口

当前状态：已切换到aita_xushijie@172.19.164.161（gpu04）。共享目录无需搬运；四帧线上/缓存、三头真实权重保存加载及56条原生评价等价检查通过，同硬件时延已测量。P1/P2状态另见阶段回执。沿用本卡分支，不新选权重或场景。

## 评分模块

`code/scoring.py`提供同一FrozenScorer接口：固定原1024候选几何，只对原动作计算Q；S直接使用Q，F05使用全帧float64总体标准差的1:1标准化融合。只使用候选、group256、score token64、原动作4维及冻结fit统计量，不读取评测真值。

`code/infer.py`提供已有特征缓存和原始EG数据/固定点云样本两条入口。两者都调用同一评分模块。线上入口复用原始EG模型和数据加载器，捕获中间特征；不生成额外姿态。原始预处理仍读取基准的seg/meta等文件，未声称整个系统完全不依赖标注。

以下为同一评分模块的两种入口。单实例固定种子20261018；三个头均值不是该实例的单模型成绩。输出文件名需未存在。

```bash
export GRASP_ROOT=/aita_data/home/aita_xushijie/work/grasp
export LQR_ROOT="$GRASP_ROOT/eg_lqr_build_confirm_v3_20261019"
source "$LQR_ROOT/activate_gpu04.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 固定种子20261018的原始缓存入口
python "$LQR_ROOT/code/infer.py" \
  --head "$GRASP_ROOT/eg_pose_select_pilot_v1_20261018/heads/20261018/step1000.pt" \
  --seed 20261018 --family S \
  --cache "$GRASP_ROOT/eg_nqs_pilot_v2_20261018/exports/subset/scene_0037/0000.npz" \
  --output "$LQR_ROOT/preflight/cache_0037_0000.npz"

# 同一原始点云样本的线上EG入口：GPU编号需先检查当前资源
CUDA_VISIBLE_DEVICES=<当前获准GPU> python "$LQR_ROOT/code/infer.py" \
  --head "$GRASP_ROOT/eg_pose_select_pilot_v1_20261018/heads/20261018/step1000.pt" \
  --seed 20261018 --family S \
  --eg-code "$GRASP_ROOT/eg_sub25_v1_20261015/runtime/EconomicGrasp" \
  --dataset-root "$GRASP_ROOT/eg_sub25_v1_20261015/data/graspnet" \
  --backbone "$GRASP_ROOT/eg_sub25_v1_20261015/runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar" \
  --scene 37 --frame 0 \
  --input-sample "$GRASP_ROOT/eg_nqs_pilot_v2_20261018/inputs/scene_0037/0000.npz" \
  --device cuda:0 --verify-hook \
  --output "$LQR_ROOT/preflight/online_0037_0000.npz"
```

评价入口已对接既有原生适配器，evaluate.py始终重走完整1024候选的NMS/物体分配与质量计算；run_stage.py管理并行与预算，report.py执行预登记的配对统计和S优先规则。不得用旧Top50简单重排替代完整评价。

```bash
CUDA_VISIBLE_DEVICES='' python "$LQR_ROOT/code/evaluate.py" --stage P1 --scene 104
CUDA_VISIBLE_DEVICES='' python "$LQR_ROOT/code/run_stage.py" --stage P1 --workers 7
```

新节点独立venv继承原PyTorch2.5.1+cu121/MinkowskiEngine0.5.4，覆盖Open3D0.19以兼容GLIBC2.34，并补齐其导入依赖；原共享环境没有修改。兼容后固定P0结果精确一致。

同硬件时延使用固定输入、batch1和GPU同步；头位于单线程CPU，EG位于GPU。EG+score+CD计时包含特征传输、归一化/写回和原始点云检测器构建，未含磁盘I/O、NMS/官方评价或机器人执行，不能称跨硬件加速倍数。

## 继续执行时的顺序

1. 当前计算节点为gpu04；每轮重新核验当前代码/资源，复用单个保活SSH。原数据、权重与历史结果直接读取共享目录。
2. 四个P0帧已通过。其余环境或源码变动仍需做针对性等价核验，不覆盖历史结果。
3. 初始原生P0使用4个低优先级单线程worker；P1可按当前负载扩至7个worker加1个控制进程，总计算进程最多8。无新训练。
4. 按S优先规则生成method_lock.json。若两族均不满足，停止并报告迁移负结果。
5. 只在锁定后审计C1已知性能暴露；已暴露就暂停C1，不换场景。不把以前的冻结清单当作当前未暴露证明。
6. 仅在允许的条件下执行P2并做预登记统计、重复观测敏感性与同硬件时延。

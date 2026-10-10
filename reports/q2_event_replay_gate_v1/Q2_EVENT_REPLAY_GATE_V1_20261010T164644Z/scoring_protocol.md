# Q2 受控评分与搜索协议

协议ID为CPU_FP32_NATIVE_ONE_STATE_DETERMINISTIC_V1。这是新受控协议：原GPU、最多48图一批改为CPU、每次真实调用只输入一个图。三权重、状态表示和原生算术不改；使用原cli_ext.make_evaluator→NeuralEval.evaluate([state])→PddlSerialModel.state_values，没有切换FastValue算术，没有取整、量化、均值或容差并列。低分优先，实际插入序号FIFO，FP32精确相等为并列。

D0@820、D0@2460与T1-S@2460（工厂名T1@2460）的完整SHA见registration.json。原load_checkpoint核对sidecar哈希后严格载入整个model状态；没有载入optimizer。全部参数CPU/FP32，eval、requires_grad=False、inference_mode；Torch intra/inter-op线程各1，确定性算法开启，matmul precision highest。运行Python3.10.19、Torch2.7.1、PyG2.6.1，其余依赖见runtime.json。CUDA_VISIBLE_DEVICES为空，本轮GPU为0。

缓存完整语义键为任务SHA、域SHA、全状态bitmask、模型SHA、协议ID。快照用身份元数据作为外层命名空间，state hex作为内层键，值为原始小端FP32字节。恢复原样复制；未来新状态仍使用相同原生单状态函数。非D0预筛用独立诊断值表，后续D0搜索不读取它。资格三轮对完整有效OPEN、三模型逐状态真实前向，绕过所有状态评分缓存；第三轮反向请求顺序。已观察的1062个任务—模型—状态键重复值均逐位一致，不外推跨设备、批大小或未知环境。

任务模板、图边和目标静态关系缓存从固定任务、源码、权重重建，不存状态评分。NeuralEval的n/calls/infer_seconds仅用于性能统计，不参与分数或选择。本次dense模式不进入rand掩码分支；构造seed固定、随后严格载入全权重，受控前向无随机抽样。因此未使用的RNG状态、Python对象地址不属于语义快照。预算计时是独立截止条件，由登记和账本绑定，不参与队列排序。本轮重复使用同一冻结模型实例进行真实前向；未单独测试另起模型进程的一致性。

搜索保留原eager GBFS：动作索引次序，完整状态重复检测，CLOSED不重开，OPEN较小g只更新父链接且不重评分，生成后继时检查目标，所有新后继评分/入队完成后才再次出队。三任务根均非目标。快照保存任务/域/模型/协议身份、实际heap数组及原始键/有效标记/插入序号、CLOSED、全部node父指针/g/action、serial、生成/重复/路径更新/评分批次计数、root/best、完整D0缓存及缓存计数、空pending_scores、status/plan。无异步队列或延迟评分；429个项目源码SHA绑定实现。

G2先反序列化并核对完整规范现场，再比较下一出队和基线后缀。p12为62步至前缀64步上限，Struct为6步至解出；每步完整dict内容相等，包括实际heap排列、所有父记录、缓存和计数，独立复核实际最小队列键。没有仅用摘要或Jaccard替代。三份快照全部248条父动作边合法、回到根；两份已有完整计划通过动作合同回放，最优性未查询。

旧p12 k16只在保留旧逻辑现场的独立副本重赋177个OPEN分数，历史root/best和累计计数仍为旧元数据，不冒充全程CPU轨迹。D0仍选151，T1由134变132；历史关系稳定而具体旧事件身份改变。该副本未做G2、未选入阶段C；新p12 k2才是当前协议下实际根搜索的事件。

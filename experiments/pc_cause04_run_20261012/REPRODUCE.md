# PC-CAUSE-04 CPU复核与测量自检复现

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

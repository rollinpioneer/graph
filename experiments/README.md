# CP-DISR v2.1 Experimental Agent Kit

主文件为 `docs/CP_DISR_v2.1_Minimal_Experimental_Execution_Plan.md`。
本包根目录可以作为EXPERIMENT_ROOT；手册的 `experiments/` 是这一根目录的概念名称，不要求重命名用户已有目录。

包含完整手册、17张自包含Stage卡、Experiment Manifest v0、17份Stage配置、七方法配置、71个默认新增训练job的计划矩阵（另含9行复用Full索引）、状态/selection模板和只读计划/预检工具。

`configs/planned_jobs.*` 是计划，不是实验结果。`stage_cards`包含共同协议，离线可读。

本包没有完整PPO训练器、相机驱动、机器人controller、真实task资产、VLM缓存或训练checkpoint。0A必须绑定/实现这些资源，不能把建议CLI当已存在程序。`pyproject.proposed.toml` 不是已成功解析的uv.lock。

已有v2.1原件逐字复制到sources。本文只改实验执行规范，不覆盖方法原件。

可运行的准备工具：
```bash
python tools/plan_stage.py --stage 1A
python tools/preflight.py --root . --context conversation_sandbox
python tools/validate_kit.py
```
这些工具不训练、不运行机器人、不调用VLM。实际Stage由Agent读取对应Stage卡、检查绑定并在真实项目执行。每次仅执行用户授权的Stage或Part。

`part_0_validation/stage_0a/`保存本轮仅针对聊天沙箱的资料/软件预检；不能当用户训练机验收。其他Stage保持NOT_STARTED。

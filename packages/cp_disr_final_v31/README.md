# CP-DISR Final Adjudication Package

**文档3.1 / Method2.1.1 / Experimental Plan1.1。**

入口：`review/CP_DISR_Independent_Adjudication_v1.md`（25节最终裁决）；`research/CP_DISR_Paper_Oriented_Research_Specification_v3.1.md`（完整研究维护源）；`experiments/Experimental_Plan_v1.1.md`（执行规范）。

原v3.0和Claude审查稿保留于sources。图/DP/Full Actor/schema不变；训练prefix=false、时长半衰期、B1-K、两层Set Transformer B0、A_CAT和分阶段预算是本轮明确修订。新层表和比较方法配置不应误当原source事实。

核心第一暂停点11新训练；三seed主阶梯及CAT累计29；含Q32；A_B需要时35。所有planned rows为NOT_STARTED，所有实际机器/资产/安全/API条件仍需绑定。不包含已经实现的机器人PPO训练器，不会自动发起训练。

`validation/check_algebra.py`仅验证代数见证/配置，不是生产R-GCN、PPO、VLM或机器人验收。`validation/validate_package.py`检查文档/manifest/source。阶段生产单测在0B另行实现运行。

Markdown为本次维护版，没有生成声称v3.1的旧Word副本；原Word若继续使用必须重新按v3.1同步。图目录只提供绘图规范，不包含虚构真实场景或实验曲线。

来源核验ledger区分全文相关段落、摘要主页和访问未确认；未确认OpenReview条目不作已证据。所有实验收益仍NOT_MEASURED。

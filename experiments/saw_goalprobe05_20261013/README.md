# SAW-GOALPROBE-05

完整结论见 REPRODUCTION_REPORT.md 与 HYPOTHESIS_VERDICT.md。冻结设计见 EXPERIMENT_CARD.md、PREREGISTRATION.json，原始用户设计稿见 CARD_SUPPLIED.md。

运行阶段：CPU环境与4核affinity固定 → prepare.py（不加载策略、冻结场景和物理准入）→ execute.py（只读模型和历史输出检查）→ analyze.py（簇bootstrap及图）→ audit.py（只读轨迹重算）。本次prepare导出错误由recover_export.py复用已有准入；analysis的NaN导出由finalize_analysis.py完成报告，科学CSV/图不重算。run_phase.py只监控本轮自己的子进程。

所有运行必须CUDA_VISIBLE_DEVICES空值、JAX_PLATFORMS=cpu、taskset -c 72-75；统计依赖通过uv pip install --python <原只读解释器> --target <本轮vendor/analysis_deps> --no-deps -r requirements_analysis.txt安装，并加入PYTHONPATH；不能安装覆盖原环境。首次运行创建独立输出目录；已有状态/输出拒绝直接重跑，不删除断言或覆盖旧结果。

这是CPU诊断而非训练入口。没有自动追加、优化种子或GCIVL训练。

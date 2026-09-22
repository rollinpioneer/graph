# Stage 0B — Production Mathematical and Interface Invariants (Plan v1.1 / Method 2.1.1)

Status: `PASS`.

Interpreter: `.venv-stage0a/bin/python`. PYTHONPATH=src. Package algebra check_algebra.py was not used as a substitute.

Ignored collection-only file: `tests/test_deadline_semantics.py` (robosuite import at collection). Not counted as a skipped Method invariant.

pytest returncode=0 duration=18.6s

```
........................................................................ [ 85%]
............                                                             [100%]
=============================== warnings summary ===============================
tests/test_stage0c_pipeline.py::test_provider_sdk_dispatch_no_key_read
  /home/__compress_data/xushijie/graph_cp_disr_v2_1/.venv-stage0a/lib/python3.10/site-packages/dashscope/__init__.py:63: DeprecationWarning: The Assistants API (dashscope.assistants) is deprecated and will be removed in a future release. Please migrate to the Responses API. See https://help.aliyun.com/zh/model-studio/synchronous-call-api-reference for migration details.
    from dashscope.assistants import Assistant, AssistantList, Assistants

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
84 passed, 1 warning in 17.35s

```

New-profile RL training runs this stage: **0**.


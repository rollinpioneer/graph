# 重复性核查

比较每个完整检查点的实际内容，包括队列物理布局、父指针、g/action、CLOSED、评分缓存和计数。时间截断长度差单列。

## struct_n4_004_k008

```json
{
  "event": "struct_n4_004_k008",
  "A_vs_A_repeat": {
    "common_identical_complete_steps": 6,
    "first_full_content_mismatch_zero_based": null,
    "PASS_common_prefix": true
  },
  "B_vs_B_repeat": {
    "common_identical_complete_steps": 6,
    "first_full_content_mismatch_zero_based": null,
    "PASS_common_prefix": true
  },
  "old_suffix_steps": 6,
  "old_suffix": {
    "A": {
      "steps_compared": 6,
      "first_mismatch": null,
      "all_available_old_suffix_reached": true,
      "PASS_common_prefix": true
    },
    "A_repeat": {
      "steps_compared": 6,
      "first_mismatch": null,
      "all_available_old_suffix_reached": true,
      "PASS_common_prefix": true
    }
  },
  "technical_validity": "PASS"
}
```

## ipc_p12_k002

```json
{
  "event": "ipc_p12_k002",
  "A_vs_A_repeat": {
    "common_identical_complete_steps": 110,
    "first_full_content_mismatch_zero_based": null,
    "PASS_common_prefix": true
  },
  "B_vs_B_repeat": {
    "common_identical_complete_steps": 108,
    "first_full_content_mismatch_zero_based": null,
    "PASS_common_prefix": true
  },
  "old_suffix_steps": 62,
  "old_suffix": {
    "A": {
      "steps_compared": 62,
      "first_mismatch": null,
      "all_available_old_suffix_reached": true,
      "PASS_common_prefix": true
    },
    "A_repeat": {
      "steps_compared": 62,
      "first_mismatch": null,
      "all_available_old_suffix_reached": true,
      "PASS_common_prefix": true
    }
  },
  "technical_validity": "PASS"
}
```

# CUPID Stage 2B 决策报告

## 输入审计

- Rollout：100
- Demo：192
- 成功 / 失败：71 / 29
- 官方分数重建最大误差：7.1316965e-06

## 重要限制

- 完整 100 条只是有限池参考，不是真实价值。
- 独立 50/50 池也有噪声，因此只用于压力测试。
- Bootstrap 删除概率是离线稳定性指标，不是严格置信证书。
- 本阶段不把低分 Demo 直接解释成有害 Demo。
- 本阶段不允许自动启动重训练。

## 底部 20% 边界

```text
 proportion  delete_k  last_delete_original_demo  first_keep_original_demo  full100_boundary_gap  pairwise_gap_standard_error  gap_divided_by_standard_error  demos_within_one_standard_error_of_boundary  demos_within_two_standard_errors_of_boundary  bootstrap_stable_delete_count_p_ge_0.90  bootstrap_stable_keep_count_p_le_0.10  bootstrap_ambiguous_count
        0.2        38                        250                       127                0.0002                     0.067776                       0.002956                                           57                                           121                                        7                                    116                         69
```

## 50 条 Rollout 下的底部 20% 稳定性

```text
 budget  proportion  delete_k  stable_delete_count_p_ge_0.90  stable_keep_count_p_le_0.10  ambiguous_count_p_between_0.10_and_0.90  strongly_ambiguous_count_p_between_0.20_and_0.80
     50         0.2        38                              5                          114                                       73                                                46
```

## 50/50 独立池结果

```text
                           method  evaluations  nonempty_rate  selected_count_mean  selected_count_median  precision_mean  precision_std  recall_mean  jaccard_mean  independent_rank_percentile_mean  independent_negative_score_rate_mean
                   fixed_bottom10          200           1.00               19.000                   19.0        0.542895       0.103607     0.271447      0.223116                          0.292318                              0.695263
                   fixed_bottom20          200           1.00               38.000                   38.0        0.422105       0.062644     0.422105      0.269509                          0.357269                              0.607105
matched_size_bottom_for_p_ge_0.80          200           1.00               11.005                   11.0        0.597905       0.149923     0.169342      0.151785                          0.261895                              0.734731
matched_size_bottom_for_p_ge_0.90          200           1.00                5.290                    5.0        0.631562       0.238420     0.083553      0.079280                          0.247305                              0.755652
matched_size_bottom_for_p_ge_0.95          200           0.95                2.495                    2.0        0.548246       0.365221     0.036184      0.035224                          0.294784                              0.683289
            stable_core_p_ge_0.80          200           1.00               11.005                   11.0        0.530887       0.149273     0.151447      0.133760                          0.304385                              0.674298
            stable_core_p_ge_0.90          200           1.00                5.290                    5.0        0.526538       0.245837     0.070789      0.066397                          0.297123                              0.693886
            stable_core_p_ge_0.95          200           0.95                2.495                    2.0        0.488246       0.364511     0.030526      0.029603                          0.308594                              0.658202
```

## 决策诊断

```json
{
  "core90_nonempty_rate": 1.0,
  "core90_selected_count_mean": 5.29,
  "core90_precision_mean": 0.5265376984126984,
  "fixed20_precision_mean": 0.4221052631578947,
  "matched90_precision_mean": 0.6315616883116882,
  "precision_gain_over_fixed20": 0.10443243525480367,
  "precision_gain_over_matched_size": -0.10502398989898987,
  "core90_independent_rank_percentile_mean": 0.29712314551767677,
  "stable_delete_count_at_budget50_bottom20": 5,
  "core80_nonempty_rate": 1.0,
  "core80_selected_count_mean": 11.005,
  "core80_precision_mean": 0.5308867386698268,
  "core80_precision_gain_over_fixed20": 0.10878147551193212
}
```

## 最终决策

**FAIL_SQUARE_FIXED_FILTER_BRANCH_CONSIDER_TRANSPORT**

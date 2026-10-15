# CUPID Stage 3 Transport-MH 稳定性决策报告

## 输入审计

- Rollout：100
- Demo：192
- 成功 / 失败：44 / 56
- 官方分数重建最大误差：1.468116e-06

## 重要限制

- 完整 100 条只是有限池参考，不是真实价值。
- 独立 50/50 池也有噪声，因此只用于压力测试。
- Bootstrap 删除概率是离线稳定性指标，不是严格置信证书。
- 本阶段不把低分 Demo 直接解释成有害 Demo。
- 本阶段不允许自动启动重训练。

## 底部 20% 边界

```text
 proportion  delete_k  last_delete_original_demo  first_keep_original_demo  full100_boundary_gap  pairwise_gap_standard_error  gap_divided_by_standard_error  demos_within_one_standard_error_of_boundary  demos_within_two_standard_errors_of_boundary  bootstrap_stable_delete_count_p_ge_0.90  bootstrap_stable_keep_count_p_le_0.10  bootstrap_ambiguous_count
        0.2        38                        256                       154              0.000471                     0.017961                       0.026208                                           41                                            75                                       20                                    130                         42
```

## 50 条 Rollout 下的底部 20% 稳定性

```text
 budget  proportion  delete_k  stable_delete_count_p_ge_0.90  stable_keep_count_p_le_0.10  ambiguous_count_p_between_0.10_and_0.90  strongly_ambiguous_count_p_between_0.20_and_0.80
     50         0.2        38                             18                          130                                       44                                                33
```

## 50/50 独立池结果

```text
                           method  evaluations  nonempty_rate  selected_count_mean  selected_count_median  precision_mean  precision_std  recall_mean  jaccard_mean  independent_rank_percentile_mean  independent_negative_score_rate_mean  paired_precision_gain_over_fixed20_mean  paired_precision_gain_over_matched_size_mean
                   fixed_bottom10          200            1.0               19.000                   19.0        0.810000       0.079984     0.405000      0.371653                          0.136456                              0.944474                                 0.000000                                      0.000000
                   fixed_bottom20          200            1.0               38.000                   38.0        0.621316       0.054310     0.621316      0.452890                          0.206786                              0.909737                                 0.000000                                      0.000000
matched_size_bottom_for_p_ge_0.80          200            1.0               18.585                   18.5        0.815612       0.086851     0.397632      0.365485                          0.133983                              0.947113                                 0.000000                                      0.000000
matched_size_bottom_for_p_ge_0.90          200            1.0               12.845                   13.0        0.882175       0.097423     0.296974      0.285765                          0.106428                              0.968325                                 0.000000                                      0.000000
matched_size_bottom_for_p_ge_0.95          200            1.0                9.215                    9.0        0.909992       0.101448     0.219474      0.214632                          0.093463                              0.980541                                 0.000000                                      0.000000
            stable_core_p_ge_0.80          200            1.0               18.585                   18.5        0.815661       0.089389     0.397105      0.364893                          0.130092                              0.963692                                 0.194345                                      0.000048
            stable_core_p_ge_0.90          200            1.0               12.845                   13.0        0.870824       0.097180     0.293026      0.280842                          0.107590                              0.984233                                 0.249508                                     -0.011352
            stable_core_p_ge_0.95          200            1.0                9.215                    9.0        0.880731       0.113983     0.212763      0.206836                          0.102960                              0.988631                                 0.259415                                     -0.029260
```

## 决策诊断

```json
{
  "core90_nonempty_rate": 1.0,
  "core90_selected_count_mean": 12.845,
  "core90_precision_mean": 0.8708235198753775,
  "fixed20_precision_mean": 0.6213157894736843,
  "matched90_precision_mean": 0.8821754225736425,
  "precision_gain_over_fixed20": 0.24950773040169325,
  "precision_gain_over_matched_size": -0.011351902698264933,
  "core90_independent_rank_percentile_mean": 0.10758975783839127,
  "stable_delete_count_at_budget50_bottom20": 18
}
```

## 最终决策

**PASS_VARIABLE_K_DIAGNOSIS_BOOTSTRAP_MEMBERSHIP_NOT_PROVEN**

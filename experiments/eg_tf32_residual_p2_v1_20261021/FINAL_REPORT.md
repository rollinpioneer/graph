# P2 frozen TF32 plus affine residual: complete C3 report

Decision: E_ACT_FAIL; overall STOP_DIVISION_KEEP_G.

| Method (CD) | AP | Top-1 % | Nonempty collision % |
|---|---:|---:|---:|
| EG | 40.4409 | 44.3685 | 21.1589 |
| O | 42.2805 | 42.8277 | 16.1675 |
| G | 44.7718 | 47.5260 | 12.7713 |
| Eact | 44.8352 | 46.6146 | 11.2847 |
| Epot | 45.2379 | 48.7196 | 10.1671 |
| Gcap65 | 44.8324 | 47.5803 | 13.0859 |

| Comparison (CD) | delta AP [95% CI] | delta Top-1 pp [95% CI] |
|---|---|---|
| Eact-G | +0.0634 [-0.43503614578793787, 0.5134155919005431] | -0.9115 [-3.168402777777778, 1.7363823784722183] |
| Epot-G | +0.4661 [0.2103894863658507, 0.7301029733984422] | +1.1936 [-0.14105902777777787, 2.647569444444444] |
| Gcap65-G | +0.0607 [-0.03896097528641661, 0.1534931612443451] | +0.0543 [-1.0850694444444446, 1.1395941840277737] |
| Eact-Epot | -0.4027 [-0.7499224924264951, -0.11116965217670022] | -2.1050 [-4.155815972222222, -0.05425347222222229] |
| Eact-Gcap65 | +0.0027 [-0.49943986699515586, 0.4735820166712781] | -0.9657 [-3.6024305555555554, 1.6927083333333328] |

All nine heads ran 1000 AdamW steps, lr0.001/wd0.0001, 128 a0 groups per step. Each has exactly 65 trainable residual parameters, initialized to zero; sample-stream hashes match corresponding protected G. Eact/Epot use alpha*(cached historical TF32 score + residual)+beta; alpha/beta use fit-only target moments. Gcap65 uses sigmoid(frozen G logit + residual). All residual arithmetic is float64; frozen G forward remains float32. No clipping, tuning, extra seeds, extended training or method combinations.

This is an explicitly authorized change of function class, not a numerical repair of the original pure-affine hypothesis. The original pure-affine F1 remains failed. Prior exact TF32 replay evidence was reused by verified hash; all accessed export identities were checked again. Zero-training I1-I4 passed before any training. See invariance_checks.json.

Full C3: 12 scenes x256 frames, CD/noCD. Official selection/quality pilot checked. Historical EG/O/G selections and metrics exactly reproduced every frame. P0 union and existing P1 labels are hash-verified, overlap-checked and reused; newly needed GT is evaluation-only. No C4/C5/reserve data or new training labels.

Same-seed differences; average seeds within scene, then Seen/Similar/Novel stratified paired bootstrap10000, seed20261021. Repeatedly used C3 is development evidence, not independent confirmation. Only preregistered sequential comparisons reached by earlier passing gates support mechanism claims; controls are never promoted.

{
  "status": "E_ACT_FAIL",
  "route": null,
  "performance_gates": {
    "Eact": {
      "AP_route": false,
      "Top1_route": false,
      "collision_protection_triggered": false,
      "passed": false
    },
    "Epot": {
      "AP_route": false,
      "Top1_route": false,
      "collision_protection_triggered": false,
      "passed": false
    },
    "Gcap65": {
      "AP_route": false,
      "Top1_route": false,
      "collision_protection_triggered": false,
      "passed": false
    }
  },
  "sequential_Eact_vs_Epot": null,
  "sequential_Eact_vs_Gcap65": null,
  "untested_sequential_steps_are_not_claimed": true,
  "only_E_ACT_FULL_counts": true,
  "overall": "STOP_DIVISION_KEEP_G",
  "controls_not_promoted": true,
  "C3_is_development": true,
  "C4_read": false,
  "redesign": "frozen TF32 score plus 65-parameter affine residual; not original pure affine model"
}

{
  "status": "PASS",
  "frames": 3072,
  "configurations": 32,
  "records": 98304,
  "old_EG_O_G_selection_and_metrics_exact": true,
  "selected_quality_finite": true,
  "transitions_net_matches_metric": true,
  "protected_G_and_feature_caches_unchanged": true,
  "training_began_after_all_invariants_passed": true,
  "C4_read": false,
  "new_training_labels": 0,
  "new_evaluation_quality_calls": 28134,
  "CPU_worker_hours": 1.5741732049650616,
  "CPU_process_hours": 1.1205099947038886,
  "actual_evaluation_peak_processes": 58,
  "GPU_process_hours": 0.01456211699379815,
  "new_bytes": 130940819,
  "outside_01_fraction": {
    "Eact": 0.020071453518337674,
    "Epot": 0.011607276068793403
  }
}

Offline fit/held-out MSE, decision AUROC and OLS convergence gaps are diagnostic only, with no refitting or deployment of OLS. Training/evaluation caches and checkpoints remain on servers.
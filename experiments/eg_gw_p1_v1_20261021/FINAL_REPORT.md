# G-W P1 complete C3 report

Decision: NOT_PASSED; overall STOP_DIVISION_KEEP_G. P2 blocked at F1, not trained or evaluated.

| Method (CD) | AP | Top-1 % | Nonempty collision % |
|---|---:|---:|---:|
| EG | 40.4409 | 44.3685 | 21.1589 |
| O | 42.2805 | 42.8277 | 16.1675 |
| G | 44.7718 | 47.5260 | 12.7713 |
| GW | 44.5543 | 46.7665 | 12.5326 |

GW-G AP -0.2175, 95% CI [-0.3180750772476328, -0.12063332062655949]; Top-1 -0.7595pp, 95% CI [-1.4214409722222223, -0.03255208333333336].
AP route False; Top-1 route False; collision protection False.

Top-1 paired units 9216: changed 3086, recovered 334, regressed 404, net -70. Three seeds are not independent scene samples.

All six fixed-step training tasks completed on A100. Original G reruns are tensor-byte identical to protected G; no rerun C3 needed. GW initialization and group sample streams match G. Only the prescribed loss weights changed. Checkpoint step1000 only; no tuning, extra seeds or steps.

C3 evaluation: 12 scenes x256 frames, CD and noCD, official selector/quality checks in pilot; original EG/O/G selections and metrics exactly reproduced for every frame. Newly needed quality labels used only for evaluation; union checked first. Same-seed comparisons, seed-average within scene, Seen/Similar/Novel stratified bootstrap 10000, seed20261021.

C3 has been repeatedly used for development. These intervals are developmental evidence, not independent confirmation. P2 F1 max error 0.00321946 exceeds frozen1e-4; the mathematical affine readout premise does not guarantee this numerical equivalence in existing exports. No redesign or tolerance relaxation performed.

Resources and validation:
{
  "status": "PASS",
  "frames": 3072,
  "configurations": 20,
  "records": 61440,
  "old_EG_O_G_selection_and_metrics_exact": true,
  "selected_quality_finite": true,
  "transitions_net_matches_metric": true,
  "three_G_reruns_byte_exact": true,
  "initial_and_sample_stream_match": true,
  "C4_read": false,
  "new_training_labels": 0,
  "new_evaluation_quality_calls": 7918,
  "CPU_worker_hours": 1.0007024553087023,
  "CPU_process_hours": 0.5456353817597221,
  "actual_evaluation_peak_processes": 58,
  "GPU_process_hours": 0.020896979967753092,
  "new_bytes": 72926045
}

Stop the tested loss-weight recipe if not passed. Preserve G. Do not combine methods or open C4/C5. P2 can only resume after separately approved numerical redesign.
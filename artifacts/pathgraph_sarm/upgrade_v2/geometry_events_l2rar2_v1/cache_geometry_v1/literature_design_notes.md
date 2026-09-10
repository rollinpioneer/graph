# Literature design notes (design inspiration only)

- [L1] Calandra et al. 2017, The Feeling of Success - vision and touch are
  complementary for grasp outcome prediction. This round introduces no
  tactile network and reuses no training/label conclusion.
- [L2] Dong et al., Maintaining Grasps within Slipping Bound - relative
  displacement fields motivate 'contact can persist while slipping'.
  Incipient slip is not a task-loss label.
- [L3] robosuite task docs / stack environment - lift height is used for
  task-specific stage success, not as a general stable-hold label. This
  round uses relative initial height.
- [L4] MuJoCo 3.4.0 simulation docs - state/warmstart/control relations
  inform the (unused, unauthorised) R14 replay record contract.

No paper accuracy is reused as an expected result for this cache.

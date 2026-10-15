# Evaluator contract comparison

- old ignores `role_condition` and can misclassify horizon.
- new executes guards, returns `guard_ambiguous`, and censors horizon.

# Public validation recommendation

Scope: LIBERO-Spatial, -Object, -Goal and -10 as frozen by the upstream task map (LIBERO-90 not audited).

- total tasks: 40; exactly supported: 0; supported with new binding: 0; unsupported: 40
- coverage rate (strict): 0.000; upper bound if spawn orientation were verified: 0.150
- unsupported reasons: {"NEW_SCRIPTED_SKILL_REQUIRED": 36, "NEW_LOW_LEVEL_CONTROLLER_REQUIRED": 4}
- per suite: {"libero_spatial": {"total": 10, "exact_or_binding": 0, "upper_bound": 0}, "libero_object": {"total": 10, "exact_or_binding": 0, "upper_bound": 4}, "libero_goal": {"total": 10, "exact_or_binding": 0, "upper_bound": 1}, "libero_10": {"total": 10, "exact_or_binding": 0, "upper_bound": 1}}

Label rule: UNMODIFIED_LIBERO_PUBLIC_VALIDATION needs coverage >= 0.80 of the frozen scope, official goals/inits/identities unchanged and a platform binding; otherwise LIBERO_COMPATIBLE_SUBSET; modified tasks are LIBERO_DERIVED_DIAGNOSTIC

Result: **LIBERO_COMPATIBLE_SUBSET** is the only label this scope could carry today, so the unmodified public tasks cannot be called a LIBERO benchmark result.

Why the public tasks are poor mechanism carriers as well as poor coverage: most are single-goal chains, so Full, B2 and the contract planner see the same one-step plan and the comparison cannot separate priors.

What a future public-validation run needs (not authorised here): a LIBERO environment binding (new), perception for textured objects, a BDDL-based Evaluator, handle/knob/rim skills, and an explicit statement of which tasks are covered. Report coverage per suite and never extrapolate to the whole benchmark.

# P1 continuation v2

## Consumer trace

G1 is compiled by compile_graphs, consumed by predicate inference and execute_graphs for branch and coverage diagnostics, while scripted high-level actions are produced independently by scripted_controller.action_program and executed by DynamicTabletop.perform. No G1-to-controller dispatch or G1-to-reward numeric binding exists. fixed_manipulation_path grants implicit coverage for canonical actions, so coverage is not evidence of explicit edge completeness.

## Counterexample reachability

The raw clipped cycle is a legal path in the retained legacy transport graph (in_transit -> dropped_or_misaligned -> recovery -> grasped -> in_transit). It remains a positive signed-return sensitivity result under unit-scale costs, but the locked normalized shortest-path contract makes its signed return approximately zero. The phi-reset cycle is not exactly legal in the retained graph: there is no failure/recovery pair returning to the same logical node. Its synthetic +0.5 result is retained conditionally for future runtime expansions and was not removed or parameter-tuned.

## Causal replay

One complete pilot rollout was located through the original external-artifact manifest and replayed from current-frame online/contact/oracle state. The adapter used no final outcome, scenario label, or future field; prefix outputs at 4/7/10/13 frames were stable. The normalized JSONL keeps unresolved same-node holds explicit and binds only legacy edges with evidence.

No frozen reward engine/configuration was modified. No new physics, training, or LLM calls were made.

## One-trajectory reward replay

The frozen scorer ran on the same 12 causal transitions with engine parity error 0.0. Signed returns were: SPARSE_TERMINAL 1.0, LINEAR_FIXED_ORDER 1.0, UNORDERED_COMPLETION_PROGRESS 1.0, GRAPH_COST_ONLY 0.6666666667, GRAPH_COST_PHI 1.1573635343, GRAPH_COST_DEBT 0.6666666667, FULL_FROZEN 1.1573635343, and NO_RECOVERY_CREDIT 1.1573635343. This is one existing pilot replay, not a new physical experiment or independent confirmation.

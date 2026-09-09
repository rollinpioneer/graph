# Attempt-boundary diagnostic

- Status: `L2RAR1_PARTIAL_KEEP_G1`; L3 and R4 remain closed.
- Scope: `dev_select` cached cases only; `16` cases, no physical re-execution.
- At first contact loss, the common online state is identical across the missed-grasp and short-touch groups.
- The already-arrived low-level command target is stationary for all missed-grasp cases and moving for all short-touch cases in this fixed cache: `True`.

## Interpretation

The existing hold/event interface cannot announce `attempt_end` from its current predicate set alone. The command trace contains a potentially useful causal phase signal, but its perfect separation here is limited to this fixed development cache and must not be promoted directly into a threshold-tuned retry rule.

The smallest next interface change is to expose an explicit controller-side attempt phase or attempt-end marker derived from command lifecycle. Keep `hold_unknown` separate from `retryable_missed_grasp`; do not remove short-contact protection, lower thresholds, use future action names, or enter L3/R4.

## Files

- `attempt_boundary_diagnostic.json`: machine-readable result and case rows.
- `attempt_boundary_cases.csv`: compact per-case comparison table.

# Evaluator contract hardening

- evaluator tests: `26 passed`
- scientific rerun: `false`
- API calls: `0`
- API key read: `false`
- training jobs: `0`
- explicit runtime guard fields: `terminal_failure_event`, `stable_success_event`
- diagnostic/label fields remain excluded from online context
- fresh confirmation metrics: unchanged

The hardening makes the required true/false/ambiguous conditional-role cases executable while preventing legacy `horizon`, `nonterminal`, `terminal_*`, and event-log fields from activating guards unless an explicit runtime-observable field list authorizes the supported guard field.

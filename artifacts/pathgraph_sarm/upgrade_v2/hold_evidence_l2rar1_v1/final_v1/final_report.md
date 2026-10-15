# L2RA-R1 final report

- Scientific status: **L2RAR1_PARTIAL_KEEP_G1**
- Retained graph: `G1_predicate_bound`
- Development route: `DEVELOPMENT_NOT_READY`; selected candidate: `None`
- Confirmation: `NOT_RUN_DEVELOPMENT_FAILED`; standard `NOT_RUN`, challenge `NOT_RUN`
- Primary evidence: old action-end cases plus newly generated R1 control-tick development data where collection completed.
- The direction-aware and time-aware rules remain proxy measurements in a fixed MuJoCo family.
- API calls: `0`; training jobs: `0`; API keys read: `false`.
- L3 entry: `false`; the formal retained graph remains G1.

The R1 entry replay supports the code-level diagnosis that old co-motion discards direction, but historical action-end observations alone cannot establish that every false positive or missed short hold is a sampling failure.

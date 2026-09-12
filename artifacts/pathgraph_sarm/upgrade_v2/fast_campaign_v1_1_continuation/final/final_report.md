# L2RAR2 Fast Campaign V1.1 Continuation

The parent V1 stop is preserved. Its A2 grant was consumed, but no model construction or physics step began. V1.1 reuses the approved Campaign and performs environment preflight before issuing a new A2 grant.

## Continuation result (2026-09-12)

- A2 retry: PASS, 16/16 main gates, one physical instance.
- B retry: PASS, 16/16 main gates, one physical instance.
- A2 versus B: PASS; discrete, physics, render, model, source, and environment comparisons are exact.
- C retry: PASS, 16/16 main gates, one physical instance.
- B versus C: PASS; comparisons are exact.
- Physical budget: 3 used, 37 remaining. The historical V1 pre-physics failure is not charged to scientific physical budget.

R16 calibration then executed three physical levels (I1, I2, I3). All three passed pre-hold and trace/numeric execution checks but none reached `PHYSICAL_LOSS_CONFIRMED`; no valid intervention level was selected. Per the campaign hard-stop rule the continuation is now `STOPPED_CALIBRATION_FAILED`; development, generator, and evaluation were not dispatched. Physical budget is 6 used and 34 remaining. No confirmation or L3 entry occurred; `selected_candidate_id` remains null. The A1 decision remains `A_INCONCLUSIVE_REMAIN_BLOCKED` and is not reclassified as pass.

# Minimal refinement attempts

The only refinement applied was a zero-initialized event-cost calibration head. The encoder, graph specification, and supervision split were retained. All reported values below come from fresh checkpoint-forward predictions and the Stage 5 event extractor.

| seed | failure direction | recovery direction | result |
|---:|---:|---:|---|
| 20260906 | 1.000 | 1.000 | passes event gates |
| 20260907 | 0.571 | 0.000 | fails event gates |
| 20260908 | 1.000 | 0.500 | fails recovery gate |

The required 2/3 seed model gate remains unmet. No reward parameter search, selection lock, test-based selection, or G2 claim is made.

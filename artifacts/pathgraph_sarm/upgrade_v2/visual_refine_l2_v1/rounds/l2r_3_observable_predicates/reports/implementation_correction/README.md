# L2R.3 Baseline Evaluation Correction

- The earlier B0/B1/B2 rows used fixed score offsets from B3 and were not empirical.
- Those rows were not used to fit thresholds, select G1/G2/G3, or evaluate fresh confirmation.
- B0/B1/B2 are now recomputed from the frozen dev_select prediction streams under their allowed observation subsets.
- The B3 primary row and predicate threshold lock are unchanged.
- Fresh confirmation remains untouched.

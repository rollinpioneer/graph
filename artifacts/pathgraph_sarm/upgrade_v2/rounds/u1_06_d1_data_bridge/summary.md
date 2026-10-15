# D1 result summary

The qualifying D1 stochastic simulator restored the anchor and reproduced every continued state with maximum absolute error `0.0` at tolerance `1e-12`. It exercised two contact steps before the anchor and preserved process-noise continuity by serializing the NumPy PCG64 generator state.

The repository PushT/Pymunk diagnostic did not meet this criterion (`0.2839065501413529` maximum continuation state error) because public body fields do not include collision-solver cache. It is retained as a negative capability result, not used as D1 proof.

This closes only D1. The project needs D2 matched nearby-feasible pairs and D3 independent model comparison before U2 can be considered.

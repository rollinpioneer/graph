# Superseded U2 claims

- Historical boundary/reward tables remain preserved for provenance.
- Boundary values in `boundaries_v2/` use episode-local dynamic-program matching.
- Reward values in `reward_v2/` use incoming-transition attribution and test-only family bootstrap.
- The historical current-frame/history formula comparison is `not_computed`; no gain claim is carried forward.
- `weak-only` and `budget0` retain shared small-gold calibration supervision and are not zero-gold claims.

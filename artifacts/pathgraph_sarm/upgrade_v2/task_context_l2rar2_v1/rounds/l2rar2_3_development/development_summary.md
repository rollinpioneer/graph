# L2RA-R2 development result

- Cache/logic pilot: `CACHE_MECHANISM_AND_LOGIC_PASS`; it has zero physical-family credit.
- New fit: 4 roots / 32 physical rollouts. Fit was used only for implementation and data-contract review.
- New select: 8 roots / 64 physical rollouts, with exactly one K1-K8 case per root.
- All 64 request provenance checks passed; requests were emitted by controller dispatch before action and contained no outcome/reference fields.
- M1 K1 recall was 8/8. Among reference-estimable held-loss cases, K4 was 3/3, K5 was 3/4, and K6 was 3/3.
- Each method has 64 primary event opportunities; 18 positive events were estimable and 14 positive events were reference-unresolved. M1 positive wrong-or-unknown was 1/18 (0.0556) among estimable positives, while 19/64 primary cases were reference-unresolved under the frozen physical hold proxy.
- M1 false emergency was 2/8 for K2 and 2/8 for K8. Both pairs arose after B_count2 incorrectly established hold, then emitted recover on contact loss.
- D0 false emergency was 8/8 for K2. M1 removed six generic-end retries, but the remaining 2/8 recover errors exceed the 0.10 gate.
- M1 touch false-hold evidence was 2/8; brief-hold evidence was 3/4 among estimable K5 cases.
- K3 had 3/8 estimable events and K7 had 8/8; neither produced a false emergency in estimable events. History/context mask protection passed, but aggregate gates requiring complete event windows and references did not pass.
- Development status: `DEVELOPMENT_FAILED`; selected candidate is null. Focused confirmation is not authorized.
- No threshold, candidate, seed, case program, or reference rule was changed after select.

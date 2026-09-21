# Stage 0C — VLM Cache Sanity Check

Status: `PASS`. Frozen numeric gate: 24 processed records, 24 normal requests, first-JSON-valid 24/24, legal nonempty scenes 5 including D0=2. Manual semantic review complete; 1 obvious semantic issue recorded and not written back into cache. Stage 1A was not started.

## Required answers

1. Final status: `PASS`.
2. Formal scenes: 24/24 real bound dev RGB scenes from `experiments/stage_0c_inputs`.
3. Few-shots: 3/3 frozen independent examples, used in every request.
4. Model access: verified by 24 HTTP 200 responses with request IDs for `qwen3.8-max-0902` at `https://dashscope.aliyuncs.com/api/v1`. Response objects did not echo a model name.
5. Frozen configuration was not substituted: model, region `cn-beijing`, endpoint, schema v2, prompt, temperature 0, 2048 tokens, no thinking/search/tools.
6. Requests: 24. Retries: 0. First scene `D0_dev_00` was preflight and counted in the 24.
7. JSON/schema/ID/effect valid rates: 24/24 JSON; 24/24 schema; 95/95 IDs; 95/95 effects.
8. Empty-prior rate after isolation: 19/24 (0.792). API failures were 0 and are not counted as empty priors.
9. Mean accepted relations: 0.375; median 0.0; max 2.
10. Accepted type counts: SOFT_SUPPORTS=1, SOFT_RELEVANT_TO_GOAL=8.
11. Contract-redundant raw relations: 86/95 (0.905). Obvious semantic issues: 1 (`T_C_dev_06` PICK(interferer)->PICK(target)).
12. No unauthorized skills, rewards, rankings, or new IDs were accepted. Raw outputs that restated local OPEN/PICK/PLACE contracts were isolated as CONTRACT_REDUNDANCY.
13. Each scene has immutable cache files: raw_response, parsed, accepted, rejected, prompt, redacted payload, hashes, COMPLETE marker.
14. Ready as Stage 0C cache input. Not a claim of high VLM yield. High redundancy/empty rate is recorded. Stage 1A runtime remains unbound.
15. Plot: logs/final_relation_counts.png.

## Notes

Most raw candidates repeated local contract supports. That is expected isolation, not a transport failure. The single T_C nonempty extra-contract edge used Held(interferer) as support for PICK(target); review labels it OBVIOUS_ISSUE and leaves the cache unchanged. D0 nonempty extras are OPEN/PICK goal-relevance, which the prompt allows and the contract does not already cover.

## Stage 1A remaining

Controller, verifier, TaskEvaluator, safety, timeouts, perception/calibration, and runtime factory remain MUST_BIND_BEFORE_STAGE_1A. `stage_1a_runtime_ready=false`.

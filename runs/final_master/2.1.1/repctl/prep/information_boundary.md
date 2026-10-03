# CP-DISR-TB-REP-CONTROLS-01 — information boundary

Frozen before any RL run. Base commit `9422b837cf1dfc43afc056a77bdb59d63512b3b6`.
The question is a **representation / computation-placement / optimization-bias** question. It is **not** a question about which method owns more true environment information.

## What every arm may read (identical for B2, +E, Absolute Successor, Matched Neural Composition)

| source | content |
|---|---|
| current facts F | the three-valued (TRUE / FALSE / UNKNOWN) measured fact store of the snapshot |
| Skill Contract | the frozen grounded contracts of the 7 T_B candidates: PRE_POS, PRE_NEG, ADD, DEL, UNKNOWN, conditional guards/effects, bound arguments |
| candidate set / IDs / hard mask | exactly the snapshot's `candidate_ids` and `mask` (unchanged builder, verifier, safety) |
| observation / candidate features | the snapshot's `base_input` and `candidate_features` (GRU hidden state, candidate embedding) |
| goals | the frozen T_B goals and their signs, as graph template nodes |

Nothing else: no VLM, no prior / relation edges (`prior_mode = absent`, effective R empty), no provider, no test cache, no privileged simulator state, no successor sidecar, no multi-step rollout, no planner, no learned world model.

## Absolute Successor (`B2-ABS`)

* Information sources are the **same as B2**: the same `four_views` / `nominal_overlay` (nominal_apply) produce `F_after_i = T_nom(F, a_i)` with the same current facts, candidate set, grounding/binding, PRE/ADD/DEL/UNKNOWN, conditional guards and frame semantics, encoded by the same graph encoder family (shared weights, same number of encoder calls).
* The **only** change is the parameterization of the consequence: the candidate readout receives `E(F_after_i)` instead of `E(F_after_i) − E(F)`. The current representation `E(F)` still reaches the policy through the context (`mean E(F)`), exactly as in B2.
* The latent difference is not used as the consequence representation. No parameter is added: the module set and the parameter count are identical to B2.

## Matched Neural Composition (`B1-K+NC`)

* Information sources are the **same as +E**: current F and the grounded candidate effects K(a_i) — PRE_POS/PRE_NEG, ADD, DEL, UNKNOWN, conditional guards and conditional effects (with their conditional index), and binding / effect ownership through the candidate-local contract.
* The network learns the (F, K(a_i)) → candidate-rows interaction itself: each grounded atom is embedded by the shared `NodeFeatures` (predicate, argument types and the atom's **current** TRUE/FALSE/UNKNOWN code), tagged with its contract role, mixed with the candidate's mean token and passed through one learned layer.
* No environment ground truth is added. `nominal_apply` / `nominal_overlay`, `four_views`, successor views and `F_after_i` are never built or read; there is no sidecar or successor cache (source scan + runtime trap + file-read trap tests).
* It does **not** carry +E's 4096-bucket hashed token tables (1.75 M parameters); that is a deliberate capacity decision so the control sits within the pre-registered ±5 % band of B2 (see `representation_capacity_audit.json`).

## Statement required by the card

Under **fully deterministic effects and a known frame rule**, `F_after_i` is a deterministic function of `(F, K(a_i))`. Therefore **+E / Matched Neural Composition and the successor family can be approximately information-equivalent**: a network that receives `F` and `K(a_i)` can in principle compute whatever `nominal_apply` computes. This experiment tests whether the *representation*, the *placement of computation* (symbolic application outside the network vs learned composition inside it) and the *optimization bias* of the parameterization change how early the policy learns — not who has more true information.

## What this card does not claim

No claim about general superiority, equivalence or sample-efficiency ratios of any method; no claim from three seeds and ten dev cases beyond the pre-registered case table; no claim about test30 (not read) or about structural generalization (not run).

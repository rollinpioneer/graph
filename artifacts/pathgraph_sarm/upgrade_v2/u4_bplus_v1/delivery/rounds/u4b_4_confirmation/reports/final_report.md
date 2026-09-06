# U4 B+ initial confirmation exclusion report

- status: `contaminated_implementation_correction`
- families: 12; base rollouts: 48; continuations: 96.
- exclusion: this round is not used for any final claim or G2-G1 effect.
- reason: it ran before correction of the train-only mapper and graph evaluation implementation.
- remedy: a separately locked reconfirmation used generator indices 36-47 from the same seed stream, without changing the graph, mapper, boundary rule, semantics, thresholds, or metric version.
- this was a protocol repair, not a score-chasing rerun.

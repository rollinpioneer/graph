# U0 baseline v3 rescore

The v3 rescore loaded the checksum-verified legacy checkpoint
`ee3c4aea451a8be78008339826d945df9bca2b86990892fc371c301c3fa53f8a`
with the `cupid` PyTorch 2.8.0 runtime on CPU.  It scored all 380 readable
Stage6 deterministic episodes: 160 `transport_dual_order` and 220
`transport_recovery`.

The scorer derives fixed chains from GraphSpec start-to-success paths and uses
non-negative Dijkstra for topology cost.  It produced 1,300 computed
episode-level rows and two explicitly `not_computed` oracle rows.  The latter
remain unavailable because the source has no independent ground-truth node
sequence.  The graph-filter structure ablation is also explicitly
`not_computed`; no structural-benefit claim is supported.

Episode-return means are descriptive mechanism-only values: dual-order
`manual_graph_topology_v3=-0.689410002`, `A_first=-1.341148393`,
`B_first=-0.727081613`; recovery `manual_graph_topology_v3=0.034392762`,
`graph_path_1=0.035624469`.  `time_fraction_oracle=1.0` uses full episode
length and is an offline reference, not a learned or deployment baseline.

These values supersede neither physical validation nor independent
generalization evidence.  They are correctly scoped U0 legacy mechanism
measurements.

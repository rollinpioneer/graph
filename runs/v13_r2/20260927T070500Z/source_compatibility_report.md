# R2 source compatibility

Baseline: `55a3b7ce35edebbbb8a587fe1e3a98ca1967db07`. R2 training commit: `228517ed214e18bd75ab11b84f3edff6b07f14b7`.

Only the R2 adapter is new. The shared Policy, B0 two-SAB Set Transformer, PPO, collector, reward, clock, safety, candidate encoding and runtime modules are unchanged. R2 adds the B0 allowlist, output/status routing, provenance, and evaluation interval fields.

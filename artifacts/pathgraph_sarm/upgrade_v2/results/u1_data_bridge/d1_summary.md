# D1 stochastic state restore check

- status: D1_COMPLETE
- backend: explicit-state continuous stochastic simulator (position, velocity, contact, and RNG state)
- anchor/continuation equality tolerance: 1e-12
- maximum continuation state error: 0.000e+00
- contact steps before/after anchor: 2 / 0
- object displacement before/after anchor: 0.145069 / 0.092063
- non-qualifying backend diagnostic: the repository PushT/Pymunk public state is insufficient for collision-exact replay; its artifact is retained separately.
- scope: D1 is complete, while D2 and D3 remain mandatory and U2 remains ineligible.

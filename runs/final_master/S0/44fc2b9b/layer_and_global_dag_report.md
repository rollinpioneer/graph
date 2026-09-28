# Layer and global readout DAG

```text
FactStore + contract graph
        |
        v
NodeFeatures (kind/schema/typed arguments/fact truth/goal sign)
        |
        v
RGCNConv x4 (128, 4 bases, mean, root/self) -> ReLU -> node LayerNorm
        |
        v
GoalReadout: goal MLP rows + ACTION mean + PROPOSITION mean + goal mean
        |
        +--> nonlinear global row
        +--> goal rows
        |
        v
Policy recurrent observation + candidate projection + CandidateReadout
        |
        +--> phi_K / u_K
        +--> Full: D_P = (Z11-Z10) - (Z01-Z00)
        +--> A_STAT: S_R = Z10-Z00 (delta.zh-delta.zk)
        +--> A_CAT: [Z00,Z01,Z10,Z11] -> 512-to-128 projection
        |
        v
Anchored prior readout -> bounded Delta -> Actor logits / V / all-candidate Q
```

`four_views` preserves the same facts/template and applies a read-only nominal successor only for methods that explicitly use it. B1-K and B1-K+E do not call that path. `Q` is forwarded for every candidate but PPO supervision selects only the executed candidate.

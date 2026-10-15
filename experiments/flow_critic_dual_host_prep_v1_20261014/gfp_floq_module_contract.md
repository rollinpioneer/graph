# GFP + floq Critic module contract

Frozen host: GFP `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`.
Frozen value source: floq `3d60e638b42bbef018c56cb1199ff37c3470520d`.

`agents/floq_adapter.create_agent('gfp', ...)` routes `critic_type=scalar` directly to the unchanged official GFP constructor. `critic_type=floq` initializes the namespaced floq value modules and the original GFP ActorVectorField definitions. The host agent name remains `gfp`; the official CSV matcher runs before configuration export. No host MLP/Value/network utility file is changed.

| Key | Role | Parameter gradients |
|---|---|---|
| `floq` | Online expected-return velocity | Flow matching loss only |
| `target_floq` | Euler TD backup | None |
| `critic` | Online scalar distilled readout | Scalar distillation only |
| `target_critic` | EMA scalar readout for guidance | None; not a TD-trained Critic |
| `actor` | Original one-step policy | Endpoint distillation and Q loss |
| `vabc` | Original guided multi-step BC policy | VaBC loss |

Actor loss, guidance formula, reference aggregation, Q normalization, policy methods and endpoint distillation are inherited verbatim from GFP. The host Actor uses the online scalar Critic and guidance uses `target_critic`. There is no duplicate `target_distill` and no FA/FAN.

For TD targets, both policy action generation and value evaluation states are explicit. `target_use_next=False` preserves current-observation VaBC action generation followed by next-observation value evaluation. Actor actions are generated at next observations. The same host RNG is used for Actor/VaBC calls, preserving the host branch convention while adding the card's repeated sample dimension.

The order is: integrate each target Flow ensemble, aggregate ensembles for each sample, aggregate Actor/VaBC sources for that sample, compute reward plus discounted masked return for each sample, then average samples. The original Flow core uses a noise sum for velocity error; scalar distillation averages online Flow returns over noise then ensemble and applies stop-gradient. Flow and scalar loss matrices are added before the final mean.

This is an expected-value Critic transplant, not a distributional backup. Source `embed_time=True` actually uses `cos(t)` with one channel; `time_embed_dim=64` remains an inert inherited configuration field. The reward support is rebuilt from actual task rewards and GFP discount.

Target definitions use the fixed source's pre-optimizer online parameters: `tau*old_online + (1-tau)*old_target`, retaining its one-update lag. Both `floq→target_floq` and `critic→target_critic` mappings are specified. During this card, Agent updates and target updates are blocked; the formula is tested only on synthetic arrays. No training dispatcher exists in the zero-training entry.

Original GFP weights restore only in their original environment. New random combination states can round-trip separately; an old complete GFP Agent is not declared compatible with the new parameter/optimizer tree.

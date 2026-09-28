# Full stall phase reconciliation

- Classification: **PPO_INTERNAL_STALL** (high confidence).
- The frozen training log contains `Full PPO complete update transitions=1024 N=3072 ...` immediately before the 30-minute no-transition window.
- The authoritative durable boundary remains N=2048, T=9771.400000000713, two complete updates.
- N=3072 was observed in the transition log but `update_complete_3.pt` was not persisted; the N=3072 tail is not treated as a completed update.
- No long reproduction was performed; no Full resume is authorized in this recovery.

## Evidence

```text
[stage1a_v11] 2026-09-27T16:28:14Z Full PPO complete update transitions=1024 N=1024 T=4950.8000000004395 num_envs=1
[stage1a_v11] 2026-09-27T17:11:36Z Full eval N=2048
[stage1a_v11] 2026-09-27T17:12:39Z Full PPO complete update transitions=1024 N=2048 T=9771.400000000713 num_envs=1
[stage1a_v11] 2026-09-27T18:03:08Z Full PPO complete update transitions=1024 N=3072 T=14758.050000000949 num_envs=1
```

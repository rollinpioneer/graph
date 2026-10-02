# PPO wall-time spikes observed 15:26-16:10Z (recorded 16:25Z, after the 30-min window)

Pre-registered flag (amendment): an existing worker's PPO update > 1.5x its previous update.
Observed (published - PPO start): E-0 update 6144: 7.95 min (prev 3.68, 2.2x) -> update 7168: 3.77 min (recovered);
DK-1 update 3072: 35.4 min (prev 19.5, 1.8x; first was 27.1); K-1 update 3072: 9.95 min (prev 4.1, 2.4x).
Context from throughput/samples.jsonl: 15:20-16:00Z GPU0 utilisation rose from ~3% to 76-84% with E-0 (alone ~3%) -> external tenant;
GPU1/GPU2 were at 80-98% in the same span; by 16:10Z all three GPUs were back to ~42-44% and E-0's next update was 3.77 min.
Machine load1 13-40 of 112 cores, iowait 3.4-5.5%; collection rates per 10 min stayed flat (E-0 ~34/min, DK-1 ~18-28/min, K-1 ~25-39/min).
Each of my workers has its own GPU, so the only shared resources among them are CPU/RAM/disk (not saturated).
Assessment: one-off spikes that coincide with external GPU contention, not persistent degradation of an existing worker and not
attributable to the third worker; no stop. Not proven causal (no per-process GPU profiling). Keep watching DK-1/K-1 PPO times.

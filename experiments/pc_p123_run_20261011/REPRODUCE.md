# PC-P123-03 reproduction

Execution branch: `codex/pc-p123-03-20261011-card`, inherited card commit `8b575e44363bf2fb6b1a765de5994ac2004fa3f4`. P1 registration commit `c6e079ae5`; P3 registration commit `cab22312d`. Frozen worker/config hashes, rather than the current HEAD alone, identify executed code. Exact frozen source checks are in `freeze_P1.json` and `freeze_validation.json`.

Use one persistent, keepalive-enabled SSH to `lab-xushijie3`. Python is `/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011/env/bin/python` (Python 3.8.20, Torch 2.1.2+cu118, NumPy 1.23.5, Isaac Gym 1.0rc4). Official EC-Diffuser HEAD and all three asset identities are reverified in `results/identity.json`. Model EMA, encoder, normalizer and original physics are frozen; no training.

Read-only assets and official source are in `/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011`. Current raw data are under `/home/xushijie3/work/graph_pc_p123_card_20261011/experiments/pc_p123_run_20261011/runs`. Large arrays, CUDA caches and full process/GPU logs stay on the server. See `evidence_inventory.json` for absolute paths, byte lengths and SHA-256. Source scene IDs are not DLP particle IDs.

Copy the tracked experiment subtree into a **fresh** server directory without copying `runs` or `logs`. Preserve original frozen files byte for byte. Use the portable `reproduce_job.sh`, which resolves that new directory, while leaving original frozen `run_job.sh` bytes unchanged. Select the currently least busy suitable GPU from all eight cards, set `STAGE5_GPU_INDEX` through `guard.py`, and enforce 11264 MiB aggregate safety threshold plus total 7200 GPU seconds. Multiple jobs share only read-only assets and each has isolated vector environments and output directories. Do not invoke commands against the archived run names: guards refuse existing directories.

For the original frozen replication (same scene distribution, not a new independent study):

```bash
cd <fresh_experiment_directory>
PY=/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011/env/bin/python
# Choose GPU from current nvidia-smi; replace <gpu>.
$PY guard.py --out runs/repro_sparse --seconds 1170 --gpu <gpu> -- bash reproduce_job.sh --out runs/repro_sparse/data --n 6 --mode P1 --condition 6_sparse --seed 951124 --steps 200 --parallel 8 --groups 0,1,2,3,4,5,6,7
# Run crowded under the same phase sum cap 2340, using a distinct suitable GPU if parallel.
$PY guard.py --out runs/repro_crowded --seconds 1170 --gpu <gpu> -- bash reproduce_job.sh --out runs/repro_crowded/data --n 6 --mode P1 --condition 6_crowded --seed 951124 --steps 200 --parallel 8 --groups 0,1,2,3,4,5,6,7
# P3 comparison; both arms receive 200 model calls and 200 control steps.
PC_P123_WORKER=evaluate_validation.py $PY guard.py --out runs/repro_P3 --seconds 2300 --gpu <gpu> -- bash reproduce_job.sh --out runs/repro_P3/data --n 6 --mode validation --condition 6_crowded --seed 951126 --steps 200 --parallel 8 --groups 8,9,10,11 --arms B0,B1
```

If shared GPU load needs a reservation transfer, the current `takeover_guard.py` is scoped to original P1 job names and known own monitor PIDs. Do not blindly use those PIDs in a reproduction. Original transfer preserved the crowded worker and the 2340-second combined P1 cap; no samples or scientific settings changed.

P1/diagnostic contact collection is the official CC_NEVER (0): all-zero forces mean unavailable information. P3 uses CC_LAST_SUBSTEP (1) for readout only; original solver, time step, two internal substeps and physical parameters are equal. Effective collection is one last-internal-substep sample per 0.01667 simulation second, not complete contact-pair or transient coverage. Ordinary action damping scales xyz commands by 0.5 throughout the fixed budget; it is not a guaranteed physical velocity limit and uses no ground-truth trigger.

Original four stall windows and five fixed reruns are discovery evidence. Fixed reruns use a new five-slot vector batch, so seed equality cannot reproduce the original policy noise stream or strict same state. Exactly one rerun per selected scene was performed; the four old stalls did not recur. The fifth selected scene was the original new fall case, merged into that same batch. P2-B, snapshot repair and new network development were not executed.

`post.py`, `P3_audit.py`, `diagnose_new.py`, `resource_union.py`, `validate.py` and the finalizer reproduce lightweight summaries from available server arrays. Main metric remains identity-matched XY distance strictly less than 0.04 m. Fall is a separate z<0.975 for five control observations, with first sustained detection timestamp. Bootstrap intervals are exploratory at n=8 layout pairs / n=4 control pairs. Do not treat frames, objects or time samples as additional independent episodes.

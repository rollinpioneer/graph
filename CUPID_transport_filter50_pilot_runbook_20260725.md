# CUPID Transport-MH 50% Filter Pilot Runbook

Version date: 2026-07-25

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: UNVERIFIED
- Version Label: transport_filter50_pilot_runbook_v1

## Objective

Test whether the CUPID paper's direct 50% filtering rule improves policy
success after retraining. The previous Bootstrap branch remains a frozen
negative result and is not reclassified by this experiment.

## Frozen arms

| Arm | Action | GPU |
|---|---|---:|
| baseline | Re-evaluate the completed unfiltered seed-0 checkpoint | 2 |
| cupid50 | Remove the 96 lowest CUPID performance-influence demos, then train | 1 |
| quality50 | Remove the 96 lowest official CUPID-Quality demos, then train | 4 |
| random50 | Remove 96 demos selected without replacement using seed 20260725, then train | 5 |

All arms use dataset seed 0, the same frozen `192/12/96` original split, and
the same new evaluation seeds `200000..200099`. Exact filtering leaves
`96/12/96`; validation and holdout membership remain bitwise unchanged.

## Scoring

- CUPID: `sum_of_sum-net-all`.
- CUPID-Quality: min-max normalize each of `sum_of_sum-net-all`,
  `min_of_max-net-all`, and `max_of_min-net-all`, then combine with weights
  `[0.50, 0.25, 0.25]`.
- Random: NumPy `default_rng(20260725)`, sampling 96 original training IDs
  without replacement.
- Score ties are resolved by ascending dataset episode ID.

The official CUPID score reconstruction maximum absolute error is
`1.8789673e-7`. CUPID and CUPID-Quality filter sets overlap on 83 of 96
members (Jaccard `0.76147`).

## Training

Filtered arms follow the repository's Transport-MH 50% curation schedule:

```text
training.seed=0
training.num_epochs=2301
training.resume=true
training.checkpoint_every=50
training.rollout_every=50
checkpoint.topk.k=1
task.dataset.seed=0
task.dataset.val_ratio=0.04
task.dataset.dataset_mask_kwargs.train_ratio=0.64
task.dataset.dataset_mask_kwargs.uniform_quality=true
task.dataset.dataset_mask_kwargs.filter_episode_ids=<frozen 96 IDs>
task.env_runner.n_envs=8
dataloader.num_workers=0
val_dataloader.num_workers=0
logging.mode=offline
```

Each arm has an independent `setsid` supervisor, output directory, lock, PID,
log, and status tree. A failed training attempt resumes from `latest.ckpt`
after 30 seconds. Five consecutive failures within 300 seconds are terminal.
Checkpoint and log audits require epoch 2300 and finite values before
evaluation starts.

The prior unfiltered seed-0 training used its frozen 1751-epoch schedule and is
not retrained in this pilot. All three 50% arms use identical 2301-epoch
budgets, matching the repository's curated Transport schedule.

## Evaluation

Each arm saves and audits 100 episodes using:

```text
checkpoint=latest.ckpt
test_start_seed=200000
num_episodes=100
device=cuda:0 (mapped to the arm's physical GPU)
```

Evaluation restarts overwrite only that arm's incomplete new evaluation
directory and never alter training checkpoints or the original score pool.

## Pilot gate

A candidate advances to multi-training-seed confirmation only when all
conditions hold:

1. success-rate difference versus unfiltered baseline is at least `+0.05`;
2. success-rate difference versus Random-50% is at least `+0.03`;
3. paired wins exceed paired losses against both references.

This is a practical-effect pilot gate, not strong statistical confirmation.
The automatic finalizer reports paired win/tie/loss counts, a paired bootstrap
interval, and exact McNemar p-values. It stops after emitting either
`ADVANCE_TO_MULTI_SEED_CONFIRMATION` or `STOP_PILOT_NO_PRACTICAL_GAIN`; it does
not silently launch seeds 1 and 2.

## Notification and persistence

The independent email monitor reads SMTP settings from
`/home/__compress_data/xushijie/article/original_research/private/smtp_163.env`
and sends only experiment name, event, time, arm, and status metadata. Events
include start, restart, completion, termination, unrecoverable failure, and
supervisor loss. It runs in a separate detached session from all arm
supervisors and from Codex.

## Frozen artifacts

- `manifests/transport_filter50_pilot_20260725/experiment_manifest.json`
- `manifests/transport_filter50_pilot_20260725/filter_ids/`
- `manifests/transport_filter50_pilot_20260725/frozen_scores.csv`
- `manifests/transport_filter50_pilot_20260725/artifact_sha256.txt`
- `frozen/transport_filter50_pilot_20260725/launch_commands.txt`
- `frozen/transport_filter50_pilot_20260725/execution_sha256.txt`


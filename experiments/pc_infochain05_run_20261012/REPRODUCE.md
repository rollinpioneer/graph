# PC-INFOCHAIN-05 reproduction

Branch: codex/pc-infochain-05-20261012-card
Source card SHA-256: b76d5f3752b6d68b5ab96751dca6ff0067e131aa4c2fbaa600fe12064851ea82
Archive starting commit: c878a514710f9511189aff5bafa71fbbdd744935
Official runtime commit and compatibility overlay: receipts/official_runtime_source.json and official_compatibility.patch.

## Environment and inputs

All execution used lab-xushijie3 and one persistent SSH shell. The Python runtime is
/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011/env/bin/python.
Isaac Gym must import before Torch. Python 3.8, Torch 2.1.2+cu118, Isaac Gym GPU pipeline.
No training or checkpoint changes. Assets were rehashed this round; see receipts/assets_identity.json.
Do not copy weights, datasets, complete logs, RNG pickles or tensor outputs into Git.

Set OLD to /home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011,
PREV to the PC-CAUSE-04 experiment run, P123 to the PC-P123-03 experiment run,
ROUND to this run directory. Add OLD/env/bin to PATH, OLD/env/lib to LD_LIBRARY_PATH,
and set TORCH_EXTENSIONS_DIR=OLD/torch_extensions, OMP_NUM_THREADS=4,
MAX_JOBS=4, PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1, MPLBACKEND=Agg.

## Frozen material

prepare.py generated the twelve representative historical episodes and provisional noise cases.
The final six ABCD input cases are results/noise_input_cases.json; their frozen identity and the
cost-only reduction from twelve are recorded in receipts/freeze_offline.json. Use these exact
archived lists rather than regenerating them after seeing outcomes.

Historical DLP reencoding uses original batch size/slot and processes the two views separately.
All initial and goal saved-particle comparisons are in results/P1_input_identity_check.json.
Only stored image times are reconstructed. No missing intermediate latent state is invented.

P1 goal probes use the same historical current input, G0 and opposite equal-distance G1/G2.
runtime.py constructs goals with official assets, camera and rendering APIs in an isolated
goal-image fixture. Cube gravity and actor collisions are disabled in that fixture to prevent
incidental motion; a physics tick updates graphics. Requested XY and matching robot DOF,
changed pixels, bounds and actual cube-axis separation are checked. These fixture flags are
never applied to an executed task. These are capability probes, not historical trajectory replays.
The saved versus rerendered G0 difference is reported; G1 versus G2 is the primary contrast.

## GPU execution

Use guard.py around every GPU job, with a fresh output directory and --gpu selected from current
shared-device conditions. Guard time includes setup/failure wall time and multiplies by GPUs
with task-resident contexts. It measures aggregate memory only for this round's process groups.
Resource-reallocation receipts retain actual cost-only monitor takeovers without worker restart.

Offline runs:
- probe.py --target-n 3 or 6 --resume-decomposition <original decomposition_index.json>
- probe.py --noise-only --resume-decomposition <original decomposition_index.json>

The original decomposition is runs/offline_formal/data/decomposition_index.json and its NPZ paths.
That run failed in the goal renderer after decomposition. It was resumed without replacing those outputs.
All renderer/loader failures and the invalid identical-image goal tests remain indexed and excluded
from scientific inference. The valid goal runs are targets3 and targets6, and valid ABCD is noise_formal.

For complete recording use rollout.py with the frozen supplement or validation seeds.
Every episode is 200 closed-loop controls, batch1/slot0, H5/T100; only the first XYZ action is executed.
Every control retains actual RGB, current/goal particles, generated H5, complete 101 draws and RNG
states, actual gripper/body pose, actions and controller targets. Large output remains on server.

Validation conditions and alternative predictions were frozen in receipts/validation_freeze.json.
validation_launch.py executes the four episodes sequentially. Two conditions share exactly the same
25 whole-sampler seed streams, each used eight times: permuted time order versus contiguous eight-step
blocks. It is a temporal correlation diagnostic, not a proposed fixed-noise policy or ordinary
independent sampling baseline. Complete noise-content multisets and initial states/images must match.
Fresh paired scenes are independent mechanism probes, not restores of historical failures.

## Analysis and limitations

roi.py analyzes per-object projected/color-visible regions, foreground alpha and reconstruction.
Particle indices are never stable object identities; colors/visibility are exploratory proxies.
stationary.py compares stable object centers but keeps gripper, occlusion and rotation differences explicit.
geometry.py and geometry_selfcheck.py use actual URDF tip boxes and cube OBB separation axes.
Positive separation or contact-offset proximity never proves/excludes physical contact,
counterparts, inter-step contact or other links.
actual_readout.py / actual_readout_figures.py decode stored foreground atfixed0/100/199 and exploratory event/control frames.
post.py, new_figures.py and finalize.py produce the final lightweight tables, plots, ledgers and manifests.

All fall endpoints stay in summaries. Strict stationary events and moving-but-cancelled adjacent
behavior remain separate. External normal references are descriptive, not causal matching.
Twelve selected episodes, eight repeats per input and one matched pair per object count do not
estimate population effects or cross-training-seed scaling.


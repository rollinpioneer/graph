# GFP-GUIDANCE-DISTANCE-Z01-V1

Authorized scope: Z0 public evidence + Z1 frozen inference only. Zero optimizer updates, zero environment interactions. T1 task4 training NOT AUTHORIZED.
Branch: [codex/gfp-guidance-distance-z01-20261015](https://github.com/rollinpioneer/graph/tree/codex/gfp-guidance-distance-z01-20261015).
Server root: `/home/xushijie3/work/gfp_guidance_distance_z01_20261015`.

Z0: fixed official GFP CSVs and configurations; compare GFP, ReBRAC, FQL, VaBC and GFP-AWR with missing results explicit. Static extraction of OGBench1.1.0 maze/task definitions; BFS, Euclidean and detour geometry. Check OptiFlow and QGF primary sources, retaining protocol and uncertainty differences.

Z1: G0 task1 seeds0/1 at200k/600k/1M. Each model receives100 train and100 validation batches of256, totaling307,200 primary state evaluations. First10+10 batches are an INCLUDED pilot. Two additional RNG repeats on these20 batches/model add61,440 sensitivity state evaluations. Uniform valid-transition sampling within each original split; no distance-stratified batch normalization. All models share sampled indices. Actor and VaBC share the exact training-style Gaussian input; actor clipping, online ensemble aggregation and batch mean absolute Q normalization retained; target critic supplies data/reference scores, guidance_agg=min.

No environment construction, reset or step: the OGBench constructor itself can perform stabilization steps. Read dataset arrays and task definitions directly. Model loading uses existing tested environment. No model parameters or optimizer state updated. Hashes checked before/after.

Distance bins: BFS0-3,4-7,8-11,12+. Neutral0.4-0.6; saturation below0.01/above0.99. Report signed and absolute Q gap, normalization denominator/logit, member ranking agreement, and guidance association with recorded future BFS progress at25/100 steps, excluding trajectory boundaries. Future behavior is an imperfect proxy under changing collection goals, NOT actual advantage versus reference actions.

Trajectory-cluster bootstrap200 replicates for intervals; train/validation and each model seed/checkpoint separate. Control Euclidean distance (including square), head height, torso upright, speed, and trajectory fixed effects. Model-seed uncertainty is not replaced by millions of states. Counterfactual eta multipliers0.1/0.3/1/3/10 affect recorded-Q weights only, not policy or training.

Stop on nonfinite values, formula mismatch>0.001, changed asset identity, environment/optimizer entry or budget excess. No retries or automatic training. Joint report and GO/DIAGNOSE/NO-GO table; four mechanism figures and one geometry figure with lightweight metrics. Full arrays stay server-side. No task4 Q diagnosis with task1 checkpoints.

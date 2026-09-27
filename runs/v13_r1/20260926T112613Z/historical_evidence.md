# Historical Evidence (Read-Only)

|Profile|Methods|Preserved observation|Source|Interpretation|
|---|---|---|---|---|
|D0 stage 1A|B2 and Full|29/30 each on historical final test; mean G 0.6137862199598949|status/stage_1a.json|Earlier profile/test; not an R1 comparison and no new test-ID use here.|
|T_B historical partial run|B0|Durable N=22528, 22 complete updates; later EIO at logged N=23552|status/stage_2a.json|Keep its own source/profile; do not splice into R1 curves.|
|T_B historical partial run|B1-K|Durable N=21504, 21 complete updates; stopped after counterpart EIO|status/stage_2a.json|Do not substitute its weights for the new B1-K run.|
|T_C nonoverlap_v1|direct versus relocation|Direct contrast 10/10; relocation controller sequence passed 20 cases, with final task success verified in 10; no demonstrated relocation benefit|status/tc_nonoverlap_remediation.json|Not a useful-relocation result or a valid R1 extension.|
|Phase A stalled attempt|T_C Full/B2|Zero verified PPO updates; retained n_000000 and STOP evidence|status/phase_a.json|Not a completed E16 result.|
|Historical 0D empty-patch metric|T_B|Historical empty_patch_rate=1.0 classified unreliable/not reconstructable|tools/plan_v11_stage0d_erratum.py|Do not reuse as an R1 numerical mechanism claim.|

All historical rows retain their original profile and provenance. No historical checkpoint was loaded or re-evaluated for this R1 report.

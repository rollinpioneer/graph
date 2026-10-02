import json,os,glob,hashlib,datetime
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip(); ANA=OUT+'/zero_sample_analysis'
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
FULL='7c2f58c9844c7a8db4cd5a28fe1615e5630dfeff'
tabs=open(ANA+'/comparison_tables.md').read()
tableA=tabs.split('## Table B')[0]
cid=json.load(open(ANA+'/checkpoint_identity.json'))
L=[]
L.append('# Post-run zero-new-sample analysis — T_B final runs (R-TB-E-0, R-TB-DK-1, R-TB-K-1)\n')
L.append('Generated %s. No new episode, environment construction, provider call, physical evaluation, formal test, RL run or training step was executed for this report; the only computation was reading existing files, recomputing file hashes, and loading checkpoints to compare weights.\n'%now)
L.append('## 0. Result commit and repository state\n')
L.append('- Results commit (full SHA): `%s` — "research: record T_B launch recovery and planned comparisons".'%FULL)
L.append('- Branch `codex/cp-disr-final-tb-engineering-launch`, parent `cac2fa5b3cafb46f1f74370314189f4bca4878fa` (source/prep). **Not pushed**: no remote-tracking branch contains the commit, and nothing here claims the remote is in sync.')
L.append('- The analysis files in this folder (`zero_sample_analysis/`) are new and untracked; they are not part of `%s` and nothing was committed after it.'%FULL[:9])
L.append('- No training worker or helper process of this task is running. After the analysis all 694 files of the artifact manifest (every file of the three run directories) were re-hashed: 0 missing, 0 changed (`manifest_reverify.json`).\n')
L.append('## 1. Comparison table\n')
L.append(tableA.replace('## Table A — one row per run','### Table A — one row per run').strip()+'\n')
L.append('Tables B (budget and compute) and C (every eval point) and the caveats C1-C4 are in `comparison_tables.md`; per-case rows are in `comparison.json`. Caveat C1 matters for the "success time" column: the recorded `success_seconds` is the duration of the final successful skill, not the episode time to success; the latter is not recorded (UNVERIFIED).\n')
L.append('## 2. Final checkpoint identity (all three) and DK-1 repeated returns\n')
L.append('| run | final checkpoint | sha256 (recomputed now) | equals manifest | manifest N/T/attempt id == job_summary | fresh-load names the final generation | eval_final evaluated this file | persistence copy sha-equal | weights/Adam equal `update_fragment_1` |\n|---|---|---|---|---|---|---|---|---|')
ffr=json.load(open(ANA+'/final_vs_fragment.json'))
for p,c in cid.items():
    ck=c['checks']
    L.append('| %s | %s | `%s` | %s | %s | %s | %s | %s | %s |'%(p,c['final_generation']+'.pt',c['final_pt_sha256_recomputed_now'],ck['sha_now_equals_manifest'],all(ck[k] for k in('ckpt_manifest_N_equals_job_summary','ckpt_manifest_T_equals_job_summary','ckpt_manifest_attempt_id_equals_run')),ck['fresh_load_generation_equals_final'],ck['eval_final_ckpt_is_final_pt'],all(v['sha_equal'] for v in c['persistence_vs_checkpoints_dir'].values()),ffr[p]['final_vs_update_fragment_1']==[0.0,0] or ffr[p]['final_vs_update_fragment_1']==(0.0,0)))
L.append('\nLimit of the fresh-load evidence: the `fresh_load_final_*.json` file records only the generation name and booleans (adam, model, fresh_process), no hash; it is bound to the file by name, by being written after the `.pt` (mtime), and by the generation directory, not by content hash.\n')
L.append('DK-1: the three post-update evals (N = 4096, 8192, 14170) used three different checkpoints (distinct sha256; 66 of 141 float tensors differ, max abs diff 4.6e-02 to 1.2e-01) and returned identical per-case numbers (G, steps, reason, success_seconds). The eval rows store no action sequences, so whether the trajectories are identical is **UNVERIFIED**. Details: `dk1_repeated_eval_check.md`.\n')
L.append('## 3. Disk below 2 GiB with the guard not pausing\n')
L.append('Recorded as an execution deviation, with budget estimate (2051 MiB per run, admission check), actual pause threshold (guard stop below 1.0 GB, which never fired; minimum sampled 1.665 GB at 2026-10-02T00:16:17Z, six 30-s samples below 2 GiB) and the 2 GiB safety margin kept apart. File integrity: all log lines and JSON files parse, transition lines equal N, all checkpoints load, persistence copies agree, no write-error text in logs. No retrain, no overwrite of old logs. Details: `deviation_record_disk_below_2GiB.md`.\n')
L.append('## 4. Registration and elastic accounting\n')
L.append('- **NONTRIVIAL_LEARNING registered for R-TB-E-0 (B1-K+E, seed 0)** under Experimental Plan v3 §3.1: trigger = the final post-update dev10 file `eval_final.json` (10/10 TASK_SUCCESS; the 4096 and 8192 post-update points were 0/10; step 0 is pre-update and not counted), bound to `final_n_014512_u14.pt` (sha256 `%s`, weights and Adam state equal to `update_fragment_1.pt`), the fresh-load file and job_summary by path and sha256. Evaluator independence is established at the schema level only (adapters.EvaluationInput carries no prior, graph or nominal facts); the Evaluator implementation was not re-audited. File: `registration_NONTRIVIAL_LEARNING_R-TB-E-0.json`.'%cid['R-TB-E-0']['final_pt_sha256_recomputed_now'])
L.append('- **Elastic pool**: 4 attempts (ELASTIC-01..04, Plan v3 §7.2). No allocation record or unified ledger exists in this worktree or the 15 sibling worktrees; every ledger found records elastic used 0 (S0 manifest, five S1 ledgers, this card; the three T_B runs are base runs). Remaining by recorded evidence: **4 of 4**; check against an external Master Ledger: **UNVERIFIED**.')
L.append('- **Request** (not an authorization; nothing started, no token, no ledger change): one run, R-TB-E-1 = B1-K+E seed 1 with the same profile, taking ELASTIC-01 (remaining would be 3). Found in the code: `PLAN_TABLE` is hard-coded to three plans and `MAX_NEW_RL_ATTEMPTS = 3`, so a 4th run needs a launch-entry change, a new prep commit, new registration and a smoke decision (cumulative smoke budget is used up). Options and open decisions are in `authorization_request_R-TB-E-1_ELASTIC-01.md`.\n')
L.append('## 5. Conclusion, limited to what the records support\n')
L.append('- On this final dev10 (10 cases, deterministic argmax, one run per method, E-0 seed 0, DK-1 and K-1 seed 1): **+E (R-TB-E-0) 10/10, B2 (R-TB-DK-1) 10/10, B1-K (R-TB-K-1) 0/10.**')
L.append('- It is **not shown that B2 is better than +E, and not shown that the two are equivalent**: both are at the 10/10 ceiling on 10 cases, with different seeds, and the observed differences in mean discounted return (0.4893 vs 0.4988) and recorded final-skill duration (2.915 vs 3.54 s) come from one run each and are not tested.')
L.append('- Method 2.1.1 is unchanged; Family B and RoboCasa engineering branches were not resumed.\n')
L.append('## 6. UNVERIFIED / NOT_MEASURED\n')
L.append('- Episode time to success (not recorded; `success_seconds` is the final skill\'s duration).\n- Whether DK-1\'s three post-update evals took identical trajectories (no action sequences stored).\n- Evaluator independence beyond the input schema.\n- Elastic usage against an external Master Ledger.\n- True minimum free space and exact duration between 30-s samples; per-process attribution of the free-space drop; filesystem quota.\n- Fresh-load evidence has no content hash.\n- Full repository test suite (not run in this card).\n')
L.append('## 7. File index (`zero_sample_analysis/`)\n')
files=sorted(os.listdir(ANA))
for f in files:
    if f=='post_run_analysis_report.md': continue
    L.append('- `%s` — sha256 `%s`'%(f,hashlib.sha256(open(ANA+'/'+f,'rb').read()).hexdigest()))
open(ANA+'/post_run_analysis_report.md','w').write('\n'.join(L))
print('\n'.join(L)[:1800])

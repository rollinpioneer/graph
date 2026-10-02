import json,os,glob,csv
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip(); ANA=OUT+'/zero_sample_analysis'
res=json.load(open(ANA+'/comparison.json'))
order=['R-TB-E-0','R-TB-DK-1','R-TB-K-1']
by={r['plan']:r for r in res}
def f(x,nd=4): return 'NA' if x is None else (('%.'+str(nd)+'f')%x)
L=[]
L.append('## Table A — one row per run (existing evaluations only; no episode was run for this table)\n')
L.append('Eval points are the pre-registered 0 / 4096 / 8192 / final (Plan v3 §7.1). Success is n of 10 dev10 cases, deterministic argmax. "Frozen discounted return" = recorded `mean_discounted_return` of the eval file (frozen H = 23.1 s). "Final-skill duration" = the recorded `success_seconds` field (see caveat C1); NA = no successful case.\n')
L.append('| run | success n/10 at 0 / 4096 / 8192 / final | mean discounted return at 0 / 4096 / 8192 / final | recorded success_seconds (mean over successes) at 0 / 4096 / 8192 / final |')
L.append('|---|---|---|---|')
for p in order:
    r=by[p]; pts=r['eval_points']
    L.append('| %s %s s%d | %s | %s | %s |'%(r['plan'],r['method'],r['seed'],' / '.join(str(x['success_n']) for x in pts),' / '.join(f(x['mean_discounted_return']) for x in pts),' / '.join(f(x['mean_success_seconds'],3) for x in pts)))
L.append('\n## Table B — actual budget and compute (job_summary / launch_state; training only)\n')
L.append('| run | N actual (Ncap 16384) | T actual s (Tcap 68812.8) | stop | complete + fragment updates | optimizer steps | train success episodes | wall h (start -> finish UTC) | PPO min (sum, complete updates) | collection min (sum, between updates) | first-update profile (ppo s / transitions per wall s) |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|')
for p in order:
    r=by[p]; t=r['train']; c=r['compute']; pr=c.get('profiling_summary_first_update',{})
    L.append('| %s %s s%d | %d | %.2f | %s | %d + %d | %d | %d | %.2f (%s -> %s) | %.1f | %.1f | %.0f / %.3f |'%(r['plan'],r['method'],r['seed'],t['valid_transitions_N'],t['interaction_seconds_T'],t['stop_reason'],t['complete_updates'],t['fragment_updates'],t['optimizer_steps'],t['train_success_episodes'],c['wall_hours'],c['started_utc'][5:16],c['finished_utc'][5:16],c['sum_complete_update_ppo_minutes'],c['sum_collection_minutes_between_updates'],pr.get('ppo_forward_backward_optimizer_seconds',float('nan')),pr.get('transitions_per_wall_second',float('nan'))))
L.append('\nAll three: NaN_n 0, hard_fail null, HEAD cac2fa5b3. Compute columns are wall clock on a shared 112-core server with shared GPUs and 1-3 concurrent workers (caveat C2).\n')
L.append('## Table C — every eval point (from `eval_n_*.json` / `eval_final.json`)\n')
L.append('| run | eval file | update / N | success | mean disc. return | recorded success_seconds mean | failure reasons | checkpoint (sha256 first 12) |')
L.append('|---|---|---|---|---|---|---|---|')
for p in order:
    r=by[p]
    for x in r['eval_points']:
        L.append('| %s | %s | %s / %s | %d/%d | %s | %s | %s | %s (%s) |'%(r['plan'],x['file'],x['update'],x['skill_transitions'],x['success_n'],x['n'],f(x['mean_discounted_return'],6),f(x['mean_success_seconds'],3),json.dumps(x['reasons'],ensure_ascii=False).replace('|','/'),x['checkpoint'],(x['checkpoint_sha256'] or 'NA')[:12]))
L.append('''
### Caveats
- **C1 `success_seconds`.** In `src/cp_disr/stage2a_v11.py` line 1044 the field is `snapshot.elapsed_seconds + t.duration` if the snapshot has `elapsed_seconds`, else `t.duration`. `rl.Snapshot` has `clock_seconds` and no `elapsed_seconds`, so the recorded value is the duration of the final (successful) skill only (it equals single-skill durations such as 2.90 s / 3.55 s). Total episode time to success is NOT recorded in the eval rows -> UNVERIFIED; it is not reconstructed here. The discounted return G is recorded and frozen (H = 23.1 s) and is the time-sensitive quantity.
- **C2 compute.** Not a controlled compute-cost measurement: concurrency differed (E-0 alone ~38 min, then 2, then 3 workers until E-0 finished at 20:55Z; DK-1 ran with 1-3 workers) and GPU 0 hosted other users' jobs.
- **C3 scope.** dev10 only (10 cases, seed per run as registered: E-0 seed 0, DK-1 seed 1, K-1 seed 1), the registered final point is the primary endpoint; no best-dev checkpoint substitutes for it. The job_summary `selected_checkpoint` (E-0 final, DK-1 final, K-1 n_000000 by the max_mean3 -> max_worst -> earlier-N rule) is listed in the JSON only and is not used here.
- **C4 failure reasons** are copied from the eval rows: all zero-success points of E-0 (0/4096) and K-1 (all four) end with `INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE`; E-0 at 8192 has 7 of that and 3 `NO_CANDIDATE_SAFE_TERMINATION`. No mechanism is inferred.
''')
open(ANA+'/comparison_tables.md','w').write('\n'.join(L))
print('\n'.join(L))
